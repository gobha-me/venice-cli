"""Unit tests for the MCP tool implementations (`venice.commands._mcp`).

These import only `_mcp` (which never imports the `mcp` SDK), so they run on every
supported Python version -- including 3.9, where the SDK can't be installed. HTTP is
mocked via the shared `FakeResp`; every tool call also asserts that **nothing leaks
to stdout**, since a real MCP stdio server owns stdout for JSON-RPC framing.
"""
import base64
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_client import FakeResp
from venice.client import VeniceClient
from venice.commands import _mcp


def _client():
    return VeniceClient(api_key="fake")


def _seq(*responses):
    """A urlopen replacement yielding the given FakeResps in order."""
    it = iter(responses)
    return lambda *a, **kw: next(it)


def _price_doc(model, usd):
    return json.dumps(
        {"data": [{"id": model, "model_spec": {"pricing": {"output": {"usd": usd}}}}]}
    ).encode()


class _ToolTest(unittest.TestCase):
    """Base with a persistent temp output dir (survives until tearDown) and a
    stdout-empty guard usable around any tool call."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.td, ignore_errors=True)

    def stdout_guard(self):
        test = self

        class _G:
            def __enter__(self_):
                self_.buf = io.StringIO()
                self_.p = mock.patch.object(sys, "stdout", self_.buf)
                self_.p.start()
                return self_

            def __exit__(self_, *a):
                self_.p.stop()
                test.assertEqual(self_.buf.getvalue(), "", "tool wrote to stdout!")
                return False

        return _G()


class TestImageTool(_ToolTest):
    def test_ok_writes_file_and_returns_path(self):
        png = b"\x89PNG\r\n\x1a\nHELLO"
        gen = json.dumps({"images": [base64.b64encode(png).decode()]}).encode()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _price_doc("venice-sd35", 0.01)), FakeResp(200, gen)),
        ), self.stdout_guard():
            res = _mcp.image_tool(_client(), "a dragon", output_dir=self.td, confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["cost_estimate_usd"], 0.01)
        self.assertTrue(os.path.isfile(res["paths"][0]))
        self.assertEqual(Path(res["paths"][0]).read_bytes(), png)

    def test_multiple_variants_write_multiple_files(self):
        imgs = [base64.b64encode(b"A").decode(), base64.b64encode(b"BB").decode()]
        gen = json.dumps({"images": imgs}).encode()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _price_doc("venice-sd35", 0.001)), FakeResp(200, gen)),
        ):
            res = _mcp.image_tool(_client(), "x", variants=2, output_dir=self.td, confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["paths"]), 2)
        for p in res["paths"]:
            self.assertTrue(os.path.isfile(p))

    def test_spend_gate_over_cap_blocks_before_generate(self):
        calls = {"n": 0}

        def counting(*a, **kw):
            calls["n"] += 1
            return FakeResp(200, _price_doc("venice-sd35", 5.0))

        with mock.patch("venice.client.urllib.request.urlopen", counting):
            res = _mcp.image_tool(
                _client(), "x", output_dir=self.td, confirm=False, max_spend=0.10
            )
        self.assertEqual(res["status"], "confirmation_required")
        self.assertEqual(res["estimated_cost_usd"], 5.0)
        self.assertEqual(res["max_spend_usd"], 0.10)
        # Only the (free) price GET happened -- no paid /image/generate.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(list(Path(self.td).iterdir()), [])

    def test_under_cap_auto_approves_without_confirm(self):
        gen = json.dumps({"images": [base64.b64encode(b"IMG").decode()]}).encode()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _price_doc("venice-sd35", 0.01)), FakeResp(200, gen)),
        ):
            res = _mcp.image_tool(
                _client(), "x", output_dir=self.td, confirm=False, max_spend=1.00
            )
        self.assertEqual(res["status"], "ok")

    def test_invalid_variants_is_error_without_http(self):
        def boom(*a, **kw):
            raise AssertionError("must not hit the network")

        with mock.patch("venice.client.urllib.request.urlopen", boom):
            res = _mcp.image_tool(_client(), "x", variants=9, confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("variants", res["message"])

    def test_api_error_becomes_error_dict(self):
        from urllib.error import HTTPError

        def boom(*a, **kw):
            raise HTTPError("u", 500, "err", {"Content-Type": "application/json"},
                            io.BytesIO(b'{"code":"X"}'))

        # price fetch swallows errors (-> None -> gate needs confirm), so pass
        # confirm=True to reach the generate call which then 500s.
        with mock.patch("venice.client.urllib.request.urlopen", boom):
            res = _mcp.image_tool(_client(), "x", confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("image failed", res["message"])


class TestTtsTool(_ToolTest):
    def test_ok_writes_audio(self):
        price = json.dumps(
            {"data": [{"id": "tts-kokoro",
                       "model_spec": {"pricing": {"input": {"usd": 3.5}}}}]}
        ).encode()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, price), FakeResp(200, b"AUDIO", "audio/mpeg")),
        ), self.stdout_guard():
            res = _mcp.tts_tool(_client(), "hello world", output_dir=self.td, confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"AUDIO")
        self.assertTrue(res["path"].endswith(".mp3"))

    def test_bad_speed_is_error(self):
        res = _mcp.tts_tool(_client(), "hi", speed=9.0, confirm=True)
        self.assertEqual(res["status"], "error")
        self.assertIn("speed", res["message"])


class TestSfxTool(_ToolTest):
    def test_ok_queue_poll_save(self):
        responses = _seq(
            FakeResp(200, b'{"quote": 0.0027}'),
            FakeResp(200, b'{"queue_id": "abcdef1234", "status": "QUEUED"}'),
            FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()),
            FakeResp(200, b"SFXBYTES", "audio/mpeg"),
            FakeResp(200, b'{"ok": true}'),  # /audio/complete cleanup
        )
        with mock.patch("venice.client.urllib.request.urlopen", responses), \
             mock.patch("venice.client.time.sleep"), self.stdout_guard():
            res = _mcp.sfx_tool(
                _client(), "thunder", output_dir=self.td, confirm=True, max_wait=10
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["queue_id"], "abcdef1234")
        self.assertEqual(Path(res["path"]).read_bytes(), b"SFXBYTES")

    def test_gate_blocks_before_queue(self):
        calls = {"n": 0}

        def counting(*a, **kw):
            calls["n"] += 1
            return FakeResp(200, b'{"quote": 2.50}')

        with mock.patch("venice.client.urllib.request.urlopen", counting):
            res = _mcp.sfx_tool(
                _client(), "thunder", output_dir=self.td, confirm=False, max_spend=0.10
            )
        self.assertEqual(res["status"], "confirmation_required")
        self.assertEqual(calls["n"], 1)  # only the quote, no queue

    def test_unknown_model_is_error(self):
        res = _mcp.sfx_tool(_client(), "x", model="nope", confirm=True)
        self.assertEqual(res["status"], "error")


class TestMusicTool(_ToolTest):
    def test_ok_queue_poll_save(self):
        music_spec = json.dumps(
            {"data": [{"id": "elevenlabs-music", "model_spec": {}}]}
        ).encode()
        responses = _seq(
            FakeResp(200, music_spec),                       # fetch_music_spec
            FakeResp(200, b'{"quote": 0.05}'),               # /audio/quote
            FakeResp(200, b'{"queue_id": "musicq123"}'),     # /audio/queue
            FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()),
            FakeResp(200, b"MUSICBYTES", "audio/mpeg"),
            FakeResp(200, b'{"ok": true}'),                  # /audio/complete
        )
        with mock.patch("venice.client.urllib.request.urlopen", responses), \
             mock.patch("venice.client.time.sleep"):
            res = _mcp.music_tool(
                _client(), "dungeon drone", output_dir=self.td, confirm=True, max_wait=10
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"MUSICBYTES")


class TestBinaryTools(_ToolTest):
    def _png(self):
        ip = Path(self.td) / "in.png"
        ip.write_bytes(b"\x89PNG\r\n" + b"x" * 32)
        return ip

    def test_upscale_needs_confirm(self):
        ip = self._png()

        def boom(*a, **kw):
            raise AssertionError("must not hit the network without confirm")

        with mock.patch("venice.client.urllib.request.urlopen", boom):
            res = _mcp.upscale_tool(_client(), str(ip), output_dir=self.td, confirm=False)
        self.assertEqual(res["status"], "confirmation_required")
        self.assertIsNone(res["estimated_cost_usd"])

    def test_upscale_ok_with_confirm(self):
        ip = self._png()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, b"UPSCALED", "image/png")),
        ), self.stdout_guard():
            res = _mcp.upscale_tool(_client(), str(ip), output_dir=self.td, confirm=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"UPSCALED")
        self.assertTrue(res["path"].endswith("-upscaled.png"))

    def test_bg_remove_from_url_ok(self):
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, b"NOBG", "image/png")),
        ):
            res = _mcp.bg_remove_tool(
                _client(), image_url="https://x/y.png", output_dir=self.td, confirm=True
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"NOBG")

    def test_bg_remove_requires_one_source(self):
        res = _mcp.bg_remove_tool(_client(), confirm=True)  # neither file nor url
        self.assertEqual(res["status"], "error")


class TestChatTool(_ToolTest):
    def test_missing_openai_returns_error(self):
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(sys, "stderr", err):
            res = _mcp.chat_tool(_client(), "hi")
        self.assertEqual(res["status"], "error")
        self.assertIn("openai", res["message"])

    def test_ok_returns_content(self):
        catalog = json.dumps(
            {"data": [{"id": "venice-uncensored", "model_spec": {"traits": ["default"]}}]}
        ).encode()
        msg = mock.Mock()
        msg.content = "hi there"
        choice = mock.Mock()
        choice.message = msg
        resp = mock.Mock()
        resp.choices = [choice]
        resp.usage = None
        oai = mock.Mock()
        oai.chat.completions.create.return_value = resp

        with mock.patch(
            "venice.client.urllib.request.urlopen", _seq(FakeResp(200, catalog))
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.chat_tool(_client(), "hello", model="venice-uncensored")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["content"], "hi there")
        self.assertEqual(res["model"], "venice-uncensored")


class TestSpendHelpers(unittest.TestCase):
    def test_resolve_max_spend_precedence(self):
        self.assertEqual(_mcp.resolve_max_spend(0.5), 0.5)  # explicit wins
        with mock.patch.dict(os.environ, {"VENICE_MCP_MAX_SPEND": "0.25"}):
            self.assertEqual(_mcp.resolve_max_spend(None), 0.25)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_mcp.resolve_max_spend(None), _mcp.DEFAULT_MCP_MAX_SPEND)

    def test_check_spend_paths(self):
        self.assertIsNone(_mcp.check_spend(0.05, confirm=False, max_spend=0.10, label="x"))
        self.assertIsNone(_mcp.check_spend(9.9, confirm=True, max_spend=0.10, label="x"))
        gate = _mcp.check_spend(0.5, confirm=False, max_spend=0.10, label="x")
        self.assertEqual(gate["status"], "confirmation_required")
        gate2 = _mcp.check_spend(None, confirm=False, max_spend=0.10, label="x")
        self.assertEqual(gate2["status"], "confirmation_required")  # unknown -> confirm

    def test_output_dir_env_default(self):
        with mock.patch.dict(os.environ, {"VENICE_MCP_OUTPUT_DIR": "/tmp/venice-x"}):
            self.assertEqual(_mcp.resolve_output_dir(None), Path("/tmp/venice-x"))
        self.assertEqual(_mcp.resolve_output_dir("/explicit"), Path("/explicit"))


class TestRetrieveBytesParity(unittest.TestCase):
    def test_returns_ctype_and_bytes(self):
        from venice.commands import _audio

        responses = _seq(
            FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()),
            FakeResp(200, b"DONEBYTES", "audio/mpeg"),
        )
        with mock.patch("venice.client.urllib.request.urlopen", responses), \
             mock.patch("venice.client.time.sleep"):
            ctype, data = _audio.retrieve_bytes(
                _client(), "m", "q", poll_interval=0, max_wait=10
            )
        self.assertEqual(ctype, "audio/mpeg")
        self.assertEqual(data, b"DONEBYTES")

    def test_timeout_raises(self):
        from venice.commands import _audio

        with mock.patch(
            "venice.client.urllib.request.urlopen",
            lambda *a, **kw: FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()),
        ), mock.patch("venice.client.time.sleep"):
            with self.assertRaises(TimeoutError):
                _audio.retrieve_bytes(_client(), "m", "q", poll_interval=0, max_wait=-1)


if __name__ == "__main__":
    unittest.main()
