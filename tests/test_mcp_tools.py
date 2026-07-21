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


def _video_catalog(model="seedance-2-0-text-to-video"):
    """A /models?type=video catalog whose first id carries the 'default' trait."""
    return json.dumps(
        {"data": [{"id": model, "model_spec": {"traits": ["default"]}}]}
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


class TestVideoTool(_ToolTest):
    def test_ok_queue_poll_save(self):
        responses = _seq(
            FakeResp(200, _video_catalog()),               # /models?type=video
            FakeResp(200, b'{"quote": 0.5}'),              # /video/quote
            FakeResp(200, b'{"queue_id": "vidq123"}'),     # /video/queue
            FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()),
            FakeResp(200, b"MP4BYTES", "video/mp4"),       # /video/retrieve
            FakeResp(200, b'{"ok": true}'),                # /video/complete
        )
        with mock.patch("venice.client.urllib.request.urlopen", responses), \
             mock.patch("venice.client.time.sleep"), self.stdout_guard():
            res = _mcp.video_tool(
                _client(), "a koi pond at dawn",
                output_dir=self.td, confirm=True, max_wait=10,
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["queue_id"], "vidq123")
        self.assertEqual(Path(res["path"]).read_bytes(), b"MP4BYTES")
        self.assertTrue(res["path"].endswith(".mp4"))
        self.assertEqual(res["cost_estimate_usd"], 0.5)

    def test_vps_model_fetches_download_url(self):
        url = "https://cdn.example.com/presigned?sig=abc"
        responses = _seq(
            FakeResp(200, _video_catalog()),
            FakeResp(200, b'{"quote": 0.5}'),
            FakeResp(200, json.dumps({"queue_id": "vpsq1", "download_url": url}).encode()),
            FakeResp(200, json.dumps({"status": "COMPLETED"}).encode()),
            FakeResp(200, b"PRESIGNEDMP4", "video/mp4"),   # get_url_bytes
            FakeResp(200, b'{"ok": true}'),
        )
        seen = []

        def fake(req, timeout=None):
            seen.append(req.full_url)
            return responses(req, timeout)

        with mock.patch("venice.client.urllib.request.urlopen", fake), \
             mock.patch("venice.client.time.sleep"), self.stdout_guard():
            res = _mcp.video_tool(
                _client(), "a koi pond", output_dir=self.td, confirm=True, max_wait=10
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"PRESIGNEDMP4")
        self.assertIn(url, seen)  # presigned URL fetched verbatim

    def test_gate_blocks_before_queue(self):
        seen = []

        def fake(req, timeout=None):
            seen.append(req.full_url)
            if "type=video" in req.full_url:
                return FakeResp(200, _video_catalog())
            if req.full_url.endswith("/video/quote"):
                return FakeResp(200, b'{"quote": 2.50}')
            raise AssertionError(f"unexpected call past the gate: {req.full_url}")

        with mock.patch("venice.client.urllib.request.urlopen", fake):
            res = _mcp.video_tool(
                _client(), "a koi pond",
                output_dir=self.td, confirm=False, max_spend=0.10,
            )
        self.assertEqual(res["status"], "confirmation_required")
        self.assertTrue(all("/video/queue" not in u for u in seen))

    def test_empty_prompt_is_error(self):
        res = _mcp.video_tool(_client(), "   ", confirm=True)
        self.assertEqual(res["status"], "error")


class TestImageEditTool(_ToolTest):
    def _png(self):
        ip = Path(self.td) / "base.png"
        ip.write_bytes(b"\x89PNG\r\n" + b"x" * 16)
        return ip

    def test_needs_confirm(self):
        ip = self._png()

        def boom(*a, **kw):
            raise AssertionError("must not hit the network without confirm")

        with mock.patch("venice.client.urllib.request.urlopen", boom):
            res = _mcp.image_edit_tool(
                _client(), "make it blue", input_path=str(ip),
                output_dir=self.td, confirm=False,
            )
        self.assertEqual(res["status"], "confirmation_required")
        self.assertIsNone(res["estimated_cost_usd"])

    def test_edit_ok_with_confirm(self):
        ip = self._png()
        captured = {}

        def fake(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            return FakeResp(200, b"EDITED", "image/png")

        with mock.patch("venice.client.urllib.request.urlopen", fake), self.stdout_guard():
            res = _mcp.image_edit_tool(
                _client(), "make it blue", input_path=str(ip),
                output_dir=self.td, confirm=True,
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"EDITED")
        self.assertTrue(captured["url"].endswith("/image/edit"))
        self.assertIn("image", captured["body"])  # single-edit sends "image"
        self.assertEqual(base64.b64decode(captured["body"]["image"]), ip.read_bytes())

    def test_multi_edit_with_layers(self):
        ip = self._png()
        layer = Path(self.td) / "mask.png"
        layer.write_bytes(b"\x89PNG\r\nMASK")
        captured = {}

        def fake(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            return FakeResp(200, b"MULTI", "image/png")

        with mock.patch("venice.client.urllib.request.urlopen", fake), self.stdout_guard():
            res = _mcp.image_edit_tool(
                _client(), "composite the mask", input_path=str(ip),
                layer_paths=[str(layer)], output_dir=self.td, confirm=True,
            )
        self.assertEqual(res["status"], "ok")
        self.assertTrue(captured["url"].endswith("/image/multi-edit"))
        self.assertEqual(len(captured["body"]["images"]), 2)  # base + 1 layer

    def test_missing_source_is_error(self):
        res = _mcp.image_edit_tool(_client(), "edit", confirm=True)  # no input/url
        self.assertEqual(res["status"], "error")


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


class TestModelsTool(_ToolTest):
    def _catalog(self, *ids):
        return json.dumps({"data": [{"id": i} for i in ids]}).encode()

    def test_lists_ids_for_one_type(self):
        with mock.patch("venice.client.urllib.request.urlopen",
                        _seq(FakeResp(200, self._catalog("m-a", "m-b")))), \
                self.stdout_guard():
            out = _mcp.models_tool(_client(), type="image")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["type"], "image")
        self.assertEqual(out["models"], ["m-a", "m-b"])
        self.assertEqual(out["count"], 2)

    def test_all_returns_map_keyed_by_type(self):
        from venice.commands.models import MODEL_TYPES
        resps = _seq(*[FakeResp(200, self._catalog(f"{t}-1")) for t in MODEL_TYPES])
        with mock.patch("venice.client.urllib.request.urlopen", resps), \
                self.stdout_guard():
            out = _mcp.models_tool(_client(), type="all")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(set(out["models"]), set(MODEL_TYPES))
        self.assertEqual(out["count"], len(MODEL_TYPES))

    def test_unknown_type_errors_without_http(self):
        def boom(*a, **kw):
            raise AssertionError("a bad type must not trigger an HTTP call")
        with mock.patch("venice.client.urllib.request.urlopen", boom), \
                self.stdout_guard():
            out = _mcp.models_tool(_client(), type="bogus")
        self.assertEqual(out["status"], "error")
        self.assertIn("unknown type", out["message"])


class TestModelDetailsTool(_ToolTest):
    def test_returns_curated_details(self):
        # `text` is first in MODEL_TYPES, so _find_model matches on the first GET.
        spec = {"name": "Big", "pricing": {"input": {"usd": 1.5}},
                "availableContextTokens": 131072, "promptCharacterLimit": 1500,
                "capabilities": {"supportsFunctionCalling": True},
                "traits": ["default"]}
        catalog = json.dumps(
            {"data": [{"id": "big-model", "type": "text", "model_spec": spec}]}
        ).encode()
        with mock.patch("venice.client.urllib.request.urlopen",
                        _seq(FakeResp(200, catalog))), self.stdout_guard():
            out = _mcp.model_details_tool(_client(), model="big-model")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["id"], "big-model")
        self.assertEqual(out["available_context_tokens"], 131072)
        self.assertEqual(out["prompt_character_limit"], 1500)
        self.assertEqual(out["pricing"], {"input": {"usd": 1.5}})
        self.assertTrue(out["capabilities"]["supportsFunctionCalling"])

    def test_unknown_model_errors(self):
        from venice.commands.models import MODEL_TYPES
        empty = json.dumps({"data": []}).encode()
        resps = _seq(*[FakeResp(200, empty) for _ in MODEL_TYPES])
        with mock.patch("venice.client.urllib.request.urlopen", resps), \
                self.stdout_guard():
            out = _mcp.model_details_tool(_client(), model="nope")
        self.assertEqual(out["status"], "error")
        self.assertIn("no model", out["message"])


if __name__ == "__main__":
    unittest.main()
