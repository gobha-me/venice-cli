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
from urllib.error import HTTPError

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

    def test_background_returns_queue_id(self):
        # #62: background=True queues (charging up front) then returns a handle
        # WITHOUT polling/retrieving or writing a file.
        seen = []

        def fake(req, timeout=None):
            seen.append(req.full_url)
            if req.full_url.endswith("/audio/quote"):
                return FakeResp(200, b'{"quote": 0.0027}')
            if req.full_url.endswith("/audio/queue"):
                return FakeResp(200, b'{"queue_id": "bgsfx123"}')
            raise AssertionError(f"background must not poll: {req.full_url}")

        with mock.patch("venice.client.urllib.request.urlopen", fake), self.stdout_guard():
            res = _mcp.sfx_tool(
                _client(), "thunder", output_dir=self.td, confirm=True, background=True
            )
        self.assertEqual(res["status"], "queued")
        self.assertEqual(res["queue_id"], "bgsfx123")
        self.assertEqual(res["type"], "sfx")
        self.assertEqual(res["model"], _mcp._sfx.DEFAULT_SFX_MODEL)
        self.assertEqual(res["cost_estimate_usd"], 0.0027)
        self.assertEqual(os.listdir(self.td), [])            # nothing written
        self.assertTrue(all("/audio/retrieve" not in u for u in seen))


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

    def test_background_returns_queue_id(self):
        def fake(req, timeout=None):
            u = req.full_url
            if "type=music" in u:                            # fetch_music_spec
                return FakeResp(200, json.dumps(
                    {"data": [{"id": "elevenlabs-music", "model_spec": {}}]}).encode())
            if u.endswith("/audio/quote"):
                return FakeResp(200, b'{"quote": 0.05}')
            if u.endswith("/audio/queue"):
                return FakeResp(200, b'{"queue_id": "bgmus123"}')
            raise AssertionError(f"background must not poll: {u}")

        with mock.patch("venice.client.urllib.request.urlopen", fake), self.stdout_guard():
            res = _mcp.music_tool(
                _client(), "dungeon drone", output_dir=self.td, confirm=True, background=True
            )
        self.assertEqual(res["status"], "queued")
        self.assertEqual(res["queue_id"], "bgmus123")
        self.assertEqual(res["type"], "music")
        self.assertEqual(os.listdir(self.td), [])


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


def _vision_catalog(default_vision=False):
    """A /models?type=text catalog: a default-trait model (vision per the flag)
    and a non-default vision-capable one."""
    return json.dumps({"data": [
        {"id": "venice-uncensored", "model_spec": {
            "traits": ["default"],
            "capabilities": {"supportsVision": default_vision},
        }},
        {"id": "qwen-vl", "model_spec": {
            "traits": [],
            "capabilities": {"supportsVision": True},
        }},
    ]}).encode()


class TestVisionTool(_ToolTest):
    def _png(self):
        ip = Path(self.td) / "shot.png"
        ip.write_bytes(b"\x89PNG\r\n" + b"x" * 16)
        return ip

    def _oai(self, content="a red square"):
        msg = mock.Mock()
        msg.content = content
        choice = mock.Mock()
        choice.message = msg
        resp = mock.Mock()
        resp.choices = [choice]
        resp.usage = None
        oai = mock.Mock()
        oai.chat.completions.create.return_value = resp
        return oai

    def test_missing_openai_returns_error(self):
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(sys, "stderr", err):
            res = _mcp.vision_tool(_client(), str(self._png()))
        self.assertEqual(res["status"], "error")
        self.assertIn("openai", res["message"])

    def test_requires_exactly_one_source(self):
        def boom(*a, **kw):
            raise AssertionError("must not hit the network on bad args")

        with mock.patch("venice.client.urllib.request.urlopen", boom), \
             self.stdout_guard():
            neither = _mcp.vision_tool(_client())
            both = _mcp.vision_tool(
                _client(), str(self._png()), image_url="https://x/y.png")
        self.assertEqual(neither["status"], "error")
        self.assertEqual(both["status"], "error")

    def test_missing_input_file_errors(self):
        err = io.StringIO()
        oai = self._oai()
        with mock.patch("openai.OpenAI", return_value=oai), \
             mock.patch.object(sys, "stderr", err), self.stdout_guard():
            res = _mcp.vision_tool(_client(), str(Path(self.td) / "nope.png"))
        self.assertEqual(res["status"], "error")
        oai.chat.completions.create.assert_not_called()

    def test_ok_local_image_sends_data_url(self):
        oai = self._oai()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _vision_catalog())),
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(_client(), str(self._png()), model="qwen-vl")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["content"], "a red square")
        self.assertEqual(res["model"], "qwen-vl")
        kwargs = oai.chat.completions.create.call_args.kwargs
        parts = kwargs["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": _mcp.DEFAULT_VISION_PROMPT})
        self.assertTrue(
            parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("max_tokens", kwargs)

    def test_ok_image_url_and_custom_prompt(self):
        oai = self._oai("blue")
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _vision_catalog())),
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(
                _client(), image_url="https://x/y.png",
                prompt="What color?", model="qwen-vl", max_tokens=64,
            )
        self.assertEqual(res["status"], "ok")
        kwargs = oai.chat.completions.create.call_args.kwargs
        parts = kwargs["messages"][0]["content"]
        self.assertEqual(parts[0]["text"], "What color?")
        self.assertEqual(parts[1]["image_url"]["url"], "https://x/y.png")
        self.assertEqual(kwargs["max_tokens"], 64)

    def test_explicit_non_vision_model_errors(self):
        oai = self._oai()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _vision_catalog())),
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(
                _client(), str(self._png()), model="venice-uncensored")
        self.assertEqual(res["status"], "error")
        self.assertIn("supportsVision", res["message"])
        oai.chat.completions.create.assert_not_called()

    def test_unknown_capabilities_proceeds(self):
        # No capabilities field on the model spec -> tri-state None -> attempt.
        catalog = json.dumps(
            {"data": [{"id": "mystery", "model_spec": {"traits": []}}]}
        ).encode()
        oai = self._oai()
        with mock.patch(
            "venice.client.urllib.request.urlopen", _seq(FakeResp(200, catalog))
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(_client(), str(self._png()), model="mystery")
        self.assertEqual(res["status"], "ok")

    def test_default_skips_non_vision_default_trait(self):
        oai = self._oai()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _vision_catalog(default_vision=False))),
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(_client(), str(self._png()))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["model"], "qwen-vl")

    def test_default_prefers_vision_capable_default_trait(self):
        oai = self._oai()
        with mock.patch(
            "venice.client.urllib.request.urlopen",
            _seq(FakeResp(200, _vision_catalog(default_vision=True))),
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(_client(), str(self._png()))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["model"], "venice-uncensored")

    def test_no_vision_model_in_catalog_errors(self):
        catalog = json.dumps({"data": [
            {"id": "venice-uncensored", "model_spec": {
                "traits": ["default"],
                "capabilities": {"supportsVision": False},
            }},
        ]}).encode()
        oai = self._oai()
        with mock.patch(
            "venice.client.urllib.request.urlopen", _seq(FakeResp(200, catalog))
        ), mock.patch("openai.OpenAI", return_value=oai), self.stdout_guard():
            res = _mcp.vision_tool(_client(), str(self._png()))
        self.assertEqual(res["status"], "error")
        self.assertIn("vision-capable", res["message"])
        oai.chat.completions.create.assert_not_called()


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

    def test_background_returns_queue_id_with_download_url(self):
        url = "https://cdn.example.com/presigned?sig=abc"

        def fake(req, timeout=None):
            u = req.full_url
            if "type=video" in u:
                return FakeResp(200, _video_catalog())
            if u.endswith("/video/quote"):
                return FakeResp(200, b'{"quote": 0.5}')
            if u.endswith("/video/queue"):
                return FakeResp(200, json.dumps(
                    {"queue_id": "bgvid1", "download_url": url}).encode())
            raise AssertionError(f"background must not poll: {u}")

        with mock.patch("venice.client.urllib.request.urlopen", fake), self.stdout_guard():
            res = _mcp.video_tool(
                _client(), "a koi pond", output_dir=self.td, confirm=True, background=True
            )
        self.assertEqual(res["status"], "queued")
        self.assertEqual(res["queue_id"], "bgvid1")
        self.assertEqual(res["type"], "video")
        self.assertEqual(res["download_url"], url)
        self.assertEqual(os.listdir(self.td), [])


class TestJobStatusTool(_ToolTest):
    def test_processing(self):
        responses = _seq(FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()))
        with mock.patch("venice.client.urllib.request.urlopen", responses), self.stdout_guard():
            res = _mcp.job_status_tool(
                _client(), queue_id="q1", type="sfx", model="elevenlabs-sound-effects-v2"
            )
        self.assertEqual(res["status"], "processing")
        self.assertEqual(res["queue_id"], "q1")

    def test_done_when_bytes_ready(self):
        responses = _seq(FakeResp(200, b"AUDIOBYTES", "audio/mpeg"))
        with mock.patch("venice.client.urllib.request.urlopen", responses), self.stdout_guard():
            res = _mcp.job_status_tool(
                _client(), queue_id="q2", type="music", model="elevenlabs-music"
            )
        self.assertEqual(res["status"], "done")
        self.assertTrue(res["ready"])
        self.assertEqual(res["bytes_available"], len(b"AUDIOBYTES"))
        # a status probe never writes a file
        self.assertEqual(os.listdir(self.td), [])

    def test_video_completed_is_done(self):
        responses = _seq(FakeResp(200, json.dumps({"status": "COMPLETED"}).encode()))
        with mock.patch("venice.client.urllib.request.urlopen", responses), self.stdout_guard():
            res = _mcp.job_status_tool(
                _client(), queue_id="q3", type="video", model="seedance-2-0-text-to-video"
            )
        self.assertEqual(res["status"], "done")

    def test_not_found_on_404(self):
        def boom(*a, **kw):
            raise HTTPError(
                url="https://api.venice.ai/api/v1/audio/retrieve", code=404,
                msg="Not Found", hdrs={"Content-Type": "application/json"},  # type: ignore[arg-type]
                fp=io.BytesIO(b'{"message": "no such job"}'),
            )

        with mock.patch("venice.client.urllib.request.urlopen", boom), self.stdout_guard():
            res = _mcp.job_status_tool(
                _client(), queue_id="gone", type="sfx", model="elevenlabs-sound-effects-v2"
            )
        self.assertEqual(res["status"], "not_found")

    def test_unknown_type_is_error(self):
        res = _mcp.job_status_tool(_client(), queue_id="q", type="bogus", model="m")
        self.assertEqual(res["status"], "error")


class TestJobResultTool(_ToolTest):
    def test_ok_writes_file_and_cleans_up(self):
        seen = []

        def fake(req, timeout=None):
            seen.append(req.full_url)
            if req.full_url.endswith("/audio/retrieve"):
                return FakeResp(200, b"SFXBYTES", "audio/mpeg")
            if req.full_url.endswith("/audio/complete"):
                return FakeResp(200, b'{"ok": true}')
            raise AssertionError(f"unexpected: {req.full_url}")

        with mock.patch("venice.client.urllib.request.urlopen", fake), \
             mock.patch.dict(os.environ, {"VENICE_MCP_OUTPUT_DIR": self.td}), \
             self.stdout_guard():
            res = _mcp.job_result_tool(
                _client(), queue_id="abcdef1234", type="sfx",
                model="elevenlabs-sound-effects-v2",
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["queue_id"], "abcdef1234")
        self.assertEqual(Path(res["path"]).read_bytes(), b"SFXBYTES")
        self.assertTrue(res["path"].startswith(self.td))
        self.assertTrue(any(u.endswith("/audio/complete") for u in seen))

    def test_processing_when_not_ready(self):
        # max_wait=0 -> a single probe; still PROCESSING -> no file, status back.
        responses = _seq(FakeResp(200, json.dumps({"status": "PROCESSING"}).encode()))
        with mock.patch("venice.client.urllib.request.urlopen", responses), \
             mock.patch.dict(os.environ, {"VENICE_MCP_OUTPUT_DIR": self.td}), \
             self.stdout_guard():
            res = _mcp.job_result_tool(
                _client(), queue_id="q1", type="music", model="elevenlabs-music", max_wait=0
            )
        self.assertEqual(res["status"], "processing")
        self.assertEqual(os.listdir(self.td), [])

    def test_video_fetches_download_url(self):
        url = "https://cdn.example.com/presigned?sig=xyz"
        seen = []

        def fake(req, timeout=None):
            seen.append(req.full_url)
            if req.full_url.endswith("/video/retrieve"):
                return FakeResp(200, json.dumps({"status": "COMPLETED"}).encode())
            if url in req.full_url:
                return FakeResp(200, b"PRESIGNEDMP4", "video/mp4")
            if req.full_url.endswith("/video/complete"):
                return FakeResp(200, b'{"ok": true}')
            raise AssertionError(f"unexpected: {req.full_url}")

        with mock.patch("venice.client.urllib.request.urlopen", fake), \
             mock.patch.dict(os.environ, {"VENICE_MCP_OUTPUT_DIR": self.td}), \
             self.stdout_guard():
            res = _mcp.job_result_tool(
                _client(), queue_id="vpsq1", type="video",
                model="seedance-2-0-text-to-video", download_url=url,
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(Path(res["path"]).read_bytes(), b"PRESIGNEDMP4")
        self.assertTrue(res["path"].endswith(".mp4"))
        self.assertIn(url, seen)

    def test_video_completed_without_download_url_is_error(self):
        responses = _seq(FakeResp(200, json.dumps({"status": "COMPLETED"}).encode()))
        with mock.patch("venice.client.urllib.request.urlopen", responses), self.stdout_guard():
            res = _mcp.job_result_tool(
                _client(), queue_id="vpsq2", type="video",
                model="seedance-2-0-text-to-video",  # no download_url
            )
        self.assertEqual(res["status"], "error")

    def test_unknown_type_is_error(self):
        res = _mcp.job_result_tool(_client(), queue_id="q", type="bogus", model="m")
        self.assertEqual(res["status"], "error")

    def test_max_wait_clamped_to_ceiling(self):
        # A model asking for an absurd block must not exceed the audio ceiling.
        captured = {}

        def fake_retrieve(client, model, queue_id, *, poll_interval, max_wait, on_tick=None):
            captured["max_wait"] = max_wait
            return "audio/mpeg", b"BYTES"

        with mock.patch.object(_mcp._audio, "retrieve_bytes", fake_retrieve), \
             mock.patch("venice.client.urllib.request.urlopen",
                        _seq(FakeResp(200, b'{"ok": true}'))), \
             mock.patch.dict(os.environ, {"VENICE_MCP_OUTPUT_DIR": self.td}), \
             self.stdout_guard():
            res = _mcp.job_result_tool(
                _client(), queue_id="q1", type="sfx",
                model="elevenlabs-sound-effects-v2", max_wait=999999,
            )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(captured["max_wait"], _mcp._sfx.config.SFX_POLL_MAX_WAIT_SEC)


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
    def _details(self, spec, mtype="text"):
        # _find_model matches by id on the first catalog it queries, so one
        # response containing the model is enough regardless of `type`.
        catalog = json.dumps(
            {"data": [{"id": "m1", "type": mtype, "model_spec": spec}]}
        ).encode()
        with mock.patch("venice.client.urllib.request.urlopen",
                        _seq(FakeResp(200, catalog))), self.stdout_guard():
            return _mcp.model_details_tool(_client(), model="m1")

    def test_image_model_surfaces_constraints(self):
        # Image models: metadata lives under model_spec.constraints, and
        # promptCharacterLimit is nested there (the bug kimi-k3 hit).
        spec = {"name": "SD35", "pricing": {"input": {"usd": 0.01}},
                "capabilities": None,
                "constraints": {"promptCharacterLimit": 1500,
                                "aspectRatios": ["1:1", "16:9"],
                                "resolutions": ["1K", "2K", "4K"]},
                "traits": ["eliza-default"]}
        out = self._details(spec, mtype="image")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["prompt_character_limit"], 1500)  # from constraints
        self.assertEqual(out["constraints"]["aspectRatios"], ["1:1", "16:9"])
        self.assertEqual(out["pricing"], {"input": {"usd": 0.01}})
        self.assertIsNone(out["capabilities"])          # image -> null (API reality)
        self.assertEqual(out["traits"], ["eliza-default"])
        self.assertEqual(out["model_spec"], spec)       # full spec, nothing dropped

    def test_text_model_capabilities_and_context(self):
        spec = {"name": "LLM", "availableContextTokens": 131072,
                "capabilities": {"supportsFunctionCalling": True,
                                 "supportsVision": True},
                "constraints": {"temperature": {"default": 0.8}}}
        out = self._details(spec, mtype="text")
        self.assertEqual(out["available_context_tokens"], 131072)
        self.assertTrue(out["capabilities"]["supportsVision"])
        self.assertIsNone(out["prompt_character_limit"])  # not applicable to text

    def test_prompt_limit_top_level_fallback(self):
        # if a model ever puts promptCharacterLimit at the top level, still read it
        out = self._details({"promptCharacterLimit": 800, "constraints": {}})
        self.assertEqual(out["prompt_character_limit"], 800)

    def test_tts_model_surfaces_voices(self):
        # #64: TTS models expose model_spec.voices (flat list of ids); surface it so
        # the agent can pick a valid voice for venice_tts without guessing.
        spec = {"name": "Kokoro", "pricing": {"input": {"usd": 3.5}},
                "voices": ["Achernar", "Aiden", "Alex"]}
        out = self._details(spec, mtype="tts")
        self.assertEqual(out["voices"], ["Achernar", "Aiden", "Alex"])

    def test_non_voice_model_voices_is_none(self):
        # image/text models have no voices -> null, like capabilities/constraints.
        out = self._details({"name": "SD35", "constraints": {}}, mtype="image")
        self.assertIsNone(out["voices"])

    def test_unknown_model_errors(self):
        from venice.commands.models import MODEL_TYPES
        empty = json.dumps({"data": []}).encode()
        resps = _seq(*[FakeResp(200, empty) for _ in MODEL_TYPES])
        with mock.patch("venice.client.urllib.request.urlopen", resps), \
                self.stdout_guard():
            out = _mcp.model_details_tool(_client(), model="nope")
        self.assertEqual(out["status"], "error")
        self.assertIn("no model", out["message"])


class TestReindexTool(_ToolTest):
    """#44: reindex rebuilds the discovered .venice index; paid, confirm-gated.

    Exercises the real discover/load/meta-recovery plumbing against an on-disk
    store (pointed at via $VENICE_INDEX_DIR) and mocks only the paid `build_index`.
    """

    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ)  # snapshot/restore $VENICE_INDEX_DIR
        env.start()
        self.addCleanup(env.stop)

    def _make_index(self, meta):
        root = Path(self.td)
        store_dir = root / ".venice" / "index"
        store_dir.mkdir(parents=True)
        (store_dir / "index.json").write_text(
            json.dumps({"meta": meta, "files": {}}), encoding="utf-8")
        os.environ["VENICE_INDEX_DIR"] = str(store_dir)
        return root

    def test_no_index_errors(self):
        with mock.patch.object(_mcp._index, "discover_store", return_value=None), \
                self.stdout_guard():
            r = _mcp.reindex_tool(_client(), confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("venice index", r["message"])

    def test_unconfirmed_gates_without_building(self):
        self._make_index({"backend": "venice", "model": "m1"})
        calls = []
        with mock.patch.object(_mcp._index, "build_index",
                               lambda *a, **k: calls.append(k)), \
                self.stdout_guard():
            r = _mcp.reindex_tool(_client(), confirm=False)
        self.assertEqual(r["status"], "confirmation_required")
        self.assertEqual(calls, [])  # no side effect before confirmation

    def test_confirm_venice_backend_recovers_meta(self):
        root = self._make_index(
            {"backend": "venice", "model": "text-embed-x", "requested_dimensions": 256})
        captured = {}

        def fake_build(r, **k):
            captured["root"] = r
            captured["kw"] = k
            return {"indexed": 2, "reused": 3, "removed": 0, "files": 5,
                    "chunks": 10, "backend": "venice", "model": k.get("model"),
                    "dimensions": 256, "store": "x"}

        with mock.patch.object(_mcp._index, "build_index", fake_build), \
                self.stdout_guard():
            r = _mcp.reindex_tool(_client(), confirm=True)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["indexed"], 2)
        self.assertEqual(r["reused"], 3)
        self.assertEqual(captured["kw"]["model"], "text-embed-x")
        self.assertEqual(captured["kw"]["dimensions"], 256)
        self.assertNotIn("embed_base_url", captured["kw"])  # venice, not local
        self.assertEqual(Path(captured["root"]).resolve(), root.resolve())

    def test_confirm_local_backend_recovers_base_url(self):
        self._make_index(
            {"backend": "local", "model": "bge-small", "base_url": "http://h:8080/v1"})
        captured = {}

        def fake_build(r, **k):
            captured.update(k)
            return {"indexed": 0, "reused": 0, "removed": 0, "files": 0,
                    "chunks": 0, "backend": "local", "model": "bge-small",
                    "dimensions": None, "store": "x"}

        with mock.patch.object(_mcp._index, "build_index", fake_build), \
                self.stdout_guard():
            r = _mcp.reindex_tool(_client(), confirm=True)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(captured["embed_base_url"], "http://h:8080/v1")
        self.assertEqual(captured["embed_model"], "bge-small")
        self.assertNotIn("model", captured)  # local path uses embed_model, not model

    def test_build_error_becomes_error_dict(self):
        self._make_index({"backend": "venice", "model": "m"})

        def boom(*a, **k):
            raise _mcp._index.IndexingError("embeddings backend unreachable", 5)

        with mock.patch.object(_mcp._index, "build_index", boom), \
                self.stdout_guard():
            r = _mcp.reindex_tool(_client(), confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("unreachable", r["message"])


if __name__ == "__main__":
    unittest.main()
