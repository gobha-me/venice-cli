"""End-to-end video flow with mocked HTTP. Drives the command handler with --yes.

Each generate sequence starts with the free /models?type=video catalog GET used
to resolve the default model (mirrors test_chat.py's catalog mock).
"""
import argparse
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_client import FakeResp

DEFAULT_MODEL = "seedance-2-0-text-to-video"


def _catalog(*ids_with_default):
    """Build a /models?type=video response; first id carries the 'default' trait."""
    data = []
    for i, mid in enumerate(ids_with_default):
        traits = ["default"] if i == 0 else []
        data.append({"id": mid, "model_spec": {"traits": traits}})
    return FakeResp(200, json.dumps({"data": data}).encode(), "application/json")


def _build_args(**overrides):
    base = dict(
        prompt="a koi pond at dawn",
        model=None,
        duration="5s",
        resolution=None,
        aspect_ratio=None,
        negative_prompt=None,
        no_audio=False,
        # media inputs (#18)
        image=None,
        end_image=None,
        video=None,
        audio_input=None,
        reference_image=None,
        reference_video=None,
        reference_audio=None,
        scene_image=None,
        reference_video_duration=None,
        element=None,
        output=None,
        yes=True,
        background=False,
        dry_run=False,
        no_cleanup=False,
        max_spend=None,
        no_balance=True,
        poll_interval=0,
        max_wait=10,
        download_url=None,
        command="video",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestVideoFlow(unittest.TestCase):

    def setUp(self):
        # Hermetic: never read the developer's real ~/.config/venice/config.json.
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_streaming_model_writes_mp4(self):
        from venice.commands import video

        responses = iter([
            _catalog(DEFAULT_MODEL),
            FakeResp(200, b'{"quote": 0.5}', "application/json"),
            FakeResp(
                200,
                b'{"model":"' + DEFAULT_MODEL.encode() + b'","queue_id":"vid12345678"}',
                "application/json",
            ),
            FakeResp(
                200,
                json.dumps({
                    "status": "PROCESSING",
                    "average_execution_time": 145000,
                    "execution_duration": 50000,
                }).encode(),
                "application/json",
            ),
            FakeResp(200, b"FAKEMP4BYTES", "video/mp4"),
            FakeResp(200, b'{"success": true}', "application/json"),
        ])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)), \
             mock.patch("venice.client.time.sleep"):
            rc = video._run_generate(_build_args())

        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-video-*.mp4"))
        self.assertEqual(len(written), 1, f"expected 1 mp4, got {written}")
        self.assertEqual(written[0].read_bytes(), b"FAKEMP4BYTES")
        self.assertTrue(written[0].name.startswith("venice-video-vid12345"))

    def test_vps_model_fetches_download_url(self):
        from venice.commands import video

        url = "https://cdn.example.com/presigned?sig=abc"
        responses = iter([
            _catalog(DEFAULT_MODEL),
            FakeResp(200, b'{"quote": 0.5}', "application/json"),
            FakeResp(
                200,
                json.dumps({
                    "model": DEFAULT_MODEL,
                    "queue_id": "vpsjob123456",
                    "download_url": url,
                }).encode(),
                "application/json",
            ),
            FakeResp(
                200,
                json.dumps({
                    "status": "COMPLETED",
                    "average_execution_time": 145000,
                    "execution_duration": 144000,
                }).encode(),
                "application/json",
            ),
            FakeResp(200, b"PRESIGNEDMP4", "video/mp4"),
            FakeResp(200, b'{"success": true}', "application/json"),
        ])
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(req.full_url)
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             mock.patch("venice.client.time.sleep"):
            rc = video._run_generate(_build_args())

        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-video-*.mp4"))
        self.assertEqual(len(written), 1, f"expected 1 mp4, got {written}")
        self.assertEqual(written[0].read_bytes(), b"PRESIGNEDMP4")
        # the presigned URL was fetched verbatim (no base_url prefix)
        self.assertIn(url, seen)

    def test_dry_run_quotes_and_exits_zero(self):
        from venice.commands import video

        calls = []
        responses = iter([
            _catalog(DEFAULT_MODEL),
            FakeResp(200, b'{"quote": 0.5}', "application/json"),
        ])

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = video._run_generate(_build_args(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[-1].endswith("/video/quote"))
        self.assertEqual(list(Path(".").glob("venice-video-*")), [])

    def test_background_prints_queue_id_to_stdout(self):
        from venice.commands import video

        responses = iter([
            _catalog(DEFAULT_MODEL),
            FakeResp(200, b'{"quote": 0.5}', "application/json"),
            FakeResp(
                200,
                b'{"model":"' + DEFAULT_MODEL.encode() + b'","queue_id":"BGVID99999"}',
                "application/json",
            ),
        ])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)), \
             mock.patch("sys.stdout") as out:
            rc = video._run_generate(_build_args(background=True))

        self.assertEqual(rc, 0)
        writes = "".join(c.args[0] for c in out.write.call_args_list)
        self.assertIn("BGVID99999", writes)

    def test_unknown_model_returns_exit_6(self):
        from venice.commands import video

        responses = iter([_catalog(DEFAULT_MODEL)])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)):
            rc = video._run_generate(_build_args(model="no-such-model"))

        self.assertEqual(rc, 6)
        self.assertEqual(list(Path(".").glob("venice-video-*")), [])

    def test_missing_api_key_returns_exit_2(self):
        from venice.commands import video

        with mock.patch.dict(os.environ, {}, clear=True):
            empty = tempfile.TemporaryDirectory()
            os.environ["HOME"] = empty.name
            try:
                import importlib
                import venice.auth as _auth
                import venice.config as _cfg
                importlib.reload(_cfg)
                importlib.reload(_auth)
                rc = video._run_generate(_build_args())
            finally:
                empty.cleanup()
        self.assertEqual(rc, 2)


class TestVideoMediaInputs(unittest.TestCase):
    """Issue #18: image-to-video, reference inputs, and @Element JSON."""

    def setUp(self):
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: os.chdir(self.cwd))

    def _run_full(self, args):
        """Drive the whole quote->queue->retrieve->save flow, capturing the JSON
        bodies POSTed to /video/quote and /video/queue."""
        from venice.commands import video

        responses = iter([
            _catalog(DEFAULT_MODEL),
            FakeResp(200, b'{"quote": 0.5}', "application/json"),
            FakeResp(
                200,
                b'{"model":"' + DEFAULT_MODEL.encode() + b'","queue_id":"vid12345678"}',
                "application/json",
            ),
            FakeResp(
                200,
                json.dumps({"status": "PROCESSING", "average_execution_time": 1000,
                            "execution_duration": 500}).encode(),
                "application/json",
            ),
            FakeResp(200, b"FAKEMP4BYTES", "video/mp4"),
            FakeResp(200, b'{"success": true}', "application/json"),
        ])
        captured = {}

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/video/quote"):
                captured["quote"] = json.loads(req.data.decode("utf-8"))
            elif req.full_url.endswith("/video/queue"):
                captured["queue"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             mock.patch("venice.client.time.sleep"):
            rc = video._run_generate(args)
        return rc, captured

    def _run_prevalidation(self, args):
        """Run with only the catalog GET stubbed; used for inputs expected to
        fail in _collect_media, before any quote POST. Returns (rc, urls_hit)."""
        from venice.commands import video

        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return _catalog(DEFAULT_MODEL)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = video._run_generate(args)
        return rc, calls

    def test_image_file_encoded_as_data_url(self):
        Path("frame.png").write_bytes(b"PNGDATA")
        rc, cap = self._run_full(_build_args(image="frame.png"))
        self.assertEqual(rc, 0)
        iu = cap["queue"]["image_url"]
        self.assertTrue(iu.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(iu.split(",", 1)[1]), b"PNGDATA")
        # image conditioning is queue-only (QuoteVideoRequest rejects it)
        self.assertNotIn("image_url", cap["quote"])

    def test_image_url_passthrough(self):
        rc, cap = self._run_full(_build_args(image="https://x.test/a.png"))
        self.assertEqual(rc, 0)
        self.assertEqual(cap["queue"]["image_url"], "https://x.test/a.png")

    def test_reference_images_array_mixed_file_and_url(self):
        Path("a.png").write_bytes(b"AAA")
        rc, cap = self._run_full(
            _build_args(reference_image=["a.png", "https://x.test/b.png"])
        )
        self.assertEqual(rc, 0)
        refs = cap["queue"]["reference_image_urls"]
        self.assertEqual(len(refs), 2)
        self.assertTrue(refs[0].startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(refs[0].split(",", 1)[1]), b"AAA")
        self.assertEqual(refs[1], "https://x.test/b.png")

    def test_reference_video_duration_is_quote_only(self):
        rc, cap = self._run_full(_build_args(
            video="https://x.test/in.mp4",
            reference_video=["https://x.test/ref.mp4"],
            reference_video_duration=7.0,
        ))
        self.assertEqual(rc, 0)
        # duration prices the R2V tier -> quote only, never on the queue body
        self.assertEqual(cap["quote"]["reference_video_total_duration"], 7.0)
        self.assertNotIn("reference_video_total_duration", cap["queue"])
        # video_url is valid on both endpoints
        self.assertEqual(cap["quote"]["video_url"], "https://x.test/in.mp4")
        self.assertEqual(cap["queue"]["video_url"], "https://x.test/in.mp4")
        self.assertEqual(cap["queue"]["reference_video_urls"], ["https://x.test/ref.mp4"])

    def test_element_json_encodes_nested_path(self):
        Path("hero.png").write_bytes(b"HERO")
        el = json.dumps({
            "frontal_image_url": "hero.png",
            "reference_image_urls": ["https://x.test/r.png"],
        })
        rc, cap = self._run_full(_build_args(element=[el]))
        self.assertEqual(rc, 0)
        elements = cap["queue"]["elements"]
        self.assertEqual(len(elements), 1)
        front = elements[0]["frontal_image_url"]
        self.assertTrue(front.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(front.split(",", 1)[1]), b"HERO")
        self.assertEqual(elements[0]["reference_image_urls"], ["https://x.test/r.png"])

    def test_scene_image_over_cap_exits_2_before_quote(self):
        rc, calls = self._run_prevalidation(
            _build_args(scene_image=[f"https://x.test/{i}.png" for i in range(5)])
        )
        self.assertEqual(rc, 2)
        self.assertTrue(all(not c.endswith("/video/quote") for c in calls))

    def test_missing_image_file_exits_2_before_quote(self):
        rc, calls = self._run_prevalidation(_build_args(image="does-not-exist.png"))
        self.assertEqual(rc, 2)
        self.assertTrue(all(not c.endswith("/video/quote") for c in calls))

    def test_bad_element_json_exits_2_before_quote(self):
        rc, calls = self._run_prevalidation(_build_args(element=["{not valid json"]))
        self.assertEqual(rc, 2)
        self.assertTrue(all(not c.endswith("/video/quote") for c in calls))


if __name__ == "__main__":
    unittest.main()
