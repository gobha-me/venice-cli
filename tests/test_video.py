"""End-to-end video flow with mocked HTTP. Drives the command handler with --yes.

Each generate sequence starts with the free /models?type=video catalog GET used
to resolve the default model (mirrors test_chat.py's catalog mock).
"""
import argparse
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


if __name__ == "__main__":
    unittest.main()
