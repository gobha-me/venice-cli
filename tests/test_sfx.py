"""End-to-end SFX flow with mocked HTTP. Drives the command handler with --yes."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_client import FakeResp


def _build_args(**overrides):
    base = dict(
        prompt="thunder",
        model="elevenlabs-sound-effects-v2",
        duration=3,
        output=None,
        play=False,
        yes=True,
        background=False,
        dry_run=False,
        no_cleanup=False,
        poll_interval=0,
        max_wait=10,
        command="sfx",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestSfxFullFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_generate_writes_mp3(self):
        from venice.commands import sfx

        responses = iter([
            FakeResp(200, b'{"quote": 0.0027}', "application/json"),
            FakeResp(
                200,
                b'{"model":"elevenlabs-sound-effects-v2","queue_id":"abcdef1234567890","status":"QUEUED"}',
                "application/json",
            ),
            FakeResp(
                200,
                json.dumps(
                    {
                        "status": "PROCESSING",
                        "average_execution_time": 2000,
                        "execution_duration": 500,
                    }
                ).encode(),
                "application/json",
            ),
            FakeResp(200, b"FAKEMP3BYTES", "audio/mpeg"),
            FakeResp(200, b'{"success": true}', "application/json"),
        ])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)), \
             mock.patch("venice.client.time.sleep"):
            rc = sfx._run_generate(_build_args())

        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-sfx-*.mp3"))
        self.assertEqual(len(written), 1, f"expected 1 mp3, got {written}")
        self.assertEqual(written[0].read_bytes(), b"FAKEMP3BYTES")
        self.assertTrue(written[0].name.startswith("venice-sfx-abcdef12"))

    def test_dry_run_only_quotes_and_exits_zero(self):
        from venice.commands import sfx

        calls = []
        responses = iter([FakeResp(200, b'{"quote": 0.0027}', "application/json")])

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = sfx._run_generate(_build_args(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/audio/quote"))
        self.assertEqual(list(Path(".").glob("venice-sfx-*")), [])

    def test_background_prints_queue_id_to_stdout(self):
        from venice.commands import sfx

        responses = iter([
            FakeResp(200, b'{"quote": 0.0027}', "application/json"),
            FakeResp(
                200,
                b'{"model":"elevenlabs-sound-effects-v2","queue_id":"BGID12345","status":"QUEUED"}',
                "application/json",
            ),
        ])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)), \
             mock.patch("sys.stdout") as out:
            rc = sfx._run_generate(_build_args(background=True))

        self.assertEqual(rc, 0)
        # stdout should have received the queue_id on a line by itself
        writes = "".join(c.args[0] for c in out.write.call_args_list)
        self.assertIn("BGID12345", writes)

    def test_missing_api_key_returns_exit_2(self):
        from venice.commands import sfx

        with mock.patch.dict(os.environ, {}, clear=True):
            # ensure VENICE_API_KEY isn't in os.environ; also point HOME at empty tmpdir
            empty = tempfile.TemporaryDirectory()
            os.environ["HOME"] = empty.name
            try:
                import importlib
                import venice.config as _cfg
                import venice.auth as _auth
                importlib.reload(_cfg)
                importlib.reload(_auth)
                rc = sfx._run_generate(_build_args())
            finally:
                empty.cleanup()
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
