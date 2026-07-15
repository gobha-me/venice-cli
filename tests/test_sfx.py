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
        max_spend=None,
        no_balance=True,
        poll_interval=0,
        max_wait=10,
        command="sfx",
        master=False,
        lufs=-16.0,
        true_peak=-1.0,
        sample_rate=48000,
        bit_depth=24,
        loop=False,
        loop_crossfade=2.0,
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

    def test_master_flag_masters_saved_file(self):
        from venice.commands import sfx

        responses = iter([
            FakeResp(200, b'{"quote": 0.0027}', "application/json"),
            FakeResp(
                200,
                b'{"model":"elevenlabs-sound-effects-v2","queue_id":"abcdef1234567890","status":"QUEUED"}',
                "application/json",
            ),
            FakeResp(200, b"FAKEMP3BYTES", "audio/mpeg"),
            FakeResp(200, b'{"success": true}', "application/json"),
        ])
        mastered = []

        def fake_master(inp, out, **kw):
            mastered.append((Path(inp), Path(out), kw))
            return 0

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)), \
             mock.patch("venice.client.time.sleep"), \
             mock.patch("venice.audio_post.has_ffmpeg", lambda: True), \
             mock.patch("venice.audio_post.master", fake_master):
            rc = sfx._run_generate(_build_args(master=True))

        self.assertEqual(rc, 0)
        self.assertEqual(len(mastered), 1)
        inp, out, _ = mastered[0]
        self.assertTrue(inp.name.startswith("venice-sfx-"))
        self.assertTrue(out.name.endswith(".mastered.wav"))

    def test_master_without_ffmpeg_aborts_before_spend(self):
        from venice.commands import sfx

        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            raise AssertionError("should not reach the network")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             mock.patch("venice.audio_post.has_ffmpeg", lambda: False):
            rc = sfx._run_generate(_build_args(master=True))

        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])

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
