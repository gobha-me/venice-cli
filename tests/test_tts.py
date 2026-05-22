"""Unit tests for `venice tts` (mocks urlopen)."""
import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_client import FakeResp


def _build_args(**ov):
    base = dict(
        text="hello world",
        from_file=None,
        stdin=False,
        model="tts-kokoro",
        voice=None,
        format="mp3",
        speed=None,
        output=None,
        play=False,
        yes=True,
        dry_run=False,
        max_spend=None,
        no_balance=True,
        command="tts",
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _tts_models_payload():
    """Mimics /models?type=tts response with two entries."""
    return json.dumps({
        "object": "list",
        "data": [
            {
                "id": "tts-kokoro",
                "type": "tts",
                "model_spec": {
                    "name": "Kokoro",
                    "pricing": {"input": {"usd": 3.5}},
                    "voices": ["af_sky", "am_michael"],
                },
            },
            {
                "id": "tts-xai-v1",
                "type": "tts",
                "model_spec": {
                    "pricing": {"input": {"usd": 18.75}},
                    "voices": ["voice_a"],
                },
            },
        ],
    }).encode()


class TestTtsFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_generate_writes_mp3(self):
        from venice.commands import tts

        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"FAKEMP3", "audio/mpeg"),
        ])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)):
            rc = tts._run(_build_args(text="Hello, voice."))
        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-tts-*.mp3"))
        self.assertEqual(len(written), 1, f"expected 1 mp3, got {written}")
        self.assertEqual(written[0].read_bytes(), b"FAKEMP3")

    def test_dry_run_does_not_call_speech(self):
        from venice.commands import tts

        calls = []
        responses = iter([FakeResp(200, _tts_models_payload(), "application/json")])

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = tts._run(_build_args(text="hi", dry_run=True))
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/models?type=tts"))
        self.assertEqual(list(Path(".").glob("venice-tts-*")), [])

    def test_max_spend_aborts_when_estimate_too_high(self):
        from venice.commands import tts

        # 200000 chars * $3.50/M = $0.70; cap at $0.10 -> abort.
        text = "x" * 200_000
        responses = iter([FakeResp(200, _tts_models_payload(), "application/json")])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)):
            rc = tts._run(_build_args(text=text, max_spend=0.10))
        self.assertEqual(rc, 1)

    def test_from_file_reads_input(self):
        from venice.commands import tts

        f = Path(self.tmp.name) / "speech.txt"
        f.write_text("file contents go here", encoding="utf-8")
        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"WAVBYTES", "audio/wav"),
        ])
        captured = {}

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/audio/speech"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = tts._run(_build_args(text=None, from_file=f, format="wav"))
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["input"], "file contents go here")
        self.assertEqual(captured["body"]["model"], "tts-kokoro")
        self.assertEqual(captured["body"]["response_format"], "wav")
        self.assertNotIn("voice", captured["body"])  # omitted when not set

    def test_stdin_reads_input(self):
        from venice.commands import tts

        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"MP3STDIN", "audio/mpeg"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", lambda *a, **kw: next(responses)), \
             mock.patch.object(sys, "stdin", io.StringIO("piped text")):
            rc = tts._run(_build_args(text=None, stdin=True))
        self.assertEqual(rc, 0)

    def test_empty_input_returns_exit_2(self):
        from venice.commands import tts

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = tts._run(_build_args(text="   "))
        self.assertEqual(rc, 2)

    def test_no_input_source_returns_exit_2(self):
        from venice.commands import tts

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = tts._run(_build_args(text=None))
        self.assertEqual(rc, 2)

    def test_invalid_speed_returns_exit_2(self):
        from venice.commands import tts

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = tts._run(_build_args(text="hi", speed=5.0))
        self.assertEqual(rc, 2)

    def test_voice_included_in_body_when_set(self):
        from venice.commands import tts

        captured = {}
        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"X", "audio/mpeg"),
        ])

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/audio/speech"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = tts._run(_build_args(text="x", voice="af_sky", speed=1.25))
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["voice"], "af_sky")
        self.assertEqual(captured["body"]["speed"], 1.25)


if __name__ == "__main__":
    unittest.main()
