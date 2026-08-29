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
        # #57 Class C1: both default None on the parser now; `_run` resolves
        # them, so the tests below double as proof the fallback fires.
        model=None,
        voice=None,
        format=None,
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
    """Mimics /models?type=tts with model-specific format contracts."""
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
                    "supported_formats": ["mp3", "wav"],
                    "default_format": "mp3",
                },
            },
            {
                "id": "tts-xai-v1",
                "type": "tts",
                "model_spec": {
                    "pricing": {"input": {"usd": 18.75}},
                    "voices": ["voice_a"],
                    "supported_formats": ["mp3", "wav"],
                    "default_format": "mp3",
                },
            },
            {
                "id": "tts-inworld-1-5-max",
                "type": "tts",
                "model_spec": {
                    "pricing": {"input": {"usd": 12.5}},
                    "supported_formats": ["wav"],
                    "default_format": "wav",
                },
            },
            {
                "id": "tts-gradium-v1",
                "type": "tts",
                "model_spec": {
                    "pricing": {"input": {"usd": 20.0}},
                    "supported_formats": ["wav", "pcm", "opus"],
                    "default_format": "wav",
                },
            },
        ],
    }).encode()


class TestTtsFlow(unittest.TestCase):

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

    def test_generate_writes_mp3(self):
        from venice.commands import tts

        captured = {}
        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"FAKEMP3", "audio/mpeg"),
        ])

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/audio/speech"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = tts._run(_build_args(text="Hello, voice."))
        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-tts-*.mp3"))
        self.assertEqual(len(written), 1, f"expected 1 mp3, got {written}")
        self.assertEqual(written[0].read_bytes(), b"FAKEMP3")
        self.assertNotIn("response_format", captured["body"])

    def test_wav_only_model_uses_catalog_default(self):
        from venice.commands import tts

        captured = {}
        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"WAV", "audio/wav; charset=binary"),
        ])

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/audio/speech"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = tts._run(_build_args(model="tts-inworld-1-5-max"))
        self.assertEqual(rc, 0)
        self.assertNotIn("response_format", captured["body"])
        self.assertEqual(len(list(Path(".").glob("venice-tts-*.wav"))), 1)

    def test_response_content_type_wins_for_generated_extension(self):
        from venice.commands import tts

        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"WAV", "audio/wav"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = tts._run(_build_args(format="mp3"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(list(Path(".").glob("venice-tts-*.wav"))), 1)

    def test_explicit_output_filename_is_preserved(self):
        from venice.commands import tts

        output = Path("custom.audio")
        responses = iter([
            FakeResp(200, _tts_models_payload(), "application/json"),
            FakeResp(200, b"WAV", "audio/wav"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = tts._run(_build_args(output=output))
        self.assertEqual(rc, 0)
        self.assertEqual(output.read_bytes(), b"WAV")

    def test_unsupported_format_fails_before_speech_request(self):
        from venice.commands import tts

        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return FakeResp(200, _tts_models_payload(), "application/json")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = tts._run(_build_args(
                model="tts-inworld-1-5-max", format="mp3"
            ))
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/models?type=tts"))

    def test_unknown_model_fails_before_speech_request(self):
        from venice.commands import tts

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: FakeResp(
                            200, _tts_models_payload(), "application/json"
                        )):
            rc = tts._run(_build_args(model="tts-nope"))
        self.assertEqual(rc, 2)

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


class TestTtsPricingValidation(unittest.TestCase):
    def test_catalog_api_failure_is_rejected(self):
        from venice.client import VeniceAPIError
        from venice.commands import tts

        client = mock.Mock()
        client.get_json.side_effect = VeniceAPIError(0, "/models", "offline")
        with self.assertRaisesRegex(ValueError, "could not fetch live TTS catalog"):
            tts._resolve_tts(client, "m", None)

    def test_non_finite_catalog_price_is_rejected(self):
        from venice.commands import tts

        class Client:
            @staticmethod
            def get_json(*args, **kwargs):
                return {"data": [{
                    "id": "m",
                    "model_spec": {
                        "pricing": {"input": {"usd": float("nan")}},
                        "supported_formats": ["wav"],
                        "default_format": "wav",
                    },
                }]}

        with self.assertRaisesRegex(ValueError, "invalid TTS price"):
            tts._resolve_tts(Client(), "m", None)

    def test_catalog_contract_failures_are_rejected(self):
        from venice.commands import tts

        cases = (
            ({}, "no data list"),
            ({"data": []}, "not in the live TTS catalog"),
            ({"data": [{"id": "m"}]}, "has no model_spec"),
            ({"data": [{"id": "m", "model_spec": {}}]},
             "invalid supported_formats"),
            ({"data": [{"id": "m", "model_spec": {
                "supported_formats": ["wav"]}}]}, "has no default_format"),
            ({"data": [{"id": "m", "model_spec": {
                "supported_formats": ["wav"], "default_format": "mp3"}}]},
             "is not in supported_formats"),
        )
        for doc, message in cases:
            with self.subTest(message=message):
                client = mock.Mock()
                client.get_json.return_value = doc
                with self.assertRaisesRegex(ValueError, message):
                    tts._resolve_tts(client, "m", None)

    def test_parser_accepts_live_catalog_values(self):
        from venice import cli

        args = cli.build_parser().parse_args([
            "tts", "hello", "--model", "tts-gradium-v1", "--format", "opus"
        ])
        self.assertEqual(args.model, "tts-gradium-v1")
        self.assertEqual(args.format, "opus")


if __name__ == "__main__":
    unittest.main()
