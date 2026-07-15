"""End-to-end `venice music` flow with mocked HTTP. Drives the handler with --yes.

Covers the /models validation gate, the quote->queue->retrieve->complete happy
path, budget/dry-run, capability gating, and graceful degrade when /models is
unreachable.
"""
import argparse
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from tests.test_client import FakeResp


def _build_args(**overrides):
    base = dict(
        prompt="tense dungeon drone",
        model="elevenlabs-music",
        duration=60,
        instrumental=False,
        lyrics=None,
        speed=None,
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
        command="music",
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


def _models_payload(caps_override=None, spec_override=None):
    spec = {
        "name": "ElevenLabs Music",
        "pricing": {"output": {"usd": 0.1}},
        # camelCase capabilities (as the live API returns them)
        "capabilities": {
            "supportsForceInstrumental": True,
            "supportsLyrics": True,
            "supportsSpeed": True,
        },
        # snake_case constraints at the spec level
        "min_prompt_length": 1,
        "prompt_character_limit": 500,
        "min_duration": 10,
        "max_duration": 300,
        "min_speed": 0.5,
        "max_speed": 2.0,
    }
    if caps_override:
        spec["capabilities"].update(caps_override)
    if spec_override:
        spec.update(spec_override)
    return json.dumps({
        "object": "list",
        "data": [{"id": "elevenlabs-music", "type": "music", "model_spec": spec}],
    }).encode()


def _router(models_payload, responses, *, captured=None, calls=None):
    """urlopen mock: serves /models from `models_payload`, everything else from
    the `responses` iterator in order. Records urls in `calls` and request bodies
    in `captured` (keyed by audio endpoint)."""
    it = iter(responses)

    def _urlopen(req, timeout=None):
        url = req.full_url
        if calls is not None:
            calls.append(url)
        if captured is not None and req.data:
            try:
                body = json.loads(req.data.decode("utf-8"))
            except Exception:
                body = None
            for key in ("quote", "queue", "retrieve", "complete"):
                if url.endswith("/audio/" + key):
                    captured[key] = body
        if "/models" in url:
            return FakeResp(200, models_payload, "application/json")
        return next(it)

    return _urlopen


def _quote(usd=0.5):
    return FakeResp(200, json.dumps({"quote": usd}).encode(), "application/json")


def _queue(qid="musicqueue123456"):
    return FakeResp(
        200,
        json.dumps({"model": "elevenlabs-music", "queue_id": qid, "status": "QUEUED"}).encode(),
        "application/json",
    )


def _processing():
    return FakeResp(
        200,
        json.dumps({"status": "PROCESSING", "average_execution_time": 2000,
                    "execution_duration": 500}).encode(),
        "application/json",
    )


def _audio_bytes(data=b"FAKEMUSICBYTES"):
    return FakeResp(200, data, "audio/mpeg")


def _complete():
    return FakeResp(200, b'{"success": true}', "application/json")


class TestMusicFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_generate_writes_mp3_and_sends_music_params(self):
        from venice.commands import music

        captured = {}
        responses = [_quote(), _queue(), _processing(), _audio_bytes(), _complete()]
        urlopen = _router(_models_payload(), responses, captured=captured)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen), \
             mock.patch("venice.client.time.sleep"):
            rc = music._run_generate(_build_args(instrumental=True, duration=90))

        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-music-*.mp3"))
        self.assertEqual(len(written), 1, f"expected 1 mp3, got {written}")
        self.assertEqual(written[0].read_bytes(), b"FAKEMUSICBYTES")
        self.assertTrue(written[0].name.startswith("venice-music-musicque"))
        # queue body carried the music-only params
        self.assertEqual(captured["queue"]["prompt"], "tense dungeon drone")
        self.assertEqual(captured["queue"]["duration_seconds"], 90)
        self.assertTrue(captured["queue"]["force_instrumental"])

    def test_dry_run_stops_after_quote(self):
        from venice.commands import music

        calls = []
        urlopen = _router(_models_payload(), [_quote()], calls=calls)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen):
            rc = music._run_generate(_build_args(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("/models", calls[0])
        self.assertTrue(calls[1].endswith("/audio/quote"))
        self.assertEqual(list(Path(".").glob("venice-music-*")), [])

    def test_max_spend_aborts(self):
        from venice.commands import music

        urlopen = _router(_models_payload(), [_quote(0.5)])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen):
            rc = music._run_generate(_build_args(max_spend=0.001))

        self.assertEqual(rc, 1)

    def test_duration_out_of_range_errors_before_quote(self):
        from venice.commands import music

        calls = []
        # duration below model min (10); quote must never be reached
        urlopen = _router(_models_payload(), [], calls=calls)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen):
            rc = music._run_generate(_build_args(duration=5))

        self.assertEqual(rc, 2)
        self.assertTrue(all("/audio/quote" not in c for c in calls))

    def test_lyrics_and_instrumental_conflict(self):
        from venice.commands import music

        urlopen = _router(_models_payload(), [])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen):
            rc = music._run_generate(_build_args(lyrics="la la", instrumental=True))

        self.assertEqual(rc, 2)

    def test_instrumental_gated_off_by_model(self):
        from venice.commands import music

        payload = _models_payload(caps_override={"supportsForceInstrumental": False})
        urlopen = _router(payload, [])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen):
            rc = music._run_generate(_build_args(instrumental=True))

        self.assertEqual(rc, 2)

    def test_background_prints_queue_id(self):
        from venice.commands import music

        urlopen = _router(_models_payload(), [_quote(), _queue("BGMUSIC99")])

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen), \
             mock.patch("sys.stdout") as out:
            rc = music._run_generate(_build_args(background=True))

        self.assertEqual(rc, 0)
        writes = "".join(c.args[0] for c in out.write.call_args_list)
        self.assertIn("BGMUSIC99", writes)

    def test_models_unreachable_degrades_and_proceeds(self):
        from venice.commands import music

        responses = iter([_quote(), _queue(), _processing(), _audio_bytes(), _complete()])

        def urlopen(req, timeout=None):
            if "/models" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 500, "err", {}, io.BytesIO(b"{}")
                )
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen), \
             mock.patch("venice.client.time.sleep"):
            rc = music._run_generate(_build_args())

        self.assertEqual(rc, 0)
        self.assertEqual(len(sorted(Path(".").glob("venice-music-*.mp3"))), 1)

    def test_master_flag_masters_saved_file(self):
        from venice.commands import music

        responses = [_quote(), _queue(), _processing(), _audio_bytes(), _complete()]
        urlopen = _router(_models_payload(), responses)
        mastered = []

        def fake_master(inp, out, **kw):
            mastered.append((Path(inp), Path(out), kw))
            return 0

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen), \
             mock.patch("venice.client.time.sleep"), \
             mock.patch("venice.audio_post.has_ffmpeg", lambda: True), \
             mock.patch("venice.audio_post.master", fake_master):
            rc = music._run_generate(_build_args(master=True, loop=True))

        self.assertEqual(rc, 0)
        self.assertEqual(len(mastered), 1)
        inp, out, kw = mastered[0]
        self.assertTrue(inp.name.startswith("venice-music-"))
        self.assertTrue(out.name.endswith(".mastered.wav"))
        self.assertTrue(kw["loop"])
        self.assertEqual(kw["sample_rate"], 48000)

    def test_master_without_ffmpeg_aborts_before_spend(self):
        from venice.commands import music

        calls = []
        urlopen = _router(_models_payload(), [], calls=calls)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen), \
             mock.patch("venice.audio_post.has_ffmpeg", lambda: False):
            rc = music._run_generate(_build_args(master=True))

        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])  # no /models, no /quote, no /queue -> no spend

    def test_missing_api_key_returns_exit_2(self):
        from venice.commands import music

        with mock.patch.dict(os.environ, {}, clear=True):
            empty = tempfile.TemporaryDirectory()
            os.environ["HOME"] = empty.name
            try:
                import importlib
                import venice.config as _cfg
                import venice.auth as _auth
                importlib.reload(_cfg)
                importlib.reload(_auth)
                rc = music._run_generate(_build_args())
            finally:
                empty.cleanup()
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
