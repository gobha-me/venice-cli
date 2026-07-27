"""End-to-end SFX flow with mocked HTTP. Drives the command handler with --yes."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice import config
from tests.test_client import FakeResp


def _build_args(**overrides):
    base = dict(
        prompt="thunder",
        # #57 Class C1: None on the parser now; `_run_generate` resolves it, so
        # the tests below double as proof the fallback fires. `duration` stays
        # concrete -- it models an explicitly-passed `--duration 3`.
        model=None,
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

    def test_status_handler_applies_config_defaults(self):
        """#57 Class B: `sfx-status` never called apply_defaults, so a config key
        would have applied to `venice sfx` but silently not to `venice sfx-status`."""
        from venice.commands import sfx

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"sfx": {"no_cleanup": True}}}
        args = argparse.Namespace(
            queue_id="abcdef1234567890", model="elevenlabs-sound-effects-v2",
            output=None, poll_interval=0.0, max_wait=1.0, no_cleanup=None,
            play=False, command="sfx-status",
        )
        seen = {}

        def fake_retrieve(client, model, qid, *a, **kw):
            seen["no_cleanup"] = a[3]  # 7th positional overall
            return 0

        with mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.commands._audio.retrieve_and_save", fake_retrieve):
            rc = sfx._run_status(args)

        self.assertEqual(rc, 0)
        self.assertIs(seen["no_cleanup"], True)

    def test_config_master_triggers_the_ffmpeg_precheck(self):
        """#57 Class B: defaults.sfx.master reaches args.master, proven by the
        pre-spend ffmpeg guard firing without an inline --master."""
        from venice.commands import sfx

        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            raise AssertionError("should not reach the network")

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"sfx": {"master": True}}}
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             mock.patch("venice.audio_post.has_ffmpeg", lambda: False):
            rc = sfx._run_generate(_build_args(master=None))

        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])

    def test_explicit_no_master_beats_config(self):
        from venice.commands import sfx

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"sfx": {"master": True}}}
        args = _build_args(master=False, dry_run=True)

        # urlopen MUST be patched: --dry-run still POSTs /audio/quote before it
        # returns, so an unmocked run would hit the real API (CONTRIBUTING: "No
        # test should ever make a real API call or need a real key.").
        def fake_urlopen(req, timeout=None):
            return FakeResp(200, b'{"quote": 0.0027}', "application/json")

        with mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             mock.patch("venice.audio_post.has_ffmpeg", lambda: False):
            sfx._run_generate(args)
        self.assertIs(args.master, False)  # config did not overwrite it

    def test_config_model_reaches_the_body_and_the_status_hint(self):
        """#57 Class C1. `defaults.sfx.model` must reach the queue body -- and the
        printed follow-up must carry `--model`, because on the status side the
        model is job IDENTITY. Without it, a job queued under a config model is
        fetched back with the built-in model and 404s something already paid for.
        """
        from venice.commands import sfx

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"sfx": {"model": "mmaudio-v2-text-to-audio"}}}
        bodies = []
        responses = iter([
            FakeResp(200, b'{"quote": 0.0027}', "application/json"),
            FakeResp(200, b'{"queue_id":"BGID12345","status":"QUEUED"}',
                     "application/json"),
        ])

        def fake_urlopen(req, timeout=None):
            bodies.append(json.loads(req.data.decode("utf-8")))
            return next(responses)

        err = io.StringIO()
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             contextlib.redirect_stderr(err):
            rc = sfx._run_generate(_build_args(background=True))

        self.assertEqual(rc, 0)
        for body in bodies:
            self.assertEqual(body["model"], "mmaudio-v2-text-to-audio")
        self.assertIn("venice sfx-status BGID12345 "
                      "--model mmaudio-v2-text-to-audio", err.getvalue())

    def test_zero_duration_from_config_still_warns(self):
        """#57 Class C1: `apply_literals` fills with `is not None`, never `or`.
        A config-set 0 is a value the user typed and must reach _clamp_duration's
        own "must be > 0" warning rather than be silently rewritten to 5."""
        from venice.commands import sfx

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"sfx": {"duration": 0}}}
        args = _build_args(duration=None, dry_run=True)

        def fake_urlopen(req, timeout=None):
            return FakeResp(200, b'{"quote": 0.0027}', "application/json")

        err = io.StringIO()
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen), \
             contextlib.redirect_stderr(err):
            sfx._run_generate(args)

        self.assertIn("--duration must be > 0", err.getvalue())

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


class TestPollCadenceReachesRetrieve(unittest.TestCase):
    """#57 Class C2: the poll cadence literals now live in
    `_shared.apply_poll_defaults`, called by the handler -- not on the parser.

    Every other test in this file hand-builds a Namespace with
    `poll_interval`/`max_wait` already set, so all of them keep passing if a
    handler forgets that call. These parse the REAL parser and let the handler
    fill, which is the only way the omission is visible. It matters because a
    leftover None is a TypeError inside the poll loop -- and for
    `poll_interval` only on the SECOND iteration, so a fast job hides it and a
    slow one crashes after the spend.
    """

    def setUp(self):
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)

    @staticmethod
    def _parse(*argv):
        from venice import cli
        return cli.build_parser().parse_args(list(argv))

    def _spy(self):
        seen = {}

        def fake(client, model, queue_id, output, poll_interval, max_wait,
                 *a, **kw):
            seen.update(poll_interval=poll_interval, max_wait=max_wait)
            return 0

        return seen, fake

    def test_generate_uses_the_builtin_cadence(self):
        from venice.commands import sfx as cmd
        from venice.commands import _audio

        responses = iter([
            FakeResp(200, b'{"quote": 0.01}', "application/json"),
            FakeResp(200, b'{"queue_id":"abcdef1234567890","status":"QUEUED"}',
                     "application/json"),
        ])
        seen, fake = self._spy()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)), \
             mock.patch.object(_audio, "retrieve_and_save", fake):
            rc = cmd._run_generate(
                self._parse("sfx", "a prompt", "--yes", "--no-balance"))
        self.assertEqual(rc, 0)
        self.assertEqual(seen["poll_interval"], config.SFX_POLL_INTERVAL_SEC)
        self.assertEqual(seen["max_wait"], config.SFX_POLL_MAX_WAIT_SEC)

    def test_status_uses_the_builtin_cadence(self):
        from venice.commands import sfx as cmd
        from venice.commands import _audio, _queue

        seen, fake = self._spy()
        with mock.patch.object(_queue, "build_client", lambda: (object(), 0)), \
             mock.patch.object(_audio, "retrieve_and_save", fake):
            rc = cmd._run_status(self._parse("sfx-status", "abcdef1234567890"))
        self.assertEqual(rc, 0)
        self.assertEqual(seen["poll_interval"], config.SFX_POLL_INTERVAL_SEC)
        self.assertEqual(seen["max_wait"], config.SFX_POLL_MAX_WAIT_SEC)

    def test_config_cadence_reaches_both_halves(self):
        from venice.commands import sfx as cmd
        from venice.commands import _audio, _queue

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"sfx": {"poll_interval": 0.5, "max_wait": 60}}}
        seen, fake = self._spy()
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch.object(_queue, "build_client", lambda: (object(), 0)), \
             mock.patch.object(_audio, "retrieve_and_save", fake):
            rc = cmd._run_status(self._parse("sfx-status", "abcdef1234567890"))
        self.assertEqual(rc, 0)
        self.assertEqual(seen["poll_interval"], 0.5)
        self.assertEqual(seen["max_wait"], 60.0)


if __name__ == "__main__":
    unittest.main()
