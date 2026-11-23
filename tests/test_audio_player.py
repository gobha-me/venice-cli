"""Exercise audio playback through real temporary executables (issue #264)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from venice import audio_player

# README's documented precedence, independent of the implementation's table.
PLAYERS = ("paplay", "aplay", "ffplay", "mpg123", "play", "afplay")


def _python_launcher(script):
    # A shebang splits interpreter paths at spaces. The shell preserves both
    # the interpreter and the Python program as single quoted arguments.
    return (
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(script)} "$@"\n'
    )


class AudioPlayerBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="audio player ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "player tools"
        self.bin.mkdir()
        self.record = self.root / "invocation.json"
        self.audio = self.root / "sound with spaces.wav"
        self.audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        # PATH is the executable-discovery boundary. Never replace the player
        # module, shutil.which, or subprocess.run with a test double.
        environment = patch.dict(os.environ, {"PATH": str(self.bin)})
        environment.start()
        self.addCleanup(environment.stop)

    def install_player(self, name, *, status=0, executable=True):
        path = self.bin / name
        path.write_text(
            _python_launcher(
                "import json\n"
                "from pathlib import Path\n"
                "import sys\n"
                f"Path({str(self.record)!r}).write_text("
                f"json.dumps({{'player': {name!r}, 'argv': sys.argv[1:]}}))\n"
                f"raise SystemExit({status!r})\n"
            ),
            encoding="utf-8",
        )
        path.chmod(0o755 if executable else 0o644)
        return path

    def recorded_call(self):
        return json.loads(self.record.read_text(encoding="utf-8"))

    def test_empty_path_has_no_player_and_play_returns_false(self):
        self.assertIsNone(audio_player.find_player())
        self.assertIs(audio_player.has_player(), False)
        errors, output = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(output):
            self.assertIs(audio_player.play(self.audio), False)
        self.assertEqual(
            errors.getvalue(),
            "no audio player found (tried " + "/".join(PLAYERS) + ")\n"
            f"saved to {self.audio}; play manually with your preferred tool.\n",
        )
        self.assertEqual(output.getvalue(), "")
        self.assertFalse(self.record.exists())

    def test_first_available_player_wins_over_all_later_candidates(self):
        for name in PLAYERS:
            self.install_player(name)
        self.assertIs(audio_player.has_player(), True)
        self.assertIs(audio_player.play(self.audio), True)
        self.assertEqual(self.recorded_call()["player"], "paplay")

    def test_missing_first_candidate_selects_second(self):
        self.install_player("aplay")
        self.install_player("ffplay")
        self.assertIs(audio_player.has_player(), True)
        self.assertIs(audio_player.play(self.audio), True)
        self.assertEqual(self.recorded_call()["player"], "aplay")

    def test_each_documented_player_can_be_selected_without_earlier_players(self):
        for name in PLAYERS:
            with self.subTest(player=name):
                path = self.install_player(name)
                self.assertIsNotNone(audio_player.find_player())
                self.assertIs(audio_player.has_player(), True)
                self.assertIs(audio_player.play(self.audio), True)
                call = self.recorded_call()
                self.assertEqual(call["player"], name)
                self.assertEqual(call["argv"][-1], str(self.audio))
                self.assertEqual(call["argv"].count(str(self.audio)), 1)
                path.unlink()

    def test_non_executable_first_candidate_falls_back_to_second(self):
        self.install_player("paplay", executable=False)
        self.install_player("aplay")
        self.assertIs(audio_player.play(self.audio), True)
        self.assertEqual(self.recorded_call()["player"], "aplay")

    def test_only_non_executable_candidate_is_not_a_player(self):
        self.install_player("paplay", executable=False)
        self.assertIsNone(audio_player.find_player())
        self.assertIs(audio_player.has_player(), False)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIs(audio_player.play(self.audio), False)
        self.assertFalse(self.record.exists())

    def test_nonzero_exit_returns_false_without_trying_another_player(self):
        self.install_player("aplay")
        for status in (1, 7, 127):
            with self.subTest(status=status):
                self.install_player("paplay", status=status)
                errors, output = io.StringIO(), io.StringIO()
                with (
                    contextlib.redirect_stderr(errors),
                    contextlib.redirect_stdout(output),
                ):
                    self.assertIs(audio_player.play(self.audio), False)
                self.assertEqual(
                    errors.getvalue(),
                    f"paplay exited {status}; saved to {self.audio}\n",
                )
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(self.recorded_call()["player"], "paplay")

    def test_executable_with_invalid_format_returns_false(self):
        path = self.bin / "paplay"
        path.write_bytes(b"this is not an executable format\n")
        path.chmod(0o755)
        self.assertIs(audio_player.has_player(), True)
        errors, output = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(output):
            self.assertIs(audio_player.play(self.audio), False)
        # The OS error text varies; the diagnostic must still identify the
        # aborted playback and the saved file without contaminating stdout.
        self.assertTrue(errors.getvalue().startswith("playback aborted ("))
        self.assertTrue(errors.getvalue().endswith(f"); file is at {self.audio}\n"))
        self.assertIn("paplay", errors.getvalue())
        self.assertEqual(output.getvalue(), "")
        self.assertFalse(self.record.exists())

    def test_broken_player_symlink_is_unavailable(self):
        (self.bin / "paplay").symlink_to(self.bin / "missing-target")
        self.assertIsNone(audio_player.find_player())
        self.assertIs(audio_player.has_player(), False)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIs(audio_player.play(self.audio), False)

    def test_spaces_in_player_and_audio_paths_are_single_arguments(self):
        self.install_player("paplay")
        self.assertIs(audio_player.play(self.audio), True)
        self.assertEqual(
            self.recorded_call(),
            {"player": "paplay", "argv": [str(self.audio)]},
        )

    def test_player_cannot_consume_cli_input_or_write_to_cli_streams(self):
        player = self.install_player("paplay")
        player.write_text(
            _python_launcher(
                "import json\n"
                "from pathlib import Path\n"
                "import sys\n"
                "print('player stdout must stay private')\n"
                "print('player stderr must stay private', file=sys.stderr)\n"
                f"Path({str(self.record)!r}).write_text("
                "json.dumps({'stdin': sys.stdin.read()}))\n"
            ),
            encoding="utf-8",
        )
        # Change actual descriptors: Python's redirect_stdout cannot capture
        # output inherited by a child process. All descriptors are restored
        # even if playback or an assertion fails.
        with (
            tempfile.TemporaryFile() as incoming,
            tempfile.TemporaryFile() as outgoing,
            tempfile.TemporaryFile() as errors,
        ):
            incoming.write(b"CLI input belongs to the user\n")
            incoming.seek(0)
            sys.stdout.flush()
            sys.stderr.flush()
            saved = []
            try:
                for descriptor, stream in enumerate((incoming, outgoing, errors)):
                    saved.append((descriptor, os.dup(descriptor)))
                    os.dup2(stream.fileno(), descriptor)
                result = audio_player.play(self.audio)
            finally:
                for descriptor, original in reversed(saved):
                    os.dup2(original, descriptor)
                    os.close(original)
            self.assertIs(result, True)
            self.assertEqual(self.recorded_call(), {"stdin": ""})
            outgoing.seek(0)
            errors.seek(0)
            self.assertEqual(outgoing.read(), b"")
            self.assertEqual(errors.read(), b"")


if __name__ == "__main__":
    unittest.main()
