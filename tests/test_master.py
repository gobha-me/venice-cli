"""`venice master` / audio_post: ffmpeg mastering with mocked subprocess.

No real ffmpeg is invoked -- subprocess.run and shutil.which are patched. Tests
assert the constructed ffmpeg argv (2-pass loudnorm, WAV codec/rate, seamless
loop filtergraph), the pre-flight guards, and the command-layer exit codes.
"""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice import audio_post

_LOUDNORM_JSON = json.dumps({
    "input_i": "-18.50",
    "input_tp": "-3.20",
    "input_lra": "5.40",
    "input_thresh": "-28.70",
    "output_i": "-16.00",
    "output_tp": "-1.50",
    "output_lra": "5.10",
    "output_thresh": "-26.10",
    "normalization_type": "dynamic",
    "target_offset": "0.30",
})


class _FakeCP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run(calls, *, duration="5.000000", pass2_rc=0, pass1_stderr=None):
    """Fake subprocess.run: records argv, fakes ffprobe duration + loudnorm JSON."""
    def run(cmd, **kw):
        calls.append(cmd)
        if cmd and cmd[0] == "ffprobe":
            return _FakeCP(0, stdout=duration + "\n")
        joined = " ".join(cmd)
        if "print_format=json" in joined:  # pass 1 (measure)
            stderr = _LOUDNORM_JSON if pass1_stderr is None else pass1_stderr
            return _FakeCP(0, stderr="ffmpeg noise...\n" + stderr + "\n")
        return _FakeCP(pass2_rc, stderr="" if pass2_rc == 0 else "encode boom")
    return run


def _which_all(name):
    return "/usr/bin/" + name


class TestAudioPostHelpers(unittest.TestCase):
    def test_parse_loudnorm_json(self):
        out = audio_post._parse_loudnorm_json("blah blah\n" + _LOUDNORM_JSON + "\ntrailing")
        self.assertEqual(out["input_i"], "-18.50")
        self.assertEqual(out["target_offset"], "0.30")

    def test_parse_loudnorm_json_missing(self):
        with self.assertRaises(ValueError):
            audio_post._parse_loudnorm_json("no json here at all")

    def test_loudnorm_pass1_vs_pass2(self):
        p1 = audio_post._loudnorm(-16.0, -1.0, None)
        self.assertIn("print_format=json", p1)
        self.assertNotIn("measured_I", p1)
        p2 = audio_post._loudnorm(-16.0, -1.0, {
            "input_i": "-18.50", "input_tp": "-3.20", "input_lra": "5.40",
            "input_thresh": "-28.70", "target_offset": "0.30",
        })
        self.assertIn("measured_I=-18.50", p2)
        self.assertIn("offset=0.30", p2)
        self.assertIn("linear=true", p2)

    def test_loop_filter_boundaries(self):
        g = audio_post._loop_filter("m", 10.0, 2.0)
        self.assertIn("asplit=3", g)
        self.assertIn("atrim=8:10", g)   # tail = dur-cf : dur
        self.assertIn("atrim=2:8", g)    # middle = cf : dur-cf
        self.assertIn("concat=n=2", g)

    def test_default_output(self):
        self.assertEqual(audio_post.default_output(Path("a/b.mp3")).name, "b.mastered.wav")


class TestMasterFunction(unittest.TestCase):
    def test_two_pass_argv(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", _which_all), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls)):
            rc = audio_post.master(Path("in.mp3"), Path("out.wav"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)  # no ffprobe without --loop
        self.assertIn("print_format=json", " ".join(calls[0]))
        p2 = " ".join(calls[1])
        self.assertIn("-ar 48000", p2)
        self.assertIn("pcm_s24le", p2)
        self.assertIn("measured_I=-18.50", p2)
        self.assertIn("measured_TP=-3.20", p2)
        self.assertIn("offset=0.30", p2)

    def test_bit_depth_maps_to_codec(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", _which_all), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls)):
            rc = audio_post.master(Path("in.mp3"), Path("out.wav"),
                                   bit_depth=16, sample_rate=44100)
        self.assertEqual(rc, 0)
        p2 = " ".join(calls[1])
        self.assertIn("pcm_s16le", p2)
        self.assertIn("-ar 44100", p2)

    def test_bad_bit_depth(self):
        rc = audio_post.master(Path("in.mp3"), Path("out.wav"), bit_depth=20)
        self.assertEqual(rc, 2)

    def test_missing_ffmpeg_no_run(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", lambda n: None), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls)):
            rc = audio_post.master(Path("in.mp3"), Path("out.wav"))
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])

    def test_loop_argv(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", _which_all), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls, duration="5.0")):
            rc = audio_post.master(Path("in.wav"), Path("out.wav"),
                                   loop=True, loop_crossfade=1.0)
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "ffprobe")
        p2 = " ".join(calls[-1])
        self.assertIn("-filter_complex", p2)
        self.assertIn("asplit=3", p2)
        self.assertIn("concat=n=2", p2)
        self.assertIn("-map [out]", p2)

    def test_loop_too_short(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", _which_all), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls, duration="1.0")):
            rc = audio_post.master(Path("in.wav"), Path("out.wav"), loop=True)
        self.assertEqual(rc, 2)  # 1.0s <= 2*2.0s crossfade
        self.assertEqual(len(calls), 1)  # only the ffprobe
        self.assertEqual(calls[0][0], "ffprobe")

    def test_pass2_failure_is_5(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", _which_all), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls, pass2_rc=1)):
            rc = audio_post.master(Path("in.mp3"), Path("out.wav"))
        self.assertEqual(rc, 5)

    def test_pass1_unparseable_is_5(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", _which_all), \
             mock.patch("venice.audio_post.subprocess.run",
                        _fake_run(calls, pass1_stderr="no json")):
            rc = audio_post.master(Path("in.mp3"), Path("out.wav"))
        self.assertEqual(rc, 5)

    def test_dry_run_runs_nothing_even_without_ffmpeg(self):
        calls = []
        with mock.patch("venice.audio_post.shutil.which", lambda n: None), \
             mock.patch("venice.audio_post.subprocess.run", _fake_run(calls)):
            rc = audio_post.master(Path("in.mp3"), Path("out.wav"),
                                   loop=True, dry_run=True)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])


def _cmd_args(**overrides):
    base = dict(
        input=Path("in.wav"), output=None, dry_run=False,
        lufs=-16.0, true_peak=-1.0, sample_rate=48000, bit_depth=24,
        loop=False, loop_crossfade=2.0, command="master",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestMasterCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_input_not_found_is_6(self):
        from venice.commands import master as master_cmd
        rc = master_cmd._run(_cmd_args(input=Path("nope.wav")))
        self.assertEqual(rc, 6)

    def test_dry_run_on_existing_file(self):
        from venice.commands import master as master_cmd
        p = Path("in.wav")
        p.write_bytes(b"x")
        rc = master_cmd._run(_cmd_args(input=p, dry_run=True))
        self.assertEqual(rc, 0)  # prints commands, spawns no ffmpeg


if __name__ == "__main__":
    unittest.main()
