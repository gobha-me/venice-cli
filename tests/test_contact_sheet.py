"""`venice contact-sheet` / image_montage: engine shell-out with mocked subprocess.

No real montage/ffmpeg is invoked -- subprocess.run and shutil.which are patched.
Tests assert the constructed argv (montage vs ffmpeg tile filter, --label), the
engine auto-detection, the missing-tool guard, and command-layer exit codes.
"""
import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice import image_montage


class _FakeCP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run(calls, *, rc=0):
    def run(cmd, **kw):
        calls.append(cmd)
        return _FakeCP(rc, stderr="" if rc == 0 else "montage/ffmpeg boom")
    return run


def _which(*present):
    """which() that only 'finds' the named binaries."""
    def which(name):
        return "/usr/bin/" + name if name in present else None
    return which


class TestHelpers(unittest.TestCase):
    def test_parse_cell(self):
        self.assertEqual(image_montage._parse_cell("256x320"), (256, 320))
        self.assertIsNone(image_montage._parse_cell("256"))
        self.assertIsNone(image_montage._parse_cell("0x10"))
        self.assertIsNone(image_montage._parse_cell("axb"))

    def test_label_text_sanitizes(self):
        self.assertEqual(image_montage._label_text(Path("card-1.png")), "card-1")
        self.assertEqual(image_montage._label_text(Path("a:b'c.png")), "a_b_c")

    def test_select_engine_prefers_montage(self):
        with mock.patch("venice.image_montage.shutil.which", _which("montage", "ffmpeg")):
            self.assertEqual(image_montage.select_engine("auto"), "montage")
        with mock.patch("venice.image_montage.shutil.which", _which("ffmpeg")):
            self.assertEqual(image_montage.select_engine("auto"), "ffmpeg")
        with mock.patch("venice.image_montage.shutil.which", _which()):
            self.assertIsNone(image_montage.select_engine("auto"))

    def test_select_engine_forced(self):
        with mock.patch("venice.image_montage.shutil.which", _which("ffmpeg")):
            self.assertIsNone(image_montage.select_engine("montage"))
            self.assertEqual(image_montage.select_engine("ffmpeg"), "ffmpeg")


class TestCollectInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_dir_expands_to_sorted_images(self):
        for name in ("b.png", "a.jpg", "notes.txt", "c.webp"):
            Path(name).write_bytes(b"x")
        got = [p.name for p in image_montage.collect_inputs(["."])]
        self.assertEqual(got, ["a.jpg", "b.png", "c.webp"])  # txt dropped, sorted

    def test_glob_and_explicit_files(self):
        for name in ("card-1.png", "card-2.png", "other.png"):
            Path(name).write_bytes(b"x")
        got = [p.name for p in image_montage.collect_inputs(["card-*.png"])]
        self.assertEqual(got, ["card-1.png", "card-2.png"])

    def test_no_images_is_empty(self):
        Path("readme.txt").write_bytes(b"x")
        self.assertEqual(image_montage.collect_inputs(["."]), [])


class TestContactSheetArgv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        for name in ("card-1.png", "card-2.png", "card-3.png"):
            Path(name).write_bytes(b"x")

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_montage_argv(self):
        calls = []
        with mock.patch("venice.image_montage.shutil.which", _which("montage", "ffmpeg")), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls)):
            rc = image_montage.contact_sheet(["."], Path("sheet.png"),
                                             cols=2, label=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        argv = calls[0]
        self.assertEqual(argv[0], "montage")
        joined = " ".join(argv)
        self.assertIn("-tile 2x", joined)
        self.assertIn("-geometry 256x320+4+4", joined)
        self.assertIn("-label", argv)
        self.assertEqual(argv[-1], "sheet.png")

    def test_ffmpeg_fallback_argv(self):
        calls = []
        with mock.patch("venice.image_montage.shutil.which", _which("ffmpeg")), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls)):
            rc = image_montage.contact_sheet(["."], Path("sheet.png"),
                                             cols=2, label=True)
        self.assertEqual(rc, 0)
        argv = calls[0]
        self.assertEqual(argv[0], "ffmpeg")
        joined = " ".join(argv)
        self.assertEqual(argv.count("-i"), 3)        # one per image
        self.assertIn("tile=2x2", joined)            # 3 imgs, 2 cols -> 2 rows
        self.assertIn("concat=n=3", joined)
        self.assertIn("drawtext", joined)            # --label
        self.assertIn("-map [out]", joined)

    def test_ffmpeg_no_label_has_no_drawtext(self):
        calls = []
        with mock.patch("venice.image_montage.shutil.which", _which("ffmpeg")), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls)):
            rc = image_montage.contact_sheet(["."], Path("sheet.png"), cols=3)
        self.assertEqual(rc, 0)
        self.assertNotIn("drawtext", " ".join(calls[0]))
        self.assertIn("tile=3x1", " ".join(calls[0]))

    def test_no_engine_is_2(self):
        calls = []
        with mock.patch("venice.image_montage.shutil.which", _which()), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls)):
            rc = image_montage.contact_sheet(["."], Path("sheet.png"))
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])

    def test_no_inputs_is_6(self):
        calls = []
        os.mkdir("empty")
        with mock.patch("venice.image_montage.shutil.which", _which("ffmpeg")), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls)):
            rc = image_montage.contact_sheet(["empty"], Path("sheet.png"))
        self.assertEqual(rc, 6)
        self.assertEqual(calls, [])

    def test_bad_cell_is_2(self):
        rc = image_montage.contact_sheet(["."], Path("sheet.png"), cell="nope")
        self.assertEqual(rc, 2)

    def test_bad_cols_is_2(self):
        rc = image_montage.contact_sheet(["."], Path("sheet.png"), cols=0)
        self.assertEqual(rc, 2)

    def test_dry_run_runs_nothing_even_without_engine(self):
        calls = []
        with mock.patch("venice.image_montage.shutil.which", _which()), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls)):
            rc = image_montage.contact_sheet(["."], Path("sheet.png"), dry_run=True)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [])

    def test_engine_failure_is_5(self):
        calls = []
        with mock.patch("venice.image_montage.shutil.which", _which("ffmpeg")), \
             mock.patch("venice.image_montage.subprocess.run", _fake_run(calls, rc=1)):
            rc = image_montage.contact_sheet(["."], Path("sheet.png"))
        self.assertEqual(rc, 5)


def _cmd_args(**overrides):
    base = dict(
        inputs=["."], output=None, cols=4, cell="256x320", label=False,
        background="white", padding=4, engine="auto", dry_run=False,
        command="contact-sheet",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestContactSheetCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_no_inputs_is_6(self):
        from venice.commands import contact_sheet as cs_cmd
        rc = cs_cmd._run(_cmd_args(inputs=["."]))  # empty dir
        self.assertEqual(rc, 6)

    def test_dry_run_on_existing_images(self):
        from venice.commands import contact_sheet as cs_cmd
        Path("card-1.png").write_bytes(b"x")
        rc = cs_cmd._run(_cmd_args(inputs=["."], dry_run=True))
        self.assertEqual(rc, 0)  # prints command, spawns nothing


if __name__ == "__main__":
    unittest.main()
