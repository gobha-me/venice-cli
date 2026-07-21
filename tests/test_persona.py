"""Unit tests for the persona store (`venice.commands._persona`, #68).

Personas are local files under ~/.config/venice/personas/. These tests point
config.PERSONAS_DIR at a tempdir and never touch the real config dir. The
security-critical property is that a name can only ever resolve to a file inside
the personas dir -- traversal (``../credentials``) and symlink escapes must fail.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _persona


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "personas"
        self.dir.mkdir()
        p = mock.patch("venice.config.PERSONAS_DIR", self.dir)
        p.start()
        self.addCleanup(p.stop)

    def _write(self, name, text):
        (self.dir / name).write_text(text, encoding="utf-8")


class TestAvailable(_Base):
    def test_empty_dir_lists_nothing(self):
        self.assertEqual(_persona.available(), [])

    def test_missing_dir_lists_nothing(self):
        # Point at a dir that does not exist -- must not raise.
        with mock.patch("venice.config.PERSONAS_DIR", self.dir / "nope"):
            self.assertEqual(_persona.available(), [])

    def test_lists_name_and_first_line_sorted(self):
        self._write("pirate.md", "\n  You are a pirate.\nmore\n")
        self._write("coach.txt", "Motivate the user.")
        self.assertEqual(
            _persona.available(),
            [("coach", "Motivate the user."), ("pirate", "You are a pirate.")],
        )

    def test_ignores_other_extensions(self):
        self._write("notes.org", "nope")
        self._write("real.md", "yes")
        self.assertEqual(_persona.available(), [("real", "yes")])

    def test_md_wins_over_txt_for_same_stem(self):
        self._write("dup.txt", "from txt")
        self._write("dup.md", "from md")
        self.assertEqual(_persona.available(), [("dup", "from md")])


class TestLoad(_Base):
    def test_load_reads_content(self):
        self._write("pirate.md", "You are a pirate.\n")
        self.assertEqual(_persona.load("pirate"), "You are a pirate.\n")

    def test_load_prefers_md_over_txt(self):
        self._write("dup.txt", "from txt")
        self._write("dup.md", "from md")
        self.assertEqual(_persona.load("dup"), "from md")

    def test_load_txt_when_no_md(self):
        self._write("only.txt", "plain text")
        self.assertEqual(_persona.load("only"), "plain text")

    def test_missing_persona_raises(self):
        with self.assertRaises(_persona.PersonaError) as cm:
            _persona.load("ghost")
        self.assertIn("ghost", str(cm.exception))

    def test_resolve_path_returns_md_path(self):
        self._write("pirate.md", "x")
        self.assertEqual(_persona.resolve_path("pirate"), self.dir / "pirate.md")


class TestTraversalRejection(_Base):
    def test_parent_traversal_rejected(self):
        # A credentials file one level up must be unreachable.
        (self.dir.parent / "credentials").write_text("SECRET", encoding="utf-8")
        for bad in ("../credentials", "..", "a/b", "a\\b", "/etc/passwd", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(_persona.PersonaError):
                    _persona.load(bad)

    def test_symlink_escape_rejected(self):
        # A bare name (no separators) that is a symlink out of the dir must still
        # be refused by the realpath containment check.
        secret = self.dir.parent / "secret.md"
        secret.write_text("SECRET", encoding="utf-8")
        link = self.dir / "evil.md"
        try:
            os.symlink(secret, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unsupported on this platform")
        with self.assertRaises(_persona.PersonaError) as cm:
            _persona.load("evil")
        self.assertIn("escapes", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
