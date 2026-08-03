"""Guard for the invisible-character scanner (#109).

The scan itself lives in `tests/_hygiene.py`; this asserts it actually bites.

Two halves, and the second matters as much as the first:

* every invisible class is caught -- otherwise the guard is decoration;
* every *visible* punctuation mark already in the repo stays legal -- otherwise
  the guard is noise, and a noisy guard gets deleted the first time someone
  types an em dash.

No literal suspect character appears in this file. They are all built with
`chr()` at runtime, so this source stays plain ASCII and is scanned by
`test_the_repository_is_clean` on the same terms as everything else. A guard
that has to exempt itself from its own rule is not a guard.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests import _hygiene

# (codepoint, expected label) -- one representative per class the scanner
# claims to cover. Named rather than inlined so a failure says which attack
# slipped through.
SUSPECT = [
    (0x200B, "INVISIBLE_FORMAT"),  # zero width space
    (0x200D, "INVISIBLE_FORMAT"),  # zero width joiner
    (0xFEFF, "INVISIBLE_FORMAT"),  # BOM used mid-file
    (0x202E, "INVISIBLE_FORMAT"),  # right-to-left override (Trojan Source)
    (0x2066, "INVISIBLE_FORMAT"),  # left-to-right isolate
    (0xE0041, "INVISIBLE_FORMAT"),  # tag capital A -- the glyphless alphabet
    (0xFE0F, "VARIATION_SELECTOR"),
    (0xE0101, "VARIATION_SELECTOR"),
    (0xE000, "PRIVATE_USE"),
    (0xF0001, "PRIVATE_USE"),
    (0x0007, "CONTROL"),  # bell
    (0x00A0, "UNUSUAL_SPACE"),  # non-breaking space
    (0x2003, "UNUSUAL_SPACE"),  # em space
    (0x2028, "LINE_SEPARATOR"),
]

# Everything the repo legitimately uses today. If a change to `classify` starts
# flagging these, the scan has become unusable and this test says so.
LEGITIMATE = [0x2014, 0x2192, 0x2026, 0x00B7, 0x2264, 0x2265, 0x00D7, 0x2022]


class TestClassify(unittest.TestCase):
    def test_every_suspect_class_is_caught(self):
        for code, label in SUSPECT:
            with self.subTest(codepoint=f"U+{code:04X}"):
                self.assertEqual(_hygiene.classify(chr(code)), label)

    def test_visible_punctuation_stays_legal(self):
        for code in LEGITIMATE:
            with self.subTest(codepoint=f"U+{code:04X}"):
                self.assertIsNone(_hygiene.classify(chr(code)))

    def test_plain_ascii_is_legal(self):
        for ch in " \t\n\rabcXYZ0189_-#\"'()[]{}<>=/\\|@$%&*+~`^!?.,;:":
            with self.subTest(char=repr(ch)):
                self.assertIsNone(_hygiene.classify(ch))


class TestScanFile(unittest.TestCase):
    def test_reports_position_and_codepoint(self):
        text = "alpha = 1\nbeta =" + chr(0x200B) + " 2\n"
        findings = _hygiene.scan_file("x.py", text.encode("utf-8"))
        self.assertEqual(len(findings), 1)
        found = findings[0]
        self.assertEqual((found.path, found.line, found.col), ("x.py", 2, 7))
        self.assertEqual(found.label, "INVISIBLE_FORMAT")
        self.assertIn("U+200B", found.detail)

    def test_clean_source_yields_nothing(self):
        source = '"""Doc with an em dash ' + chr(0x2014) + ' and ok."""\nx = 1\n'
        self.assertEqual(_hygiene.scan_file("x.py", source.encode("utf-8")), [])

    def test_leading_bom_is_a_finding(self):
        findings = _hygiene.scan_file("x.py", b"\xef\xbb\xbfx = 1\n")
        self.assertEqual([f.label for f in findings], ["UTF8_BOM"])

    def test_invalid_utf8_is_a_finding(self):
        findings = _hygiene.scan_file("x.py", b"x = '\xff\xfe'\n")
        self.assertEqual([f.label for f in findings], ["NOT_UTF8"])

    def test_homoglyph_identifier_is_caught(self):
        # Cyrillic 'a' (U+0430) inside an otherwise ordinary name.
        source = "api_key = 1\n" + chr(0x0430) + "pi_key = 2\n"
        findings = _hygiene.scan_file("x.py", source.encode("utf-8"))
        self.assertEqual([f.label for f in findings], ["NON_ASCII_IDENTIFIER"])
        self.assertEqual(findings[0].line, 2)

    def test_visible_punctuation_in_a_docstring_is_not_an_identifier_finding(self):
        # The reason the identifier pass tokenizes instead of pattern-matching.
        source = '"""Ellipsis ' + chr(0x2026) + ' and arrow ' + chr(0x2192) + '."""\n'
        self.assertEqual(_hygiene.scan_file("x.py", source.encode("utf-8")), [])

    def test_identifier_pass_is_python_only(self):
        source = chr(0x0430) + "pi_key: not python\n"
        self.assertEqual(_hygiene.scan_file("notes.md", source.encode("utf-8")), [])

    def test_unparseable_python_still_gets_the_character_sweep(self):
        # `make lint` owns syntax errors; the character sweep must not be
        # skippable by handing the tokenizer something it cannot read.
        source = "def broken(\n" + chr(0x200B) + "\n"
        labels = [f.label for f in _hygiene.scan_file("x.py", source.encode("utf-8"))]
        self.assertIn("INVISIBLE_FORMAT", labels)


class TestScanRepo(unittest.TestCase):
    def test_the_repository_is_clean(self):
        findings = _hygiene.scan_repo()
        if findings is None:
            self.skipTest("not a git checkout")
        self.assertEqual(findings, [], "\n" + _hygiene.format_findings(findings))

    def test_the_scan_covers_the_whole_tracked_set(self):
        # A scan that silently looked at nothing would pass the test above.
        names = _hygiene.tracked_files()
        if names is None:
            self.skipTest("not a git checkout")
        self.assertIn("src/venice/__init__.py", names)
        self.assertIn("tests/_hygiene.py", names)
        self.assertIn(".github/workflows/test.yml", names)
        self.assertGreater(len(names), 100)

    def test_a_planted_payload_is_found_end_to_end(self):
        # Proves the git-driven path reports, not just `scan_file`.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            payload = root / "payload.py"
            payload.write_text("x = 1  #" + chr(0xE0041) + "\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", "payload.py"], check=True
            )
            findings = _hygiene.scan_repo(root)
        self.assertEqual([f.path for f in findings], ["payload.py"])
        self.assertEqual(findings[0].label, "INVISIBLE_FORMAT")

    def test_without_git_the_scan_says_so_rather_than_passing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_hygiene.scan_repo(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
