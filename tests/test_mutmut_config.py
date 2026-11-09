"""The live mutmut config must copy scripts/ into mutants/ (#271)."""
from __future__ import annotations

import ast
import configparser
import re
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11; the test suite stays stdlib-only.
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]


def _parse_tool_mutmut_without_tomllib(text):
    section = re.search(
        r"^\[tool\.mutmut\][ \t]*(?:#[^\n]*)?$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL
    )
    if section is None:
        return None
    also_copy = re.search(r"^also_copy\s*=\s*(\[.*?\])", section.group(1), re.MULTILINE | re.DOTALL)
    if also_copy is None:
        return {}
    return {"also_copy": ast.literal_eval(also_copy.group(1))}


def _tool_mutmut_table(text):
    if tomllib is None:
        return _parse_tool_mutmut_without_tomllib(text)
    return tomllib.loads(text).get("tool", {}).get("mutmut")


def _read_also_copy(root):
    """Resolve also_copy in mutmut 3.8.1's order: pyproject table, else setup.cfg."""
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        table = _tool_mutmut_table(pyproject.read_text(encoding="utf-8"))
        if table is not None:
            return table.get("also_copy", [])

    parser = configparser.ConfigParser()
    parser.read(root / "setup.cfg", encoding="utf-8")
    try:
        value = parser.get("mutmut", "also_copy")
    except (configparser.NoSectionError, configparser.NoOptionError):
        return []
    return [line for line in value.split("\n") if line]


class LiveMutmutConfigTests(unittest.TestCase):
    def test_also_copy_includes_the_scripts_directory(self):
        entries = _read_also_copy(ROOT)
        self.assertTrue(entries, "mutmut also_copy is not set")
        self.assertIn(Path("scripts"), [Path(entry) for entry in entries])

    def test_contract_script_exists_to_be_copied(self):
        self.assertTrue((ROOT / "scripts" / "openapi_contract.py").is_file())


class MutmutConfigResolutionTests(unittest.TestCase):
    def _root(self, **files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for name, content in files.items():
            (root / name.replace("_", ".")).write_text(content, encoding="utf-8")
        return root

    def test_pyproject_table_takes_precedence_over_setup_cfg(self):
        root = self._root(
            pyproject_toml='[tool.mutmut]\nalso_copy = ["from-pyproject/"]\n',
            setup_cfg="[mutmut]\nalso_copy = from-setup-cfg/\n",
        )
        self.assertEqual(_read_also_copy(root), ["from-pyproject/"])

    def test_pyproject_table_without_also_copy_does_not_fall_back(self):
        root = self._root(
            pyproject_toml="[tool.mutmut]\n",
            setup_cfg="[mutmut]\nalso_copy = from-setup-cfg/\n",
        )
        self.assertEqual(_read_also_copy(root), [])

    def test_setup_cfg_is_used_without_a_pyproject_table(self):
        root = self._root(
            pyproject_toml='[project]\nname = "x"\n',
            setup_cfg="[mutmut]\nalso_copy =\n    scripts/\n    data/\n",
        )
        self.assertEqual(_read_also_copy(root), ["scripts/", "data/"])

    def test_unset_also_copy_resolves_to_empty(self):
        self.assertEqual(_read_also_copy(self._root()), [])

    def test_stdlib_fallback_reads_only_the_mutmut_table(self):
        text = (
            '[tool.other]\nalso_copy = ["wrong/"]\n\n'
            '[tool.mutmut]  # comment\nalso_copy = [\n  "scripts/",  # why\n]\n\n'
            '[tool.later]\nalso_copy = ["late/"]\n'
        )
        self.assertEqual(_parse_tool_mutmut_without_tomllib(text), {"also_copy": ["scripts/"]})
        self.assertEqual(_parse_tool_mutmut_without_tomllib("[tool.mutmut]\n"), {})
        self.assertIsNone(_parse_tool_mutmut_without_tomllib("[tool.other]\n"))


if __name__ == "__main__":
    unittest.main()
