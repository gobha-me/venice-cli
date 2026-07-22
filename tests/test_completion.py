"""Tests for `venice completion [bash|zsh]` (#41).

Pure-Python + hermetic: exercises the generator through `cli.main` and asserts the
emitted scripts stay in sync with the real parser (the drift guard). No network, no
key, no subprocess.
"""
import argparse
import io
import sys
import unittest
from unittest import mock

from venice import cli
from venice.commands import completion


def _capture(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
    return rc, out.getvalue(), err.getvalue()


def _top_level_names():
    """Canonical top-level subcommand names straight from the live parser."""
    parser = cli.build_parser()
    action = next(a for a in parser._actions
                  if isinstance(a, argparse._SubParsersAction))
    return [pa.dest for pa in action._choices_actions]


class TestBashCompletion(unittest.TestCase):
    def setUp(self):
        rc, self.out, _ = _capture(cli.main, ["completion", "bash"])
        self.assertEqual(rc, 0)

    def test_registers_the_function(self):
        self.assertIn("_venice()", self.out)
        self.assertIn("complete -F _venice venice", self.out)

    def test_mentions_representative_commands(self):
        for name in ("chat", "mcp-serve", "bg-remove", "config", "secret",
                     "sessions", "completion"):
            self.assertIn(name, self.out)

    def test_nested_actions_and_aliases(self):
        # config actions
        self.assertIn("add list remove show get set unset", self.out)
        # secret canonical + aliases both present
        for tok in ("set ls list rm remove",):
            self.assertIn(tok, self.out)
        # sessions actions + aliases (#47)
        self.assertIn("ls list show cat rm remove", self.out)

    def test_drift_guard_every_command_present(self):
        """The generator must emit every registered top-level command."""
        missing = [n for n in _top_level_names() if n not in self.out]
        self.assertEqual(missing, [], f"commands missing from bash script: {missing}")


class TestZshCompletion(unittest.TestCase):
    def setUp(self):
        rc, self.out, _ = _capture(cli.main, ["completion", "zsh"])
        self.assertEqual(rc, 0)

    def test_compdef_header(self):
        self.assertTrue(self.out.startswith("#compdef venice"))
        self.assertIn('_venice "$@"', self.out)

    def test_has_command_descriptions(self):
        # _describe entries carry help text, sanitized of ':'.
        self.assertIn("_describe", self.out)
        self.assertIn("'chat:", self.out)

    def test_drift_guard_every_command_present(self):
        missing = [n for n in _top_level_names() if n not in self.out]
        self.assertEqual(missing, [], f"commands missing from zsh script: {missing}")


class TestArgparseWiring(unittest.TestCase):
    def test_invalid_shell_exits_2(self):
        with self.assertRaises(SystemExit) as cm:
            _capture(cli.main, ["completion", "fish"])
        self.assertEqual(cm.exception.code, 2)

    def test_bare_completion_is_an_error(self):
        # `shell` is a required positional -> argparse exits 2.
        with self.assertRaises(SystemExit) as cm:
            _capture(cli.main, ["completion"])
        self.assertEqual(cm.exception.code, 2)


class TestIntrospection(unittest.TestCase):
    """Lock the alias-recovery behavior the generator depends on."""

    def test_secret_aliases_recovered(self):
        model = completion._walk_parser(cli.build_parser())
        secret = next(c for c in model["commands"] if c["name"] == "secret")
        actions = {s["name"]: s["aliases"] for s in secret["subcommands"]}
        self.assertEqual(actions.get("ls"), ["list"])
        self.assertEqual(actions.get("rm"), ["remove"])
        # `set` has no alias.
        self.assertEqual(actions.get("set"), [])

    def test_flags_are_option_strings_only(self):
        model = completion._walk_parser(cli.build_parser())
        login = next(c for c in model["commands"] if c["name"] == "login")
        self.assertIn("--embed", login["flags"])
        # positionals (no option strings) never leak in.
        self.assertNotIn("shell", login["flags"])


if __name__ == "__main__":
    unittest.main()
