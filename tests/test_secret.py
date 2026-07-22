"""Tests for `venice secret` CLI + wiring (#43). Placeholder secrets only.

Redirects config.CONFIG_DIR / SECRETS_FILE to a tmpdir (test_config `_Base` style),
mocks getpass + isatty so nothing reads a real terminal or the real store.
"""
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import venice.config as cfg
from venice import auth, cli


def _capture(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
    return rc, out.getvalue(), err.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name)
        self.cfg_dir = home / ".config" / "venice"
        self.secrets = self.cfg_dir / "secrets.json"
        for name, val in (("CONFIG_DIR", self.cfg_dir), ("SECRETS_FILE", self.secrets)):
            p = mock.patch.object(cfg, name, val)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ)  # snapshot/restore
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("VENICE_EMBED_API_KEY", None)

    def _set(self, name, value):
        """Drive `venice secret set <name>` with getpass returning `value`."""
        with mock.patch.object(auth.sys.stdin, "isatty", return_value=True), \
             mock.patch.object(auth.getpass, "getpass", return_value=value):
            return _capture(cli.main, ["secret", "set", name])


class TestSecretCLI(_Base):
    def test_set_then_ls_shows_length_not_value(self):
        rc, out, err = self._set("embed", "SUPER-SECRET-9")
        self.assertEqual(rc, 0)
        self.assertNotIn("SUPER-SECRET-9", out + err)   # never echoed

        rc, out, err = _capture(cli.main, ["secret", "ls"])
        self.assertEqual(rc, 0)
        self.assertIn("embed", out)
        self.assertIn("14 chars", out)                  # len("SUPER-SECRET-9")
        self.assertNotIn("SUPER-SECRET-9", out)         # value never printed

    def test_set_writes_0600(self):
        self._set("embed", "abc")
        self.assertEqual(stat.S_IMODE(self.secrets.stat().st_mode), 0o600)

    def test_ls_empty(self):
        rc, out, err = _capture(cli.main, ["secret", "ls"])
        self.assertEqual(rc, 0)
        self.assertIn("no secrets stored", err)

    def test_list_alias(self):
        self._set("embed", "abc")
        rc, out, _ = _capture(cli.main, ["secret", "list"])
        self.assertEqual(rc, 0)
        self.assertIn("embed", out)

    def test_rm(self):
        self._set("embed", "abc")
        rc, out, err = _capture(cli.main, ["secret", "rm", "embed"])
        self.assertEqual(rc, 0)
        self.assertIn("removed", err)
        # gone now -> rc 1
        rc, out, err = _capture(cli.main, ["secret", "rm", "embed"])
        self.assertEqual(rc, 1)
        self.assertIn("no secret named", err)

    def test_bare_secret_prints_help_rc2(self):
        rc, out, err = _capture(cli.main, ["secret"])
        self.assertEqual(rc, 2)
        self.assertIn("ACTION", err)

    def test_set_bad_name_rc1(self):
        rc, out, err = self._set("bad name", "x")
        self.assertEqual(rc, 1)
        self.assertIn("secret name must be", err)


if __name__ == "__main__":
    unittest.main()
