"""Tests for `venice login` (+ --embed, #43). First getpass/login coverage.

Placeholder keys only; nothing reads a real terminal or the real credentials.
"""
import io
import os
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


class TestLogin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        home = Path(self.tmp.name)
        self.cfg_dir = home / ".config" / "venice"
        for name, val in (
            ("CONFIG_DIR", self.cfg_dir),
            ("CREDS_FILE", self.cfg_dir / "credentials"),
            ("SECRETS_FILE", self.cfg_dir / "secrets.json"),
        ):
            p = mock.patch.object(cfg, name, val)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("VENICE_API_KEY", None)
        os.environ.pop("VENICE_EMBED_API_KEY", None)

    def _login(self, argv, value):
        with mock.patch.object(auth.sys.stdin, "isatty", return_value=True), \
             mock.patch.object(auth.getpass, "getpass", return_value=value):
            return _capture(cli.main, argv)

    def test_login_stores_main_key_length_only(self):
        rc, out, err = self._login(["login"], "MAIN-KEY-1234")
        self.assertEqual(rc, 0)
        self.assertNotIn("MAIN-KEY-1234", out + err)
        self.assertIn("13-char", err)
        self.assertEqual(auth.load_key(), "MAIN-KEY-1234")

    def test_login_embed_stores_named_secret(self):
        rc, out, err = self._login(["login", "--embed"], "EMBED-KEY-9")
        self.assertEqual(rc, 0)
        self.assertNotIn("EMBED-KEY-9", out + err)
        self.assertIn("embed", err)
        self.assertEqual(auth.load_secret("embed"), "EMBED-KEY-9")
        # --embed must NOT touch the main credentials file
        self.assertFalse(cfg.CREDS_FILE.exists())

    def test_login_requires_tty(self):
        with mock.patch.object(auth.sys.stdin, "isatty", return_value=False):
            rc, out, err = _capture(cli.main, ["login"])
        self.assertEqual(rc, 1)
        self.assertIn("TTY", err)


if __name__ == "__main__":
    unittest.main()
