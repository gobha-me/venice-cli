"""Credential read/write tests. Patches HOME to a tmpdir + reloads modules."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestAuth(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self._env = mock.patch.dict(
            os.environ,
            {"HOME": str(self.home)},
            clear=False,
        )
        self._env.start()
        if os.environ.get("VENICE_API_KEY"):
            os.environ.pop("VENICE_API_KEY", None)
        import venice.config as _cfg
        import venice.auth as _auth
        importlib.reload(_cfg)
        importlib.reload(_auth)
        self.config = _cfg
        self.auth = _auth

    def tearDown(self):
        self._env.stop()
        self.tmp.cleanup()

    def test_env_var_overrides_file(self):
        with mock.patch.dict(os.environ, {self.config.ENV_API_KEY: "envkey"}):
            self.assertEqual(self.auth.load_key(), "envkey")

    def test_save_key_chmod_0600(self):
        self.auth.save_key("abc123")
        mode = self.config.CREDS_FILE.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.assertEqual(self.config.CREDS_FILE.read_text().strip(), "abc123")

    def test_save_key_creates_config_dir_0700(self):
        self.auth.save_key("xyz")
        dir_mode = self.config.CONFIG_DIR.stat().st_mode & 0o777
        self.assertEqual(dir_mode, 0o700)

    def test_save_key_refuses_empty(self):
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_key("")
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_key("   ")

    def test_save_key_refuses_whitespace(self):
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_key("abc def")
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_key("abc\tdef")

    def test_load_key_raises_when_missing(self):
        with self.assertRaises(self.auth.AuthError):
            self.auth.load_key()

    def test_load_key_returns_saved(self):
        self.auth.save_key("paste-this-key")
        self.assertEqual(self.auth.load_key(), "paste-this-key")


if __name__ == "__main__":
    unittest.main()
