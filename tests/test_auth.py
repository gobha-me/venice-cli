"""Credential read/write tests. Patches HOME to a tmpdir + reloads modules."""
import importlib
import io
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


class TestSecretStore(unittest.TestCase):
    """The named-secret store (secrets.json) -- #43. Placeholder values only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self._env = mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=False)
        self._env.start()
        for var in ("VENICE_API_KEY", "VENICE_EMBED_API_KEY"):
            os.environ.pop(var, None)
        import venice.config as _cfg
        import venice.auth as _auth
        importlib.reload(_cfg)
        importlib.reload(_auth)
        self.config = _cfg
        self.auth = _auth

    def tearDown(self):
        self._env.stop()
        self.tmp.cleanup()

    def test_round_trip(self):
        self.auth.save_secret("embed", "placeholder-embed-XYZ")
        self.assertEqual(self.auth.load_secret("embed"), "placeholder-embed-XYZ")

    def test_secrets_file_0600_and_dir_0700(self):
        self.auth.save_secret("embed", "sekret")
        import stat
        self.assertEqual(
            stat.S_IMODE(self.config.SECRETS_FILE.stat().st_mode), 0o600)
        self.assertEqual(
            stat.S_IMODE(self.config.CONFIG_DIR.stat().st_mode), 0o700)
        self.assertEqual(list(self.config.CONFIG_DIR.glob("*.tmp")), [])

    def test_env_var_beats_store_for_embed(self):
        self.auth.save_secret("embed", "from-store")
        with mock.patch.dict(os.environ, {"VENICE_EMBED_API_KEY": "from-env"}):
            self.assertEqual(self.auth.load_secret("embed"), "from-env")
        # env gone -> falls back to the stored value
        self.assertEqual(self.auth.load_secret("embed"), "from-store")

    def test_name_without_canonical_env_is_store_only(self):
        self.auth.save_secret("cluster_token", "abc")
        # a same-name env var must NOT be consulted (no mapping for it)
        with mock.patch.dict(os.environ, {"cluster_token": "nope"}):
            self.assertEqual(self.auth.load_secret("cluster_token"), "abc")

    def test_list_shows_lengths_only(self):
        self.auth.save_secret("embed", "abcd")
        self.auth.save_secret("tok", "abcdefg")
        self.assertEqual(self.auth.list_secrets(), [("embed", 4), ("tok", 7)])

    def test_delete(self):
        self.auth.save_secret("embed", "x")
        self.assertTrue(self.auth.delete_secret("embed"))
        self.assertFalse(self.auth.delete_secret("embed"))
        self.assertIsNone(self.auth.load_secret("embed"))

    def test_load_missing_returns_none_no_raise(self):
        self.assertIsNone(self.auth.load_secret("nope"))
        self.assertEqual(self.auth.list_secrets(), [])

    def test_reject_bad_name_and_empty_value(self):
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_secret("bad name", "x")     # space
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_secret("../escape", "x")    # path-ish
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_secret("embed", "   ")      # blank value

    def test_corrupt_store_tolerant_read_strict_write(self):
        self.config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.config.SECRETS_FILE.write_text("{ not json", encoding="utf-8")
        # tolerant read -> empty, no raise
        self.assertEqual(self.auth.list_secrets(), [])
        self.assertIsNone(self.auth.load_secret("embed"))
        # strict-before-write -> refuse to clobber
        with self.assertRaises(self.auth.AuthError):
            self.auth.save_secret("embed", "x")

    def test_prompt_and_save_secret_length_only(self):
        # getpass + TTY mocked; assert the value never appears, only its length.
        err = io.StringIO()
        with mock.patch.object(self.auth.sys.stdin, "isatty", return_value=True), \
             mock.patch.object(self.auth.getpass, "getpass",
                               return_value="TOP-SECRET-VALUE"), \
             mock.patch.object(self.auth.sys, "stderr", err):
            self.auth.prompt_and_save_secret("embed")
        out = err.getvalue()
        self.assertNotIn("TOP-SECRET-VALUE", out)
        self.assertIn("16-char", out)
        self.assertEqual(self.auth.load_secret("embed"), "TOP-SECRET-VALUE")


if __name__ == "__main__":
    unittest.main()
