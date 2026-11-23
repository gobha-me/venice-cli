"""Real CLI and filesystem regressions for sessions unhappy paths (#265)."""

import contextlib
import errno
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._venice_fake_server import FakeVenice
from venice import cli
from venice.commands import _session as sessions

EMPTY_STORE = (
    "no saved sessions yet (start `venice chat` or `venice code`; "
    "auto-save is on unless --ephemeral).\n"
)
NO_CODE_SESSION = "sessions: no code session to steer (start `venice code`).\n"


class SessionsErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = FakeVenice().start()
        cls.addClassCleanup(cls.api.stop)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / "sessions"
        env = {
            "HOME": str(self.root),
            "XDG_CONFIG_HOME": str(self.root / "config"),
            "VENICE_SESSIONS_DIR": str(self.store),
            "VENICE_API_KEY": "test-fake-key",
            "VENICE_BASE_URL": self.api.base_url,
        }
        isolated = mock.patch.dict(os.environ, env, clear=True)
        isolated.start()
        self.addCleanup(isolated.stop)
        self.addCleanup(lambda: self.assertEqual(self.api.paths, []))

    def _run(self, *args, stdin=""):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.object(sys, "stdin", io.StringIO(stdin)):
            code = cli.main(["sessions", *args])
        return code, out.getvalue(), err.getvalue()

    def _saved(self, command="code"):
        session = sessions.new_session(
            command, model="fixture-model",
            messages=[{"role": "user", "content": "retain this conversation"}],
        )
        sessions.save(session)
        return session

    def _snapshot(self):
        if not self.store.exists():
            return None
        return {
            str(path.relative_to(self.store)): (
                stat.S_IMODE(path.stat().st_mode),
                path.read_bytes() if path.is_file() else None,
            )
            for path in self.store.rglob("*")
        }

    def test_missing_show_send_and_rm_do_not_create_a_store(self):
        expected = {
            "show": f"sessions: no session 'absent' in {self.store}\n",
            "send": "sessions: no session named 'absent'\n",
            "rm": "sessions: no session named 'absent'\n",
        }
        for action, message in expected.items():
            with self.subTest(action=action):
                args = (action, "absent", "hello") if action == "send" else (action, "absent")
                self.assertEqual(self._run(*args), (1, "", message))
                self.assertFalse(self.store.exists())

    def test_missing_ids_preserve_an_unrelated_saved_session(self):
        self._saved()
        before = self._snapshot()
        for args, message in (
            (("show", "absent"), f"sessions: no session 'absent' in {self.store}\n"),
            (("send", "absent", "hello"), "sessions: no session named 'absent'\n"),
            (("rm", "absent"), "sessions: no session named 'absent'\n"),
        ):
            with self.subTest(args=args):
                self.assertEqual(self._run(*args), (1, "", message))
                self.assertEqual(self._snapshot(), before)

    def test_empty_plain_listing_has_only_the_guidance_message(self):
        self.assertEqual(self._run("ls"), (0, "", EMPTY_STORE))
        self.assertFalse(self.store.exists())

    def test_empty_json_listing_is_an_empty_array_without_diagnostics(self):
        self.assertEqual(self._run("ls", "--json"), (0, "[]\n", ""))
        self.assertFalse(self.store.exists())

    def test_send_latest_with_no_sessions_does_not_create_a_mailbox(self):
        self.assertEqual(self._run("send", "latest", "hello"), (1, "", NO_CODE_SESSION))
        self.assertFalse(self.store.exists())

    def test_send_latest_with_only_chat_sessions_preserves_them(self):
        self._saved(command="chat")
        before = self._snapshot()
        self.assertEqual(self._run("send", "latest", "hello"), (1, "", NO_CODE_SESSION))
        self.assertEqual(self._snapshot(), before)

    def test_show_invalid_json_reports_the_error_without_changing_the_file(self):
        self.store.mkdir()
        path = self.store / "broken.json"
        for payload, diagnostic in (
            ("", "Expecting value: line 1 column 1 (char 0)"),
            ("{", "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
            ('{"truncated":', "Expecting value: line 1 column 14 (char 13)"),
        ):
            with self.subTest(payload=payload):
                path.write_text(payload, encoding="utf-8")
                before = self._snapshot()
                self.assertEqual(
                    self._run("show", "broken"),
                    (1, "", f"sessions: invalid session JSON in {path}: {diagnostic}\n"),
                )
                self.assertEqual(self._snapshot(), before)

    def test_show_non_object_json_reports_a_malformed_envelope(self):
        self.store.mkdir()
        path = self.store / "broken.json"
        for payload in ("null", "[]", '"not an envelope"'):
            with self.subTest(payload=payload):
                path.write_text(payload, encoding="utf-8")
                before = self._snapshot()
                self.assertEqual(
                    self._run("show", "broken"),
                    (1, "", "sessions: session 'broken' is malformed (expected an envelope)\n"),
                )
                self.assertEqual(self._snapshot(), before)

    def test_corrupt_only_store_lists_as_empty_and_preserves_bad_bytes(self):
        self.store.mkdir()
        (self.store / "broken.json").write_bytes(b'{"unfinished":')
        before = self._snapshot()
        self.assertEqual(self._run("ls"), (0, "", EMPTY_STORE))
        self.assertEqual(self._run("ls", "--json"), (0, "[]\n", ""))
        self.assertEqual(self._snapshot(), before)

    def test_mixed_store_lists_only_the_valid_session_without_repairing_corruption(self):
        saved = self._saved()
        (self.store / "broken.json").write_bytes(b'{"unfinished":')
        before = self._snapshot()
        code, out, err = self._run("ls", "--json")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out), [{
            "id": saved.id, "command": saved.command, "updated": saved.updated,
            "messages": len(saved.messages), "model": saved.model, "pending": 0,
        }])
        self.assertEqual(self._snapshot(), before)

    def test_rm_corrupt_session_removes_only_the_requested_bad_file(self):
        self._saved()
        broken = self.store / "broken.json"
        broken.write_bytes(b'{"unfinished":')
        expected = self._snapshot()
        del expected["broken.json"]
        self.assertEqual(self._run("rm", "broken"), (0, "", "removed session 'broken'.\n"))
        self.assertFalse(broken.exists())
        self.assertEqual(self._snapshot(), expected)

    def _assert_empty_message_rejected(self, text, *, stdin=""):
        saved = self._saved()
        before = self._snapshot()
        self.assertEqual(
            self._run("send", saved.id, text, stdin=stdin),
            (2, "", "sessions: empty steering message\n"),
        )
        self.assertEqual(self._snapshot(), before)
        self.assertFalse((self.store / saved.id).exists())

    def test_send_empty_argument_does_not_create_a_mailbox(self):
        self._assert_empty_message_rejected("")

    def test_send_whitespace_does_not_create_a_mailbox(self):
        self._assert_empty_message_rejected(" \t\n ")

    def test_send_empty_stdin_does_not_create_a_mailbox(self):
        self._assert_empty_message_rejected("-")

    def test_send_whitespace_stdin_does_not_create_a_mailbox(self):
        self._assert_empty_message_rejected("-", stdin=" \t\n ")

    def test_send_read_only_store_reports_permission_error_without_partial_mailbox(self):
        saved = self._saved()
        before = self._snapshot()
        original_mode = stat.S_IMODE(self.store.stat().st_mode)
        self.store.chmod(0o555)
        try:
            self.assertFalse(os.access(self.store, os.W_OK), "fixture must really deny writes")
            sidecar = self.store / saved.id
            denied = PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(sidecar))
            self.assertEqual(
                self._run("send", saved.id, "valid steering message"),
                (1, "", f"sessions: could not queue message: {denied}\n"),
            )
            self.assertEqual(self._snapshot(), before)
            self.assertFalse(sidecar.exists())
        finally:
            self.store.chmod(original_mode)

    def test_rm_read_only_store_reports_permission_error_and_preserves_session(self):
        saved = self._saved()
        self.assertEqual(self._run("send", saved.id, "keep this queued")[0], 0)
        before = self._snapshot()
        original_mode = stat.S_IMODE(self.store.stat().st_mode)
        self.store.chmod(0o555)
        try:
            self.assertFalse(os.access(self.store, os.W_OK), "fixture must really deny writes")
            path = self.store / f"{saved.id}.json"
            denied = PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
            self.assertEqual(
                self._run("rm", saved.id),
                (1, "", f"sessions: cannot delete session '{saved.id}': {denied}\n"),
            )
            self.assertEqual(self._snapshot(), before)
        finally:
            self.store.chmod(original_mode)


if __name__ == "__main__":
    unittest.main()
