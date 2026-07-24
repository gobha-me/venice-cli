"""Tests for the per-session steering mailbox + `venice sessions send` (#78).

Redirects the session store to a throwaway dir via ``$VENICE_SESSIONS_DIR`` (the
mailbox lives under it), so nothing touches ~/.config/venice/sessions. Covers the
store (perms/atomicity/order), the CLI verb, the `run_loop` drain checkpoint, and
`sessions rm` cleanup of the sidecar dir. No network, no key.
"""
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice import cli
from venice.commands import _agent, _mailbox
from venice.commands import _session as S
from tests.test_agent import _fake_oai, _free_tool
from tests.test_chat import FakeToolCompletion, _FnCall


def _capture(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
    return rc, out.getvalue(), err.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zone = Path(self.tmp.name) / "sessions"
        env = mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": str(self.zone)})
        env.start()
        self.addCleanup(env.stop)

    def _mk(self, command="code", **ov):
        """Mint + persist a session so it's a valid steering target."""
        sess = S.new_session(command, label=f"venice {command}", model="m-1", **ov)
        S.save(sess)
        return sess


class TestMailboxStore(_Base):
    def test_deposit_perms_and_layout(self):
        sess = self._mk()
        path = _mailbox.deposit(sess.id, "steer me")
        # 0600 message file in a 0700 mailbox dir under a 0700 per-session sidecar.
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        box = _mailbox.mailbox_dir(sess.id)
        self.assertEqual(stat.S_IMODE(os.stat(box).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(box.parent).st_mode), 0o700)
        # The mailbox is a sibling of <id>.json, invisible to the store's *.json glob.
        self.assertEqual(box.parent, self.zone / sess.id)
        self.assertTrue((self.zone / (sess.id + ".json")).is_file())
        self.assertEqual([r[0] for r in S.list_sessions()], [sess.id])  # 1 session, not 2

    def test_deposit_is_atomic_no_tmp_left(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "a")
        box = _mailbox.mailbox_dir(sess.id)
        self.assertFalse([p for p in box.iterdir() if p.suffix == ".tmp"])

    def test_drain_returns_order_and_deletes(self):
        sess = self._mk()
        for t in ("first", "second", "third\nmultiline"):
            _mailbox.deposit(sess.id, t)
        self.assertEqual(_mailbox.pending(sess.id), 3)
        self.assertEqual(_mailbox.drain(sess.id), ["first", "second", "third\nmultiline"])
        self.assertEqual(_mailbox.pending(sess.id), 0)
        self.assertEqual(_mailbox.drain(sess.id), [])  # idempotent when empty

    def test_drain_missing_mailbox_is_empty(self):
        sess = self._mk()
        self.assertEqual(_mailbox.drain(sess.id), [])
        self.assertEqual(_mailbox.pending(sess.id), 0)

    def test_pending_does_not_consume(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "x")
        self.assertEqual(_mailbox.pending(sess.id), 1)
        self.assertEqual(_mailbox.pending(sess.id), 1)  # counting twice keeps it queued

    def test_unsafe_id_rejected(self):
        with self.assertRaises(S.SessionError):
            _mailbox.deposit("../evil", "x")
        # pending/drain swallow the unsafe id rather than raising (best-effort readers).
        self.assertEqual(_mailbox.pending("../evil"), 0)
        self.assertEqual(_mailbox.drain("../evil"), [])

    def test_no_key_in_message_files(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "just a plain steer")
        for root, _, files in os.walk(self.zone):
            for f in files:
                self.assertNotIn(
                    "VENICE_API_KEY", Path(root, f).read_text(encoding="utf-8"))


class TestSessionsSendCLI(_Base):
    def test_send_by_id(self):
        sess = self._mk()
        rc, out, err = _capture(cli.main, ["sessions", "send", sess.id, "do the thing"])
        self.assertEqual(rc, 0)
        self.assertIn("1 pending", err)
        self.assertEqual(_mailbox.drain(sess.id), ["do the thing"])

    def test_send_latest_targets_newest_code_session(self):
        old = self._mk()  # noqa: F841 -- older session
        new = self._mk()
        rc, out, err = _capture(cli.main, ["sessions", "send", "latest", "hi"])
        self.assertEqual(rc, 0)
        self.assertEqual(_mailbox.pending(new.id), 1)
        self.assertEqual(_mailbox.pending(old.id), 0)

    def test_send_latest_no_code_session(self):
        self._mk(command="chat")  # a chat session doesn't satisfy `latest` (code-scoped)
        rc, out, err = _capture(cli.main, ["sessions", "send", "latest", "hi"])
        self.assertEqual(rc, 1)
        self.assertIn("no code session", err)

    def test_send_nonexistent_id(self):
        rc, out, err = _capture(cli.main, ["sessions", "send", "20200101T000000-abc", "x"])
        self.assertEqual(rc, 1)
        self.assertIn("no session named", err)

    def test_send_unsafe_id(self):
        rc, out, err = _capture(cli.main, ["sessions", "send", "../evil", "x"])
        self.assertEqual(rc, 1)
        self.assertIn("invalid session id", err)

    def test_send_empty_message_rejected(self):
        sess = self._mk()
        rc, out, err = _capture(cli.main, ["sessions", "send", sess.id, "   "])
        self.assertEqual(rc, 2)
        self.assertIn("empty steering message", err)

    def test_send_reads_stdin_dash(self):
        sess = self._mk()
        with mock.patch.object(sys, "stdin", io.StringIO("piped steer\n")):
            rc, out, err = _capture(cli.main, ["sessions", "send", sess.id, "-"])
        self.assertEqual(rc, 0)
        self.assertEqual(_mailbox.drain(sess.id), ["piped steer"])

    def test_ls_json_includes_pending(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "queued")
        rc, out, err = _capture(cli.main, ["sessions", "ls", "--json"])
        self.assertEqual(rc, 0)
        import json
        rows = json.loads(out)
        self.assertEqual(rows[0]["id"], sess.id)
        self.assertEqual(rows[0]["pending"], 1)

    def test_ls_plain_flags_pending(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "queued")
        rc, out, err = _capture(cli.main, ["sessions", "ls"])
        self.assertEqual(rc, 0)
        self.assertIn("1 pending", out)

    def test_show_reports_pending(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "queued")
        rc, out, err = _capture(cli.main, ["sessions", "show", sess.id])
        self.assertEqual(rc, 0)
        self.assertIn("pending steers: 1", out)

    def test_rm_removes_mailbox_sidecar(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "queued")
        sidecar = self.zone / sess.id
        self.assertTrue(sidecar.is_dir())
        rc, out, err = _capture(cli.main, ["sessions", "rm", sess.id])
        self.assertEqual(rc, 0)
        self.assertFalse(sidecar.exists())                     # sidecar gone
        self.assertFalse((self.zone / (sess.id + ".json")).exists())  # envelope gone


class TestRunLoopSteerDrain(_Base):
    """`run_loop` drains the mailbox at the checkpoint boundary (top of each turn)."""

    def _run(self, seq, messages, steer_drain):
        fake, calls = _fake_oai(seq)
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", messages, {}, [_free_tool()],
                max_tool_calls=0, yes=True, json_out=False, steer_drain=steer_drain,
            )
        return rc, calls

    def test_preseeded_steer_is_tagged_user_turn(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "please also update the README")
        messages = [{"role": "user", "content": "go"}]
        seq = [FakeToolCompletion("done")]  # stops immediately after the drain
        rc, calls = self._run(seq, messages, lambda: _mailbox.drain(sess.id))
        self.assertEqual(rc, 0)
        steer = [m for m in messages if m["role"] == "user"
                 and "steering message received mid-run" in m["content"]]
        self.assertEqual(len(steer), 1)
        self.assertIn("update the README", steer[0]["content"])
        self.assertEqual(_mailbox.pending(sess.id), 0)  # consumed

    def test_steer_deposited_midrun_picked_up_next_turn(self):
        sess = self._mk()
        messages = [{"role": "user", "content": "go"}]

        # A message arrives DURING the first model turn; it must be consumed at the
        # top of the *next* turn -- after the first turn's tool result is appended.
        def _create(**kw):
            if not getattr(_create, "fired", False):
                _create.fired = True
                _mailbox.deposit(sess.id, "changed my mind, do X instead")
                return FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")])
            return FakeToolCompletion("done")

        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = _create
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", messages, {}, [_free_tool()],
                max_tool_calls=0, yes=True, json_out=False,
                steer_drain=lambda: _mailbox.drain(sess.id),
            )
        self.assertEqual(rc, 0)
        roles = [m["role"] for m in messages]
        # Contract: the steer (user) lands only AFTER the tool result of the prior turn.
        steer_idx = next(i for i, m in enumerate(messages)
                         if m["role"] == "user"
                         and "steering message received mid-run" in m["content"])
        self.assertEqual(messages[steer_idx - 1]["role"], "tool")
        self.assertIn("do X instead", messages[steer_idx]["content"])

    def test_no_drain_callback_is_inert(self):
        messages = [{"role": "user", "content": "go"}]
        seq = [FakeToolCompletion("done")]
        rc, calls = self._run(seq, messages, None)
        self.assertEqual(rc, 0)
        self.assertFalse([m for m in messages
                          if "steering message received mid-run" in str(m.get("content"))])


if __name__ == "__main__":
    unittest.main()
