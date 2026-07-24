"""Tests for attached-terminal Ctrl+C steering (#79).

Exercises `_steer` without OS signal delivery (which is timing-fragile): the SIGINT
handler is invoked directly via the disposition `pause_and_steer` installs, and the
composed drain is driven with a fake `input`. A run_loop integration test proves the
composed drain feeds through the same `[steering message received mid-run]` path as a
mailbox steer. No network, no key.
"""
import io
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _agent, _mailbox, _steer
from venice.commands import _session as S
from tests.test_agent import _fake_oai, _free_tool
from tests.test_chat import FakeToolCompletion, _FnCall


class _Base(unittest.TestCase):
    def setUp(self):
        # Always leave the process's SIGINT disposition as we found it, even if a test
        # asserts mid-`with` and unwinds.
        self._orig = signal.getsignal(signal.SIGINT)
        self.addCleanup(lambda: signal.signal(signal.SIGINT, self._orig))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zone = Path(self.tmp.name) / "sessions"
        env = mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": str(self.zone)})
        env.start()
        self.addCleanup(env.stop)

    def _mk(self, command="code", **ov):
        sess = S.new_session(command, label=f"venice {command}", model="m-1", **ov)
        S.save(sess)
        return sess


class TestPauseAndSteerScoping(_Base):
    def test_disabled_installs_no_handler_and_none_drain(self):
        before = signal.getsignal(signal.SIGINT)
        with _steer.pause_and_steer(None, enabled=False) as drain:
            self.assertEqual(signal.getsignal(signal.SIGINT), before)  # untouched
            self.assertIsNone(drain)  # no session -> nothing to drain
        self.assertEqual(signal.getsignal(signal.SIGINT), before)

    def test_disabled_with_session_is_plain_mailbox_drain(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "hello")
        before = signal.getsignal(signal.SIGINT)
        with _steer.pause_and_steer(sess.id, enabled=False) as drain:
            self.assertEqual(signal.getsignal(signal.SIGINT), before)
            self.assertEqual(drain(), ["hello"])  # #78 behavior, verbatim
        self.assertEqual(signal.getsignal(signal.SIGINT), before)

    def test_enabled_installs_and_restores_handler(self):
        before = signal.getsignal(signal.SIGINT)
        with _steer.pause_and_steer(None, enabled=True) as drain:
            inside = signal.getsignal(signal.SIGINT)
            self.assertTrue(callable(inside))
            self.assertIsNot(inside, before)  # our handler is installed
            self.assertTrue(callable(drain))
        self.assertEqual(signal.getsignal(signal.SIGINT), before)  # restored

    def test_handler_restored_even_on_abort(self):
        before = signal.getsignal(signal.SIGINT)
        with self.assertRaises(RuntimeError):
            with _steer.pause_and_steer(None, enabled=True):
                raise RuntimeError("boom")
        self.assertEqual(signal.getsignal(signal.SIGINT), before)


class TestHandlerStateMachine(_Base):
    def test_first_ctrlc_arms_then_drain_prompts(self):
        with _steer.pause_and_steer(None, enabled=True) as drain:
            handler = signal.getsignal(signal.SIGINT)
            with mock.patch.object(sys, "stderr", io.StringIO()) as err:
                handler(signal.SIGINT, None)  # first Ctrl+C -> arm
            self.assertIn("Ctrl+C", err.getvalue())  # operator got a hint
            with mock.patch("builtins.input", return_value="pivot to plan B"):
                out = drain()
            self.assertEqual(out, ["pivot to plan B"])
            self.assertEqual(drain(), [])  # flag was reset -> inert next time

    def test_second_ctrlc_before_checkpoint_aborts(self):
        with _steer.pause_and_steer(None, enabled=True):
            handler = signal.getsignal(signal.SIGINT)
            with mock.patch.object(sys, "stderr", io.StringIO()):
                handler(signal.SIGINT, None)  # arm
                with self.assertRaises(KeyboardInterrupt):
                    handler(signal.SIGINT, None)  # mash -> today's kill

    def test_real_sigint_is_delivered_to_handler(self):
        # One real delivery to prove the wiring (single, deterministic: arm only).
        with _steer.pause_and_steer(None, enabled=True) as drain:
            with mock.patch.object(sys, "stderr", io.StringIO()):
                signal.raise_signal(signal.SIGINT)
            with mock.patch("builtins.input", return_value="steer via real signal"):
                out = drain()
        self.assertEqual(out, ["steer via real signal"])


class TestComposedDrain(_Base):
    def _armed(self):
        st = _steer._Pending()
        st.requested = True
        return st

    def test_typed_line_appended_and_flag_reset(self):
        st = self._armed()
        drain = _steer._make_drain(None, st, prompt=lambda p: "do the thing")
        self.assertEqual(drain(), ["do the thing"])
        self.assertFalse(st.requested)

    def test_empty_or_whitespace_line_injects_nothing(self):
        for text in ("", "   ", "\n"):
            st = self._armed()
            drain = _steer._make_drain(None, st, prompt=lambda p, t=text: t)
            self.assertEqual(drain(), [])

    def test_not_armed_never_prompts(self):
        st = _steer._Pending()  # requested False

        def _boom(p):
            raise AssertionError("prompt must not be called when not armed")

        drain = _steer._make_drain(None, st, prompt=_boom)
        self.assertEqual(drain(), [])

    def test_eof_at_prompt_resumes(self):
        def _eof(p):
            raise EOFError

        st = self._armed()
        drain = _steer._make_drain(None, st, prompt=_eof)
        self.assertEqual(drain(), [])
        self.assertFalse(st.requested)

    def test_ctrlc_at_prompt_propagates(self):
        def _ctrlc(p):
            raise KeyboardInterrupt

        st = self._armed()
        drain = _steer._make_drain(None, st, prompt=_ctrlc)
        with self.assertRaises(KeyboardInterrupt):
            drain()

    def test_mailbox_and_inline_combine_in_order(self):
        sess = self._mk()
        _mailbox.deposit(sess.id, "from-mailbox")
        st = self._armed()
        drain = _steer._make_drain(sess.id, st, prompt=lambda p: "from-ctrlc")
        self.assertEqual(drain(), ["from-mailbox", "from-ctrlc"])

    def test_none_session_skips_mailbox(self):
        # Attached-ephemeral: no session id -> inline-only, never touches the store.
        st = self._armed()
        drain = _steer._make_drain(None, st, prompt=lambda p: "just this")
        self.assertEqual(drain(), ["just this"])


class TestDefaultSigint(_Base):
    def test_swaps_to_default_and_restores(self):
        marker = signal.getsignal(signal.SIGINT)
        with _steer.default_sigint():
            self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)
        self.assertEqual(signal.getsignal(signal.SIGINT), marker)

    def test_restores_even_on_raise(self):
        marker = signal.getsignal(signal.SIGINT)
        with self.assertRaises(ValueError):
            with _steer.default_sigint():
                raise ValueError
        self.assertEqual(signal.getsignal(signal.SIGINT), marker)


class TestRunLoopWithCtrlCSteer(_Base):
    """The composed drain feeds run_loop exactly like a mailbox steer (#78 contract)."""

    def test_armed_steer_is_tagged_user_turn_after_prior_tool(self):
        st = _steer._Pending()
        messages = [{"role": "user", "content": "go"}]

        # A Ctrl+C arms during the first model turn; the composed drain must inject the
        # typed line at the top of the *next* turn, after the first turn's tool result.
        def _create(**kw):
            if not getattr(_create, "fired", False):
                _create.fired = True
                st.requested = True  # simulate a SIGINT landing during this call
                return FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")])
            return FakeToolCompletion("done")

        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = _create
        drain = _steer._make_drain(None, st, prompt=lambda p: "changed my mind, do X")
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", messages, {}, [_free_tool()],
                max_tool_calls=0, yes=True, json_out=False, steer_drain=drain,
            )
        self.assertEqual(rc, 0)
        steer_idx = next(i for i, m in enumerate(messages)
                         if m["role"] == "user"
                         and "steering message received mid-run" in m["content"])
        self.assertEqual(messages[steer_idx - 1]["role"], "tool")  # contract-valid point
        self.assertIn("do X", messages[steer_idx]["content"])


if __name__ == "__main__":
    unittest.main()
