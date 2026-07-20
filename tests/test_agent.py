"""Unit tests for the agent loop helpers + run_loop budget/gate/progress.

Covers the ergonomics work: unlimited `--max-tool-calls` (#53), the TTY-gated
progress feedback (#54), and the `all`/auto-accept confirm gate (#55). Reuses
`test_chat`'s fake completions so the fakes stay in lock-step. No network/key.
"""
import io
import sys
import unittest
from unittest import mock

from venice.commands import _agent
from tests.test_chat import FakeToolCompletion, _FnCall


def _fake_oai(seq):
    """A fake `oai` whose chat.completions.create() returns queued completions
    and records each call's kwargs."""
    calls = []
    it = iter(seq)

    def _create(**kw):
        calls.append(kw)
        return next(it)

    fake = mock.MagicMock()
    fake.chat.completions.create.side_effect = _create
    return fake, calls


def _tool(name, impl, *, paid=False):
    return _agent.Tool(name, name, {"type": "object", "properties": {}}, impl, paid=paid)


def _free_tool():
    return _tool("t", lambda a, *, confirm=False: {"status": "ok"})


def _tty(value=True):
    m = mock.MagicMock()
    m.isatty.return_value = value
    return m


class TestShortArgs(unittest.TestCase):
    def test_prefers_informative_field(self):
        self.assertEqual(
            _agent._short_args('{"path": "a/b.py", "data": "x"}'), "path=a/b.py"
        )

    def test_truncates_long_values(self):
        s = _agent._short_args('{"command": "%s"}' % ("x" * 100))
        self.assertTrue(s.startswith("command=") and s.endswith("..."))

    def test_bad_or_nonobject_json_is_empty(self):
        self.assertEqual(_agent._short_args("{not json"), "")
        self.assertEqual(_agent._short_args("[1,2,3]"), "")

    def test_falls_back_to_sorted_keys(self):
        self.assertEqual(_agent._short_args('{"z": {"k": 1}, "a": [1]}'), "a, z")


class TestPromptYes(unittest.TestCase):
    def test_all(self):
        with mock.patch("builtins.input", return_value="a"):
            self.assertEqual(_agent._prompt_yes(), "all")

    def test_yes(self):
        with mock.patch("builtins.input", return_value="yes"):
            self.assertEqual(_agent._prompt_yes(), "yes")

    def test_no_and_eof(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(_agent._prompt_yes(), "no")
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(_agent._prompt_yes(), "no")


class TestConfirmGate(unittest.TestCase):
    def _paid_tool(self, seen):
        def impl(arguments, *, confirm=False):
            seen.append(confirm)
            return {"status": "ok"} if confirm else {
                "status": "confirmation_required", "message": "spend?"}
        return _tool("venice_image", impl, paid=True)

    def test_all_runs_call_and_flips_gate_sticky(self):
        seen = []
        dispatch = {"venice_image": self._paid_tool(seen)}
        gate = {"auto": False}
        with mock.patch.object(sys, "stdin", _tty()), \
             mock.patch("builtins.input", return_value="a"), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            result = _agent._run_one_call(_FnCall("c1", "venice_image", "{}"),
                                          dispatch, gate)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(gate["auto"])          # "all" made auto sticky
            self.assertEqual(seen, [False, True])  # gated, then re-run confirmed
            # a subsequent paid call now runs with confirm=True and never prompts
            seen.clear()
            result2 = _agent._run_one_call(_FnCall("c2", "venice_image", "{}"),
                                           dispatch, gate)
        self.assertEqual(result2["status"], "ok")
        self.assertEqual(seen, [True])

    def test_no_declines_and_feeds_gate_back(self):
        seen = []
        dispatch = {"venice_image": self._paid_tool(seen)}
        gate = {"auto": False}
        with mock.patch.object(sys, "stdin", _tty()), \
             mock.patch("builtins.input", return_value="n"), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            result = _agent._run_one_call(_FnCall("c1", "venice_image", "{}"),
                                          dispatch, gate)
        self.assertEqual(result["status"], "confirmation_required")
        self.assertFalse(gate["auto"])


class TestRunLoopBudget(unittest.TestCase):
    def _run(self, seq, *, max_tool_calls, stderr=None, tty_err=False):
        fake, calls = _fake_oai(seq)
        err = stderr or io.StringIO()
        if tty_err:
            err.isatty = lambda: True
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [_free_tool()], max_tool_calls=max_tool_calls, yes=True, json_out=False,
            )
        return rc, calls, err

    def test_unlimited_runs_past_default(self):
        # Five tool rounds then a final answer -- unlimited must not force-stop.
        seq = [FakeToolCompletion(tool_calls=[_FnCall(f"c{i}", "t", "{}")])
               for i in range(5)]
        seq.append(FakeToolCompletion("done"))
        rc, calls, err = self._run(seq, max_tool_calls=0)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 6)  # 5 tool turns + 1 that stops
        self.assertNotIn("max-tool-calls", err.getvalue())  # no cap message
        self.assertTrue(all(c.get("tool_choice") == "auto" for c in calls))

    def test_none_is_also_unlimited(self):
        seq = [FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")]),
               FakeToolCompletion("done")]
        rc, calls, err = self._run(seq, max_tool_calls=None)
        self.assertEqual(rc, 0)
        self.assertNotIn("max-tool-calls", err.getvalue())

    def test_positive_cap_forces_final(self):
        seq = [FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")]),
               FakeToolCompletion("done")]
        rc, calls, err = self._run(seq, max_tool_calls=1)
        self.assertEqual(rc, 0)
        self.assertIn("max-tool-calls", err.getvalue())
        self.assertEqual(calls[-1]["tool_choice"], "none")  # forced final answer


class TestProgress(unittest.TestCase):
    def test_progress_prints_on_tty(self):
        err = io.StringIO()
        err.isatty = lambda: True
        with mock.patch.object(sys, "stderr", err):
            _agent._progress("· hi", enabled=True)
        self.assertIn("· hi", err.getvalue())

    def test_progress_silent_off_tty(self):
        err = io.StringIO()  # StringIO.isatty() -> False
        with mock.patch.object(sys, "stderr", err):
            _agent._progress("· hi", enabled=True)
        self.assertEqual(err.getvalue(), "")

    def test_progress_silent_when_disabled(self):
        err = io.StringIO()
        err.isatty = lambda: True
        with mock.patch.object(sys, "stderr", err):
            _agent._progress("· hi", enabled=False)
        self.assertEqual(err.getvalue(), "")

    def test_run_loop_emits_activity_line_on_tty(self):
        seq = [FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")]),
               FakeToolCompletion("done")]
        fake, calls = _fake_oai(seq)
        err = io.StringIO()
        err.isatty = lambda: True
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [_free_tool()], max_tool_calls=0, yes=True, json_out=False,
            )
        self.assertEqual(rc, 0)
        self.assertIn("· t", err.getvalue())  # per-tool-call activity line


if __name__ == "__main__":
    unittest.main()
