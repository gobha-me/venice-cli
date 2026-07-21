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
    and records each call's kwargs. A queued `Exception` instance is raised
    (not returned), so tests can exercise a failing API call."""
    calls = []
    it = iter(seq)

    def _create(**kw):
        calls.append(kw)
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

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


class TestCostLedger(unittest.TestCase):
    """The #66 session spend ledger."""

    def test_accumulates_input_and_output_cost(self):
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 1.5}, "output": {"usd": 4.0}})
        c = L.record({"prompt_tokens": 1000, "completion_tokens": 500})
        # 1000*1.5/1e6 + 500*4.0/1e6 = 0.0015 + 0.0020
        self.assertAlmostEqual(c, 0.0035)
        self.assertAlmostEqual(L.total, 0.0035)
        self.assertEqual(L.prompt_tokens, 1000)
        self.assertEqual(L.completion_tokens, 500)

    def test_over_only_when_capped(self):
        L = _agent.CostLedger()  # uncapped
        L.bind_pricing({"input": {"usd": 100.0}, "output": {"usd": 100.0}})
        L.record({"prompt_tokens": 10**6, "completion_tokens": 10**6})
        self.assertFalse(L.over())  # huge spend, but no cap set
        L2 = _agent.CostLedger(max_spend=0.001)
        L2.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 1.0}})
        L2.record({"prompt_tokens": 2000, "completion_tokens": 0})  # $0.002 > cap
        self.assertTrue(L2.over())

    def test_unpriced_model_counts_tokens_without_charge(self):
        L = _agent.CostLedger(max_spend=0.0)
        # no bind_pricing -> unknown rate
        c = L.record({"prompt_tokens": 5000, "completion_tokens": 100})
        self.assertEqual(c, 0.0)
        self.assertTrue(L.unpriced)
        self.assertEqual(L.prompt_tokens, 5000)
        self.assertFalse(L.over())  # nothing charged, so cap never trips
        self.assertIn("unpriced", L.summary())

    def test_record_tolerates_sdk_objects_and_garbage(self):
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 1.0}})
        usage = mock.MagicMock()
        usage.model_dump.return_value = {"prompt_tokens": 100, "completion_tokens": 10}
        L.record(usage)
        L.record(None)
        L.record({"prompt_tokens": "nope"})
        self.assertEqual(L.prompt_tokens, 100)
        self.assertAlmostEqual(L.total, 0.00011)

    def test_factory_none_without_cap(self):
        args = type("A", (), {"session_max_spend": None})()
        self.assertIsNone(_agent.ledger_from_args(args, [], "m"))

    def test_factory_binds_catalog_pricing(self):
        args = type("A", (), {"session_max_spend": 0.5})()
        models = [{"id": "m", "model_spec": {"pricing": {"input": {"usd": 2.0}}}}]
        L = _agent.ledger_from_args(args, models, "m")
        self.assertEqual(L.max_spend, 0.5)
        self.assertEqual(L._in, 2.0 / 1e6)
        self.assertIsNone(L._out)  # no output price advertised


class TestRunLoopSpendGate(unittest.TestCase):
    """The loop stops starting paid turns once the session cap is hit (#66)."""

    def _tool(self):
        return _agent.Tool("t", "t", {"type": "object", "properties": {}},
                           lambda a, *, confirm=False: {"status": "ok"})

    def test_forces_final_when_cap_crossed(self):
        # Turn 1 calls a tool AND costs enough to cross the cap; the next
        # iteration must force a final answer instead of another paid turn.
        usage = {"prompt_tokens": 9000, "completion_tokens": 1000,
                 "total_tokens": 10000}
        seq = [
            FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")], usage=usage),
            FakeToolCompletion("wrapped up"),  # the forced-final turn
        ]
        fake, calls = _fake_oai(seq)
        ledger = _agent.CostLedger(max_spend=0.001)
        ledger.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 1.0}})
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=ledger,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)                      # turn 1 + forced final
        self.assertEqual(calls[-1]["tool_choice"], "none")   # forced, no tools
        self.assertTrue(ledger.over())

    def test_no_ledger_means_no_gate(self):
        usage = {"prompt_tokens": 10**9, "completion_tokens": 10**9}
        seq = [FakeToolCompletion("done", usage=usage)]
        fake, calls = _fake_oai(seq)
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)  # no forced final despite huge usage

    def test_unpriced_ledger_never_gates_but_counts(self):
        usage = {"prompt_tokens": 5000, "completion_tokens": 500}
        seq = [FakeToolCompletion("done", usage=usage)]
        fake, calls = _fake_oai(seq)
        ledger = _agent.CostLedger(max_spend=0.0)  # cap 0, but no price bound
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=ledger,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)  # unpriced -> counted, not gated
        self.assertEqual(ledger.prompt_tokens, 5000)


class TestAutoCompact(unittest.TestCase):
    """Auto-compaction in run_loop (#48)."""

    def _big_history(self, pairs=8):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(pairs):
            msgs.append({"role": "user", "content": f"u{i} " + "x" * 200})
            msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 200})
        return msgs

    def test_compacts_before_capped_turn(self):
        # Turn 1 answers with usage over threshold; the run then needs a second
        # (forced-final) turn -- which must compact first instead of sending the
        # full history again.
        history = self._big_history()
        usage = {"prompt_tokens": 5000, "completion_tokens": 3, "total_tokens": 5003}
        seq = [
            FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")], usage=usage),
            FakeToolCompletion("summary of the work so far"),  # compaction turn
            FakeToolCompletion("done"),                        # forced final
        ]
        fake, calls = _fake_oai(seq)
        budget = _agent._compact.Budget(threshold_tokens=1000, keep_turns=2)
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", history, {}, [_free_tool()],
                max_tool_calls=1, yes=True, json_out=False, budget=budget,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        compact_call, final_call = calls[1], calls[2]
        self.assertEqual(compact_call["tool_choice"], "none")
        self.assertNotIn("tools", compact_call)
        # The final turn saw the compacted history: summary system message, not
        # the full original prefix.
        final_msgs = final_call["messages"]
        self.assertLess(len(final_msgs), 17)
        self.assertTrue(any(
            m.get("role") == "system" and "summary of the work" in str(m.get("content"))
            for m in final_msgs
        ))
        # The caller's history was compacted in place too.
        self.assertEqual(history[1]["role"], "system")
        self.assertIn("summary of the work", history[1]["content"])

    def test_no_budget_means_no_compaction(self):
        history = self._big_history()
        before = list(history)
        seq = [FakeToolCompletion("done")]
        fake, calls = _fake_oai(seq)
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", history, {}, [_free_tool()],
                max_tool_calls=0, yes=True, json_out=False,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)  # no summarization call snuck in
        self.assertEqual(history[: len(before)], before)  # only appended after
        self.assertEqual(history[1]["role"], "user")

    def test_under_budget_no_compaction(self):
        history = self._big_history(pairs=3)
        seq = [FakeToolCompletion("done")]
        fake, calls = _fake_oai(seq)
        budget = _agent._compact.Budget(threshold_tokens=10**9, keep_turns=2)
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(
                fake, "m", history, {}, [_free_tool()],
                max_tool_calls=0, yes=True, json_out=False, budget=budget,
            )
        self.assertEqual(len(calls), 1)

    def test_failed_compaction_run_continues(self):
        history = self._big_history()
        usage = {"prompt_tokens": 5000, "completion_tokens": 3, "total_tokens": 5003}
        seq = [
            FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")], usage=usage),
            RuntimeError("summary boom"),   # compaction call raises
            FakeToolCompletion("done"),
        ]
        fake, calls = _fake_oai(seq)
        budget = _agent._compact.Budget(threshold_tokens=1000, keep_turns=2)
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", history, {}, [_free_tool()],
                max_tool_calls=1, yes=True, json_out=False, budget=budget,
            )
        self.assertEqual(rc, 0)
        # History NOT compacted (the failed summary changed nothing).
        self.assertEqual(history[1]["role"], "user")
        self.assertNotIn("[Summary", str(history))


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


class TestToolSection(unittest.TestCase):
    def test_section_derivation(self):
        self.assertEqual(_agent._tool_section("venice_image"), "image")
        self.assertEqual(_agent._tool_section("venice_image_edit"), "image_edit")
        self.assertEqual(_agent._tool_section("venice_vision"), "vision")
        self.assertEqual(_agent._tool_section("project_search"), "project_search")


class TestConfigDefaults(unittest.TestCase):
    """#58: defaults.<cmd>.* are layered UNDER a tool's model-supplied args."""

    def _spy(self):
        captured = {}

        def image_tool(client, prompt=None, *, hide_watermark=None, steps=None,
                       safe_mode=None, confirm=False, max_spend=None,
                       output_dir=None, **kw):
            captured.update(hide_watermark=hide_watermark, steps=steps,
                            safe_mode=safe_mode)
            captured.update(kw)
            return {"status": "ok"}

        return captured, image_tool

    def _tool(self, spy, doc):
        from venice.commands import _mcp
        with mock.patch.object(_mcp, "image_tool", spy):
            return _agent.builtin_tools(object(), config=doc,
                                        only={"venice_image"})[0]

    def test_injected_and_model_wins(self):
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"hide_watermark": True, "steps": 40}}}
        tool = self._tool(spy, doc)
        tool.invoke({"prompt": "p"})                     # model set no preference
        self.assertIs(captured["hide_watermark"], True)  # from config
        self.assertEqual(captured["steps"], 40)
        captured.clear()
        tool.invoke({"prompt": "p", "hide_watermark": False, "steps": 5})
        self.assertIs(captured["hide_watermark"], False)  # explicit model arg wins
        self.assertEqual(captured["steps"], 5)

    def test_no_config_no_injection(self):
        captured, spy = self._spy()
        tool = self._tool(spy, None)
        tool.invoke({"prompt": "p"})
        self.assertIsNone(captured["hide_watermark"])  # tool's own default applies
        self.assertIsNone(captured["steps"])

    def test_only_accepted_allowlisted_keys_inject(self):
        # `preset` is config-backable for image (#57) but image_tool takes no such
        # param -> must NOT be injected; the accepted key still is.
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"preset": "foo", "safe_mode": False}}}
        tool = self._tool(spy, doc)
        tool.invoke({"prompt": "p"})
        self.assertNotIn("preset", captured)
        self.assertIs(captured["safe_mode"], False)

    def test_string_config_value_is_coerced(self):
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"hide_watermark": "true", "steps": "12"}}}
        tool = self._tool(spy, doc)
        tool.invoke({"prompt": "p"})
        self.assertIs(captured["hide_watermark"], True)  # _as_bool
        self.assertEqual(captured["steps"], 12)          # int

    def test_safety_flags_flow_through_and_model_wins(self):
        # #61: safe_mode/hide_watermark are now on _IMAGE_SCHEMA, so a model-supplied
        # value must reach image_tool and override a conflicting config default.
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"safe_mode": True, "hide_watermark": False}}}
        tool = self._tool(spy, doc)
        tool.invoke({"prompt": "p", "safe_mode": False, "hide_watermark": True})
        self.assertIs(captured["safe_mode"], False)      # explicit model arg wins
        self.assertIs(captured["hide_watermark"], True)


if __name__ == "__main__":
    unittest.main()
