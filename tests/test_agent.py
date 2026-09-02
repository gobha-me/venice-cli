"""Unit tests for the agent loop helpers + run_loop budget/gate/progress.

Covers the ergonomics work: unlimited `--max-tool-calls` (#53), the TTY-gated
progress feedback (#54), and the `all`/auto-accept confirm gate (#55). Reuses
`test_chat`'s fake completions so the fakes stay in lock-step. No network/key.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
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


class TestAssistantReplay(unittest.TestCase):
    """#129/#146: reasoning and thought signatures survive buffered history."""

    def test_synthetic_sdk_message_replays_reasoning_and_exact_tool_call(self):
        from openai.types.chat import ChatCompletion

        response = ChatCompletion.model_validate({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "kimi-k3",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "I will inspect it.",
                    "reasoning_content": "reasoning bytes: \u2603",
                    "thought_signature": "sig-exact",
                    "tool_calls": [{
                        "id": "call-123",
                        "type": "function",
                        "function": {"name": "t", "arguments": '{"x":1}'},
                    }],
                },
            }],
        })

        self.assertEqual(_agent._assistant_dict(response.choices[0].message), {
            "role": "assistant",
            "content": "I will inspect it.",
            "reasoning_content": "reasoning bytes: \u2603",
            "thought_signature": "sig-exact",
            "tool_calls": [{
                "id": "call-123",
                "type": "function",
                "function": {"name": "t", "arguments": '{"x":1}'},
            }],
        })

    def test_alias_precedence_is_explicit_and_response_metadata_is_dropped(self):
        msg = SimpleNamespace(
            content="visible",
            tool_calls=None,
            reasoning_content="preferred",
            reasoning_details=[{"type": "summary", "text": "other"}],
            reasoning="last",
            thought_signature="sig-independent",
            refusal="response-only",
        )
        self.assertEqual(_agent._assistant_dict(msg), {
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "preferred",
            "thought_signature": "sig-independent",
        })

    def test_non_reasoning_message_shape_is_unchanged(self):
        msg = SimpleNamespace(content=None, tool_calls=None)
        self.assertEqual(_agent._assistant_dict(msg), {
            "role": "assistant", "content": "",
        })

    def test_next_tool_request_includes_reasoning_byte_for_byte(self):
        first = FakeToolCompletion(
            tool_calls=[_FnCall("c1", "t", "{}")],
            reasoning_content="keep this exactly \u2603",
            thought_signature="sig-exact",
        )
        fake, calls = _fake_oai([first, FakeToolCompletion("done")])
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(
                fake, "kimi-k3", messages, {}, [_free_tool()],
                max_tool_calls=0, yes=True, json_out=False,
            )
        self.assertEqual(
            calls[1]["messages"][1]["reasoning_content"],
            "keep this exactly \u2603",
        )
        self.assertEqual(
            calls[1]["messages"][1]["thought_signature"],
            "sig-exact",
        )

    def test_forced_final_response_keeps_reasoning_in_history(self):
        seq = [
            FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")]),
            FakeToolCompletion(
                "wrapped",
                reasoning_content="final thought",
                thought_signature="final-sig",
            ),
        ]
        fake, _calls = _fake_oai(seq)
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(
                fake, "kimi-k3", messages, {}, [_free_tool()],
                max_tool_calls=1, yes=True, json_out=False,
            )
        self.assertEqual(messages[-1]["reasoning_content"], "final thought")
        self.assertEqual(messages[-1]["thought_signature"], "final-sig")


def _tty(value=True):
    m = mock.MagicMock()
    m.isatty.return_value = value
    return m


@contextlib.contextmanager
def _driven_clock():
    """A monotonic clock that only the FAKES advance (#82).

    `_repl._fake_clock` auto-increments on every read because it only needs a non-zero
    duration. Here the durations ARE the assertion, so an auto-incrementing counter
    would make every expected number a function of how many times production happens to
    read the clock -- exactly the tripwire that helper's docstring warns about. Reads are
    free; a fake tool impl or a mocked `input()` moves `now[0]` deliberately.
    """
    now = [0.0]
    with mock.patch("venice.commands._agent.time.monotonic", lambda: now[0]):
        yield now


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


class TestToolTiming(unittest.TestCase):
    """#82: what `_run_one_call` does and does not count as tool time."""

    def _timed_tool(self, now, secs, *, raises=False, name="t"):
        def impl(arguments, *, confirm=False):
            now[0] += secs
            if raises:
                raise RuntimeError("boom")
            return {"status": "ok"}
        return _tool(name, impl)

    @staticmethod
    def _sink(into):
        """The `(name, seconds)` sink shape, collecting into `into` as tuples."""
        return lambda name, seconds: into.append((name, seconds))

    def _call(self, dispatch, sink, *, args="{}", name="t"):
        return _agent._run_one_call(_FnCall("c1", name, args), dispatch,
                                    {"auto": True}, on_tool=self._sink(sink))

    def test_run_one_call_records_the_invoke_window(self):
        sink = []
        with _driven_clock() as now:
            self._call({"t": self._timed_tool(now, 3.0)}, sink)
        self.assertEqual(sink, [("t", 3.0)])

    def test_validation_returns_are_not_timed(self):
        # None of the three entered a tool, so none is tool time. Stamping them would
        # land ~0.0s rows and a turn of rejected calls would report fast tools.
        with _driven_clock() as now:
            dispatch = {"t": self._timed_tool(now, 3.0)}
            unknown, bad_json, not_obj = [], [], []
            self._call(dispatch, unknown, name="nope")
            self._call(dispatch, bad_json, args="{not json")
            self._call(dispatch, not_obj, args="[1,2,3]")
        self.assertEqual((unknown, bad_json, not_obj), ([], [], []))

    def test_a_tool_that_raised_is_still_timed(self):
        # Real waiting. A crash loop that self-reports 0.0s hides its own cost.
        sink = []
        with _driven_clock() as now:
            result = self._call({"t": self._timed_tool(now, 3.0, raises=True)}, sink)
        self.assertEqual(sink, [("t", 3.0)])
        self.assertEqual(result["status"], "error")

    def test_the_confirm_prompt_is_not_tool_time(self):
        # The load-bearing one. Each invoke costs 1s; the operator stares at the
        # `Proceed?` prompt for 100s. That wait falls BETWEEN the two windows, so the
        # row is 2.0s. Kills both "bracket all of _run_one_call" (-> 102.0) and a
        # missing window on the confirm re-invoke (-> 1.0).
        sink = []
        with _driven_clock() as now:
            def impl(arguments, *, confirm=False):
                now[0] += 1.0
                return {"status": "ok"} if confirm else {
                    "status": "confirmation_required", "message": "spend?"}

            def slow_operator(*a, **kw):
                now[0] += 100.0
                return "y"

            dispatch = {"venice_image": _tool("venice_image", impl, paid=True)}
            with mock.patch.object(sys, "stdin", _tty()), \
                 mock.patch("builtins.input", slow_operator), \
                 mock.patch.object(sys, "stderr", io.StringIO()):
                _agent._run_one_call(_FnCall("c1", "venice_image", "{}"),
                                     dispatch, {"auto": False},
                                     on_tool=self._sink(sink))
        self.assertEqual(sink, [("venice_image", 2.0)])

    def test_a_gated_call_counts_once_despite_two_invokes(self):
        # One CALL, not one invoke: `calls` counts what the model asked for.
        L = _agent.CostLedger()
        with _driven_clock() as now:
            def impl(arguments, *, confirm=False):
                now[0] += 1.0
                return {"status": "ok"} if confirm else {
                    "status": "confirmation_required", "message": "spend?"}

            dispatch = {"venice_image": _tool("venice_image", impl, paid=True)}
            with mock.patch.object(sys, "stdin", _tty()), \
                 mock.patch("builtins.input", return_value="y"), \
                 mock.patch.object(sys, "stderr", io.StringIO()):
                _agent._run_one_call(_FnCall("c1", "venice_image", "{}"),
                                     dispatch, {"auto": False},
                                     on_tool=L.record_tool)
        self.assertEqual(L.tools, {"venice_image": {"seconds": 2.0, "calls": 1}})

    def test_no_sink_is_a_no_op(self):
        # `run_loop(ledger=None)` passes on_tool=None; the call must still work.
        with _driven_clock() as now:
            r = _agent._run_one_call(_FnCall("c1", "t", "{}"),
                                     {"t": self._timed_tool(now, 3.0)}, {"auto": True})
        self.assertEqual(r["status"], "ok")

    def test_record_tool_mutates_under_the_lock(self):
        # Deterministic, and deliberately structural.
        #
        # The obvious test -- hammer `record_tool` from N threads and assert no lost
        # updates -- does NOT work: measured at 16 threads x 20,000 increments with a
        # 1ns switch interval, CPython's GIL never actually loses one, so that test
        # passes with the lock DELETED. It would be a vacuous guard, so it isn't here.
        #
        # The lock is still correct: `row["calls"] += 1` is a load/add/store, and
        # nothing about the GIL's current bytecode-boundary behaviour is a language
        # guarantee (free-threaded builds drop it outright). So pin the property that
        # is actually checkable -- the mutation happens while the lock is held.
        L = _agent.CostLedger()
        held = []
        real = L._tools_lock

        class Watched:
            def __enter__(self):
                real.acquire()
                held.append("in")
                return self

            def __exit__(self, *exc):
                held.append("out")
                real.release()
                return False

        L._tools_lock = Watched()
        L.record_tool("shell", 1.0)
        L.record_tool("shell", 1.0)
        self.assertEqual(held, ["in", "out", "in", "out"])

    def test_record_tool_is_correct_under_real_threads(self):
        # Not a race detector (see above) -- a smoke test that concurrent workers,
        # which is how `--parallel` calls this, all land. `venice_spawn` batches hit
        # the same key from up to `_MAX_PARALLEL` threads at once.
        L = _agent.CostLedger()
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, old)

        def hammer():
            for _ in range(500):
                L.record_tool("shell", 0.001)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(L.tools["shell"]["calls"], 4000)


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

    def test_non_finite_session_caps_are_rejected_not_uncapped(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                _agent.CostLedger(max_spend=bad)

    def test_non_finite_catalog_pricing_trips_a_configured_cap(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            ledger = _agent.CostLedger(max_spend=1.0)
            ledger.bind_pricing({"input": {"usd": bad}})
            with self.subTest(value=bad):
                self.assertTrue(ledger.invalid_pricing)
                self.assertTrue(ledger.over())

    def test_non_finite_usage_and_restored_values_degrade_to_zero(self):
        ledger = _agent.CostLedger()
        ledger.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 1.0}})
        ledger.record({
            "prompt_tokens": float("inf"),
            "completion_tokens": float("nan"),
        })
        ledger.restore({"total": float("inf"), "elapsed_seconds": float("nan")})
        self.assertEqual(ledger.total, 0.0)
        self.assertEqual(ledger.prompt_tokens, 0)
        self.assertEqual(ledger.completion_tokens, 0)

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

    def test_model_switch_rebinds_pricing_and_partitions_usage(self):
        models = [
            {"id": "author", "model_spec": {"pricing": {
                "input": {"usd": 1.0}, "output": {"usd": 2.0},
            }}},
            {"id": "reviewer", "model_spec": {"pricing": {
                "input": {"usd": 10.0}, "output": {"usd": 20.0},
            }}},
        ]
        args = type("A", (), {"session_max_spend": None})()
        ledger = _agent.usage_ledger(args, models, "author")
        ledger.record({"prompt_tokens": 1000, "completion_tokens": 100})
        ledger.bind_model("reviewer", models)
        ledger.record({"prompt_tokens": 1000, "completion_tokens": 100})

        self.assertAlmostEqual(ledger.total, 0.0012 + 0.012)
        self.assertAlmostEqual(ledger.models["author"]["total"], 0.0012)
        self.assertAlmostEqual(ledger.models["reviewer"]["total"], 0.012)
        self.assertEqual(
            [row["model"] for row in ledger.api_calls()], ["author", "reviewer"]
        )

    def test_resume_keeps_cumulative_models_but_resets_current_run(self):
        models = [{"id": "m", "model_spec": {"pricing": {
            "input": {"usd": 1.0}, "cache_input": {"usd": 0.1},
        }}}]
        args = type("A", (), {"session_max_spend": 1.0})()
        first = _agent.usage_ledger(args, models, "m")
        first.record({
            "prompt_tokens": 100, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 90},
        })

        resumed = _agent.usage_ledger(args, models, "m")
        resumed.restore(first.to_dict(), legacy_model="m")
        self.assertEqual(resumed.current_run_models, {})
        resumed.record({
            "prompt_tokens": 50, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        })

        self.assertEqual(resumed.prompt_tokens, 150)
        self.assertEqual(resumed.models["m"]["prompt_tokens"], 150)
        self.assertEqual(resumed.current_run_models["m"]["prompt_tokens"], 50)
        self.assertEqual(
            resumed.to_dict()["current_run_models"]["m"]["cache_hit_percent"],
            0.0,
        )
        # The footer's cache clause is current-run, not the blended session rate.
        self.assertIn("cache 0.0% hit", resumed.summary(cache=True))

    def test_legacy_usage_is_attributed_to_saved_not_overridden_model(self):
        models = [{"id": "old", "model_spec": {}},
                  {"id": "new", "model_spec": {}}]
        args = type("A", (), {"session_max_spend": None})()
        ledger = _agent.usage_ledger(args, models, "new")
        ledger.restore(
            {"prompt_tokens": 25, "completion_tokens": 2, "api_calls_total": 1},
            legacy_model="old",
        )
        self.assertEqual(ledger.model_id, "new")
        self.assertEqual(ledger.models["old"]["prompt_tokens"], 25)
        self.assertNotIn("new", ledger.models)
        self.assertEqual(ledger.current_run_models, {})

    def test_model_restore_tolerates_junk_and_never_restores_current_run(self):
        ledger = _agent.CostLedger(model_id="live", models=[])
        ledger.current_run_models["preexisting"] = _agent._new_model_usage_row()
        ledger.restore({
            "models": {
                "good": {"prompt_tokens": 7, "cache_read_tokens": 3},
                "bad": "not a row",
                "": {"prompt_tokens": 99},
            },
            "current_run_models": {
                "stale": {"prompt_tokens": 1000},
            },
        })
        self.assertEqual(ledger.models["good"]["prompt_tokens"], 7)
        self.assertEqual(set(ledger.models), {"good"})
        self.assertEqual(ledger.current_run_models, {})

    def test_switch_from_valid_to_invalid_pricing_fails_closed(self):
        models = [
            {"id": "good", "model_spec": {"pricing": {
                "input": {"usd": 1.0},
            }}},
            {"id": "bad", "model_spec": {"pricing": {
                "input": {"usd": float("inf")},
            }}},
        ]
        ledger = _agent.CostLedger(max_spend=1.0, model_id="good", models=models)
        self.assertFalse(ledger.invalid_pricing)
        ledger.bind_model("bad", models)
        self.assertTrue(ledger.invalid_pricing)
        self.assertTrue(ledger.over())

    def test_legacy_trace_rows_gain_only_known_saved_model_provenance(self):
        with_model = _agent.CostLedger()
        with_model.restore(
            {"api_calls": [{"n": 1}], "api_calls_total": 1},
            legacy_model="saved",
        )
        self.assertEqual(with_model.api_calls()[0]["model"], "saved")

        unknown = _agent.CostLedger()
        unknown.restore({"api_calls": [{"n": 1}], "api_calls_total": 1})
        self.assertNotIn("model", unknown.api_calls()[0])

    def test_model_cache_guard_is_independent_after_a_switch(self):
        pricing = {"input": {"usd": 3.0}, "cache_input": {"usd": 0.3}}
        models = [
            {"id": "a", "model_spec": {"pricing": pricing}},
            {"id": "b", "model_spec": {"pricing": pricing}},
        ]
        args = type("A", (), {
            "session_max_spend": None, "cache_guard": "warn",
        })()
        ledger = _agent.usage_ledger(args, models, "a")
        usage = {
            "prompt_tokens": 3000, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        ledger.record(usage)
        self.assertIsNone(ledger.cache_guard_event("a"))
        ledger.record(usage)
        self.assertIsNotNone(ledger.cache_guard_event("a"))
        ledger.bind_model("b", models)
        ledger.record(usage)
        self.assertIsNone(ledger.cache_guard_event("b"))
        ledger.record(usage)
        self.assertIsNotNone(ledger.cache_guard_event("b"))

    def test_usage_factory_binds_code_cache_guard_only_when_requested(self):
        models = [{
            "id": "m",
            "model_spec": {"pricing": {
                "input": {"usd": 3.75}, "cache_input": {"usd": 0.375},
            }},
        }]
        args = type(
            "A", (), {"session_max_spend": None, "cache_guard": "stop"}
        )()
        self.assertEqual(_agent.usage_ledger(args, models, "m").cache_guard, "stop")
        plain = type("A", (), {"session_max_spend": None})()
        self.assertEqual(_agent.usage_ledger(plain, models, "m").cache_guard, "off")

    def test_cache_guard_uses_api_ordinal_and_three_state_usage(self):
        def usage(prompt, cached_marker):
            block = None if cached_marker is None else {
                "cached_tokens": cached_marker,
            }
            return {
                "prompt_tokens": prompt,
                "completion_tokens": 10,
                "prompt_tokens_details": block,
            }

        pricing = {
            "input": {"usd": 3.75}, "cache_input": {"usd": 0.375},
        }
        cases = (
            ("cold first call", [usage(3_000, 0)], True, None),
            ("below floor", [usage(3_000, 100), usage(1_999, 0)], True, None),
            ("unreported", [usage(3_000, 100), usage(3_000, None)], True, None),
            ("cache hit", [usage(3_000, 100), usage(3_000, 1)], True, None),
            ("not advertised", [usage(3_000, 100), usage(3_000, 0)], False, None),
        )
        for name, rows, advertised, expected in cases:
            with self.subTest(case=name):
                ledger = _agent.CostLedger(cache_guard="warn")
                ledger.bind_pricing(
                    pricing if advertised else {"input": {"usd": 3.75}}
                )
                event = None
                for row in rows:
                    ledger.record(row)
                    event = ledger.cache_guard_event("kimi-k3")
                self.assertIs(event, expected)

    def test_cache_guard_event_is_once_only_and_estimates_discount(self):
        ledger = _agent.CostLedger(cache_guard="warn")
        ledger.bind_pricing({
            "input": {"usd": 3.75}, "cache_input": {"usd": 0.375},
        })
        ledger.record({
            "prompt_tokens": 3_000, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 100},
        })
        self.assertIsNone(ledger.cache_guard_event("kimi-k3"))
        ledger.record({
            "prompt_tokens": 120_000, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 0},
        })
        event = ledger.cache_guard_event("kimi-k3")
        self.assertEqual(event.action, "warn")
        self.assertIn("10.0x cache discount", event.message)
        self.assertIn("API call 2", event.message)
        self.assertIn("+$0.4050", event.message)
        self.assertIsNone(ledger.cache_guard_event("kimi-k3"))

    def test_cache_guard_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "off, warn, stop"):
            _agent.CostLedger(cache_guard="explode")

    # -- cache-bucket accounting (#75) -------------------------------------- #

    def test_cache_buckets_priced_distinctly(self):
        # A cache-heavy turn: 9000 of 10000 input tokens are cache reads. Priced
        # with a discounted cache-read rate, it costs far less than the flat
        # input rate would imply -- the exact case the collapsed math got wrong.
        L = _agent.CostLedger()
        L.bind_pricing({
            "input": {"usd": 3.0}, "cache_input": {"usd": 0.3},
            "cache_write": {"usd": 3.75}, "output": {"usd": 15.0},
        })
        c = L.record({
            "prompt_tokens": 10000, "completion_tokens": 500,
            "prompt_tokens_details": {
                "cached_tokens": 9000, "cache_creation_input_tokens": 0,
            },
        })
        # 1000*3 + 9000*0.3 + 0 + 500*15, all /1e6 = 0.003 + 0.0027 + 0.0075
        self.assertAlmostEqual(c, 0.0132)
        self.assertEqual(L.cache_read_tokens, 9000)
        self.assertEqual(L.cache_write_tokens, 0)
        self.assertEqual(L.prompt_tokens, 10000)
        # ... and strictly cheaper than the collapsed flat-input estimate.
        flat = (10000 * 3.0 + 500 * 15.0) / 1e6
        self.assertLess(L.total, flat)

    def test_cache_write_priced_at_its_own_rate(self):
        L = _agent.CostLedger()
        L.bind_pricing({
            "input": {"usd": 3.0}, "cache_write": {"usd": 3.75},
            "output": {"usd": 15.0},
        })
        c = L.record({
            "prompt_tokens": 1000, "completion_tokens": 0,
            "prompt_tokens_details": {
                "cached_tokens": 0, "cache_creation_input_tokens": 200,
            },
        })
        # 800 uncached*3 + 200 write*3.75, /1e6
        self.assertAlmostEqual(c, (800 * 3.0 + 200 * 3.75) / 1e6)
        self.assertEqual(L.cache_write_tokens, 200)

    def test_cache_rates_fall_back_to_input_when_absent(self):
        # No cache_input/cache_write pricing -> cache tokens billed at input rate,
        # so the total matches the flat estimate (fallback keeps math consistent).
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
        L.record({
            "prompt_tokens": 10000, "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 9000},
        })
        self.assertAlmostEqual(L.total, (10000 * 3.0 + 500 * 15.0) / 1e6)
        self.assertEqual(L.cache_read_tokens, 9000)

    def test_no_cache_tokens_matches_legacy_formula(self):
        # Backward-compat: without cache detail, cost is exactly pt*in + ct*out.
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 1.5}, "output": {"usd": 4.0}})
        c = L.record({"prompt_tokens": 1000, "completion_tokens": 500})
        self.assertAlmostEqual(c, (1000 * 1.5 + 500 * 4.0) / 1e6)
        self.assertEqual(L.cache_read_tokens, 0)
        self.assertEqual(L.cache_write_tokens, 0)

    def test_cache_buckets_clamped_to_prompt_tokens(self):
        # A provider reporting the buckets additively can't drive uncached < 0.
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 1.0}})
        L.record({
            "prompt_tokens": 100, "completion_tokens": 0,
            "prompt_tokens_details": {
                "cached_tokens": 9999, "cache_creation_input_tokens": 9999,
            },
        })
        self.assertEqual(L.cache_read_tokens, 100)
        self.assertEqual(L.cache_write_tokens, 0)
        self.assertGreaterEqual(L.total, 0.0)

    # -- three-state cache reporting (#98) ---------------------------------- #
    #
    # The bug: `cache hit rate: 0.0%` came out of byte-identical code whether the
    # cache genuinely missed or the response carried no cache field at all. The
    # spec marks `prompt_tokens_details` nullable, so absence is normal -- these
    # pin that absence is recorded as UNKNOWN, and that a reported zero is not.

    def test_missing_cache_fields_are_not_a_reported_zero(self):
        # The shape the OpenAPI spec's own glm example ships.
        L = _agent.CostLedger()
        L.record({
            "prompt_tokens": 10000, "completion_tokens": 500,
            "prompt_tokens_details": None,
        })
        self.assertTrue(L.cache_read_unreported)
        self.assertTrue(L.cache_write_unreported)
        self.assertEqual(L.cache_read_tokens, 0)   # count unchanged from pre-#98

    def test_reported_zero_is_not_unreported(self):
        # THE point of #98: a provider that says "zero cached" has measured
        # something. Implementing the flag as `if not cache_read` erases exactly
        # the distinction this ticket exists to draw.
        L = _agent.CostLedger()
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 0, "cache_creation_input_tokens": 0,
            },
        })
        self.assertFalse(L.cache_read_unreported)
        self.assertFalse(L.cache_write_unreported)

    def test_null_valued_cache_field_counts_as_unreported(self):
        # The live kimi-k3/glm/deepseek shape: the key is present, the value null.
        # A `key in block` membership test would call this "reported".
        L = _agent.CostLedger()
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 13996, "cache_write_tokens": None,
            },
        })
        self.assertFalse(L.cache_read_unreported)
        self.assertTrue(L.cache_write_unreported)

    def test_non_numeric_cache_field_counts_as_unreported(self):
        # A value we cannot read is not a measurement. `bool` is an `int` subclass
        # but is never a token count (mirrors _as_int).
        for bad in ("9000", True, [], {}):
            with self.subTest(bad=bad):
                L = _agent.CostLedger()
                L.record({
                    "prompt_tokens": 100, "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": bad},
                })
                self.assertTrue(L.cache_read_unreported)
                self.assertEqual(L.cache_read_tokens, 0)

    def test_cache_write_aliases_are_recognized(self):
        # Each write key asserted INDIVIDUALLY -- an alias list checked only through
        # a response carrying both names is vacuous for whichever one loses.
        for key in ("cache_creation_input_tokens", "cache_write_tokens"):
            with self.subTest(key=key):
                L = _agent.CostLedger()
                L.record({
                    "prompt_tokens": 1000, "completion_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 0, key: 200},
                })
                self.assertFalse(L.cache_write_unreported)
                self.assertEqual(L.cache_write_tokens, 200)

    def test_documented_write_key_wins_over_the_observed_alias(self):
        # `cache_creation_input_tokens` is the spec'd name; `cache_write_tokens` is
        # observed-but-undocumented and must never take precedence.
        L = _agent.CostLedger()
        L.record({
            "prompt_tokens": 1000, "completion_tokens": 0,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_creation_input_tokens": 200,
                "cache_write_tokens": 900,
            },
        })
        self.assertEqual(L.cache_write_tokens, 200)

    def test_usage_report_says_n_a_when_no_cache_fields_were_reported(self):
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
        L.record({"prompt_tokens": 10000, "completion_tokens": 500})
        r = L.usage_report()
        self.assertIn("cache hit rate: n/a (no cache fields reported)", r)
        self.assertIn("cache breakdown not reported", r)
        self.assertNotIn("cache-read", r)   # no fabricated itemization
        self.assertNotIn("%", r.split("cost:")[0].split("input")[0])

    def test_usage_report_prints_a_real_zero_hit_rate(self):
        # The other half of the contract: don't replace one lie with another. A
        # provider-reported zero must still render as 0.0%, not n/a.
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 0, "cache_creation_input_tokens": 0,
            },
        })
        r = L.usage_report()
        self.assertIn("cache hit rate: 0.0%", r)
        # Scoped to the cache rows rather than the whole report: #99's trace block
        # legitimately renders `n/a` in its own columns (an unstamped window, an absent
        # per-row field), and a report-wide assertion would fail on an unrelated truth.
        # The claim being pinned is about the CACHE state, so pin the cache lines.
        cache_lines = [ln for ln in r.splitlines()
                       if "cache" in ln and not ln.startswith("    ")]
        self.assertEqual(cache_lines, [
            "  input          100 tok  (100 uncached + 0 cache-read + 0 cache-write)",
            "  cache hit rate: 0.0%",
        ])
        self.assertNotIn("[partially unreported]", r)

    def test_usage_report_marks_a_partially_unreported_hit_rate(self):
        # Turn 1 measured, turn 2 didn't: the rate is real but understated against
        # a prompt total that includes a turn it could not see.
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
        L.record({
            "prompt_tokens": 10000, "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 9000},
        })
        L.record({"prompt_tokens": 100, "completion_tokens": 10})
        r = L.usage_report()
        self.assertIn("cache hit rate: 89.1%  [partially unreported]", r)
        self.assertNotIn("n/a (no cache fields reported)", r)

    def test_cache_write_n_a_does_not_mark_the_hit_rate_line(self):
        # The regression `assertIn` cannot see: if both flags fed one shared marker,
        # every ordinary Venice session (read reported, write absent) would grow a
        # "[partially unreported]" on a rate that does not use the write bucket --
        # and `assertIn("cache hit rate: 90.0%", r)` would stay green throughout.
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
        L.record({
            "prompt_tokens": 10000, "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 9000},
        })
        r = L.usage_report()
        self.assertIn("cache-write n/a", r)
        self.assertNotIn("0 cache-write", r)
        self.assertIn("9,000 cache-read", r)
        rate_line = [ln for ln in r.splitlines() if "cache hit rate" in ln]
        self.assertEqual(rate_line, ["  cache hit rate: 90.0%"])

    # -- the hit rate on the one-line surfaces (#100) ------------------------ #
    #
    # #98 made the rate honest but left it visible in exactly one place (REPL
    # `/usage`), so a 94%->0% collapse ran for three days unseen. These pin the
    # rate onto `summary()`, which is what both run footers and `/cost` render.
    #
    # Every `summary` assertion below is a WHOLE-STRING assertEqual on purpose:
    # the fragment is a SUFFIX, and `assertIn("completion=5", s)` cannot see a
    # suffix appended after it (the #98 lesson, learned the expensive way).

    @staticmethod
    def _priced():
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
        return L

    def test_cache_hit_percent_is_three_state(self):
        # Unknowable two ways, and neither may render as a zero.
        blank = _agent.CostLedger()
        self.assertIsNone(blank.cache_hit_percent())        # nothing recorded
        unrep = _agent.CostLedger()
        unrep.record({"prompt_tokens": 100, "completion_tokens": 5})
        self.assertIsNone(unrep.cache_hit_percent())        # no cache field at all
        # ...but a provider-reported zero IS knowable, and is 0.0, not None.
        zero = _agent.CostLedger()
        zero.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        })
        self.assertEqual(zero.cache_hit_percent(), 0.0)
        self.assertIsNotNone(zero.cache_hit_percent())
        real = _agent.CostLedger()
        real.record({
            "prompt_tokens": 1010, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 900},
        })
        self.assertAlmostEqual(real.cache_hit_percent(), 89.10891089, places=6)
        self.assertEqual(real.cache_hit_percent(round_to=1), 89.1)

    def test_summary_stays_byte_identical_without_the_opt_in(self):
        # Default OFF is load-bearing: `run_loop`'s spend/token gates render this
        # into "why did I stop" messages on unpriced subagent ledgers.
        L = self._priced()
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 90},
        })
        self.assertEqual(
            L.summary(),
            "cost: $0.0004 (tokens prompt=100 completion=5)",
        )

    def test_summary_with_cache_renders_a_known_rate(self):
        L = self._priced()
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 90},
        })
        self.assertEqual(
            L.summary(cache=True),
            "cost: $0.0004 (tokens prompt=100 completion=5, cache 90.0% hit)",
        )

    def test_summary_with_cache_renders_a_real_zero(self):
        # The #98 contract on this surface too: a printed 0.0% means the provider
        # said zero. Collapsing this into the n/a branch is the whole bug.
        L = self._priced()
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        })
        self.assertEqual(
            L.summary(cache=True),
            "cost: $0.0004 (tokens prompt=100 completion=5, cache 0.0% hit)",
        )

    def test_summary_with_cache_says_n_a_when_nothing_was_reported(self):
        # Also pins the fragment onto the UNPRICED branch, which has its own
        # return path and no parens -- and which every command-level test renders,
        # because the catalog fake carries no `pricing`.
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 100, "completion_tokens": 5})
        self.assertEqual(
            L.summary(cache=True),
            "cost: (unpriced — model rate unknown) "
            "tokens prompt=100 completion=5, cache n/a",
        )

    def test_summary_with_cache_marks_a_partially_unreported_rate(self):
        # Same wording as `usage_report` -- an operator can run /cost and /usage
        # seconds apart, and two vocabularies for one state read as two states.
        L = self._priced()
        L.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 90},
        })
        L.record({"prompt_tokens": 100, "completion_tokens": 5})
        self.assertEqual(
            L.summary(cache=True),
            "cost: $0.0008 (tokens prompt=200 completion=10, "
            "cache 45.0% hit [partially unreported])",
        )

    def test_summary_with_cache_says_nothing_before_any_turn(self):
        # A ledger that recorded nothing has no cache claim to make -- not 0.0%,
        # and not "n/a" either, which would imply someone looked and failed.
        L = self._priced()
        self.assertEqual(
            L.summary(cache=True),
            "cost: $0.0000 (tokens prompt=0 completion=0)",
        )

    def test_json_carries_a_rounded_rate_or_null(self):
        # `venice sessions show` prints this dict straight at a human, so the
        # unrounded 89.10891089108911 is a regression there.
        L = self._priced()
        L.record({
            "prompt_tokens": 1010, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 900},
        })
        self.assertEqual(L.to_dict()["cache_hit_percent"], 89.1)
        blind = _agent.CostLedger()
        blind.record({"prompt_tokens": 100, "completion_tokens": 5})
        self.assertIn("cache_hit_percent", blind.to_dict())
        self.assertIsNone(blind.to_dict()["cache_hit_percent"])

    def test_restore_recomputes_the_rate_instead_of_accumulating_it(self):
        # Every neighbouring field in `restore` is additive because it is a tally.
        # This one is DERIVED; adding it would sum two percentages into 178.2.
        src = self._priced()
        src.record({
            "prompt_tokens": 100, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 90},
        })
        snap = src.to_dict()
        self.assertEqual(snap["cache_hit_percent"], 90.0)
        L = self._priced()
        L.restore(snap)
        L.restore(snap)
        self.assertEqual(L.prompt_tokens, 200)          # tallies DO accumulate
        self.assertEqual(L.to_dict()["cache_hit_percent"], 90.0)

    def test_usage_report_refuses_a_rate_over_zero_input(self):
        # Reachable from a partial/hand-edited envelope: `restore` (unlike
        # `record`) does not clamp the cache buckets to the prompt total. This
        # used to print "cache hit rate: 0.0%" -- a measurement of nothing, which
        # is the #98 lie one level down. Distinct wording from the flag case, so
        # the two unknowns stay tellable apart.
        L = _agent.CostLedger()
        L.restore({"completion_tokens": 5})
        L.record_turn(1.0)
        r = L.usage_report()
        rate_line = [ln for ln in r.splitlines() if "cache hit rate" in ln]
        self.assertEqual(rate_line, ["  cache hit rate: n/a (no input tokens)"])

    # -- VENICE_USAGE_RAW opt-in dump (#98) --------------------------------- #

    def _record_with_raw(self, value, usage):
        """record(usage) with $VENICE_USAGE_RAW = `value` (None = unset); -> stderr."""
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VENICE_USAGE_RAW", None)
            if value is not None:
                os.environ["VENICE_USAGE_RAW"] = value
            with contextlib.redirect_stderr(buf):
                _agent.CostLedger().record(usage)
        return buf.getvalue()

    def test_usage_raw_dumps_the_verbatim_block_to_stderr(self):
        usage = {
            "prompt_tokens": 11, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": None},
        }
        err = self._record_with_raw("1", usage)
        self.assertTrue(err.startswith("usage-raw: "))
        # Round-trips to the SAME block -- a reformatted subset would not have
        # answered the question the raw dump exists to answer.
        self.assertEqual(json.loads(err[len("usage-raw: "):]), usage)

    def test_usage_raw_dumps_null_for_a_missing_usage_block(self):
        # Fires ahead of record()'s `usage is None` bail, so "no usage block at all"
        # is distinguishable from "a block with no cache fields".
        self.assertEqual(self._record_with_raw("1", None).strip(), "usage-raw: null")

    def test_usage_raw_is_off_by_default_and_respects_a_false_value(self):
        # A bare `if os.environ.get(...)` would turn the dump ON for "0"/"false".
        for value in (None, "", "0", "false", "no", "off"):
            with self.subTest(value=value):
                self.assertEqual(self._record_with_raw(value, {"prompt_tokens": 1}), "")

    def test_usage_raw_accepts_the_documented_truthy_spellings(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value):
                err = self._record_with_raw(value, {"prompt_tokens": 1})
                self.assertIn("usage-raw:", err)

    def test_usage_raw_survives_an_unserializable_usage(self):
        # Mixed key types are unsortable, so `sort_keys=True` raises TypeError --
        # a real trigger for the fallback (`default=str` rescues odd *values*, not
        # this). A diagnostic must never take down the turn that produced it.
        err = self._record_with_raw("1", {"prompt_tokens": 1, 2: "x"})
        self.assertIn("usage-raw:", err)
        self.assertIn("prompt_tokens", err)

    def test_reasoning_tokens_captured(self):
        L = _agent.CostLedger()
        L.record({
            "prompt_tokens": 10, "completion_tokens": 200,
            "completion_tokens_details": {"reasoning_tokens": 128},
        })
        self.assertEqual(L.reasoning_tokens, 128)

    # -- per-subagent token cap (#52) --------------------------------------- #

    def test_max_tokens_defaults_off_and_normalizes_nonpositive(self):
        self.assertIsNone(_agent.CostLedger().max_tokens)          # default off
        self.assertIsNone(_agent.CostLedger(max_tokens=0).max_tokens)   # <=0 -> None
        self.assertIsNone(_agent.CostLedger(max_tokens=-5).max_tokens)
        self.assertEqual(_agent.CostLedger(max_tokens=100).max_tokens, 100)

    def test_over_tokens_counts_prompt_plus_completion_unpriced(self):
        # No bind_pricing: token counting needs no catalog, so the cap works unpriced.
        L = _agent.CostLedger(max_tokens=100)
        L.record({"prompt_tokens": 60, "completion_tokens": 39})   # 99 < 100
        self.assertFalse(L.over_tokens())
        L.record({"prompt_tokens": 1, "completion_tokens": 0})     # 100 >= 100
        self.assertTrue(L.over_tokens())
        self.assertTrue(L.unpriced)                                # never priced

    def test_token_cap_independent_of_spend_cap(self):
        # A token-capped ledger with NO max_spend never trips the USD gate, even as
        # tokens blow past the cap -- pins that over() stays USD-only.
        L = _agent.CostLedger(max_tokens=50)
        L.record({"prompt_tokens": 10**6, "completion_tokens": 10**6})
        self.assertTrue(L.over_tokens())
        self.assertFalse(L.over())
        # And the converse: a USD-only ledger never trips the token gate.
        L2 = _agent.CostLedger(max_spend=0.001)
        L2.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 1.0}})
        L2.record({"prompt_tokens": 2000, "completion_tokens": 0})
        self.assertTrue(L2.over())
        self.assertFalse(L2.over_tokens())

    def test_token_cap_not_persisted(self):
        # Caps are re-derived at construction (mirrors max_spend), so a snapshot omits
        # max_tokens and a restore never resurrects a ceiling.
        L = _agent.CostLedger(max_tokens=100)
        self.assertNotIn("max_tokens", L.to_dict())
        L2 = _agent.CostLedger()
        L2.restore({"max_tokens": 100, "prompt_tokens": 10})
        self.assertIsNone(L2.max_tokens)
        self.assertEqual(L2.prompt_tokens, 10)

    # -- usage_report + always-on ledger (#75) ------------------------------ #

    def test_usage_report_shows_cache_split_and_cost(self):
        L = _agent.CostLedger()
        L.bind_pricing({
            "input": {"usd": 3.0}, "cache_input": {"usd": 0.3},
            "output": {"usd": 15.0},
        })
        L.record({
            "prompt_tokens": 10000, "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 9000},
        })
        r = L.usage_report()
        self.assertIn("uncached", r)
        self.assertIn("cache-read", r)
        self.assertIn("1,000 uncached", r)
        self.assertIn("9,000 cache-read", r)
        self.assertIn("cache hit rate: 90.0%", r)
        self.assertIn("$0.0132", r)

    def test_usage_report_empty_before_any_turn(self):
        self.assertEqual(_agent.CostLedger().usage_report(), "(no usage recorded yet)")

    def test_usage_report_unpriced(self):
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 500, "completion_tokens": 20})
        r = L.usage_report()
        self.assertIn("model rate unknown", r)
        self.assertIn("500", r)

    def test_usage_ledger_always_on_and_priced(self):
        args = type("A", (), {"session_max_spend": None})()
        models = [{"id": "m", "model_spec": {"pricing": {"input": {"usd": 2.0}}}}]
        L = _agent.usage_ledger(args, models, "m")
        self.assertIsNotNone(L)             # unlike ledger_from_args
        self.assertIsNone(L.max_spend)      # uncapped
        self.assertEqual(L._in, 2.0 / 1e6)  # still priced

    def test_usage_ledger_honors_cap(self):
        args = type("A", (), {"session_max_spend": 0.25})()
        L = _agent.usage_ledger(args, [], "m")
        self.assertEqual(L.max_spend, 0.25)

    # -- wall-clock per turn (#81) ------------------------------------------ #

    def test_record_turn_accumulates_elapsed_and_turns(self):
        L = _agent.CostLedger()
        L.record_turn(1.25)
        L.record_turn(2.75)
        self.assertAlmostEqual(L.elapsed_seconds, 4.0)
        self.assertEqual(L.turns, 2)

    def test_record_turn_survives_garbage_but_still_counts_the_turn(self):
        # A monotonic clock can't run backwards, so these only ever mean a caller
        # bug -- but dropping the turn too would corrupt the average, which is the
        # number most likely to be read.
        L = _agent.CostLedger()
        for bad in (None, -5, True, "nope"):
            L.record_turn(bad)
        self.assertEqual(L.elapsed_seconds, 0.0)
        self.assertEqual(L.turns, 4)

    def test_timing_survives_a_resume_round_trip(self):
        # Additive, like the token tallies: `--resume` reports the TOTAL time this
        # session has kept you waiting, not just the latest leg.
        L = _agent.CostLedger()
        L.record_turn(2.0)
        L.record_turn(4.0)
        R = _agent.CostLedger()
        R.restore(L.to_dict())
        R.restore(L.to_dict())
        self.assertAlmostEqual(R.elapsed_seconds, 12.0)
        self.assertEqual(R.turns, 4)

    def test_restore_tolerates_missing_and_garbage_timing_keys(self):
        # A pre-#81 envelope has neither key; both degrade to 0 with no version bump.
        old = _agent.CostLedger()
        old.restore({"prompt_tokens": 5})
        self.assertEqual((old.elapsed_seconds, old.turns), (0.0, 0))
        junk = _agent.CostLedger()
        junk.restore({"elapsed_seconds": "x", "turns": -3})
        self.assertEqual((junk.elapsed_seconds, junk.turns), (0.0, 0))

    def test_usage_report_empty_string_survives_the_new_timing_row(self):
        # The pre-turn placeholder is pinned verbatim above; a ledger that spent
        # time but got no tokens back (a turn that raised) must still report its
        # clock -- and must NOT fabricate a cache breakdown for tokens it never saw.
        self.assertEqual(_agent.CostLedger().usage_report(), "(no usage recorded yet)")
        L = _agent.CostLedger()
        L.record_turn(2.0)
        r = L.usage_report()
        self.assertNotEqual(r, "(no usage recorded yet)")
        self.assertIn("2.0s", r)
        self.assertIn("no tokens reported", r)
        self.assertNotIn("cache hit rate", r)

    def test_usage_report_has_no_wall_row_without_a_recorded_turn(self):
        # `run_loop` in isolation and every per-subagent ledger only ever call
        # record(); they must report exactly what they reported before #81.
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 100, "completion_tokens": 5})
        self.assertNotIn("wall", L.usage_report())

    def test_usage_report_shows_wall_total_count_and_average(self):
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 100, "completion_tokens": 5})
        L.record_turn(60.0)
        L.record_turn(30.0)
        r = L.usage_report()
        self.assertIn("1m 30s", r)
        self.assertIn("over 2 turn(s)", r)
        self.assertIn("avg 45.0s", r)

    def test_format_duration_thresholds(self):
        f = _agent.format_duration
        self.assertEqual(f(0), "0.0s")
        self.assertEqual(f(4.5), "4.5s")
        self.assertEqual(f(59.9), "59.9s")
        self.assertEqual(f(60), "1m 00s")
        self.assertEqual(f(134), "2m 14s")
        self.assertEqual(f(3600), "1h 00m")
        self.assertEqual(f(7265), "2h 01m")
        # A bad value must never make a report unreadable.
        self.assertEqual(f(-5), "0.0s")
        self.assertEqual(f(None), "0.0s")

    # -- per-tool timing (#82) ----------------------------------------------- #

    def test_record_tool_accumulates_seconds_and_calls(self):
        L = _agent.CostLedger()
        L.record_tool("shell", 1.25)
        L.record_tool("shell", 2.75)
        L.record_tool("read_file", 0.5)
        # Whole-dict equality: a partial assertion here would not notice a stray key.
        self.assertEqual(L.tools, {"shell": {"seconds": 4.0, "calls": 2},
                                   "read_file": {"seconds": 0.5, "calls": 1}})
        self.assertEqual(L.tool_seconds(), 4.5)
        self.assertEqual(L.tool_calls_total(), 3)

    def test_record_tool_survives_garbage_but_still_counts_the_call(self):
        # `record_turn`'s rule one level down: the clock can't run backwards, so this
        # only ever catches a caller bug -- and dropping the call would corrupt the
        # per-call average, which is what the breakdown is read for.
        L = _agent.CostLedger()
        for bad in (None, -5, True, "nope"):
            L.record_tool("shell", bad)
        self.assertEqual(L.tools, {"shell": {"seconds": 0.0, "calls": 4}})

    def test_record_tool_drops_a_blank_name(self):
        # The one place this diverges from `record_turn`: there is no row to put an
        # anonymous call on, so it is dropped rather than counted under "".
        L = _agent.CostLedger()
        for blank in ("", "   ", None):
            L.record_tool(blank, 5.0)
        self.assertEqual(L.tools, {})
        self.assertEqual(L.tool_calls_total(), 0)

    def test_tools_survive_a_resume_round_trip(self):
        # Additive per NAME, like the token tallies: `--resume` reports where the whole
        # session's tool time went, not just the latest leg.
        L = _agent.CostLedger()
        L.record_tool("shell", 2.0)
        L.record_tool("apply_patch", 4.0)
        R = _agent.CostLedger()
        R.restore(L.to_dict())
        R.restore(L.to_dict())
        self.assertEqual(R.tools, {"shell": {"seconds": 4.0, "calls": 2},
                                   "apply_patch": {"seconds": 8.0, "calls": 2}})

    def test_tool_seconds_is_derived_and_not_restored(self):
        # The `cache_hit_percent` rule one field over: `tool_seconds` is derived from
        # the map `restore` seeds, so accumulating it too would DOUBLE every resumed
        # run's tool time. 2 restores of a 6.0s ledger -> 12.0s, never 24.0s.
        L = _agent.CostLedger()
        L.record_tool("shell", 6.0)
        R = _agent.CostLedger()
        R.restore(L.to_dict())
        R.restore(L.to_dict())
        self.assertEqual(R.tool_seconds(), 12.0)
        self.assertEqual(R.to_dict()["tool_seconds"], 12.0)

    def test_restore_tolerates_missing_and_garbage_tools_keys(self):
        # A pre-#82 envelope has no `tools` key; a hand-edited one may have anything.
        # Tolerance is PER ROW -- one bad entry must not cost the good ones.
        old = _agent.CostLedger()
        old.restore({"prompt_tokens": 5})
        self.assertEqual(old.tools, {})
        junk = _agent.CostLedger()
        junk.restore({"tools": "not-a-dict"})
        self.assertEqual(junk.tools, {})
        mixed = _agent.CostLedger()
        mixed.restore({"tools": {"shell": 5, "": {"seconds": 1, "calls": 1},
                                 "bad": {"seconds": "x", "calls": -2},
                                 "ok": {"seconds": 3.0, "calls": 2}}})
        self.assertEqual(mixed.tools, {"bad": {"seconds": 0.0, "calls": 0},
                                       "ok": {"seconds": 3.0, "calls": 2}})

    def test_to_dict_tools_is_name_sorted_and_self_describing(self):
        L = _agent.CostLedger()
        for name in ("zebra", "alpha", "middle"):
            L.record_tool(name, 1.0)
        d = L.to_dict()
        # Name-sorted so two runs of the same shape produce a diffable envelope. The
        # HUMAN block sorts by time instead; that divergence is deliberate.
        self.assertEqual(list(d["tools"]), ["alpha", "middle", "zebra"])
        self.assertEqual(d["tools"]["alpha"], {"seconds": 1.0, "calls": 1})

    def test_to_dict_tools_rows_are_copies(self):
        # The envelope reaches session files and `--json`; a caller mutating it must
        # not be able to reach back into the live ledger.
        L = _agent.CostLedger()
        L.record_tool("shell", 1.0)
        d = L.to_dict()
        d["tools"]["shell"]["calls"] = 999
        self.assertEqual(L.tools["shell"]["calls"], 1)

    def _seeded(self, pairs, *, turn=None):
        """A ledger with `(name, total_seconds, call_count)` rows.

        The whole duration goes on the first call and the rest record 0.0 -- the render
        only reads the total and the count, and splitting evenly would make the expected
        strings depend on 3-dp rounding of a repeating decimal.
        """
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 100, "completion_tokens": 5})
        for name, secs, n in pairs:
            for i in range(n):
                L.record_tool(name, secs if i == 0 else 0.0)
        if turn is not None:
            L.record_turn(turn)
        return L

    def _tool_block(self, report):
        """The tools block only: the header line and everything indented under it.

        #99 appended a `calls` block after this one, so the slice has to STOP at it --
        an open-ended tail would make every assertion below silently about two blocks.
        """
        lines = report.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("  tools "))
        rest = lines[i + 1:]
        j = next((k for k, ln in enumerate(rest) if ln.startswith("  calls ")), len(rest))
        return lines[i:i + 1 + j]

    def test_usage_report_tools_block_pins_the_whole_line(self):
        # assertIn is a substring match and cannot see a wrongly-appended suffix (the
        # `[concurrent]` marker, a stray column) -- pin whole lines.
        L = self._seeded([("shell", 118.0, 7), ("apply_patch", 31.4, 5)], turn=252.0)
        self.assertEqual(self._tool_block(L.usage_report()), [
            "  tools       2m 29s  across 12 call(s)",
            "    shell           1m 58s   7 call(s)",
            "    apply_patch      31.4s   5 call(s)",
        ])

    def test_wall_and_tools_durations_share_a_column(self):
        L = self._seeded([("shell", 118.0, 7)], turn=252.0)
        wall, tools = [ln for ln in L.usage_report().splitlines()
                       if ln.startswith(("  wall ", "  tools "))]
        self.assertEqual(wall.index("4m 12s"), tools.index("1m 58s"))

    def test_usage_report_has_no_tools_block_without_a_recorded_tool(self):
        L = self._seeded([], turn=10.0)
        self.assertNotIn("  tools", L.usage_report())

    def test_tools_block_renders_without_a_recorded_turn(self):
        # Deliberate asymmetry with `_timing_line`: a per-subagent ledger has real tool
        # time and no turn. Gating on `turns` would be a false dependency.
        L = self._seeded([("shell", 4.0, 1)])
        r = L.usage_report()
        self.assertNotIn("  wall", r)
        self.assertEqual(self._tool_block(r), [
            "  tools         4.0s  across 1 call(s)",
            "    shell             4.0s   1 call(s)",
        ])

    def test_no_tokens_branch_still_shows_the_tools_block(self):
        # A turn that burned minutes in tools then raised before any usage came back.
        L = _agent.CostLedger()
        L.record_tool("shell", 90.0)
        L.record_turn(95.0)
        self.assertEqual(L.usage_report().splitlines(), [
            "session usage:",
            "  (no tokens reported)",
            "  wall        1m 35s  over 1 turn(s)  (avg 1m 35s)",
            "  tools       1m 30s  across 1 call(s)",
            "    shell           1m 30s   1 call(s)",
        ])

    def test_tools_block_sorts_by_seconds_then_name(self):
        # Insertion order, name order and time order are all DIFFERENT here, so a
        # dropped tiebreak cannot pass on `sorted`'s stability.
        L = self._seeded([("m_mid", 5.0, 1), ("z_tied", 9.0, 1),
                          ("a_tied", 9.0, 1), ("b_low", 1.0, 1)])
        self.assertEqual([ln.split()[0] for ln in self._tool_block(L.usage_report())[1:]],
                         ["a_tied", "z_tied", "m_mid", "b_low"])

    def test_tools_block_caps_rows_and_the_residual_reconciles(self):
        # 12 tools -> 8 rows + one folded line whose seconds and calls make the rows
        # add back up to the header. A truncated block that doesn't reconcile is the
        # #98 lie in a new costume.
        L = self._seeded([(f"tool{i:02d}", float(12 - i), 1) for i in range(12)])
        block = self._tool_block(L.usage_report())
        self.assertEqual(len(block), 1 + 8 + 1)
        # Header 78s; the 8 shown rows are 12+11+...+5 = 68s, so the fold must carry
        # the remaining 4+3+2+1 = 10.0s over 4 calls for the block to add back up.
        self.assertEqual(block[0], "  tools       1m 18s  across 12 call(s)")
        self.assertEqual(block[-1], "    (+4 more)        10.0s   4 call(s)")

    def test_tools_block_marks_concurrent_only_when_it_exceeds_wall(self):
        head = lambda L: self._tool_block(L.usage_report())[0]
        over = self._seeded([("venice_spawn", 90.0, 3)], turn=40.0)
        self.assertIn("[concurrent -- exceeds wall]", head(over))
        # Exactly equal must NOT mark.
        exact = self._seeded([("shell", 40.0, 1)], turn=40.0)
        self.assertNotIn("concurrent", head(exact))
        # Exactly ON the rounding tolerance must NOT mark either. This is the case
        # that discriminates: the equal case above CANNOT tell `>` from `>=`, because
        # `40.0 > 40.001` and `40.0 >= 40.001` are both False -- the tolerance
        # dominates the operator. Here `40.001 >= 40.001` is True and `>` is False, so
        # this single case kills both `>=` AND dropping the `+ 0.001` (which would let
        # `record_turn`'s 3-dp rounding fire the marker on a plain serial run).
        edge = self._seeded([("shell", 40.001, 1)], turn=40.0)
        self.assertNotIn("concurrent", head(edge))
        # No turn recorded -> nothing to exceed, so no marker.
        noturn = self._seeded([("shell", 40.0, 1)])
        self.assertNotIn("concurrent", head(noturn))

    def test_tools_block_column_absorbs_a_long_mcp_name(self):
        long = "mcp__some_server__a_really_long_tool_name"
        L = self._seeded([(long, 4.0, 1), ("shell", 1.0, 1)])
        self.assertEqual(self._tool_block(L.usage_report())[1:], [
            f"    {long}      4.0s   1 call(s)",
            f"    {'shell':<{len(long)}}      1.0s   1 call(s)",
        ])
        self.assertNotIn("...", L.usage_report())  # the name is never truncated

    def test_tools_fragment_is_empty_without_tools(self):
        self.assertEqual(_agent.CostLedger().tools_fragment(), "")
        self.assertEqual(self._seeded([], turn=10.0).tools_fragment(), "")

    def test_tools_fragment_is_the_footer_clause(self):
        self.assertEqual(self._seeded([("shell", 161.0, 7)], turn=252.0).tools_fragment(),
                         " (2m 41s tools)")
        self.assertEqual(self._seeded([("venice_spawn", 362.0, 3)], turn=40.0)
                         .tools_fragment(), " (6m 02s tools, concurrent)")


class TestOffLoopBuckets(unittest.TestCase):
    """#101: off-loop spend (compaction today) is metered in its own partition.

    The whole point is ISOLATION: a bucketed call must be costed like any other and
    then land nowhere near the main-loop counters, because `_compact`'s summary call is
    a fresh prefix that reads ~0% cached every time and would fabricate exactly the
    cache cliff the #99 trace exists to detect.
    """

    P = {"input": {"usd": 1.0}, "output": {"usd": 2.0}}

    def _led(self, cap=None):
        L = _agent.CostLedger(max_spend=cap)
        L.bind_pricing(self.P)
        return L

    def _bucketed(self, cap=None):
        """A ledger with one main-loop turn and one compaction call."""
        L = self._led(cap)
        L.record({"prompt_tokens": 1000, "completion_tokens": 500}, seconds=2.0)
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, seconds=1.5,
                 bucket="compaction")
        return L

    # -- isolation -------------------------------------------------------------

    def test_a_bucket_record_leaves_the_main_counters_untouched(self):
        L = self._led()
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertEqual(
            (L.prompt_tokens, L.completion_tokens, L.total), (0, 0, 0.0))

    def test_a_bucket_record_never_appends_a_trace_row(self):
        # Counting the call in `api_calls_total` without a row would drive
        # `calls_dropped()` positive and print a phantom "[N row(s) dropped]".
        L = self._led()
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertEqual(
            (L.api_calls(), L.api_calls_total, L.calls_dropped()), ([], 0, 0))

    def test_a_bucket_record_does_not_set_the_cache_unreported_flags(self):
        # They drive `/usage`'s refusal to print a rate; an off-loop call reporting no
        # cache block says nothing about the conversation's cache.
        L = self._led()
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertEqual(
            (L.cache_read_unreported, L.cache_write_unreported), (False, False))

    def test_a_bucket_record_still_returns_its_cost(self):
        L = self._led()
        c = L.record({"prompt_tokens": 4000, "completion_tokens": 200},
                     bucket="compaction")
        self.assertAlmostEqual(c, 0.0044)  # 4000*1e-6 + 200*2e-6

    def test_the_bucket_row_accumulates_across_calls(self):
        L = self._led()
        for _ in range(3):
            L.record({"prompt_tokens": 1000, "completion_tokens": 100}, seconds=0.5,
                     bucket="compaction")
        self.assertEqual(L.buckets["compaction"], {
            "calls": 3, "cost": L.buckets["compaction"]["cost"],
            "prompt_tokens": 3000, "completion_tokens": 300,
            "seconds": 1.5, "unpriced": False, "unreported": False,
        })

    def test_an_unstamped_bucket_window_adds_no_seconds(self):
        L = self._led()
        L.record({"prompt_tokens": 10, "completion_tokens": 1}, bucket="compaction")
        self.assertEqual(L.buckets["compaction"]["seconds"], 0.0)

    def test_an_unpriced_bucket_call_is_flagged_not_zeroed(self):
        L = _agent.CostLedger()  # no pricing bound
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertIs(L.buckets["compaction"]["unpriced"], True)
        self.assertFalse(L.unpriced)  # the MAIN flag is untouched

    # -- totals and gates ------------------------------------------------------

    def test_billed_total_is_main_plus_buckets(self):
        L = self._bucketed()
        self.assertAlmostEqual(L.total, 0.002)      # 1000*1e-6 + 500*2e-6
        self.assertAlmostEqual(L.bucket_cost(), 0.0044)
        self.assertAlmostEqual(L.billed_total(), 0.0064)

    def test_bucket_cost_and_calls_can_be_asked_by_name(self):
        L = self._bucketed()
        self.assertAlmostEqual(L.bucket_cost("compaction"), 0.0044)
        self.assertEqual(L.bucket_calls("compaction"), 1)
        self.assertEqual(L.bucket_cost("nope"), 0.0)
        self.assertEqual(L.bucket_calls("nope"), 0)

    def test_over_counts_bucket_spend(self):
        # THE #101 ask: --session-max-spend must see off-loop money.
        L = self._led(cap=0.001)
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertEqual(L.total, 0.0)   # nothing in the main loop at all
        self.assertTrue(L.over())

    def test_over_tokens_ignores_bucket_tokens(self):
        # `--subagent-max-tokens` is #52's per-worker ceiling on the worker's OWN
        # conversation; off-loop tokens are not part of it.
        L = _agent.CostLedger(max_tokens=1000)
        L.bind_pricing(self.P)
        L.record({"prompt_tokens": 50_000, "completion_tokens": 5000},
                 bucket="compaction")
        self.assertFalse(L.over_tokens())

    # -- summary() -------------------------------------------------------------

    def test_summary_names_the_compaction_split(self):
        L = self._bucketed()
        self.assertEqual(
            L.summary(),
            "cost: $0.0064 (tokens prompt=1000 completion=500)"
            " [incl. $0.0044 compaction]")

    def test_summary_is_unchanged_without_a_bucket(self):
        # The subagent-safety pin. `run_loop`'s two stop-reason messages render this on
        # subagent ledgers, which can never acquire a bucket (they are never passed a
        # `budget`, so `maybe_compact` short-circuits).
        #
        # #117 was predicted to delete that invariant and did NOT -- stated here because
        # the issue said the opposite and a later reader will assume it landed. A
        # mirrored subagent ledger writes its PARENT's buckets and never its own, so
        # these two messages still render with no clause. See
        # `test_a_mirrored_child_never_grows_a_bucket_of_its_own`, which pins the reason.
        L = self._led()
        L.record({"prompt_tokens": 1000, "completion_tokens": 500})
        self.assertEqual(
            L.summary(), "cost: $0.0020 (tokens prompt=1000 completion=500)")

    def test_summary_on_an_unpriced_bucket_says_nothing_extra(self):
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertEqual(
            L.summary(),
            "cost: (unpriced — model rate unknown) tokens prompt=0 completion=0")

    def test_summary_before_any_turn_with_only_a_compaction(self):
        # Odd-looking but honest: `/compact` before the first turn really did spend
        # money on zero conversation tokens. Pinned so nobody "fixes" it into the
        # unpriced branch, which would print "(model rate unknown)" over a real cost.
        L = self._led()
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertEqual(
            L.summary(),
            "cost: $0.0044 (tokens prompt=0 completion=0)"
            " [incl. $0.0044 compaction]")

    # -- usage_report() --------------------------------------------------------

    def test_usage_report_renders_the_bucket_block(self):
        L = self._bucketed()
        lines = L.usage_report().split("\n")
        self.assertIn(
            "  off-loop      1.5s  across 1 API call(s)  [not in the trace above]",
            lines)
        self.assertIn(
            "    compaction  1 call(s)      4,000 in      200 out     $0.0044", lines)

    def test_the_calls_header_names_the_off_loop_calls(self):
        L = self._bucketed()
        lines = L.usage_report().split("\n")
        self.assertIn(
            "  calls         2.0s  across 1 API call(s)  [+1 off-loop]", lines)

    def test_usage_report_renders_the_bucket_block_with_no_tokens(self):
        # `/compact` before any turn lands in the "(no tokens reported)" branch -- the
        # one state where the bucket is the only money on the page. #82's "wire BOTH
        # sites" rule, third time.
        L = self._led()
        L.record({"prompt_tokens": 9000, "completion_tokens": 300}, seconds=1.1,
                 bucket="compaction")
        lines = L.usage_report().split("\n")
        self.assertIn("  (no tokens reported)", lines)
        self.assertIn(
            "  off-loop      1.1s  across 1 API call(s)  [not in the trace above]",
            lines)

    def test_no_usage_recorded_yet_yields_to_a_bucket(self):
        L = self._led()
        self.assertEqual(L.usage_report(), "(no usage recorded yet)")
        L.record({"prompt_tokens": 9000, "completion_tokens": 300}, bucket="compaction")
        self.assertNotEqual(L.usage_report(), "(no usage recorded yet)")

    def test_usage_report_cost_line_is_the_billed_total(self):
        # `/usage` and `/cost` are run seconds apart; two numbers for one question
        # reads as two questions.
        L = self._bucketed(cap=5.0)
        self.assertIn("  cost: $0.0064 / cap $5.00", L.usage_report().split("\n"))

    def test_a_no_usage_bucket_call_renders_as_n_a_not_zero(self):
        # A PRICED ledger whose response carried no usage block knows nothing: the
        # tokens are unknown and so is the cost. "0 in  0 out  $0.0000" would be a
        # measurement that never happened -- #98's rule, in the one partition that has
        # neither the trace's per-row nulls nor the "(no tokens reported)" branch.
        L = self._led()
        L.record(None, seconds=0.4, bucket="compaction")
        self.assertIs(L.buckets["compaction"]["unreported"], True)
        self.assertIn(
            "    compaction  1 call(s)        n/a in      n/a out         n/a"
            "  [unreported]",
            L.usage_report().split("\n"))

    def test_a_partly_blind_bucket_says_so_instead_of_looking_complete(self):
        # A bucket mixing a usage-less call with a usage-bearing one has real numbers
        # that are nonetheless an UNDERCOUNT. The first cut tested
        # `unreported and not (pt or ct)`, which has no partial state -- one reporting
        # call hid the marker and the shortfall rendered as a full measurement. Same
        # shape as the main token block's `[partially unreported]` since #98.
        L = self._led()
        L.record(None, bucket="compaction")
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertIs(L.buckets["compaction"]["unreported"], True)
        lines = L.usage_report().split("\n")
        self.assertIn(
            "    compaction  2 call(s)      4,000 in      200 out     $0.0044"
            "  [partially unreported]", lines)
        # ...and the cost line carries it too, so `/usage`'s total is not read as exact
        self.assertIn("  cost: $0.0044  [partially unreported]", lines)

    def test_restore_keeps_bucket_unreported_sticky(self):
        R = self._led()
        R.restore({"buckets": {"compaction": {"calls": 1, "unreported": True}}})
        R.restore({"buckets": {"compaction": {"calls": 1, "unreported": False}}})
        self.assertIs(R.buckets["compaction"]["unreported"], True)

    def test_an_unpriced_bucket_renders_as_unknown_not_zero(self):
        L = _agent.CostLedger()
        L.record({"prompt_tokens": 9000, "completion_tokens": 300}, bucket="compaction")
        self.assertIn(
            "    compaction  1 call(s)      9,000 in      300 out  (unpriced)",
            L.usage_report().split("\n"))

    # -- the event's cost clause ----------------------------------------------

    def _ev(self, **kw):
        ev = {"trigger": "auto", "after_n": 2, "messages_before": 48,
              "messages_after": 13, "est_tokens_before": 91200,
              "est_tokens_after": 8210, "observed_tokens_before": None}
        ev.update(kw)
        return ev

    def test_an_event_line_carries_the_summarization_cost(self):
        self.assertEqual(
            _agent.CostLedger._event_lines(self._ev(cost=0.0031))[0],
            "    -- compacted (auto) after #2: 48 -> 13 msgs,"
            " ~91,200 -> ~8,210 tok est, $0.0031 to summarize")

    def test_an_event_line_omits_a_zero_cost(self):
        # A static method cannot see `self.unpriced`, so a `"cost" in ev` gate would
        # print "$0.0000" for every unpriced session -- #98's fabricated zero.
        self.assertEqual(
            _agent.CostLedger._event_lines(self._ev(cost=0.0))[0],
            "    -- compacted (auto) after #2: 48 -> 13 msgs,"
            " ~91,200 -> ~8,210 tok est")

    def test_an_event_line_without_a_cost_key_is_unchanged(self):
        # Pre-#101 envelopes, restored from a session file.
        self.assertEqual(
            _agent.CostLedger._event_lines(self._ev())[0],
            "    -- compacted (auto) after #2: 48 -> 13 msgs,"
            " ~91,200 -> ~8,210 tok est")

    # -- persistence -----------------------------------------------------------

    def test_a_sub_hundredth_cent_bucket_does_not_claim_a_zero_share(self):
        # `off > 0` is true for 3.3e-05 but `f"{off:.4f}"` is "0.0000", so the clause
        # would read `[incl. $0.0000 compaction]` -- a share of nothing. #98's rule:
        # the gate has to test what will RENDER, not the raw float.
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 0.05}, "output": {"usd": 0.10}})
        L.record({"prompt_tokens": 1000, "completion_tokens": 500})
        L.record({"prompt_tokens": 500, "completion_tokens": 80}, bucket="compaction")
        self.assertGreater(L.bucket_cost(), 0)          # real money...
        self.assertNotIn("incl.", L.summary())          # ...too small to name

    def test_a_sub_hundredth_cent_event_does_not_render_a_zero(self):
        line = _agent.CostLedger._event_lines(self._ev(cost=3.3e-05))[0]
        self.assertNotIn("$0.0000", line)
        self.assertEqual(
            line,
            "    -- compacted (auto) after #2: 48 -> 13 msgs,"
            " ~91,200 -> ~8,210 tok est")

    def test_a_bucket_row_renders_sub_cent_spend_as_a_bound(self):
        L = _agent.CostLedger()
        L.bind_pricing({"input": {"usd": 0.05}, "output": {"usd": 0.10}})
        L.record({"prompt_tokens": 500, "completion_tokens": 80}, bucket="compaction")
        self.assertIn(
            "    compaction  1 call(s)        500 in       80 out    <$0.0001",
            L.usage_report().split("\n"))

    def test_an_event_cost_from_a_hand_edited_file_cannot_crash_usage(self):
        # `restore()` seeds `context_events` VERBATIM -- rows are copied, never
        # validated field by field -- so `/usage` on a resumed session is where a
        # foreign `"cost": "0.0031"` would land. A raw `:.4f` raises there.
        for junk in ("0.0031", True, {}, None, [], float("nan")):
            with self.subTest(cost=junk):
                line = _agent.CostLedger._event_lines(self._ev(cost=junk))[0]
                self.assertTrue(line.startswith("    -- compacted (auto) after #2:"))

    def test_usage_report_shows_the_cost_and_cap_when_only_a_bucket_spent(self):
        # `/usage` and `/cost` must not disagree. The no-tokens branch used to omit
        # the cost line entirely, which was harmless only while that state implied
        # zero spend -- a `/compact` before the first turn lands here with real money.
        L = self._led(cap=0.50)
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        lines = L.usage_report().split("\n")
        self.assertIn("  (no tokens reported)", lines)
        self.assertIn("  cost: $0.0044 / cap $0.50", lines)
        self.assertIn("cost: $0.0044 / cap $0.50", L.summary())  # ...and they agree

    def test_the_no_tokens_branch_stays_silent_when_nothing_was_spent(self):
        # ...but the line must NOT appear for a run that merely reported no usage:
        # `cost: $0.0000` there is the fabricated zero the branch exists to avoid.
        L = self._led()
        L.record(None, seconds=1.0)
        lines = L.usage_report().split("\n")
        self.assertIn("  (no tokens reported)", lines)
        self.assertFalse([ln for ln in lines if ln.startswith("  cost:")])

    def test_buckets_round_trip_through_to_dict(self):
        L = self._bucketed()
        R = self._led()
        R.restore(L.to_dict())
        self.assertEqual(R.to_dict()["buckets"], L.to_dict()["buckets"])

    def test_to_dict_carries_the_derived_billed_total(self):
        L = self._bucketed()
        self.assertAlmostEqual(L.to_dict()["billed_total"], 0.0064)

    def test_restore_is_additive_per_bucket(self):
        L = self._bucketed()
        R = self._led()
        d = L.to_dict()
        R.restore(d)
        R.restore(d)
        self.assertEqual(R.buckets["compaction"]["calls"], 2)
        self.assertEqual(R.buckets["compaction"]["prompt_tokens"], 8000)

    def test_restore_ignores_the_derived_billed_total(self):
        # `total` and `buckets` are both restored additively; reading `billed_total`
        # back as well would count the whole bill a second time.
        L = self._bucketed()
        R = self._led()
        d = L.to_dict()
        R.restore(d)
        R.restore(d)
        self.assertAlmostEqual(R.billed_total(), 2 * 0.0064)

    def test_restore_keeps_bucket_unpriced_sticky(self):
        # `cur[k] += v` over a heterogeneous row evaluates `False + True` to 1 and
        # silently turns a flag into a count.
        R = self._led()
        R.restore({"buckets": {"compaction": {"calls": 1, "unpriced": True}}})
        R.restore({"buckets": {"compaction": {"calls": 1, "unpriced": False}}})
        self.assertIs(R.buckets["compaction"]["unpriced"], True)

    def test_restore_tolerates_a_garbage_bucket_row(self):
        R = self._led()
        R.restore({"buckets": {"compaction": {"calls": 1}, "junk": "not a dict",
                               "": {"calls": 9}}})
        self.assertEqual(sorted(R.buckets), ["compaction"])
        self.assertEqual(R.buckets["compaction"]["calls"], 1)

    def test_a_priced_loop_beside_an_unpriced_bucket_is_marked_partial(self):
        # The bill genuinely excludes an unknown amount, so the figure must say so.
        # `self.unpriced` alone cannot answer this -- it stays main-loop-only.
        L = _agent.CostLedger()
        L.bind_pricing(self.P)
        L.record({"prompt_tokens": 1000, "completion_tokens": 500})
        L._in = L._out = L._cache_in = L._cache_write = None  # rate lost mid-session
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, bucket="compaction")
        self.assertFalse(L.unpriced)
        self.assertTrue(L.any_unpriced())
        self.assertEqual(
            L.summary(),
            "cost: $0.0020 (tokens prompt=1000 completion=500) [partially unpriced]")

    def test_a_pre_101_envelope_restores_to_no_buckets(self):
        R = self._led()
        R.restore({"total": 0.5, "prompt_tokens": 10})
        self.assertEqual(R.buckets, {})
        self.assertAlmostEqual(R.billed_total(), 0.5)

    # -- #117: more than one bucket ------------------------------------------

    def test_summary_lists_two_bucket_names(self):
        L = self._bucketed()
        L.record({"prompt_tokens": 1000, "completion_tokens": 0}, bucket="scout")
        self.assertEqual(
            L.summary(),
            "cost: $0.0074 (tokens prompt=1000 completion=500)"
            " [incl. $0.0054 compaction/scout]")

    def test_summary_collapses_more_than_two_bucket_names(self):
        # The bound. Unbounded, `--planner --review --web-search` renders
        # `compaction/review/scout/spawn/web_search` onto a footer that is also two
        # stop-reason messages -- and grows again with every future bucket.
        L = self._bucketed()
        for name in ("scout", "spawn", "review", "web_search"):
            L.record({"prompt_tokens": 1000, "completion_tokens": 0}, bucket=name)
        self.assertEqual(
            L.summary(),
            "cost: $0.0104 (tokens prompt=1000 completion=500)"
            " [incl. $0.0084 off-loop]")

    def test_a_zero_cost_bucket_does_not_count_toward_the_name_bound(self):
        # The bound counts the names that will actually PRINT. Three buckets of which
        # one is free is a two-name render, not a collapse -- otherwise an unpriced
        # rail silently degrades the other two into "off-loop".
        L = self._bucketed()
        L.record({"prompt_tokens": 1000, "completion_tokens": 0}, bucket="scout")
        L.record(None, bucket="spawn")           # no usage -> no cost
        self.assertIn("[incl. $0.0054 compaction/scout]", L.summary())

    def test_the_bucket_block_renders_a_row_per_rail(self):
        # Pin the whole block: the column width is data-driven (`max(len(k))`), so a
        # longer bucket name shifts every row and should shift in a diff, not in prod.
        L = self._led()
        L.record({"prompt_tokens": 1000, "completion_tokens": 500}, seconds=2.0)
        L.record_turn(2.0)
        L.record({"prompt_tokens": 4000, "completion_tokens": 200}, seconds=1.5,
                 bucket="compaction")
        L.record({"prompt_tokens": 800, "completion_tokens": 120}, seconds=0.4,
                 bucket="web_search")
        out = L.usage_report()
        self.assertIn(
            "  off-loop      1.9s  across 2 API call(s)  [not in the trace above]", out)
        self.assertIn(
            "    compaction  1 call(s)      4,000 in      200 out     $0.0044", out)
        self.assertIn(
            "    web_search  1 call(s)        800 in      120 out     $0.0010", out)

    def test_the_bucket_block_labels_overlapping_rail_windows(self):
        # Under `--parallel` several rails' API windows overlap, so the summed seconds
        # can exceed the wall clock. Say so, in `_tool_lines`' exact words -- one state
        # must not grow two vocabularies inside one report.
        L = self._led()
        L.record_turn(2.0)                       # 2s of wall
        L.record({"prompt_tokens": 100, "completion_tokens": 10}, seconds=5.0,
                 bucket="scout")
        L.record({"prompt_tokens": 100, "completion_tokens": 10}, seconds=5.0,
                 bucket="spawn")
        self.assertIn("[concurrent -- exceeds wall]", L.usage_report())

    def test_the_bucket_block_stays_unlabelled_within_the_wall(self):
        L = self._led()
        L.record_turn(60.0)
        L.record({"prompt_tokens": 100, "completion_tokens": 10}, seconds=5.0,
                 bucket="scout")
        self.assertNotIn("[concurrent", L.usage_report())

    def test_restore_round_trips_several_buckets_additively(self):
        # `restore` is field-by-field per name precisely because a naive `cur[k] += v`
        # evaluates `False + True` to 1 and turns a sticky flag into a count. Exercise
        # it with more than the one name that existed when it was written.
        L = self._bucketed()
        for name in ("scout", "review"):
            L.record({"prompt_tokens": 1000, "completion_tokens": 0}, bucket=name)
        R = self._led()
        R.restore(L.to_dict())
        R.restore(L.to_dict())                   # additive, twice over
        self.assertEqual(sorted(R.buckets), ["compaction", "review", "scout"])
        self.assertEqual(R.buckets["scout"]["calls"], 2)
        self.assertEqual(R.buckets["scout"]["prompt_tokens"], 2000)
        self.assertIs(R.buckets["scout"]["unpriced"], False)
        self.assertAlmostEqual(R.bucket_cost(), L.bucket_cost() * 2)


class TestMirroredSubagentLedgers(unittest.TestCase):
    """#117: a subagent ledger banks each API call into its PARENT's bucket.

    The mirror sits at `record()` -- the one place any usage is read -- rather than at
    the call sites, so the two rails that call the API outside `run_loop`
    (`_review._retry_for_verdict`, `run_web_search`) are covered without either site
    knowing a parent exists. These tests pin that, and pin that the isolation #101 built
    for compaction survives a SECOND writer arriving from a pool worker.
    """

    P = {"input": {"usd": 1.0}, "output": {"usd": 2.0}}
    #: A deliberately DIFFERENT rate, standing in for `--review-model`.
    P10 = {"input": {"usd": 10.0}, "output": {"usd": 20.0}}

    U = {"prompt_tokens": 1000, "completion_tokens": 500}

    def _parent(self, cap=None):
        L = _agent.CostLedger(max_spend=cap)
        L.bind_pricing(self.P)
        return L

    def _child(self, parent, name="scout", pricing=None):
        C = _agent.CostLedger(mirror=(parent, name))
        C.bind_pricing(pricing or self.P)
        return C

    # -- the seam ------------------------------------------------------------

    def test_a_mirrored_child_lands_in_the_parent_bucket(self):
        P = self._parent()
        self._child(P).record(self.U, seconds=1.5)
        row = P.buckets["scout"]
        self.assertEqual((row["calls"], row["prompt_tokens"],
                          row["completion_tokens"], row["seconds"]),
                         (1, 1000, 500, 1.5))
        self.assertAlmostEqual(row["cost"], 0.0020)

    def test_an_unmirrored_child_leaves_the_parent_empty(self):
        # The control. Without this, every assertion above could be passing because
        # something else populates the bucket -- and a bucket that is always written
        # is not evidence that the MIRROR wrote it.
        P = self._parent()
        C = _agent.CostLedger()
        C.bind_pricing(self.P)
        C.record(self.U, seconds=1.5)
        self.assertEqual(P.buckets, {})
        self.assertEqual(P.billed_total(), 0.0)

    def test_a_mirrored_child_never_grows_a_bucket_of_its_own(self):
        # Why `test_summary_is_unchanged_without_a_bucket` still passes: the child
        # writes the PARENT's partition. `run_loop`'s stop-reason messages render
        # `summary()` on this object and must not sprout an `[incl. ...]` clause.
        P = self._parent()
        C = self._child(P)
        C.record(self.U, seconds=1.5)
        self.assertEqual(C.buckets, {})
        self.assertEqual(C.summary(),
                         "cost: $0.0020 (tokens prompt=1000 completion=500)")

    def test_the_child_keeps_its_own_main_loop_accounting(self):
        # `--subagent-max-tokens` reads these, so mirroring must be purely additive.
        P = self._parent()
        C = self._child(P)
        C.record(self.U)
        self.assertEqual((C.prompt_tokens, C.completion_tokens), (1000, 500))
        self.assertEqual(C.api_calls_total, 1)
        self.assertEqual(len(C.api_calls()), 1)

    def test_a_mirrored_record_leaves_the_parent_main_counters_untouched(self):
        # THE isolation property. A mirror that recorded into the parent's MAIN loop
        # would fold a fresh ~0%-cached prefix into the conversation's cache rate --
        # fabricating exactly the cliff #99's trace exists to detect.
        P = self._parent()
        self._child(P).record(self.U, seconds=1.5)
        self.assertEqual((P.prompt_tokens, P.completion_tokens, P.total), (0, 0, 0.0))
        self.assertEqual((P.cache_read_unreported, P.cache_write_unreported),
                         (False, False))
        self.assertIsNone(P.cache_hit_percent())

    def test_a_mirrored_record_appends_no_parent_trace_row(self):
        # `calls_dropped()` is `api_calls_total - head - tail`; bumping the counter
        # without a row would make `/usage` report phantom dropped rows.
        P = self._parent()
        self._child(P).record(self.U)
        self.assertEqual((P.api_calls(), P.api_calls_total, P.calls_dropped()),
                         ([], 0, 0))
        self.assertEqual(P.bucket_calls("scout"), 1)

    # -- pricing (the load-bearing correction to #117's own design) -----------

    def test_the_child_is_priced_at_its_own_rate_not_the_parents(self):
        # `--review-model` is deliberately a different model. Re-costing the child's
        # usage on the parent would bill the reviewer's tokens at the author's rate --
        # a fabricated number wearing a measurement's name.
        P = self._parent()
        self._child(P, "review", pricing=self.P10).record(self.U)
        self.assertAlmostEqual(P.buckets["review"]["cost"], 0.0200)   # 10x, not 0.0020
        self.assertAlmostEqual(P.billed_total(), 0.0200)

    def test_an_unpriced_child_taints_only_the_bucket(self):
        P = self._parent()
        C = _agent.CostLedger(mirror=(P, "scout"))    # no pricing bound
        C.record(self.U)
        self.assertIs(P.buckets["scout"]["unpriced"], True)
        self.assertIs(P.unpriced, False)             # stays main-loop-only
        self.assertTrue(P.any_unpriced())

    def test_a_usage_less_child_call_marks_the_bucket_unreported(self):
        # #98's rule one partition over: absent is UNKNOWN, never a measured zero.
        P = self._parent()
        self._child(P).record(None, seconds=0.5)
        row = P.buckets["scout"]
        self.assertIs(row["unreported"], True)
        self.assertEqual((row["calls"], row["prompt_tokens"]), (1, 0))
        self.assertEqual(_agent.CostLedger.bucket_money(row), "n/a")
        self.assertTrue(P.any_unreported())

    # -- bounds --------------------------------------------------------------

    def test_the_mirror_does_not_reach_a_grandparent(self):
        # `record_bucket` is a leaf on purpose. Propagating would double-count spend
        # the parent's bucket has already banked.
        G = self._parent()
        P = _agent.CostLedger(mirror=(G, "spawn"))
        P.bind_pricing(self.P)
        self._child(P, "scout").record(self.U)
        self.assertEqual(P.buckets["scout"]["calls"], 1)
        self.assertEqual(G.buckets, {})

    def test_a_self_mirror_is_rejected(self):
        # Honest scope: a self-mirror is not reachable from `__init__`'s own call (you
        # cannot name the object being constructed), so re-init is the only way to
        # express it. The guard is cheap and states the invariant -- a ledger that
        # mirrored into itself would bank every call BOTH into its main counters and
        # into its own bucket, double-counting inside one object.
        P = self._parent()
        with self.assertRaises(ValueError):
            P.__init__(mirror=(P, "scout"))

    def test_the_raw_usage_dump_fires_once_per_api_call(self):
        # One API call must produce ONE `usage-raw:` line. Routing the mirror through
        # `record()` instead of `record_bucket()` emits two -- interleaved with other
        # workers' lines under `--parallel`, which is what the single-print contract
        # exists to prevent.
        P = self._parent()
        C = self._child(P)
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_USAGE_RAW": "1"}), \
             mock.patch.object(sys, "stderr", err):
            C.record(self.U)
        self.assertEqual(err.getvalue().count("usage-raw:"), 1)

    # -- the #117 ask --------------------------------------------------------

    def test_the_session_cap_counts_rail_spend(self):
        # THE point of the slice: `--session-max-spend` bounds a rail-heavy run.
        P = self._parent(cap=0.001)
        self.assertFalse(P.over())
        self._child(P).record(self.U)
        self.assertEqual(P.total, 0.0)               # nothing in the main loop at all
        self.assertTrue(P.over())

    def test_bucket_tokens_stay_out_of_the_parent_token_gate(self):
        # `over_tokens()` is #52's ceiling on a WORKER's own conversation, and the
        # parent's is not the worker's. Deliberately unchanged by this slice.
        P = _agent.CostLedger(max_tokens=100)
        P.bind_pricing(self.P)
        self._child(P).record(self.U)
        self.assertFalse(P.over_tokens())

    def test_the_child_token_cap_is_unaffected_by_mirroring(self):
        P = self._parent()
        C = _agent.CostLedger(max_tokens=100, mirror=(P, "scout"))
        C.bind_pricing(self.P)
        C.record(self.U)
        self.assertTrue(C.over_tokens())             # the worker's own ceiling still bites

    # -- concurrency ---------------------------------------------------------

    def test_record_bucket_mutates_under_the_lock(self):
        # Structural, exactly like `test_record_tool_mutates_under_the_lock`: CPython's
        # GIL does not actually lose these increments, so a hammer-and-count test would
        # pass with the lock DELETED and be a vacuous guard. Pin instead that the
        # mutation happens while the lock is held.
        L = _agent.CostLedger()
        held = []
        real = L._buckets_lock

        class Watched:
            def __enter__(self):
                real.acquire()
                held.append("in")
                return self

            def __exit__(self, *exc):
                held.append("out")
                real.release()
                return False

        L._buckets_lock = Watched()
        L.record_bucket("scout", cost=0.1)
        L.record_bucket("scout", cost=0.1)
        self.assertEqual(held, ["in", "out", "in", "out"])

    def test_mirrored_children_are_correct_under_real_threads(self):
        # Not a race detector -- a smoke test for how `--parallel` actually calls this:
        # up to `_MAX_PARALLEL` scouts mirroring into one parent row at once.
        P = self._parent()
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, old)

        def hammer():
            C = self._child(P)
            for _ in range(200):
                C.record(self.U)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(P.buckets["scout"]["calls"], 1600)
        self.assertEqual(P.buckets["scout"]["prompt_tokens"], 1600 * 1000)

    # -- the constructor -----------------------------------------------------

    def test_subagent_ledger_binds_the_childs_own_pricing(self):
        models = [{"id": "reviewer", "model_spec": {"pricing": self.P10}},
                  {"id": "author", "model_spec": {"pricing": self.P}}]
        P = self._parent()
        C = _agent.subagent_ledger(models, "reviewer", max_tokens=50,
                                   mirror=(P, "review"))
        self.assertEqual(C.max_tokens, 50)
        self.assertIsNone(C.max_spend)               # the parent owns the money gate
        C.record(self.U)
        self.assertAlmostEqual(P.buckets["review"]["cost"], 0.0200)

    def test_subagent_ledger_without_a_parent_is_standalone(self):
        # `venice review` (the standalone CLI) has no parent ledger and must keep
        # working -- `mirror=None` is a supported state, not an oversight.
        C = _agent.subagent_ledger([], "m")
        C.record(self.U)
        self.assertEqual(C.prompt_tokens, 1000)
        self.assertEqual(C.buckets, {})

    def test_an_unknown_model_still_meters_tokens(self):
        C = _agent.subagent_ledger([], "not-in-catalog")
        C.record(self.U)
        self.assertTrue(C.unpriced)
        self.assertEqual(C.prompt_tokens + C.completion_tokens, 1500)


class TestCallTrace(unittest.TestCase):
    """#99: the per-API-call trace and the context-event log.

    Pure-function assertions throughout, like `TestCostLedger` above: the ledger never
    reads a clock, so `seconds` is always a value the test hands in.
    """

    @staticmethod
    def _usage(pt=100, ct=5, cached=None, written=None):
        u = {"prompt_tokens": pt, "completion_tokens": ct}
        details = {}
        if cached is not None:
            details["cached_tokens"] = cached
        if written is not None:
            details["cache_creation_input_tokens"] = written
        if details:
            u["prompt_tokens_details"] = details
        return u

    def _call_block(self, report):
        """The `calls` block only: its header and everything under it."""
        lines = report.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("  calls "))
        return lines[i:]

    # --- rows ------------------------------------------------------------- #

    def test_one_row_per_record_numbered_from_one(self):
        # Kills an append hung off `record_turn`/`record_tool` (which count DIFFERENT
        # things -- see the `turns` docstring) and a no-op that never appends.
        L = _agent.CostLedger()
        for i in range(3):
            L.record(self._usage(pt=100 + i))
        L.record_turn(5.0)
        L.record_tool("shell", 2.0)
        self.assertEqual([r["n"] for r in L.api_calls()], [1, 2, 3])
        self.assertEqual([r["prompt_tokens"] for r in L.api_calls()], [100, 101, 102])
        self.assertEqual(L.api_calls_total, 3)

    def test_an_absent_cache_field_is_null_in_the_row_not_zero(self):
        # The #98 three-state, one level down. `record` collapses None->0 for the
        # AGGREGATE two lines away; a row that copied that collapse would report a
        # confident 0% miss for a provider that said nothing at all.
        L = _agent.CostLedger()
        L.record(self._usage())
        row = L.api_calls()[0]
        self.assertIsNone(row["cache_read_tokens"])
        self.assertIsNone(row["cache_write_tokens"])

    def test_a_reported_zero_stays_zero_in_the_row(self):
        # The other half of the contract: don't replace one lie with another.
        L = _agent.CostLedger()
        L.record(self._usage(cached=0, written=0))
        row = L.api_calls()[0]
        self.assertEqual(row["cache_read_tokens"], 0)
        self.assertEqual(row["cache_write_tokens"], 0)

    def test_a_usage_less_call_still_gets_a_row(self):
        # THE test for the early-return hole. `record` bails on each of these shapes
        # before any tally; a row appended at the bottom would be skipped for exactly
        # the call whose `seconds` matter most -- one that blocked, then returned
        # nothing. Every field is null, and the window it cost is still on the record.
        L = _agent.CostLedger()
        L.record(None, seconds=40.0)
        L.record("garbage", seconds=1.0)
        L.record({"prompt_tokens": "nope"}, seconds=2.0)
        rows = L.api_calls()
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["seconds"] for r in rows], [40.0, 1.0, 2.0])
        for r in rows[:2]:
            self.assertIsNone(r["prompt_tokens"])
            self.assertIsNone(r["completion_tokens"])
        # A dict with a garbage value is a real usage block: it parses to 0, not null.
        self.assertEqual(rows[2]["prompt_tokens"], 0)

    def test_row_buckets_are_clamped_like_the_aggregate(self):
        # A row has to reconcile with the total it fed, so it reports the CLAMPED
        # bucket -- but an absent field stays null through the clamp rather than
        # becoming a confident 0.
        L = _agent.CostLedger()
        L.record(self._usage(pt=100, cached=900))
        row = L.api_calls()[0]
        self.assertEqual(row["cache_read_tokens"], 100)
        self.assertIsNone(row["cache_write_tokens"])
        self.assertEqual(L.cache_read_tokens, 100)

    def test_an_unstamped_window_is_null_not_zero(self):
        # `n/a`, never `0.0s`: an unbracketed call site is UNKNOWN, and a fabricated
        # 0.0s would read as an instant response -- #98's lie in the time dimension.
        L = _agent.CostLedger()
        L.record(self._usage())
        self.assertIsNone(L.api_calls()[0]["seconds"])
        self.assertEqual(self._call_block(L.usage_report())[0],
                         "  calls         0.0s  across 1 API call(s)  [1 untimed]")

    def test_the_ledger_never_reads_a_clock(self):
        # The house contract (`record_turn`/`record_tool` docstrings). If a reflex
        # `time.monotonic()` ever appears in `record`, this raises instead of passing.
        L = _agent.CostLedger()
        with mock.patch("venice.commands._agent.time.monotonic",
                        side_effect=AssertionError("the ledger read a clock")):
            L.record(self._usage(), seconds=3.0)
        self.assertEqual(L.api_calls()[0]["seconds"], 3.0)

    def test_garbage_seconds_degrade_to_zero_but_none_stays_none(self):
        L = _agent.CostLedger()
        L.record(self._usage(), seconds="nope")
        L.record(self._usage(), seconds=None)
        self.assertEqual(L.api_calls()[0]["seconds"], 0.0)
        self.assertIsNone(L.api_calls()[1]["seconds"])

    # --- the cap ---------------------------------------------------------- #

    def test_the_cap_keeps_the_head_and_the_tail(self):
        # Kills a plain ring buffer (which would drop the cold-start evidence that is
        # the whole reason #99 exists) AND a head-only cap (which would go blind to
        # what the run is doing now). The seam is self-describing: `n` jumps.
        L = _agent.CostLedger()
        for _ in range(300):
            L.record(self._usage())
        rows = L.api_calls()
        self.assertEqual(len(rows), 250)
        self.assertEqual([r["n"] for r in rows[:3]], [1, 2, 3])
        self.assertEqual(rows[49]["n"], 50)
        self.assertEqual(rows[50]["n"], 101)  # the gap IS the drop marker
        self.assertEqual(rows[-1]["n"], 300)
        self.assertEqual(L.api_calls_total, 300)
        self.assertEqual(L.calls_dropped(), 50)

    def test_a_dropped_row_is_never_silent(self):
        L = _agent.CostLedger()
        for _ in range(300):
            L.record(self._usage(), seconds=1.0)
        self.assertEqual(self._call_block(L.usage_report())[0],
                         "  calls       4m 10s  across 300 API call(s)"
                         "  [50 row(s) dropped]")

    def test_to_dict_rows_are_copies(self):
        # `tools` already copies per row; a list of dicts is the easy miss, and
        # `code --json` hands this envelope to json.dump while the ledger is live.
        L = _agent.CostLedger()
        L.record(self._usage())
        L.record_compaction({"messages_before": 4, "messages_after": 2})
        d = L.to_dict()
        d["api_calls"][0]["prompt_tokens"] = 999999
        d["context_events"][0]["messages_before"] = 999999
        self.assertEqual(L.api_calls()[0]["prompt_tokens"], 100)
        self.assertEqual(L.context_events[0]["messages_before"], 4)

    # --- restore ---------------------------------------------------------- #

    def test_restore_is_seed_once_not_additive(self):
        # The third `restore` category. Additive (the reflex every tally above invites)
        # would CONCATENATE the rows into duplicates on a second call.
        L = _agent.CostLedger()
        for _ in range(4):
            L.record(self._usage())
        L.record_compaction({"messages_before": 8, "messages_after": 3})
        snap = L.to_dict()
        R = _agent.CostLedger()
        R.restore(snap)
        R.restore(snap)
        self.assertEqual(len(R.api_calls()), 4)
        self.assertEqual(len(R.context_events), 1)
        self.assertEqual(R.api_calls_total, 8)  # the TALLY is additive, the list is not

    def test_a_resumed_ledger_continues_the_numbering(self):
        # Restarting `n` at 1 would collide with the rows just restored and make the
        # ordinal useless as a key.
        L = _agent.CostLedger()
        for _ in range(4):
            L.record(self._usage())
        R = _agent.CostLedger()
        R.restore(L.to_dict())
        R.record(self._usage())
        self.assertEqual(R.api_calls()[-1]["n"], 5)

    def test_rows_with_a_truncated_total_still_number_uniquely(self):
        # A hand-edited envelope must not be able to make `n` collide or drive
        # `calls_dropped` negative.
        R = _agent.CostLedger()
        R.restore({"api_calls": [{"n": 1}, {"n": 2}], "api_calls_total": 0})
        self.assertEqual(R.api_calls_total, 2)
        self.assertEqual(R.calls_dropped(), 0)
        R.record(self._usage())
        self.assertEqual(R.api_calls()[-1]["n"], 3)

    def test_a_pre_99_envelope_restores_with_no_trace(self):
        # Mirrors `test_a_pre_82_envelope_restores_with_no_tools`: no key, no crash,
        # and crucially no fabricated empty section in the report.
        R = _agent.CostLedger()
        R.restore({"prompt_tokens": 100, "completion_tokens": 5})
        self.assertEqual(R.api_calls(), [])
        self.assertEqual(R.context_events, [])
        self.assertEqual(R.api_calls_total, 0)
        self.assertNotIn("calls ", R.usage_report())

    def test_restore_tolerates_junk_rows(self):
        R = _agent.CostLedger()
        R.restore({"api_calls": ["nope", {"n": 1}, 7], "context_events": [None, {}]})
        self.assertEqual(len(R.api_calls()), 1)
        self.assertEqual(len(R.context_events), 1)

    # --- render ----------------------------------------------------------- #

    def test_the_trace_block_pins_the_whole_line(self):
        # Whole-line assertEqual: a suffix appended to a row is invisible to assertIn.
        L = _agent.CostLedger()
        L.record(self._usage(pt=11204, ct=312, cached=0), seconds=4.2)
        L.record(self._usage(pt=11530, ct=88, cached=11069), seconds=1.1)
        self.assertEqual(self._call_block(L.usage_report()), [
            "  calls         5.3s  across 2 API call(s)",
            "    #1       11,204 in    0% cached      312 out      4.2s",
            "    #2       11,530 in   96% cached       88 out      1.1s",
        ])

    def test_the_elided_span_carries_its_own_totals(self):
        # The `_tool_lines` rule: a truncated block whose parts do not add up to its
        # header is the #98 lie in a new costume.
        L = _agent.CostLedger()
        for _ in range(12):
            L.record(self._usage(pt=1000, ct=10, cached=500), seconds=1.0)
        block = self._call_block(L.usage_report())
        self.assertEqual(block[0], "  calls        12.0s  across 12 API call(s)")
        self.assertEqual(len(block), 1 + 3 + 1 + 5)
        self.assertEqual(block[4],
                         "    (+4 elided)      4,000 in   50% cached"
                         "       40 out      4.0s")
        self.assertEqual(block[5],
                         "    #8               1,000 in   50% cached"
                         "       10 out      1.0s")

    def test_a_null_row_renders_n_a_in_every_column(self):
        L = _agent.CostLedger()
        L.record(None)
        self.assertEqual(self._call_block(L.usage_report())[1],
                         "    #1          n/a in   n/a cached      n/a out       n/a")

    # --- context events --------------------------------------------------- #

    def test_a_compaction_event_anchors_to_the_last_call(self):
        L = _agent.CostLedger()
        L.record(self._usage())
        L.record(self._usage())
        L.record_compaction({"trigger": "auto", "messages_before": 48,
                             "messages_after": 13, "est_tokens_before": 91200,
                             "est_tokens_after": 8210,
                             "observed_tokens_before": 88110})
        ev = L.context_events[0]
        self.assertEqual(ev["kind"], "compaction")
        self.assertEqual(ev["after_n"], 2)

    def test_the_marker_renders_after_its_anchor(self):
        L = _agent.CostLedger()
        L.record(self._usage(pt=1000, ct=10, cached=0), seconds=1.0)
        L.record_compaction({"trigger": "auto", "messages_before": 48,
                             "messages_after": 13, "est_tokens_before": 91200,
                             "est_tokens_after": 8210,
                             "observed_tokens_before": 88110})
        L.record(self._usage(pt=800, ct=10, cached=0), seconds=1.0)
        self.assertEqual(self._call_block(L.usage_report()), [
            "  calls         2.0s  across 2 API call(s)",
            "    #1        1,000 in    0% cached       10 out      1.0s",
            "    -- compacted (auto) after #1: 48 -> 13 msgs, ~91,200 -> ~8,210 tok est",
            "       (88,110 tok measured before, lower bound)",
            "    #2          800 in    0% cached       10 out      1.0s",
        ])

    def test_an_unmeasured_event_omits_the_lower_bound_line(self):
        # `/compact` without a budget: the estimate must not wear a measurement's name.
        L = _agent.CostLedger()
        L.record(self._usage(pt=1000, ct=10, cached=0), seconds=1.0)
        L.record_compaction({"trigger": "manual", "messages_before": 48,
                             "messages_after": 13, "est_tokens_before": 91200,
                             "est_tokens_after": 8210,
                             "observed_tokens_before": None})
        self.assertEqual(self._call_block(L.usage_report())[2],
                         "    -- compacted (manual) after #1: 48 -> 13 msgs, "
                         "~91,200 -> ~8,210 tok est")
        self.assertNotIn("measured", L.usage_report())

    def test_a_marker_inside_the_elided_span_still_renders(self):
        # Markers are rare and are the highest-signal line in the block; folding one
        # away silently would violate the no-silent-caps rule the header follows.
        L = _agent.CostLedger()
        for _ in range(12):
            L.record(self._usage(pt=1000, ct=10, cached=0), seconds=1.0)
            if L.api_calls_total == 6:
                L.record_compaction({"trigger": "auto", "messages_before": 40,
                                     "messages_after": 9, "est_tokens_before": 8000,
                                     "est_tokens_after": 900})
        block = self._call_block(L.usage_report())
        self.assertEqual(block[5], "    -- compacted (auto) after #6: 40 -> 9 msgs, "
                                   "~8,000 -> ~900 tok est")
        self.assertTrue(block[4].startswith("    (+4 elided)"))

    def test_an_event_before_any_call_still_renders(self):
        # `after_n` 0 has no row to hang off; without the tail sweep it would vanish.
        L = _agent.CostLedger()
        L.record_compaction({"trigger": "manual", "messages_before": 6,
                             "messages_after": 2, "est_tokens_before": 500,
                             "est_tokens_after": 90})
        L.record(self._usage(pt=100, ct=5, cached=0), seconds=1.0)
        self.assertIn("    -- compacted (manual) after #0: 6 -> 2 msgs, "
                      "~500 -> ~90 tok est",
                      self._call_block(L.usage_report()))

    def test_the_event_row_carries_no_cost(self):
        # #101 owns the summary call's cost. Tokens-saved beside a cost would invite
        # "the compaction paid for itself", which this data cannot support.
        L = _agent.CostLedger()
        L.record_compaction({"messages_before": 4, "messages_after": 2})
        self.assertNotIn("cost", L.context_events[0])


class TestRunLoopCacheGuard(unittest.TestCase):
    """#105: cache collapse is detected per API call inside one tool loop."""

    @staticmethod
    def _usage(*, cached, prompt=3_000):
        return {
            "prompt_tokens": prompt,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": cached},
        }

    @staticmethod
    def _tool():
        return _agent.Tool(
            "t", "t", {"type": "object", "properties": {}},
            lambda a, *, confirm=False: {"status": "ok"},
        )

    @staticmethod
    def _ledger(policy):
        ledger = _agent.CostLedger(cache_guard=policy)
        ledger.bind_pricing({
            "input": {"usd": 3.75}, "cache_input": {"usd": 0.375},
        })
        return ledger

    def test_warn_emits_once_and_continues_the_same_tool_loop(self):
        seq = [
            FakeToolCompletion(
                tool_calls=[_FnCall("c1", "t", "{}")],
                usage=self._usage(cached=500),
            ),
            FakeToolCompletion(
                tool_calls=[_FnCall("c2", "t", "{}")],
                usage=self._usage(cached=0),
            ),
            FakeToolCompletion("done", usage=self._usage(cached=0)),
        ]
        fake, calls = _fake_oai(seq)
        err = io.StringIO()
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "kimi-k3", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=self._ledger("warn"),
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(err.getvalue().count("returned 0 cached tokens"), 1)
        self.assertIn("API call 2", err.getvalue())

    def test_stop_finishes_returned_tools_then_uses_no_tools_final(self):
        seq = [
            FakeToolCompletion(
                tool_calls=[_FnCall("c1", "t", "{}")],
                usage=self._usage(cached=500),
            ),
            FakeToolCompletion(
                tool_calls=[_FnCall("c2", "t", "{}")],
                usage=self._usage(cached=0),
            ),
            FakeToolCompletion("wrapped up", usage=self._usage(cached=0)),
        ]
        fake, calls = _fake_oai(seq)
        err = io.StringIO()
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "kimi-k3", messages, {}, [self._tool()],
                max_tool_calls=0, yes=True, json_out=False,
                ledger=self._ledger("stop"),
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1]["tool_choice"], "none")
        self.assertTrue(any(
            m.get("role") == "tool" and m.get("tool_call_id") == "c2"
            for m in calls[-1]["messages"]
        ))
        self.assertEqual(err.getvalue().count("returned 0 cached tokens"), 1)
        self.assertIn("requesting a final answer", err.getvalue())

    def test_stop_buys_no_extra_call_when_triggering_response_is_final(self):
        seq = [
            FakeToolCompletion(
                tool_calls=[_FnCall("c1", "t", "{}")],
                usage=self._usage(cached=500),
            ),
            FakeToolCompletion("done", usage=self._usage(cached=0)),
        ]
        fake, calls = _fake_oai(seq)
        err = io.StringIO()
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "kimi-k3", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=self._ledger("stop"),
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("response is already final", err.getvalue())


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
        err = io.StringIO()
        final = []
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=ledger,
                final_emitter=lambda response: final.append(response) or 0,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)                      # turn 1 + forced final
        self.assertEqual(calls[-1]["tool_choice"], "none")   # forced, no tools
        self.assertTrue(ledger.over())
        self.assertEqual(final, [seq[-1]])
        self.assertIn("reached --session-max-spend", err.getvalue())
        self.assertNotIn("chat: reached", err.getvalue())
        self.assertNotIn("reached --max-spend", err.getvalue())

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

    def test_invalid_catalog_pricing_aborts_without_a_model_call(self):
        fake, calls = _fake_oai([])
        ledger = _agent.CostLedger(max_spend=1.0)
        ledger.bind_pricing({"input": {"usd": float("nan")}})
        err = io.StringIO()
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=ledger,
            )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertIn("invalid non-finite pricing", err.getvalue())

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

    def test_token_gate_forces_final_unpriced(self):
        # #52: turn 1 calls a tool AND its usage crosses the token cap; the next
        # iteration forces a final answer. Unpriced ledger -> proves the token gate
        # is independent of the (USD) spend gate.
        usage = {"prompt_tokens": 900, "completion_tokens": 200}   # 1100 >= 1000
        seq = [
            FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")], usage=usage),
            FakeToolCompletion("wrapped up"),  # the forced-final turn
        ]
        fake, calls = _fake_oai(seq)
        ledger = _agent.CostLedger(max_tokens=1000)  # no pricing bound
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
        self.assertTrue(ledger.over_tokens())
        self.assertFalse(ledger.over())                      # USD gate never fired

    def test_stop_reason_messages_carry_no_cache_claim(self):
        # #100 kept `summary()`'s cache clause opt-in for exactly this: both of
        # `run_loop`'s gates render it into a "why did I stop" line, on subagent and
        # review ledgers that are unpriced AND near-always cache-unreported. Making
        # the clause unconditional would tag every token-capped worker with a cache
        # warning nobody asked for. Both gates asserted -- one is not the other.
        usage = {"prompt_tokens": 900, "completion_tokens": 200}
        for kw, expect in ((dict(max_tokens=1000), "token cap"),
                           (dict(max_spend=0.0001), "--session-max-spend")):
            with self.subTest(gate=expect):
                seq = [
                    FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")],
                                       usage=usage),
                    FakeToolCompletion("wrapped up"),
                ]
                fake, _calls = _fake_oai(seq)
                ledger = _agent.CostLedger(**kw)
                if "max_spend" in kw:
                    ledger.bind_pricing({"input": {"usd": 3.0}, "output": {"usd": 15.0}})
                err = io.StringIO()
                with mock.patch.object(sys, "stdout", io.StringIO()), \
                     mock.patch.object(sys, "stderr", err):
                    _agent.run_loop(
                        fake, "m", [{"role": "user", "content": "go"}], {},
                        [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                        ledger=ledger,
                    )
                self.assertIn(expect, err.getvalue())        # the gate really fired
                self.assertNotIn("cache", err.getvalue())

    def test_no_token_cap_means_no_token_gate(self):
        # A ledger with max_tokens=None (the parent chat/REPL case) never token-gates,
        # even under huge usage -> the new gate is a no-op for chat.
        usage = {"prompt_tokens": 10**9, "completion_tokens": 10**9}
        seq = [FakeToolCompletion("done", usage=usage)]
        fake, calls = _fake_oai(seq)
        ledger = _agent.CostLedger(max_spend=0.5)  # USD cap set, token cap NOT
        ledger.bind_pricing({"input": {"usd": 0.0}, "output": {"usd": 0.0}})  # $0
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {},
                [self._tool()], max_tool_calls=0, yes=True, json_out=False,
                ledger=ledger,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)  # no forced final despite billions of tokens


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


class TestNativeVisionDispatch(unittest.TestCase):
    @staticmethod
    def _models(capability):
        spec = {"capabilities": {"supportsVision": capability}}
        return [{"id": "frontend", "model_spec": spec}]

    @staticmethod
    def _tool(config=None):
        return next(t for t in _agent.builtin_tools(
            object(), only={"venice_vision"}, config=config or {},
            media_path_authority=object(),
        ) if t.name == "venice_vision")

    def _call(self, arguments, capability, *, config=None):
        tool = self._tool(config)
        runtime = _agent._ToolRuntime("frontend", self._models(capability))
        return _agent._run_one_call(
            _FnCall("vision-1", "venice_vision", json.dumps(arguments)),
            {"venice_vision": tool}, {"auto": True}, runtime=runtime,
        )

    def test_model_facing_schema_exposes_only_the_three_modes(self):
        props = self._tool().parameters["properties"]
        self.assertEqual(props["mode"]["enum"], ["auto", "native", "delegate"])
        self.assertNotIn("runtime", props)

    def test_auto_known_capable_attaches_image_without_delegate(self):
        with mock.patch.object(
            _agent._mcp, "prepare_vision_input",
            return_value={"status": "ok", "image_url": "data:image/png;base64,eA=="},
        ) as prepare, mock.patch.object(_agent._mcp, "vision_tool") as delegate:
            outcome = self._call(
                {"image_url": "https://example.test/shot.png", "prompt": "Inspect"},
                True,
            )
        self.assertIsInstance(outcome, _agent._ToolOutcome)
        self.assertEqual(outcome.result["mode"], "native")
        self.assertEqual(outcome.result["model"], "frontend")
        self.assertEqual(outcome.followups, ({
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,eA==",
                }},
            ],
        },))
        prepare.assert_called_once()
        delegate.assert_not_called()

    def test_auto_false_or_unknown_capability_delegates(self):
        for capability in (False, None):
            with self.subTest(capability=capability), mock.patch.object(
                _agent._mcp, "vision_tool",
                return_value={"status": "ok", "content": "delegate saw it", "model": "vl"},
            ) as delegate, mock.patch.object(
                _agent._mcp, "prepare_vision_input"
            ) as prepare:
                result = self._call(
                    {"image_url": "https://example.test/shot.png", "model": "vl"},
                    capability,
                )
            self.assertEqual(result["mode"], "delegate")
            delegate.assert_called_once()
            prepare.assert_not_called()

    def test_explicit_native_fails_closed_on_false_or_unknown_capability(self):
        for capability in (False, None):
            with self.subTest(capability=capability), mock.patch.object(
                _agent._mcp, "vision_tool"
            ) as delegate, mock.patch.object(
                _agent._mcp, "prepare_vision_input"
            ) as prepare:
                result = self._call(
                    {"image_url": "https://example.test/shot.png", "mode": "native"},
                    capability,
                )
            self.assertEqual(result["status"], "error")
            self.assertIn("supportsVision", result["message"])
            delegate.assert_not_called()
            prepare.assert_not_called()

    def test_explicit_delegate_wins_even_on_capable_frontend(self):
        with mock.patch.object(
            _agent._mcp, "vision_tool",
            return_value={"status": "ok", "content": "delegate", "model": "vl"},
        ) as delegate:
            result = self._call(
                {"image_url": "https://example.test/shot.png", "mode": "delegate"},
                True,
            )
        self.assertEqual(result["mode"], "delegate")
        delegate.assert_called_once()

    def test_config_mode_is_lower_precedence_than_tool_argument(self):
        config = {"defaults": {"vision": {"mode": "delegate", "model": "vl"}}}
        with mock.patch.object(
            _agent._mcp, "prepare_vision_input",
            return_value={"status": "ok", "image_url": "https://example.test/shot.png"},
        ), mock.patch.object(_agent._mcp, "vision_tool") as delegate:
            outcome = self._call(
                {"image_url": "https://example.test/shot.png", "mode": "native"},
                True, config=config,
            )
        self.assertIsInstance(outcome, _agent._ToolOutcome)
        delegate.assert_not_called()

    def test_same_tool_uses_each_run_loops_current_frontend(self):
        call = _FnCall(
            "vision-1", "venice_vision",
            json.dumps({"image_url": "https://example.test/shot.png"}),
        )
        with mock.patch.object(
            _agent._mcp, "vision_tool",
            return_value={"status": "ok", "content": "delegate", "model": "vl"},
        ), mock.patch.object(
            _agent._mcp, "prepare_vision_input",
            return_value={"status": "ok", "image_url": "https://example.test/shot.png"},
        ):
            tool = self._tool()
            delegated = _agent._run_one_call(
                call, {"venice_vision": tool}, {"auto": True},
                runtime=_agent._ToolRuntime("text-only", [{
                    "id": "text-only", "model_spec": {
                        "capabilities": {"supportsVision": False},
                    },
                }]),
            )
            native = _agent._run_one_call(
                call, {"venice_vision": tool}, {"auto": True},
                runtime=_agent._ToolRuntime("vision-front", [{
                    "id": "vision-front", "model_spec": {
                        "capabilities": {"supportsVision": True},
                    },
                }]),
            )
        self.assertEqual(delegated["mode"], "delegate")
        self.assertIsInstance(native, _agent._ToolOutcome)
        self.assertEqual(native.result["model"], "vision-front")

    def test_run_loop_commits_all_tool_results_before_native_followup(self):
        def vision(arguments, *, confirm=False, runtime=None):
            self.assertEqual(runtime.model, "frontend")
            return _agent._ToolOutcome(
                {"status": "ok", "mode": "native"},
                ({"role": "user", "content": [{"type": "text", "text": "image"}]},),
            )

        tools = [
            _free_tool(),
            _agent.Tool(
                "venice_vision", "vision", {"type": "object", "properties": {}},
                vision, contextual=True,
            ),
        ]
        seq = [FakeToolCompletion(tool_calls=[
            _FnCall("c1", "venice_vision", "{}"),
            _FnCall("c2", "t", "{}"),
        ]), FakeToolCompletion("done")]
        fake, calls = _fake_oai(seq)
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(
                fake, "frontend", messages, {}, tools,
                max_tool_calls=0, yes=True, json_out=False,
                models=self._models(True),
            )
        roles = [m["role"] for m in calls[1]["messages"][-5:-1]]
        self.assertEqual(roles, ["assistant", "tool", "tool", "user"])
        self.assertEqual(calls[1]["messages"][-4]["tool_call_id"], "c1")
        self.assertEqual(calls[1]["messages"][-3]["tool_call_id"], "c2")


class TestAsyncJobSchemas(unittest.TestCase):
    """#62: background param on media schemas + the two async job-tool schemas."""

    def test_background_in_media_schemas(self):
        for schema in (_agent._SFX_SCHEMA, _agent._MUSIC_SCHEMA, _agent._VIDEO_SCHEMA):
            props = schema["properties"]
            self.assertIn("background", props)
            self.assertEqual(props["background"]["type"], "boolean")

    def test_video_schema_exposes_bounded_reference_inputs(self):
        props = _agent._VIDEO_SCHEMA["properties"]
        expected = {
            "reference_image_urls": _agent._video.REF_IMAGE_MAX,
            "reference_video_urls": _agent._video.REF_VIDEO_MAX,
            "reference_audio_urls": _agent._video.REF_AUDIO_MAX,
            "scene_image_urls": _agent._video.SCENE_MAX,
        }
        for name, maximum in expected.items():
            self.assertEqual(props[name]["type"], "array")
            self.assertEqual(props[name]["maxItems"], maximum)
        for name in ("video_url", "audio_url", "reference_video_duration"):
            self.assertIn(name, props)

    def test_image_schemas_expose_native_controls(self):
        image_props = _agent._IMAGE_SCHEMA["properties"]
        for name in (
            "style_references", "embed_exif_metadata", "lora_strength", "quality",
            "enable_web_search", "disable_prompt_optimization_thinking",
            "enhance_prompt", "aspect_ratio", "resolution",
        ):
            self.assertIn(name, image_props)
        refs = image_props["style_references"]
        self.assertEqual(refs["type"], "array")
        self.assertEqual(
            set(refs["items"]["properties"]), {"image", "strength"}
        )
        self.assertEqual(refs["items"]["properties"]["strength"]["minimum"], 0.1)
        self.assertEqual(refs["items"]["properties"]["strength"]["maximum"], 1.0)
        self.assertEqual(image_props["lora_strength"]["minimum"], 0)
        self.assertEqual(image_props["lora_strength"]["maximum"], 100)
        self.assertEqual(image_props["quality"]["enum"], ["low", "medium", "high"])
        edit_props = _agent._IMAGE_EDIT_SCHEMA["properties"]
        for name in (
            "quality", "disable_prompt_optimization_thinking", "enhance_prompt"
        ):
            self.assertIn(name, edit_props)

    def test_job_schemas_require_handle_fields_and_hide_controls(self):
        for schema in (_agent._JOB_STATUS_SCHEMA, _agent._JOB_RESULT_SCHEMA):
            self.assertEqual(schema.get("required"), ["queue_id", "type", "model"])
            props = schema["properties"]
            for banned in ("confirm", "max_spend", "output_dir", "download_url"):
                self.assertNotIn(banned, props)
        # only job_result exposes max_wait (block-poll seconds)
        self.assertIn("max_wait", _agent._JOB_RESULT_SCHEMA["properties"])
        self.assertNotIn("max_wait", _agent._JOB_STATUS_SCHEMA["properties"])

    def test_job_tools_are_free(self):
        by = {t.name: t for t in _agent.builtin_tools(
            object(), only={"venice_job_status", "venice_job_result"})}
        self.assertFalse(by["venice_job_status"].paid)
        self.assertFalse(by["venice_job_result"].paid)

    def test_fabricated_download_url_is_stripped_before_dispatch(self):
        self.assertNotIn(
            "download_url",
            _agent._clean({"queue_id": "q", "download_url": "file:///tmp/x"}),
        )


class TestMediaPathAuthorityWiring(unittest.TestCase):

    def test_free_and_paid_media_tools_receive_only_host_authority(self):
        authority = object()
        with mock.patch.object(
            _agent._mcp, "vision_tool", return_value={"status": "ok"}
        ) as vision, mock.patch.object(
            _agent._mcp, "upscale_tool", return_value={"status": "ok"}
        ) as upscale, mock.patch.object(
            _agent._mcp, "image_tool", return_value={"status": "ok"}
        ) as image:
            tools = {tool.name: tool for tool in _agent.builtin_tools(
                object(), only={"venice_vision", "venice_upscale", "venice_image"},
                media_path_authority=authority,
            )}
            tools["venice_vision"].invoke({
                "input_path": "frame.png", "path_authority": "model-controlled"
            })
            tools["venice_upscale"].invoke(
                {"input_path": "frame.png", "path_authority": "model-controlled"},
                confirm=True,
            )
            tools["venice_image"].invoke(
                {"prompt": "p", "path_authority": "model-controlled"},
                confirm=True,
            )
        self.assertIs(vision.call_args.kwargs["path_authority"], authority)
        self.assertIs(upscale.call_args.kwargs["path_authority"], authority)
        self.assertIs(image.call_args.kwargs["path_authority"], authority)

    def test_upscale_schema_matches_current_contract(self):
        tool = next(t for t in _agent.builtin_tools(
            object(), only={"venice_upscale"}
        ) if t.name == "venice_upscale")
        props = tool.parameters["properties"]
        self.assertEqual(set(props), {"input_path", "scale", "creativity"})
        self.assertEqual(props["scale"]["enum"], [2.0, 4.0])
        self.assertEqual(props["creativity"]["minimum"], 0.0)
        self.assertEqual(props["creativity"]["maximum"], 0.02)

    def test_replayed_retired_upscale_argument_fails_before_impl(self):
        with mock.patch.object(_agent._mcp, "upscale_tool") as impl:
            tool = next(t for t in _agent.builtin_tools(
                object(), only={"venice_upscale"}
            ) if t.name == "venice_upscale")
            result = tool.invoke({"input_path": "frame.png", "enhance": True})
        self.assertEqual(result["status"], "error")
        self.assertIn("retired argument", result["message"])
        impl.assert_not_called()

    def test_retired_upscale_config_blocks_only_upscale(self):
        doc = {"defaults": {"upscale": {"replication": 0.3}}}
        with mock.patch.object(_agent._mcp, "upscale_tool") as upscale_impl, \
             mock.patch.object(
                 _agent._mcp, "bg_remove_tool", return_value={"status": "ok"}
             ) as bg_impl:
            tools = {t.name: t for t in _agent.builtin_tools(
                object(), only={"venice_upscale", "venice_bg_remove"}, config=doc
            )}
            blocked = tools["venice_upscale"].invoke({"input_path": "frame.png"})
            allowed = tools["venice_bg_remove"].invoke({"image_url": "https://x/y"})
        self.assertEqual(blocked["status"], "error")
        self.assertIn("defaults.upscale.replication", blocked["message"])
        self.assertEqual(allowed["status"], "ok")
        upscale_impl.assert_not_called()
        bg_impl.assert_called_once()


class TestReindexBuiltin(unittest.TestCase):
    """#44: reindex is a paid, no-arg builtin advertised by chat's default set."""

    def test_in_default_set_and_paid(self):
        by = {t.name: t for t in _agent.builtin_tools(object())}
        self.assertIn("reindex", by)                 # advertised by default
        self.assertTrue(by["reindex"].paid)          # routes through the confirm gate

    def test_schema_takes_no_arguments(self):
        by = {t.name: t for t in _agent.builtin_tools(object(), only={"reindex"})}
        props = by["reindex"].parameters["properties"]
        self.assertEqual(props, {})
        for banned in ("confirm", "max_spend", "output_dir"):
            self.assertNotIn(banned, props)


class TestShellTool(unittest.TestCase):
    """#33: the opt-in gated `shell` exec tool appended by builtin_tools."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _shell(self, **kw):
        tools = _agent.builtin_tools(object(), shell=True, shell_root=self.root, **kw)
        return {t.name: t for t in tools}["shell"]

    def test_absent_by_default(self):
        names = {t.name for t in _agent.builtin_tools(object())}
        self.assertNotIn("shell", names)

    def test_present_paid_and_hides_controls(self):
        tool = self._shell()
        self.assertTrue(tool.paid)  # routes through the confirm gate
        for banned in ("confirm", "max_spend", "output_dir"):
            self.assertNotIn(banned, tool.parameters["properties"])
        self.assertEqual(tool.parameters["required"], ["command"])

    def test_survives_only_filter(self):
        # shell is a rail, not a selectable builtin: `only` narrows the venice tools
        # but the shell tool is still appended.
        tools = _agent.builtin_tools(
            object(), only={"venice_chat"}, shell=True, shell_root=self.root)
        names = {t.name for t in tools}
        self.assertEqual(names, {"venice_chat", "shell"})

    def test_gate_then_run_with_allow(self):
        tool = self._shell(shell_allow=["echo"])
        gate = tool.invoke({"command": "echo hi"})
        self.assertEqual(gate["status"], "confirmation_required")
        r = tool.invoke({"command": "echo hi"}, confirm=True)
        self.assertEqual(r["status"], "ok")
        self.assertIn("hi", r["stdout"])

    def test_deny_refused_before_confirm(self):
        tool = self._shell(shell_deny=["sudo"])
        r = tool.invoke({"command": "sudo reboot"})
        self.assertEqual(r["status"], "error")
        self.assertIn("deny", r["message"])

    def test_model_cannot_self_approve_via_confirm_arg(self):
        # A model smuggling confirm=True in its arguments must not bypass the gate.
        tool = self._shell(shell_allow=["echo"])
        r = tool.invoke({"command": "echo hi", "confirm": True})
        self.assertEqual(r["status"], "confirmation_required")


class TestBrowserTools(unittest.TestCase):
    """Browser rails are opt-in and their authority is operator-bound."""

    def test_absent_by_default(self):
        names = {t.name for t in _agent.builtin_tools(object())}
        self.assertNotIn("web_fetch", names)
        self.assertNotIn("browser_capture", names)

    def test_present_only_when_enabled(self):
        names = {t.name for t in _agent.builtin_tools(object(), browser=True)}
        self.assertIn("web_fetch", names)
        self.assertIn("browser_capture", names)

    def test_rail_is_appended_after_only_filter(self):
        tools = _agent.builtin_tools(object(), only={"venice_chat"}, browser=True)
        self.assertEqual({t.name for t in tools}, {"venice_chat", "web_fetch", "browser_capture"})

    def test_model_cannot_replace_operator_policy(self):
        with mock.patch.object(_agent._mcp, "web_fetch_tool", return_value={"status": "ok"}) as impl:
            tool = next(t for t in _agent.browser_tools(
                allow=["example.com"], deny=["internal"],
                private_hosts=["dev.local"], private_ranges=["10.0.0.0/8"],
                output_dir="/tmp", config={}) if t.name == "web_fetch")
            tool.invoke({"url": "https://example.com", "allow": ["evil"],
                         "private_ranges": ["0.0.0.0/0"]})
        self.assertEqual(impl.call_args.kwargs["allow"], ["example.com"])
        self.assertEqual(impl.call_args.kwargs["private_ranges"], ["10.0.0.0/8"])


class TestConfigDefaults(unittest.TestCase):
    """#58: defaults.<cmd>.* are layered UNDER a tool's model-supplied args."""

    def _spy(self):
        captured = {}

        def image_tool(client, prompt=None, *, hide_watermark=None, steps=None,
                       safe_mode=None, style_references=None,
                       embed_exif_metadata=None, quality=None,
                       enable_web_search=None,
                       disable_prompt_optimization_thinking=None,
                       enhance_prompt=None, confirm=False, max_spend=None,
                       output_dir=None, **kw):
            captured.update(hide_watermark=hide_watermark, steps=steps,
                            safe_mode=safe_mode,
                            style_references=style_references,
                            embed_exif_metadata=embed_exif_metadata,
                            quality=quality,
                            enable_web_search=enable_web_search,
                            disable_prompt_optimization_thinking=(
                                disable_prompt_optimization_thinking
                            ),
                            enhance_prompt=enhance_prompt)
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

    def test_native_image_config_is_coerced_and_injected(self):
        captured, spy = self._spy()
        doc = {"defaults": {"image": {
            "style_references": {
                "image": "https://x.test/style.png", "strength": 0.5
            },
            "embed_exif_metadata": "false",
            "quality": "high",
            "enable_web_search": "true",
            "disable_prompt_optimization_thinking": "false",
            "enhance_prompt": "true",
        }}}
        tool = self._tool(spy, doc)
        tool.invoke({"prompt": "p"})
        self.assertEqual(captured["style_references"], [{
            "image": "https://x.test/style.png", "strength": 0.5
        }])
        self.assertIs(captured["embed_exif_metadata"], False)
        self.assertEqual(captured["quality"], "high")
        self.assertIs(captured["enable_web_search"], True)
        self.assertIs(captured["disable_prompt_optimization_thinking"], False)
        self.assertIs(captured["enhance_prompt"], True)

    def test_safety_flags_flow_through_and_model_wins(self):
        # #61: safe_mode/hide_watermark are now on _IMAGE_SCHEMA, so a model-supplied
        # value must reach image_tool and override a conflicting config default.
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"safe_mode": True, "hide_watermark": False}}}
        tool = self._tool(spy, doc)
        tool.invoke({"prompt": "p", "safe_mode": False, "hide_watermark": True})
        self.assertIs(captured["safe_mode"], False)      # explicit model arg wins
        self.assertIs(captured["hide_watermark"], True)


class TestToolRegistry(unittest.TestCase):
    """#50: the category axis + the select/tools_in/list_categories/get API.

    The core invariant is that `select(categories=...)` reproduces the exact
    hand-maintained `only=` name-sets `code_tools` used to pass, so the refactor is
    behavior-preserving. The drift guard catches a future tool added without a
    category (or into a bogus one).
    """

    # The two legacy `_code.code_tools` only= sets, as of the refactor.
    _CATALOG = {"venice_models", "venice_model_details", "venice_vision",
                "venice_job_status", "venice_job_result"}
    _ASSETS = {"venice_image", "venice_image_edit", "venice_sfx", "venice_music",
               "venice_tts", "venice_upscale", "venice_bg_remove", "venice_video"}

    def test_select_reproduces_catalog_block(self):
        self.assertEqual(
            _agent.select(categories={"catalog", "vision", "jobs"}), self._CATALOG)

    def test_select_reproduces_asset_block(self):
        # spans the union: venice_image_edit/venice_video live in _CODE_ASSET_BUILTINS
        self.assertEqual(
            _agent.select(categories={"image", "audio", "video"}), self._ASSETS)

    def test_select_all_when_unfiltered(self):
        names = {s.name for s in _agent._REGISTRY}
        self.assertEqual(_agent.select(), names)
        self.assertEqual(len(names), 16)

    def test_select_names_ignores_unknown(self):
        self.assertEqual(
            _agent.select(names={"venice_tts", "does_not_exist"}), {"venice_tts"})

    def test_select_exclude_by_name_and_category(self):
        self.assertEqual(
            _agent.select(categories={"image"}, exclude={"venice_image_edit"}),
            {"venice_image", "venice_upscale", "venice_bg_remove"})
        # excluding a whole category subtracts all its members
        self.assertNotIn(
            "venice_tts", _agent.select(categories={"audio", "image"},
                                        exclude={"audio"}))

    def test_tools_in_and_list_categories(self):
        self.assertEqual(_agent.tools_in("catalog"),
                         {"venice_models", "venice_model_details"})
        self.assertEqual(_agent.tools_in("video"), {"venice_video"})
        self.assertEqual(_agent.tools_in("nope"), set())
        self.assertEqual(
            _agent.list_categories(),
            {"image", "audio", "video", "text", "catalog", "vision", "search", "jobs"})

    def test_get_returns_spec_or_none(self):
        spec = _agent.get("venice_image")
        self.assertEqual(spec.category, "image")
        self.assertTrue(spec.paid)
        self.assertIsNone(_agent.get("venice_nope"))

    def test_tts_schema_documents_live_format_default(self):
        spec = _agent.get("venice_tts")
        desc = spec.parameters["properties"]["format"]["description"]
        self.assertIn("catalog default", desc)

    def test_models_schema_exposes_current_catalog_types(self):
        spec = _agent.get("venice_models")
        model_types = spec.parameters["properties"]["type"]["enum"]
        self.assertIn("asr", model_types)
        self.assertIn("inpaint", model_types)

    def test_every_registry_tool_has_a_category(self):
        # drift guard: a new tool with no/empty category fails here.
        for spec in _agent._REGISTRY:
            self.assertTrue(spec.category, f"{spec.name} has no category")

    def test_categories_partition_the_registry(self):
        # drift guard: union of every category == every registered name (no orphan,
        # no name leaking into a category that doesn't round-trip).
        allnames = {s.name for s in _agent._REGISTRY}
        union = set().union(
            *(_agent.tools_in(c) for c in _agent.list_categories()))
        self.assertEqual(union, allnames)

    def test_built_tool_carries_category(self):
        by = {t.name: t for t in _agent.builtin_tools(object())}
        self.assertEqual(by["venice_image"].category, "image")
        self.assertEqual(by["venice_chat"].category, "text")

    def test_select_output_drives_builtin_tools_unchanged(self):
        # the actual call code_tools now makes: select(...) fed straight to only=.
        names = {t.name for t in _agent.builtin_tools(
            object(), only=_agent.select(categories={"image", "audio", "video"}))}
        self.assertEqual(names, self._ASSETS)

    def test_system_and_selected_tool_bytes_ignore_python_hash_seed(self):
        # #102: select() deliberately returns a set, but builtin_tools() filters the
        # ordered registry by membership. Serializing in fresh processes pins that
        # boundary: iterating the set instead makes this output seed-dependent.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = """
import json
from venice.commands import _agent

selected = _agent.select(categories={"image", "audio", "video"})
tools = _agent.builtin_tools(object(), only=selected)
print(json.dumps(
    {"system": _agent.SPAWN_SYSTEM, "tools": _agent.to_openai_tools(tools)},
    separators=(",", ":"),
))
"""
        outputs = []
        for seed in ("1", "2"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (os.path.join(root, "src"), env.get("PYTHONPATH")))
            )
            proc = subprocess.run(
                [sys.executable, "-c", script], cwd=root, env=env,
                check=True, capture_output=True, text=True,
            )
            outputs.append(proc.stdout)

        self.assertEqual(outputs[0], outputs[1])
        payload = json.loads(outputs[0])
        self.assertEqual(payload["system"], _agent.SPAWN_SYSTEM)
        selected = _agent.select(categories={"image", "audio", "video"})
        expected = [spec.name for spec in _agent._REGISTRY if spec.name in selected]
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]], expected,
        )


class TestToolSchemaStability(unittest.TestCase):
    def test_tool_array_is_snapshotted_once_for_the_run(self):
        tools = []

        def mutate(_arguments, *, confirm=False):
            tools.append(_tool("late", lambda a, *, confirm=False: {"status": "ok"}))
            return {"status": "ok"}

        tools.append(_tool("early", mutate))
        fake, calls = _fake_oai([
            FakeToolCompletion(tool_calls=[_FnCall("c1", "early", "{}")]),
            FakeToolCompletion("done"),
        ])
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(
                fake, "m", [{"role": "user", "content": "go"}], {}, tools,
                max_tool_calls=0, yes=True, json_out=False,
            )

        self.assertEqual([t.name for t in tools], ["early", "late"])
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(
                [tool["function"]["name"] for tool in call["tools"]], ["early"],
            )
        self.assertIs(calls[0]["tools"], calls[1]["tools"])


# --------------------------------------------------------------------------- #
# #52 --parallel: thread-safe stdout router + concurrent subagent dispatch
# --------------------------------------------------------------------------- #
class TestStdoutRouter(unittest.TestCase):
    """The thread-local stdout router that replaces the old global-swap capture."""

    def test_install_is_idempotent(self):
        with mock.patch.object(sys, "stdout", io.StringIO()):
            r1 = _agent._install_router()
            r2 = _agent._install_router()
            self.assertIs(r1, r2)
            self.assertIsInstance(sys.stdout, _agent._StdoutRouter)

    def test_idle_router_delegates_to_base(self):
        base = io.StringIO()
        base.isatty = lambda: True  # attribute delegation via __getattr__/isatty
        with mock.patch.object(sys, "stdout", base):
            _agent._install_router()
            print("straight through")          # no target pushed -> base
            self.assertTrue(sys.stdout.isatty())  # delegates to base.isatty()
        self.assertEqual(base.getvalue(), "straight through\n")

    def test_single_thread_capture_still_works(self):
        with mock.patch.object(sys, "stdout", io.StringIO()) as base:
            with _agent._capture_stdout() as buf:
                print("captured")
            self.assertEqual(buf.getvalue(), "captured\n")
            print("after")                     # target popped -> base
        self.assertEqual(base.getvalue(), "after\n")

    def test_nested_capture_restores_outer(self):
        with mock.patch.object(sys, "stdout", io.StringIO()) as base:
            with _agent._capture_stdout() as outer:
                print("O1")
                with _agent._capture_stdout() as inner:
                    print("I")
                print("O2")
            self.assertEqual(inner.getvalue(), "I\n")
            self.assertEqual(outer.getvalue(), "O1\nO2\n")
        self.assertEqual(base.getvalue(), "")  # nothing leaked to base

    def test_concurrent_captures_are_isolated(self):
        # Two threads capture at the same time -> each buffer gets ONLY its own writes,
        # and the base stdout gets neither. This is the property the old global swap
        # could not provide.
        with mock.patch.object(sys, "stdout", io.StringIO()) as base:
            _agent._install_router()
            results = {}
            start = threading.Barrier(2)

            def worker(tag):
                with _agent._capture_stdout() as buf:
                    start.wait()
                    for _ in range(50):
                        print(tag)
                        time.sleep(0)  # yield to interleave the threads
                results[tag] = buf.getvalue()

            ta = threading.Thread(target=worker, args=("A",))
            tb = threading.Thread(target=worker, args=("B",))
            ta.start(); tb.start(); ta.join(); tb.join()

            self.assertEqual(results["A"], "A\n" * 50)  # no B leaked in
            self.assertEqual(results["B"], "B\n" * 50)  # no A leaked in
        self.assertEqual(base.getvalue(), "")           # nothing reached base


def _sub_tool(name, *, record=None, sleep_arg=None):
    """A fake subagent Tool (venice_scout/venice_spawn) whose invoke records its call
    and echoes a per-call report. `sleep_arg` (a JSON key) lets a call sleep so tests
    can force out-of-order completion."""
    def inv(a, *, confirm=False):
        if record is not None:
            record.append(name)
        if sleep_arg and isinstance(a.get(sleep_arg), (int, float)):
            time.sleep(a[sleep_arg] / 1000.0)
        return {"status": "ok", "report": a.get("tag", name)}
    return _agent.Tool(name, name, {"type": "object", "properties": {}}, inv,
                       paid=False, category="agent", tags=("spawn",))


def _tool_msgs(messages):
    return [m for m in messages if m.get("role") == "tool"]


class TestParallelDispatch(unittest.TestCase):
    """`run_loop(parallel=True)`: subagent calls fan out, bookkeeping stays serial."""

    def _spawn_call(self, cid, tag, **extra):
        args = {"tag": tag, **extra}
        import json as _json
        return _FnCall(cid, _agent.SPAWN_TOOL_NAME, _json.dumps(args))

    def test_predicate_selects_only_scout_and_spawn(self):
        mk = lambda n: _FnCall("x", n, "{}")
        self.assertTrue(_agent._is_parallelizable(mk(_agent.SCOUT_TOOL_NAME)))
        self.assertTrue(_agent._is_parallelizable(mk(_agent.SPAWN_TOOL_NAME)))
        self.assertFalse(_agent._is_parallelizable(mk(_agent.MERGE_TOOL_NAME)))
        self.assertFalse(_agent._is_parallelizable(mk("write_file")))

    def _run(self, seq, tools, *, max_tool_calls, parallel, ledger=None):
        fake, calls = _fake_oai(seq)
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(fake, "m", messages, {}, tools,
                            max_tool_calls=max_tool_calls, yes=True, json_out=False,
                            parallel=parallel, ledger=ledger)
        return messages, calls

    # -- per-tool timing through run_loop (#82) ------------------------------ #

    def test_every_worker_window_reaches_the_one_ledger(self):
        # Three concurrent spawns land on the SAME key from three pool threads.
        L = _agent.CostLedger()
        tools = [_sub_tool(_agent.SPAWN_TOOL_NAME)]
        turn = FakeToolCompletion(tool_calls=[
            self._spawn_call("c1", "A"),
            self._spawn_call("c2", "B"),
            self._spawn_call("c3", "C"),
        ])
        self._run([turn, FakeToolCompletion("done")], tools,
                  max_tool_calls=0, parallel=True, ledger=L)
        self.assertEqual(L.tools[_agent.SPAWN_TOOL_NAME]["calls"], 3)

    def test_parallel_budget_overflow_is_not_timed(self):
        # The guard here is the ABSENCE of a `_run_one_call`, which is exactly what a
        # refactor deletes silently -- so pin it. slots=2: c3 is never executed and
        # must not appear as a 0.0s call.
        L = _agent.CostLedger()
        tools = [_sub_tool(_agent.SPAWN_TOOL_NAME), _free_tool()]
        turn = FakeToolCompletion(tool_calls=[
            self._spawn_call("c1", "A"),
            self._spawn_call("c2", "B"),
            _FnCall("c3", "t", "{}"),
        ])
        self._run([turn, FakeToolCompletion("done")], tools,
                  max_tool_calls=2, parallel=True, ledger=L)
        self.assertEqual(L.tool_calls_total(), 2)
        self.assertNotIn("t", L.tools)

    def test_serial_budget_overflow_is_not_timed(self):
        L = _agent.CostLedger()
        turn = FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}"),
                                              _FnCall("c2", "t", "{}")])
        self._run([turn, FakeToolCompletion("done")], [_free_tool()],
                  max_tool_calls=1, parallel=False, ledger=L)
        self.assertEqual(L.tools["t"]["calls"], 1)

    def test_an_unmetered_run_records_nothing_and_still_runs(self):
        record = []
        tools = [_sub_tool(_agent.SPAWN_TOOL_NAME, record=record)]
        turn = FakeToolCompletion(tool_calls=[self._spawn_call("c1", "A")])
        self._run([turn, FakeToolCompletion("done")], tools,
                  max_tool_calls=0, parallel=True, ledger=None)
        self.assertEqual(record, [_agent.SPAWN_TOOL_NAME])

    def test_batch_runs_and_appends_in_original_order(self):
        record = []
        tools = [_sub_tool(_agent.SPAWN_TOOL_NAME, record=record)]
        turn = FakeToolCompletion(tool_calls=[
            self._spawn_call("c1", "A"),
            self._spawn_call("c2", "B"),
            self._spawn_call("c3", "C"),
        ])
        messages, _ = self._run([turn, FakeToolCompletion("done")], tools,
                                max_tool_calls=0, parallel=True)
        tms = _tool_msgs(messages)
        self.assertEqual([m["tool_call_id"] for m in tms], ["c1", "c2", "c3"])
        self.assertEqual(sorted(record), ["venice_spawn"] * 3)  # all three ran
        for m, tag in zip(tms, ["A", "B", "C"]):
            self.assertIn(tag, m["content"])

    def test_mixed_batch_commits_native_followup_after_every_tool_result(self):
        def vision(arguments, *, confirm=False, runtime=None):
            return _agent._ToolOutcome(
                {"status": "ok", "mode": "native"},
                ({"role": "user", "content": [
                    {"type": "text", "text": "image"},
                ]},),
            )

        tools = [
            _sub_tool(_agent.SPAWN_TOOL_NAME),
            _agent.Tool(
                "venice_vision", "vision", {"type": "object", "properties": {}},
                vision, contextual=True,
            ),
        ]
        turn = FakeToolCompletion(tool_calls=[
            self._spawn_call("c1", "worker"),
            _FnCall("c2", "venice_vision", "{}"),
        ])
        _messages, calls = self._run(
            [turn, FakeToolCompletion("done")], tools,
            max_tool_calls=0, parallel=True,
        )
        self.assertEqual(
            [m["role"] for m in calls[1]["messages"][-5:-1]],
            ["assistant", "tool", "tool", "user"],
        )
        self.assertEqual(
            [m["tool_call_id"] for m in calls[1]["messages"][-4:-2]],
            ["c1", "c2"],
        )

    def test_budget_marks_overflow_not_executed_without_running(self):
        record = []
        tools = [_sub_tool(_agent.SPAWN_TOOL_NAME, record=record),
                 _free_tool()]  # a non-subagent serial tool named "t"
        turn = FakeToolCompletion(tool_calls=[
            self._spawn_call("c1", "A"),
            self._spawn_call("c2", "B"),
            _FnCall("c3", "t", "{}"),        # serial, position 2
        ])
        # slots = 2 -> positions 0,1 run; position 2 is over budget, never executed.
        messages, calls = self._run([turn, FakeToolCompletion("done")], tools,
                                    max_tool_calls=2, parallel=True)
        tms = _tool_msgs(messages)
        self.assertEqual([m["tool_call_id"] for m in tms], ["c1", "c2", "c3"])
        self.assertIn("not executed", tms[2]["content"])
        self.assertEqual(len(record), 2)                 # only the 2 within budget ran
        self.assertEqual(calls[-1]["tool_choice"], "none")  # cap -> forced final

    def test_parallel_matches_serial_for_independent_calls(self):
        def build():
            return [_sub_tool(_agent.SPAWN_TOOL_NAME)]

        def turns():
            return [FakeToolCompletion(tool_calls=[
                        self._spawn_call("c1", "A"),
                        self._spawn_call("c2", "B")]),
                    FakeToolCompletion("done")]

        ser_msgs, _ = self._run(turns(), build(), max_tool_calls=0, parallel=False)
        par_msgs, _ = self._run(turns(), build(), max_tool_calls=0, parallel=True)
        strip = lambda ms: [(m["tool_call_id"], m["name"], m["content"])
                            for m in _tool_msgs(ms)]
        self.assertEqual(strip(ser_msgs), strip(par_msgs))  # byte-identical results

    def test_out_of_order_completion_keeps_submission_order(self):
        # c1 sleeps longer than c2, so c2 finishes first -- appended order must still be
        # c1, c2 (original tool_calls order), not completion order.
        tools = [_sub_tool(_agent.SPAWN_TOOL_NAME, sleep_arg="ms")]
        turn = FakeToolCompletion(tool_calls=[
            self._spawn_call("c1", "SLOW", ms=80),
            self._spawn_call("c2", "FAST", ms=1),
        ])
        messages, _ = self._run([turn, FakeToolCompletion("done")], tools,
                                max_tool_calls=0, parallel=True)
        tms = _tool_msgs(messages)
        self.assertEqual([m["tool_call_id"] for m in tms], ["c1", "c2"])
        self.assertIn("SLOW", tms[0]["content"])
        self.assertIn("FAST", tms[1]["content"])

    def test_non_subagent_turn_falls_through_to_serial(self):
        # parallel=True but the turn has no scout/spawn call -> the serial path runs
        # (identical result), proving the predicate gate.
        record = []
        tools = [_tool("t", lambda a, *, confirm=False: (record.append("t"),
                                                         {"status": "ok"})[1])]
        turn = FakeToolCompletion(tool_calls=[_FnCall("c1", "t", "{}")])
        messages, _ = self._run([turn, FakeToolCompletion("done")], tools,
                                max_tool_calls=0, parallel=True)
        self.assertEqual([m["tool_call_id"] for m in _tool_msgs(messages)], ["c1"])
        self.assertEqual(record, ["t"])


if __name__ == "__main__":
    unittest.main()
