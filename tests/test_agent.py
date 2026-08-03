"""Unit tests for the agent loop helpers + run_loop budget/gate/progress.

Covers the ergonomics work: unlimited `--max-tool-calls` (#53), the TTY-gated
progress feedback (#54), and the `all`/auto-accept confirm gate (#55). Reuses
`test_chat`'s fake completions so the fakes stay in lock-step. No network/key.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
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
        self.assertNotIn("n/a", r)
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
                           (dict(max_spend=0.0001), "--max-spend")):
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


class TestAsyncJobSchemas(unittest.TestCase):
    """#62: background param on media schemas + the two async job-tool schemas."""

    def test_background_in_media_schemas(self):
        for schema in (_agent._SFX_SCHEMA, _agent._MUSIC_SCHEMA, _agent._VIDEO_SCHEMA):
            props = schema["properties"]
            self.assertIn("background", props)
            self.assertEqual(props["background"]["type"], "boolean")

    def test_job_schemas_require_handle_fields_and_hide_controls(self):
        for schema in (_agent._JOB_STATUS_SCHEMA, _agent._JOB_RESULT_SCHEMA):
            self.assertEqual(schema.get("required"), ["queue_id", "type", "model"])
            props = schema["properties"]
            for banned in ("confirm", "max_spend", "output_dir"):
                self.assertNotIn(banned, props)
        # only job_result exposes max_wait (block-poll seconds)
        self.assertIn("max_wait", _agent._JOB_RESULT_SCHEMA["properties"])
        self.assertNotIn("max_wait", _agent._JOB_STATUS_SCHEMA["properties"])

    def test_job_tools_are_free(self):
        by = {t.name: t for t in _agent.builtin_tools(
            object(), only={"venice_job_status", "venice_job_result"})}
        self.assertFalse(by["venice_job_status"].paid)
        self.assertFalse(by["venice_job_result"].paid)


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
    """#71: the opt-in web_fetch/browser_capture rails; URL policy bound by the wiring."""

    def _tools(self, **kw):
        return {t.name: t for t in _agent.builtin_tools(object(), browser=True, **kw)}

    def test_absent_by_default(self):
        names = {t.name for t in _agent.builtin_tools(object())}
        self.assertNotIn("web_fetch", names)
        self.assertNotIn("browser_capture", names)

    def test_present_free_and_hide_controls(self):
        tools = self._tools()
        for name in ("web_fetch", "browser_capture"):
            self.assertIn(name, tools)
            self.assertFalse(tools[name].paid)          # no confirm gate; URL policy guards
            self.assertEqual(tools[name].parameters["required"], ["url"])
            props = tools[name].parameters["properties"]
            for banned in ("allow", "deny", "confirm", "max_spend", "output_dir"):
                self.assertNotIn(banned, props)         # model can't set policy/controls

    def test_survives_only_filter(self):
        tools = _agent.builtin_tools(object(), only={"venice_chat"}, browser=True)
        self.assertEqual({t.name for t in tools},
                         {"venice_chat", "web_fetch", "browser_capture"})

    def test_model_cannot_widen_deny_policy(self):
        # deny is bound by the operator; a model smuggling deny=[] must not override it.
        tool = self._tools(browser_deny=["evil.com"])["web_fetch"]
        r = tool.invoke({"url": "http://evil.com/x", "deny": []})
        self.assertEqual(r["status"], "error")
        self.assertIn("deny", r["message"])

    def test_model_cannot_widen_allow_policy(self):
        tool = self._tools(browser_allow=["good.com"])["web_fetch"]
        r = tool.invoke({"url": "http://evil.com/x", "allow": ["evil.com"]})
        self.assertEqual(r["status"], "error")
        self.assertIn("allowlist", r["message"])


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

    def _run(self, seq, tools, *, max_tool_calls, parallel):
        fake, calls = _fake_oai(seq)
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _agent.run_loop(fake, "m", messages, {}, tools,
                            max_tool_calls=max_tool_calls, yes=True, json_out=False,
                            parallel=parallel)
        return messages, calls

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
