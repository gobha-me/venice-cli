"""Unit tests for context compaction (issue #48).

Covers the pure helpers in `_compact` (token estimate, group-boundary split,
synthetic-message shape), the best-effort `compact_messages` turn, and the
`Budget` usage tracker. All OpenAI calls are faked -- no network, no key.
"""
import unittest
from unittest import mock

from venice.commands import _compact, _context_archive


def _fake_oai(summary="A concise summary.", fail=False, usage=None):
    """A fake `oai` whose create() returns a canned summary (or raises).

    `usage` is set on the response EXPLICITLY, and defaults to None -- "the response
    carried no usage block". Without this the response is a bare `MagicMock`, whose
    auto-created `.usage` attribute normalizes to None through `_usage_dict` anyway, so
    a priced `record()` returns 0.0: every cost assertion written against the old
    fixture passed vacuously. Pass a real dict to bill a real number (#101).
    """
    calls = []

    def _create(**kw):
        calls.append(kw)
        if fail:
            raise RuntimeError("boom")
        msg = mock.MagicMock()
        msg.content = summary
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock(message=msg)]
        resp.usage = usage
        return resp

    fake = mock.MagicMock()
    fake.chat.completions.create.side_effect = _create
    return fake, calls


class _Rec:
    """A duck-typed stand-in for `CostLedger` (`_compact` imports nothing from `_agent`).

    Implements BOTH hooks `compact_messages` calls, deliberately: the module invokes
    `record` and `record_compaction` unguarded, so a stand-in carrying only one of them
    would let a wiring bug read as "nothing to record" rather than fail. `seq` logs the
    two in call order, which is the only way to pin that the bill is taken before the
    event that quotes it. `test_a_real_ledger_satisfies_the_duck_type` is what stops this
    class and the real ledger drifting apart.
    """

    def __init__(self, cost=0.0):
        self.events = []
        self.calls = []
        self.seq = []
        self.cost = cost

    def record(self, usage, *, seconds=None, bucket=None):
        self.calls.append({"usage": usage, "seconds": seconds, "bucket": bucket})
        self.seq.append("call")
        return self.cost

    def record_compaction(self, ev):
        self.events.append(ev)
        self.seq.append("event")


def _history(pairs, *, system=True):
    """[sys?, u/a, u/a, ...] with short contents."""
    msgs = [{"role": "system", "content": "sys"}] if system else []
    for i in range(pairs):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _tooly_history():
    """A history whose middle turns carry tool_calls + tool results."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path":"x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file",
         "content": "file contents"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_compact.estimate_tokens([]), 0)

    def test_scales_with_content(self):
        small = _compact.estimate_tokens([{"role": "user", "content": "hi"}])
        big = _compact.estimate_tokens([{"role": "user", "content": "x" * 4000}])
        self.assertGreater(big, small)
        # ~1000 content tokens + per-message overhead
        self.assertGreaterEqual(big, 1000)

    def test_counts_tool_call_arguments(self):
        plain = _compact.estimate_tokens([{"role": "assistant", "content": ""}])
        with_tc = _compact.estimate_tokens([{
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": "t", "arguments": "y" * 400}}],
        }])
        self.assertGreaterEqual(with_tc - plain, 100)

    def test_content_parts(self):
        n = _compact.estimate_tokens([
            {"role": "user", "content": [{"type": "text", "text": "x" * 40}]},
        ])
        self.assertGreaterEqual(n, 10)


class TestBudget(unittest.TestCase):
    def test_disabled_when_threshold_nonpositive(self):
        b = _compact.Budget(threshold_tokens=0)
        self.assertFalse(b.over([{"role": "user", "content": "x" * 10**6}]))

    def test_uses_observed_prompt_tokens_over_estimate(self):
        b = _compact.Budget(threshold_tokens=1000)
        b.observe({"prompt_tokens": 1500, "completion_tokens": 5})
        # A tiny history is still "over" because the server said the prompt was big.
        self.assertTrue(b.over([{"role": "user", "content": "hi"}]))
        b.observe({"prompt_tokens": 10})
        self.assertFalse(b.over([{"role": "user", "content": "hi"}]))

    def test_observe_accepts_sdk_objects_and_garbage(self):
        b = _compact.Budget(threshold_tokens=100)
        usage = mock.MagicMock()
        usage.model_dump.return_value = {"prompt_tokens": 500}
        b.observe(usage)
        self.assertEqual(b.last_prompt_tokens, 500)
        b.observe(None)
        b.observe({"prompt_tokens": "nope"})
        self.assertEqual(b.last_prompt_tokens, 500)  # unchanged

    def test_falls_back_to_estimate(self):
        b = _compact.Budget(threshold_tokens=50)
        self.assertFalse(b.over([{"role": "user", "content": "short"}]))
        self.assertTrue(b.over([{"role": "user", "content": "x" * 4000}]))


class TestSplitForCompaction(unittest.TestCase):
    def test_too_few_turns_returns_none(self):
        msgs = _history(3)
        self.assertIsNone(_compact.split_for_compaction(msgs, keep_turns=3))
        self.assertIsNone(_compact.split_for_compaction(msgs, keep_turns=10))

    def test_split_keeps_tail_and_separates_prefix(self):
        msgs = _history(6)
        prefix, tail = _compact.split_for_compaction(msgs, keep_turns=2)
        # 4 older pairs summarized, 2 newest kept; system is in neither half.
        self.assertEqual(len(tail), 4)
        self.assertEqual(tail[0]["content"], "u4")
        self.assertEqual(len(prefix), 8)
        self.assertNotIn("sys", [m.get("content") for m in prefix + tail])

    def test_never_orphans_a_tool_message(self):
        msgs = _tooly_history()
        for keep in (1, 2):
            split = _compact.split_for_compaction(msgs, keep_turns=keep)
            self.assertIsNotNone(split)
            _prefix, tail = split
            # No `tool` message may start the kept tail, and every tool_call_id
            # in the tail must be answered within the tail.
            self.assertNotEqual(tail[0].get("role"), "tool")
            ids = set()
            for m in tail:
                for tc in m.get("tool_calls") or []:
                    ids.add(tc["id"])
                if m.get("role") == "tool":
                    self.assertIn(m["tool_call_id"], ids)

    def test_no_system_prefix(self):
        msgs = _history(4, system=False)
        prefix, tail = _compact.split_for_compaction(msgs, keep_turns=1)
        self.assertEqual(len(tail), 2)
        self.assertEqual(len(prefix), 6)


class TestCompactMessages(unittest.TestCase):
    def test_evidence_policy_archives_every_removed_message_exactly(self):
        msgs = _tooly_history()
        prefix, _tail = _compact.split_for_compaction(msgs, keep_turns=1)
        expected = list(prefix)
        archive = _context_archive.ContextArchive()
        fake, calls = _fake_oai("faithful summary")
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=1, loss_policy="evidence", archive=archive,
        ))
        self.assertEqual([e["message"] for e in archive.entries], expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(msgs[1]["role"], "system")
        self.assertIn("[Archived context evidence index]", msgs[2]["content"])
        # The kept tail remains group-valid.
        self.assertEqual([m["role"] for m in msgs[-2:]], ["user", "assistant"])

    def test_evidence_capacity_refuses_before_summary_and_changes_nothing(self):
        msgs = _history(6)
        snapshot = list(msgs)
        archive = _context_archive.ContextArchive()
        fake, calls = _fake_oai("unused")
        with mock.patch.object(_context_archive, "MAX_ARCHIVE_ENTRIES", 1):
            self.assertFalse(_compact.compact_messages(
                fake, "m", msgs, keep_turns=2,
                loss_policy="evidence", archive=archive,
            ))
        self.assertEqual(calls, [])
        self.assertEqual(msgs, snapshot)
        self.assertEqual(archive.entries, [])
        self.assertIn("archive full", archive.last_error)

    def test_evidence_summary_failure_is_transactional(self):
        msgs = _history(6)
        snapshot = list(msgs)
        archive = _context_archive.ContextArchive()
        fake, _calls = _fake_oai(fail=True)
        self.assertFalse(_compact.compact_messages(
            fake, "m", msgs, keep_turns=2,
            loss_policy="evidence", archive=archive,
        ))
        self.assertEqual(msgs, snapshot)
        self.assertEqual(archive.entries, [])

    def test_repeated_evidence_compaction_keeps_all_prior_entries(self):
        msgs = _history(4)
        archive = _context_archive.ContextArchive()
        fake, _calls = _fake_oai("first")
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=1,
            loss_policy="evidence", archive=archive, protected_system_messages=1,
        ))
        first_ids = [e["id"] for e in archive.entries]
        msgs.extend(_history(3, system=False))
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=1,
            loss_policy="evidence", archive=archive, protected_system_messages=1,
        ))
        self.assertEqual([e["id"] for e in archive.entries[:len(first_ids)]], first_ids)
        self.assertEqual(len({e["id"] for e in archive.entries}), len(archive.entries))
        generated = [m for m in msgs[1:] if m.get("role") == "system"]
        self.assertEqual(len(generated), 2)

    def test_operator_system_prompt_that_looks_generated_is_never_removed(self):
        policy = "[Summary of earlier conversation]\noperator safety policy"
        msgs = [{"role": "system", "content": policy}] + _history(2, system=False)
        fake, _calls = _fake_oai("summary")
        self.assertTrue(_compact.compact_messages(fake, "m", msgs, keep_turns=1))
        self.assertEqual(msgs[0], {"role": "system", "content": policy})

    def test_replaces_prefix_with_synthetic_summary(self):
        msgs = _history(6)
        fake, calls = _fake_oai("We decided X and edited a.py.")
        changed = _compact.compact_messages(fake, "m", msgs, keep_turns=2)
        self.assertTrue(changed)
        # system + synthetic + 2 kept pairs
        self.assertEqual(len(msgs), 6)
        self.assertEqual(msgs[0]["content"], "sys")
        self.assertEqual(msgs[1]["role"], "system")
        self.assertIn("[Summary of earlier conversation]", msgs[1]["content"])
        self.assertIn("We decided X", msgs[1]["content"])
        self.assertEqual(msgs[2]["content"], "u4")
        # The summarization call is self-contained and tool-free.
        self.assertEqual(calls[0]["tool_choice"], "none")
        self.assertNotIn("tools", calls[0])
        self.assertEqual(calls[0]["model"], "m")

    def test_nothing_to_do_returns_false(self):
        msgs = _history(3)
        fake, calls = _fake_oai()
        self.assertFalse(_compact.compact_messages(fake, "m", msgs, keep_turns=5))
        self.assertEqual(calls, [])  # no wasted summarization call

    def test_failure_leaves_history_unchanged(self):
        msgs = _history(6)
        snapshot = list(msgs)
        fake, _calls = _fake_oai(fail=True)
        self.assertFalse(_compact.compact_messages(fake, "m", msgs, keep_turns=2))
        self.assertEqual(msgs, snapshot)

    def test_empty_summary_leaves_history_unchanged(self):
        msgs = _history(6)
        snapshot = list(msgs)
        fake, _calls = _fake_oai(summary="   ")
        self.assertFalse(_compact.compact_messages(fake, "m", msgs, keep_turns=2))
        self.assertEqual(msgs, snapshot)

    def test_tool_turns_survive_intact(self):
        msgs = _tooly_history()
        fake, _calls = _fake_oai("summary")
        self.assertTrue(_compact.compact_messages(fake, "m", msgs, keep_turns=2))
        # Kept tail: the tool exchange group + the last pair, fully paired.
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "system")  # synthetic summary
        tail = msgs[2:]
        self.assertNotEqual(tail[0]["role"], "tool")
        ids = set()
        for m in tail:
            for tc in m.get("tool_calls") or []:
                ids.add(tc["id"])
            if m["role"] == "tool":
                self.assertIn(m["tool_call_id"], ids)

    def test_base_kwargs_stripped_of_streaming_and_tools(self):
        msgs = _history(6)
        fake, calls = _fake_oai()
        _compact.compact_messages(
            fake, "m", msgs, keep_turns=2,
            base_kwargs={"stream": True, "stream_options": {"x": 1},
                         "tools": [1], "temperature": 0.2},
        )
        self.assertNotIn("stream", calls[0])
        self.assertNotIn("stream_options", calls[0])
        self.assertNotIn("tools", calls[0])
        self.assertEqual(calls[0]["temperature"], 0.2)
        self.assertEqual(calls[0]["max_tokens"], _compact.SUMMARY_MAX_TOKENS)

    def test_fresh_summary_does_not_reuse_parent_cache_affinity(self):
        msgs = _history(6)
        fake, calls = _fake_oai()
        original = {"extra_body": {
            "prompt_cache_key": "parent-key",
            "venice_parameters": {"include_venice_system_prompt": False},
        }}
        _compact.compact_messages(
            fake, "m", msgs, keep_turns=2, base_kwargs=original)
        self.assertNotIn("prompt_cache_key", calls[0]["extra_body"])
        self.assertEqual(
            calls[0]["extra_body"]["venice_parameters"],
            {"include_venice_system_prompt": False},
        )
        self.assertEqual(
            original["extra_body"]["prompt_cache_key"], "parent-key")


class TestBuildSummaryPrompt(unittest.TestCase):
    def test_tool_messages_rendered_as_text(self):
        prompt = _compact.build_summary_prompt([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "grep"}}]},
            {"role": "tool", "content": ""},
        ])
        self.assertEqual(prompt[0]["role"], "system")
        body = prompt[1]["content"]
        self.assertIn("user: hi", body)
        self.assertIn("called tools: grep", body)
        self.assertIn("(tool result)", body)


class TestBudgetFromArgs(unittest.TestCase):
    """The shared opt-in builder used by chat (REPL + --tools) and code (#48)."""

    def _ns(self, **kw):
        import argparse
        base = dict(auto_compact=None, compact_threshold=None,
                    compact_keep_turns=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_none_when_not_opted_in(self):
        self.assertIsNone(_compact.budget_from_args(self._ns()))
        self.assertIsNone(_compact.budget_from_args(self._ns(auto_compact=False)))

    def test_defaults_when_opted_in_without_knobs(self):
        b = _compact.budget_from_args(self._ns(auto_compact=True))
        self.assertIsInstance(b, _compact.Budget)
        self.assertEqual(b.threshold_tokens, _compact.DEFAULT_THRESHOLD_TOKENS)
        self.assertEqual(b.keep_turns, _compact.DEFAULT_KEEP_TURNS)

    def test_explicit_knobs_win(self):
        b = _compact.budget_from_args(
            self._ns(auto_compact=True, compact_threshold=1234,
                     compact_keep_turns=3))
        self.assertEqual(b.threshold_tokens, 1234)
        self.assertEqual(b.keep_turns, 3)

    def test_missing_attrs_are_safe(self):
        import argparse
        # a namespace lacking the compact attrs entirely -> None, no AttributeError
        self.assertIsNone(_compact.budget_from_args(argparse.Namespace()))


class TestCompactionEvents(unittest.TestCase):
    """#99: a compaction is logged to the ledger as a context event.

    The ledger is duck-typed here (`_compact` imports nothing from `_agent`), so these
    use a minimal recorder rather than a real `CostLedger` -- what is under test is the
    contract `_compact` calls, not the ledger's storage.
    """

    def test_a_successful_compaction_is_recorded(self):
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=2, ledger=led, trigger="manual"))
        self.assertEqual(len(led.events), 1)
        ev = led.events[0]
        self.assertEqual(ev["trigger"], "manual")
        self.assertEqual(ev["messages_before"], 13)
        self.assertEqual(ev["messages_after"], len(msgs))
        # Both estimates use the same yardstick, so the RATIO is the meaningful part.
        self.assertGreater(ev["est_tokens_before"], ev["est_tokens_after"])

    def test_a_failed_compaction_records_nothing(self):
        # The module's contract is that a failed summarization leaves history
        # untouched; an event for a compaction that did not happen would be a fresh
        # lie in the artifact that exists to stop one.
        for kw in ({"fail": True}, {"summary": "   "}):
            with self.subTest(**kw):
                msgs = _history(6)
                fake, _calls = _fake_oai(**kw)
                led = _Rec()
                self.assertFalse(_compact.compact_messages(
                    fake, "m", msgs, keep_turns=2, ledger=led))
                self.assertEqual(led.events, [])

    def test_nothing_to_compact_records_nothing(self):
        msgs = _history(3)
        fake, _calls = _fake_oai()
        led = _Rec()
        self.assertFalse(_compact.compact_messages(
            fake, "m", msgs, keep_turns=5, ledger=led))
        self.assertEqual(led.events, [])

    def test_no_ledger_is_a_no_op(self):
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        self.assertTrue(_compact.compact_messages(fake, "m", msgs, keep_turns=2))

    def test_the_event_carries_the_summarization_cost(self):
        # The inversion of #99's `test_the_event_carries_no_cost`, which existed to
        # hold this line until #101 landed. The objection then was that tokens-saved
        # beside a cost reads as "it paid for itself"; that is answered in the
        # RENDERING ("$X to summarize"), not by withholding the number.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec(cost=0.00042)
        _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led)
        self.assertEqual(led.events[0]["cost"], 0.00042)

    def test_the_summary_call_is_billed_to_the_compaction_bucket(self):
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led)
        self.assertEqual(len(led.calls), 1)
        self.assertEqual(led.calls[0]["bucket"], "compaction")

    def test_the_summary_call_window_is_stamped(self):
        # Same never-read-a-clock contract as the main loop: the caller brackets the
        # window, the ledger just adds it up.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led)
        self.assertIsNotNone(led.calls[0]["seconds"])
        self.assertGreaterEqual(led.calls[0]["seconds"], 0.0)

    def test_an_empty_summary_is_still_billed(self):
        # THE billing invariant: the call completed and the tokens are spent whether
        # or not the summary is usable. A compaction that achieved nothing is exactly
        # the one whose price an operator needs to see.
        msgs = _history(6)
        fake, _calls = _fake_oai(summary="   ")
        led = _Rec()
        self.assertFalse(
            _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led))
        self.assertEqual(len(led.calls), 1)
        self.assertEqual(led.events, [])  # ...but no event: nothing was compacted

    def test_a_raised_create_bills_nothing(self):
        # No usage block to read, and the SDK may never have reached the server.
        msgs = _history(6)
        fake, _calls = _fake_oai(fail=True)
        led = _Rec()
        self.assertFalse(
            _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led))
        self.assertEqual(led.calls, [])

    def test_nothing_to_compact_bills_nothing(self):
        msgs = _history(3)
        fake, _calls = _fake_oai()
        led = _Rec()
        self.assertFalse(
            _compact.compact_messages(fake, "m", msgs, keep_turns=5, ledger=led))
        self.assertEqual(led.calls, [])

    def test_cost_is_recorded_before_the_event(self):
        # The event quotes the cost, so the bill has to be taken first. Recording the
        # other way round yields an event whose `cost` is always 0.0.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec(cost=0.00042)
        _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led)
        self.assertEqual(led.seq, ["call", "event"])

    def test_a_real_ledger_satisfies_the_duck_type(self):
        # Anti-drift: `_Rec` stands in for `CostLedger` everywhere above, and nothing
        # else in this file would notice the two disagreeing about `record`'s
        # signature or `bucket=`'s effect. Importing `_agent` in the TEST is fine --
        # the "imports nothing from `_agent`" rule constrains the module, not the suite.
        from venice.commands import _agent
        msgs = _history(6)
        fake, _calls = _fake_oai(
            "summary", usage={"prompt_tokens": 4000, "completion_tokens": 200})
        led = _agent.CostLedger()
        led.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 2.0}})
        self.assertTrue(
            _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led))
        self.assertAlmostEqual(led.bucket_cost("compaction"), 0.0044)
        self.assertEqual(led.buckets["compaction"]["calls"], 1)
        # and the event quotes the same number the bucket accumulated
        self.assertAlmostEqual(led.context_events[0]["cost"], 0.0044)

    def test_the_summary_call_never_enters_the_call_trace(self):
        # It is a fresh prefix -- ~0% cached every time -- so inlining it would
        # manufacture the cache cliff the trace exists to detect.
        from venice.commands import _agent
        msgs = _history(6)
        fake, _calls = _fake_oai(
            "summary", usage={"prompt_tokens": 4000, "completion_tokens": 200})
        led = _agent.CostLedger()
        led.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 2.0}})
        _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led)
        self.assertEqual(
            (led.api_calls(), led.api_calls_total, led.calls_dropped()), ([], 0, 0))
        self.assertEqual(led.prompt_tokens, 0)

    def test_observed_before_is_the_pre_reset_budget_value(self):
        # THE ordering guard, seen from the automatic gate. `compact_messages` clears
        # `last_prompt_tokens` on its way out (#116); recording after that reset would
        # silently null every automatic event, leaving the estimate as the only number.
        # `TestBudgetReset.test_the_event_is_recorded_before_the_reset` is the direct
        # twin -- this one additionally pins that the reset survives the gate wrapper.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        b = _compact.Budget(threshold_tokens=1, keep_turns=2)
        b.last_prompt_tokens = 88110
        self.assertTrue(_compact.maybe_compact(fake, "m", msgs, b, {}, ledger=led))
        self.assertEqual(led.events[0]["observed_tokens_before"], 88110)
        self.assertEqual(led.events[0]["trigger"], "auto")
        self.assertIsNone(b.last_prompt_tokens)  # and it IS reset afterwards

    def test_without_a_budget_the_measured_number_is_null(self):
        # `/compact` with auto-compact off: the estimate must not wear a
        # measurement's name.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        _compact.compact_messages(fake, "m", msgs, keep_turns=2, ledger=led)
        ev = led.events[0]
        self.assertIsNone(ev["observed_tokens_before"])
        self.assertIn("est_tokens_before", ev)

    def test_an_under_budget_gate_records_nothing(self):
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        b = _compact.Budget(threshold_tokens=10_000_000, keep_turns=2)
        self.assertFalse(_compact.maybe_compact(fake, "m", msgs, b, {}, ledger=led))
        self.assertEqual(led.events, [])


class TestBudgetReset(unittest.TestCase):
    """#116: `compact_messages` owns the post-compaction budget reset.

    It used to be hand-copied into `maybe_compact` and into `/compact`, so nothing
    forced the two to agree and a third compaction site would have inherited the event
    recording for free while silently missing the reset.
    """

    def _budget(self, observed=88110):
        b = _compact.Budget(threshold_tokens=1, keep_turns=2)
        b.last_prompt_tokens = observed
        return b

    def test_compact_messages_clears_the_stale_observed_count(self):
        # Called DIRECTLY, not through `maybe_compact` -- that is the whole point of
        # the move, and it is the `/compact` path.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        b = self._budget()
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=2, budget=b))
        self.assertIsNone(b.last_prompt_tokens)

    def test_the_reset_happens_without_a_ledger(self):
        # The reset must not be nested inside the `if ledger is not None` block that
        # records the event: a `/compact` in a session with a budget and no ledger
        # still has a stale count to clear.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        b = self._budget()
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=2, ledger=None, budget=b))
        self.assertIsNone(b.last_prompt_tokens)

    def test_a_failed_compaction_leaves_the_observed_count_alone(self):
        # History is untouched on failure, so the observed count still describes it.
        for kw in ({"fail": True}, {"summary": "   "}):
            with self.subTest(**kw):
                msgs = _history(6)
                fake, _calls = _fake_oai(**kw)
                b = self._budget()
                self.assertFalse(_compact.compact_messages(
                    fake, "m", msgs, keep_turns=2, budget=b))
                self.assertEqual(b.last_prompt_tokens, 88110)

    def test_nothing_to_compact_leaves_the_observed_count_alone(self):
        msgs = _history(3)
        fake, _calls = _fake_oai()
        b = self._budget()
        self.assertFalse(_compact.compact_messages(
            fake, "m", msgs, keep_turns=5, budget=b))
        self.assertEqual(b.last_prompt_tokens, 88110)

    def test_the_event_is_recorded_before_the_reset(self):
        # The direct-call twin of `test_observed_before_is_the_pre_reset_budget_value`:
        # that one routes through `maybe_compact`, which no longer contains the reset,
        # so it can no longer see a reset hoisted above the recording.
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        led = _Rec()
        b = self._budget()
        self.assertTrue(_compact.compact_messages(
            fake, "m", msgs, keep_turns=2, ledger=led, budget=b))
        self.assertEqual(led.events[0]["observed_tokens_before"], 88110)
        self.assertIsNone(b.last_prompt_tokens)

    def test_no_budget_is_not_an_error(self):
        msgs = _history(6)
        fake, _calls = _fake_oai("summary")
        self.assertTrue(_compact.compact_messages(fake, "m", msgs, keep_turns=2))


# Stays at the BOTTOM. It used to sit above `TestCompactionEvents`, which meant a direct
# `python tests/test_compact.py` ran `unittest.main()` before that class was defined and
# silently skipped every test below it -- 23 collected instead of 31. `make test` uses
# `python -m unittest`, which imports the module and runs them all, so CI was green and
# the gap was invisible from the only place anyone looked.
if __name__ == "__main__":
    unittest.main()
