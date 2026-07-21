"""Unit tests for context compaction (issue #48).

Covers the pure helpers in `_compact` (token estimate, group-boundary split,
synthetic-message shape), the best-effort `compact_messages` turn, and the
`Budget` usage tracker. All OpenAI calls are faked -- no network, no key.
"""
import unittest
from unittest import mock

from venice.commands import _compact


def _fake_oai(summary="A concise summary.", fail=False):
    """A fake `oai` whose create() returns a canned summary (or raises)."""
    calls = []

    def _create(**kw):
        calls.append(kw)
        if fail:
            raise RuntimeError("boom")
        msg = mock.MagicMock()
        msg.content = summary
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock(message=msg)]
        return resp

    fake = mock.MagicMock()
    fake.chat.completions.create.side_effect = _create
    return fake, calls


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


if __name__ == "__main__":
    unittest.main()
