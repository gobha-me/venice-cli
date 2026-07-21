"""Context-window compaction for long chat/code sessions (issue #48).

`run_loop` and the REPL only ever append to `messages`; nothing prunes. This
module adds the missing half: estimate how large the history has grown, and
when it crosses a budget, summarize the older prefix into one synthetic
message so the session can continue instead of dying on an over-long prompt.

Design constraints (repo-wide):

- **Stdlib only, no SDK imports.** The OpenAI client is passed in and only its
  ``chat.completions.create`` is called, mirroring how `_agent.run_loop` stays
  SDK-agnostic. No tokenizer dependency: token counts are *estimated* from
  character counts, or taken from the server's own `usage` block when a
  response supplies one (see :class:`Budget`).
- **Tool-call pairing is preserved.** Trimming never orphans a ``tool``
  message from the assistant ``tool_calls`` turn that produced it: messages
  are cut on *group* boundaries (:func:`_groups`), where an assistant message
  and the tool-result messages that answer it move together.
- **Non-destructive.** Compaction mutates the live history in place (so the
  REPL's rollback markers and `run_loop`'s appends keep working), but a failed
  summarization call leaves the history unchanged -- compaction is an
  optimization, never a fatal error.
- **The summary is a system message**, inserted after the real system prompt.
  It carries no ``tool_calls``/``tool_call_id`` plumbing, so the message
  contract of the kept tail is untouched.

The summarization turn reuses the session's own model with
``tool_choice="none"`` -- the same pattern as `venice code`'s plan/verify
turns (`code.py`), and the answer to the issue's open question #2 (session
model vs a fixed cheap one): the session model preserves the conversation's
style and language, and this CLI has no separate cheap tier to hardcode.

Two behavioral notes:

- **Compaction can re-fire.** The trigger is the *observed* prompt size; if a
  run keeps appending large tool results, the history can re-cross the
  threshold after a compaction and summarize again. That's intended (each
  compaction buys headroom), not a bug.
- **Where it runs.** Auto-compaction hooks `run_loop` (per turn and before the
  forced-final turn) and the REPL's turn runner, so `venice chat -i` and
  `venice code -i` are covered. `venice code`'s one-shot plan/verify turns
  (`code.py:_no_tool_turn`) are outside `run_loop` and not compacted -- a
  one-shot's history rarely outgrows the window before its execute phase, and
  keeping this module decoupled from `code.py`'s flow is the v1 trade-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Rough chars-per-token for English/code text. Deliberately conservative
# (overestimate tokens) so the fallback triggers compaction a little early
# rather than a little late; real counts from `usage` override it anyway.
CHARS_PER_TOKEN = 4
# Fixed per-message overhead the API charges (role, separators, tool metadata).
_PER_MESSAGE_TOKENS = 4
# Cap on the summary's own length, so a pathological prefix can't make the
# summarization request itself overflow.
SUMMARY_MAX_TOKENS = 1024

DEFAULT_THRESHOLD_TOKENS = 100_000
DEFAULT_KEEP_TURNS = 10

_SUMMARY_PREFIX = "[Summary of earlier conversation]"
_INSTRUCT = (
    "Summarize the conversation so far into a compact brief for continuing it. "
    "Keep: decisions made, file paths and identifiers mentioned, code changes, "
    "pending tasks, and user preferences. Drop: chit-chat, redundant tool "
    "output, and anything later messages make obsolete. Reply with the summary "
    "only -- no preamble, no headers beyond short labels."
)


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
def _content_chars(msg: dict) -> int:
    c = msg.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):  # OpenAI content parts
        return sum(len(str(p.get("text", ""))) for p in c if isinstance(p, dict))
    return 0


def estimate_tokens(messages: List[dict]) -> int:
    """A conservative token estimate for a message list (no tokenizer dep).

    Counts content characters / CHARS_PER_TOKEN plus a per-message overhead,
    and folds in ``tool_calls`` argument JSON (which the API bills as prompt
    tokens too).
    """
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        total += _PER_MESSAGE_TOKENS
        total += math.ceil(_content_chars(m) / CHARS_PER_TOKEN)
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict):
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        total += math.ceil(len(args) / CHARS_PER_TOKEN)
    return total


@dataclass
class Budget:
    """The auto-compact budget: when to fire and how much tail to keep.

    `threshold_tokens` / `keep_turns` are the configured knobs.
    `last_prompt_tokens` is filled in from a response's `usage` block via
    :meth:`observe` -- the server's own count of the prompt we just sent,
    which is the ground truth the heuristic only approximates.
    """

    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS
    keep_turns: int = DEFAULT_KEEP_TURNS
    last_prompt_tokens: Optional[int] = None

    def observe(self, usage) -> None:
        """Record prompt tokens from a response's `usage` (dict or SDK obj)."""
        if usage is None:
            return
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            if isinstance(pt, (int, float)):
                self.last_prompt_tokens = int(pt)

    def over(self, messages: List[dict]) -> bool:
        """True when the history has crossed the compaction threshold.

        Prefers the last observed server count when available (it's exact and
        already includes system/tool overhead); else falls back to the
        character heuristic.
        """
        if self.threshold_tokens <= 0:
            return False  # auto-compact disabled
        if self.last_prompt_tokens is not None:
            return self.last_prompt_tokens >= self.threshold_tokens
        return estimate_tokens(messages) >= self.threshold_tokens


def budget_from_args(args) -> Optional["Budget"]:
    """The auto-compact Budget for a parsed-args namespace, or None when it
    isn't opted into (#48).

    Enabled by ``--auto-compact`` / ``defaults.<cmd>.auto_compact``; threshold
    and keep-turns fall back to the module defaults when unset (argparse leaves
    them None). Shared by every command surface (chat REPL, chat --tools, code)
    so opting in behaves identically everywhere.
    """
    if not getattr(args, "auto_compact", False):
        return None
    return Budget(
        threshold_tokens=(
            getattr(args, "compact_threshold", None) or DEFAULT_THRESHOLD_TOKENS
        ),
        keep_turns=(
            getattr(args, "compact_keep_turns", None) or DEFAULT_KEEP_TURNS
        ),
    )


# --------------------------------------------------------------------------- #
# Splitting on group boundaries (never orphan a tool result)
# --------------------------------------------------------------------------- #
def _groups(messages: List[dict]) -> List[List[dict]]:
    """Group the non-system tail into conversation turns.

    A group is one *exchange*: a user message plus the assistant turns that
    answer it -- including the tool-call round-trips in between (an assistant
    message and the ``tool`` messages answering its ``tool_calls`` stay glued
    together). A stray leading assistant message (e.g. a resumed transcript
    that starts mid-conversation) forms its own group. Cutting only on group
    boundaries guarantees a ``tool`` message is never separated from the
    assistant turn that produced its ``tool_call_id``, and a kept turn never
    strands the assistant's reply.
    """
    groups: List[List[dict]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "user":
            group = [m]
            i += 1
            # Absorb assistant turns (with their tool results) up to the next
            # user message -- those are this exchange's answer.
            while i < n and messages[i].get("role") != "user":
                group.append(messages[i])
                i += 1
            groups.append(group)
        else:
            # Assistant (with its tool results) or a standalone message.
            group = [m]
            i += 1
            if m.get("role") == "assistant":
                while i < n and messages[i].get("role") == "tool":
                    group.append(messages[i])
                    i += 1
            groups.append(group)
    return groups


def split_for_compaction(
    messages: List[dict], keep_turns: int
) -> Optional[Tuple[List[dict], List[dict]]]:
    """Split history into (prefix to summarize, tail to keep verbatim).

    The split preserves the system prefix (leading system messages stay out of
    both halves -- they're kept separately) and cuts the rest on group
    boundaries so at most `keep_turns` conversation turns remain. Returns None
    when there's nothing worth summarizing (too few turns).
    """
    if keep_turns < 1:
        keep_turns = 1
    sys_end = 0
    while sys_end < len(messages) and messages[sys_end].get("role") == "system":
        sys_end += 1
    tail_groups = _groups(messages[sys_end:])
    if len(tail_groups) <= keep_turns:
        return None
    cut = len(tail_groups) - keep_turns
    prefix: List[dict] = []
    for g in tail_groups[:cut]:
        prefix.extend(g)
    tail: List[dict] = []
    for g in tail_groups[cut:]:
        tail.extend(g)
    return prefix, tail


# --------------------------------------------------------------------------- #
# The summarization turn
# --------------------------------------------------------------------------- #
def build_summary_prompt(prefix: List[dict]) -> List[dict]:
    """A fresh, self-contained message list for the summarization call."""
    transcript = []
    for m in prefix:
        role = m.get("role", "?")
        text = m.get("content")
        if not isinstance(text, str) or not text:
            if role == "tool":
                text = "(tool result)"
            elif m.get("tool_calls"):
                names = [
                    (tc.get("function") or {}).get("name", "?")
                    for tc in m.get("tool_calls", [])
                    if isinstance(tc, dict)
                ]
                text = "(called tools: %s)" % ", ".join(names)
            else:
                text = ""
        transcript.append(f"{role}: {text}")
    return [
        {"role": "system", "content": _INSTRUCT},
        {"role": "user", "content": "\n".join(transcript)},
    ]


def synthetic_message(summary: str) -> dict:
    """The system-role message a summary rides in on the compacted history."""
    return {"role": "system", "content": f"{_SUMMARY_PREFIX}\n{summary.strip()}"}


def compact_messages(
    oai,
    model: str,
    messages: List[dict],
    *,
    keep_turns: int = DEFAULT_KEEP_TURNS,
    base_kwargs: Optional[dict] = None,
) -> bool:
    """Summarize the older prefix in place; keep system + last `keep_turns`.

    Returns True when the history was compacted, False when there was nothing
    to do or the summarization call failed (in which case `messages` is left
    untouched). Only the summary text is taken from the response; the model's
    own wording is never trusted with roles.
    """
    split = split_for_compaction(messages, keep_turns)
    if split is None:
        return False
    prefix, tail = split
    sys_msgs = messages[: len(messages) - len(prefix) - len(tail)]

    kwargs = dict(base_kwargs or {})
    kwargs.pop("stream", None)
    kwargs.pop("stream_options", None)
    kwargs.pop("tools", None)
    kwargs.setdefault("max_tokens", SUMMARY_MAX_TOKENS)
    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=build_summary_prompt(prefix),
            tool_choice="none",
            **kwargs,
        )
    except Exception:
        return False  # compaction is best-effort; the run continues un-compacted
    summary = ""
    if getattr(resp, "choices", None):
        summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        return False

    messages[:] = sys_msgs + [synthetic_message(summary)] + tail
    return True


def maybe_compact(oai, model: str, messages: List[dict],
                  budget: Optional[Budget], base_kwargs: Optional[dict] = None,
                  on_compact=None) -> bool:
    """Compact `messages` in place when `budget` says they're over budget.

    The shared gate for every compaction site (`run_loop`'s per-turn check, its
    forced-final turn, and the REPL). `budget=None` (auto-compact off) or an
    under-budget history is a no-op. After a successful compaction the observed
    prompt-token count is stale, so it's reset (the next turn re-observes).
    `on_compact(before, after)` is invoked on success (for progress output).
    Returns True iff the history was compacted.
    """
    if budget is None or not budget.over(messages):
        return False
    before = len(messages)
    if not compact_messages(
        oai, model, messages,
        keep_turns=budget.keep_turns, base_kwargs=base_kwargs,
    ):
        return False
    budget.last_prompt_tokens = None  # stale after compaction
    if on_compact is not None:
        on_compact(before, len(messages))
    return True
