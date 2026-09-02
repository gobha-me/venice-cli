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
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import _context_archive, _openai

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
LOSS_POLICY_CHOICES = ("aggressive", "evidence")

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
    loss_policy: str = "aggressive"
    archive: Optional[_context_archive.ContextArchive] = None
    protected_system_messages: Optional[int] = None

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


def budget_from_args(args, archive=None) -> Optional["Budget"]:
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
        loss_policy=getattr(args, "compact_loss_policy", None) or "aggressive",
        archive=archive,
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


def leading_system_count(messages: List[dict]) -> int:
    """Count the contiguous authoritative/generated system prefix."""
    count = 0
    while count < len(messages) and messages[count].get("role") == "system":
        count += 1
    return count


def split_for_compaction(
    messages: List[dict], keep_turns: int, protected_system_messages: Optional[int] = None
) -> Optional[Tuple[List[dict], List[dict]]]:
    """Split history into (prefix to summarize, tail to keep verbatim).

    The split preserves the system prefix (leading system messages stay out of
    both halves -- they're kept separately) and cuts the rest on group
    boundaries so at most `keep_turns` conversation turns remain. Returns None
    when there's nothing worth summarizing (too few turns).
    """
    if keep_turns < 1:
        keep_turns = 1
    leading_systems = leading_system_count(messages)
    # None is the conservative compatibility path: every leading system
    # message is authoritative. Session-aware callers persist the exact count,
    # which lets later generated summary/index messages be replaced without
    # classifying authority from forgeable content text.
    if protected_system_messages is None:
        sys_end = leading_systems
    else:
        sys_end = min(max(0, protected_system_messages), leading_systems)
    generated_end = leading_systems
    tail_groups = _groups(messages[generated_end:])
    if len(tail_groups) <= keep_turns:
        return None
    cut = len(tail_groups) - keep_turns
    prefix: List[dict] = list(messages[sys_end:generated_end])
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
    ledger=None,
    budget: Optional[Budget] = None,
    trigger: str = "auto",
    loss_policy: str = "aggressive",
    archive: Optional[_context_archive.ContextArchive] = None,
    protected_system_messages: Optional[int] = None,
) -> bool:
    """Summarize the older prefix in place; keep system + last `keep_turns`.

    Returns True when the history was compacted, False when there was nothing
    to do or the summarization call failed (in which case `messages` is left
    untouched). Only the summary text is taken from the response; the model's
    own wording is never trusted with roles.

    #99: when `ledger` is given, a successful compaction is logged to it as a context
    event. Recorded HERE, in the worker, rather than at the compaction sites -- the gate
    (`maybe_compact`) is only three of the four, `/compact` calls straight into this
    function, and copying the bookkeeping into each site is the shape that lets one
    site quietly forget it. (Two callers reach this function; four sites compact.)
    `ledger` is duck-typed: this module imports nothing from `_agent` and does not need
    to. `trigger` distinguishes the automatic gate from a hand-typed `/compact`, which
    answers "was this sawtooth self-inflicted".

    #101: the summarization call is billed to the ledger's "compaction" bucket, off to
    the side of the main-loop counters, and its cost rides the event. Billed on API
    success rather than compaction success: an empty summary still spent the tokens.

    #116: `budget` is now MUTATED, not merely read -- a successful compaction clears
    `last_prompt_tokens`, which the same argument moves in here. It was hand-copied at
    both call sites, so a third one added later would have inherited the event for free
    and silently missed the reset, leaving `Budget.over` reading a count larger than the
    history it now describes -- i.e. re-firing compaction immediately, a sawtooth #99's
    trace would show and nothing would prevent.
    """
    split = split_for_compaction(
        messages, keep_turns, protected_system_messages=protected_system_messages
    )
    if split is None:
        return False
    prefix, tail = split
    sys_msgs = messages[: len(messages) - len(prefix) - len(tail)]

    # Evidence mode refuses before spending a summarization call if the exact
    # removed messages cannot fit. Staging is side-effect-free; commit happens
    # only after a usable summary exists, alongside the live-history rewrite.
    staged = []
    if loss_policy == "evidence":
        if archive is None:
            return False
        archive.last_error = None
        try:
            staged = archive.stage(prefix)
        except _context_archive.ArchiveError as e:
            archive.last_error = str(e)
            return False

    # #128: compaction is a deliberately fresh summarization conversation. Reusing
    # the parent session's routing identity would mix unrelated prefixes on one cache
    # affinity key; preserve all other generation/Venice parameters while stripping it.
    kwargs = _openai.without_prompt_cache_key(base_kwargs)
    kwargs.pop("stream", None)
    kwargs.pop("stream_options", None)
    kwargs.pop("tools", None)
    kwargs.setdefault("max_tokens", SUMMARY_MAX_TOKENS)
    _t0 = time.monotonic()
    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=build_summary_prompt(prefix),
            tool_choice="none",
            **kwargs,
        )
    except Exception:
        # Nothing recorded on this path, deliberately: there is no usage block to read,
        # and the SDK may have retried or never reached the server at all. `record()`'s
        # row-on-every-path rule is about calls the caller KNOWS completed.
        return False  # compaction is best-effort; the run continues un-compacted
    # #101: billed on API SUCCESS, not on compaction success -- note this sits ABOVE the
    # empty-summary return below. The call has completed and the tokens are spent
    # whether or not the summary that comes back is usable; a compaction that achieved
    # nothing is precisely the one an operator needs to see the price of.
    #
    # `bucket=` keeps it out of the main-loop counters and out of the per-call trace:
    # this is a FRESH prefix (system + a flattened transcript), so it reads ~0% cached
    # every time and would fabricate a cache cliff in the one artifact that exists to
    # detect real ones.
    cost = 0.0
    if ledger is not None:
        cost = ledger.record(
            getattr(resp, "usage", None),
            seconds=time.monotonic() - _t0,
            bucket="compaction",
        )
    summary = ""
    if getattr(resp, "choices", None):
        summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        return False

    # #99: measure BEFORE the rewrite below, then record AFTER it, and only on this
    # success path. The module's contract is that a failed summarization leaves history
    # untouched, so an event row for a compaction that did not happen would be a fresh
    # lie in the artifact that exists to stop one.
    est_before = estimate_tokens(messages)
    msgs_before = len(messages)
    replacement = sys_msgs + [synthetic_message(summary)]
    if loss_policy == "evidence":
        archive.commit(staged)
        replacement.append(archive.live_index_message())
    messages[:] = replacement + tail
    if ledger is not None:
        ledger.record_compaction({
            "trigger": trigger,
            "messages_before": msgs_before,
            "messages_after": len(messages),
            # `est_*` vs `observed_*` carries the measured-vs-estimated distinction in
            # the KEY NAMES rather than in a `measured: true` flag -- cheaper, and it
            # cannot drift out of agreement with the value beside it. Both estimates use
            # the same 4-chars-per-token yardstick, so neither absolute means much but
            # the RATIO between them does.
            "est_tokens_before": est_before,
            "est_tokens_after": estimate_tokens(messages),
            # The server-reported prompt size of the PREVIOUS call -- a LOWER BOUND on
            # what the next one would have cost, since the history grew by an assistant
            # reply and its tool results after that number was observed. None when there
            # is no budget (the `/compact`-without-`--auto-compact` shape), which is
            # exactly the case where an unlabelled number would read as measured.
            "observed_tokens_before": (
                None if budget is None else budget.last_prompt_tokens
            ),
            # #101: what this compaction COST to perform -- the summarization call
            # above, which #99 shipped unledgered and which this key finally prices.
            #
            # #99 declined the field on the grounds that tokens-saved beside a cost
            # invites "the compaction paid for itself". That objection was about the
            # RENDERING, and it is answered there: the marker line reads "$0.0031 to
            # summarize", which cannot be misread as a saving. Per-event rather than
            # bucket-only because the bucket totals every compaction, and which ONE of
            # them was expensive is the question a sawtooth makes you ask.
            "cost": round(cost, 6),
        })
    # #116: the LAST piece of post-compaction bookkeeping to move in here, and it must
    # stay BELOW the event above -- `observed_tokens_before` reads the value this line
    # clears, so a reset hoisted above the recording silently nulls every automatic
    # event and leaves the estimate as the only number. Deliberately NOT nested inside
    # the `if ledger is not None` block one line up: a `/compact` in a session with a
    # budget and no ledger still has a stale count to clear.
    if budget is not None:
        budget.last_prompt_tokens = None  # stale after compaction
    return True


def maybe_compact(oai, model: str, messages: List[dict],
                  budget: Optional[Budget], base_kwargs: Optional[dict] = None,
                  on_compact=None, on_blocked=None, ledger=None) -> bool:
    """Compact `messages` in place when `budget` says they're over budget.

    The shared gate for every compaction site (`run_loop`'s per-turn check, its
    forced-final turn, and the REPL). `budget=None` (auto-compact off) or an
    under-budget history is a no-op. `on_compact(before, after)` is invoked on
    success (for progress output). Returns True iff the history was compacted.

    #99/#116: `ledger` and `budget` are forwarded, not consumed here. Everything that
    must happen when a compaction succeeds -- the event, the bill, and clearing the now
    stale observed prompt-token count -- belongs to `compact_messages`, because this
    gate is only three of the four compaction sites and `/compact` calls straight past
    it. `on_compact` stays here: it is presentation (the REPL and `run_loop` word it
    differently), not an invariant.
    """
    if budget is None or not budget.over(messages):
        return False
    before = len(messages)
    if not compact_messages(
        oai, model, messages,
        keep_turns=budget.keep_turns, base_kwargs=base_kwargs,
        ledger=ledger, budget=budget, trigger="auto",
        loss_policy=budget.loss_policy, archive=budget.archive,
        protected_system_messages=budget.protected_system_messages,
    ):
        if (on_blocked is not None and budget.archive is not None
                and budget.archive.last_error):
            on_blocked(budget.archive.last_error)
        return False
    if on_compact is not None:
        on_compact(before, len(messages))
    return True
