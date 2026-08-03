"""Function-calling agent loop for `venice chat --tools` (issue #15).

`venice chat` is normally one-shot. With `--tools`, the model can invoke venice's
own endpoints as **in-process** function tools and the completion runs in a loop
(model -> tool_calls -> dispatch -> tool results -> repeat until it stops). This is
the self-contained-agent foundation for the vcoder epic (#25).

Import discipline: this module reuses the print-free `*_tool` primitives in
``commands._mcp`` but NEVER imports the ``mcp``/FastMCP SDK -- the whole point of
#15 is that the agent loop needs only the ``[openai]`` extra. (`_mcp` is itself
import-clean, so pulling it in at CLI startup is cheap and mcp-free.)

Safety invariant: the loop-controlled kwargs ``confirm`` / ``max_spend`` /
``output_dir`` are injected by this module and are DELIBERATELY absent from the
advertised JSON schemas, so the model can never raise its own spending authority.
The spend gate lives inside each `_mcp.*_tool` (`check_spend`); here we only decide
what to pass and how to resolve an over-cap `confirmation_required`.

Extension point for #21 (external MCP client): the loop depends only on a
``list[Tool]`` plus :func:`dispatch_map` -- it never references `_mcp` directly.
#21 adds a sibling factory (``mcp_client_tools(session) -> list[Tool]``) whose
`Tool.invoke` routes to a remote server, concatenates it with :func:`builtin_tools`,
and passes the combined list to :func:`run_loop`. Nothing in the loop changes.
"""
from __future__ import annotations

import collections
import contextlib
import io
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

# Aliased: a bare `config` would be shadowed by the `config=None` keyword argument
# that `browser_tools` (and its callers) already take further down this module.
from .. import config as _config
from .. import userconfig
from . import _exec
from . import _mcp
from . import _memory
from . import _models
from . import _compact
from .models import MODEL_TYPES


# --------------------------------------------------------------------------- #
# Tool descriptor + derived structures (pure functions of a list[Tool])
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tool:
    """One function tool the model may call.

    ``invoke(arguments, *, confirm=False) -> dict`` takes the model-supplied
    arguments object and returns a JSON-serializable result dict. ``paid`` marks
    tools whose result can be a ``confirmation_required`` gate.

    ``category`` (e.g. ``image``/``fs``/``exec``) and ``tags`` are the capability
    axis (#50): a runtime label carried by every built tool so callers can filter a
    ``list[Tool]`` by capability. It is ORTHOGONAL to which surface advertises the
    tool (that split lives in the ``_BUILTINS``/``_CODE_ASSET_BUILTINS`` registries).
    The registry-level selectors :func:`select`/:func:`tools_in` read the same
    categories over the built-in registry; category is empty on tools with no
    registry row (e.g. remote MCP tools).
    """

    name: str
    description: str
    parameters: dict
    invoke: Callable[..., dict]
    paid: bool = False
    category: str = ""
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentProfile:
    """The seeded values that make ``venice chat`` and ``venice code`` two faces of
    one agent core (#51).

    Both commands already share the engine (:func:`run_loop` + ``_repl.run``); they
    differ only in the values seeded here. Formalizing that difference as a profile
    de-dups the surfaces and gives the multi-agent epic (#52) a clean spawn contract:
    a subagent is "run the core with profile ``P`` + task ``T``".

    ``build_gen_kwargs``/``build_system`` are injected by the owning command module
    (they reference command-local helpers), so this type stays import-clean —
    ``_agent`` never imports ``chat``/``code``. The tool axis is deliberately *not* an
    executable field here (see the ticket): chat's REPL must derive tools from
    ``args`` while code injects a prebuilt session, and ``injects_tools_session``
    records that policy without forcing either command to restructure. The executable
    tool-builder belongs to #52's non-interactive ``spawn`` core.
    """

    name: str  # session command key: "chat" | "code"
    label: str  # "venice chat" | "venice code"
    build_gen_kwargs: Callable[..., dict]  # (args) -> per-turn gen kwargs
    build_system: Callable[..., Optional[str]]  # (args, root, tools) -> system prompt
    default_max_tool_calls: int  # 8 | 25
    plan_mode: bool = False  # code's plan/accept/verify harness
    degrade_to_chat: bool = True  # non-FC model: True=plain chat, False=exit 2
    system_reseed: bool = False  # rebuild leading system message on resume
    injects_tools_session: bool = False  # code injects a prebuilt tools_session; chat must not


def wants_interactive(args, initial) -> bool:
    """Whether a chat/code command should enter the REPL: explicitly requested
    (``-i`` / ``--resume`` / ``--continue``), or no initial message/task and stdin is
    an interactive terminal. A piped or ``-`` initial is always one-shot. Shared by
    both commands so the two profiles decide interactivity identically (#51)."""
    if getattr(args, "interactive", False) or getattr(args, "resume", None) \
            or getattr(args, "cont", None):
        return True
    return initial is None and sys.stdin.isatty()


def check_function_calling(models, model, *, label, degraded_tail, unverified_tail,
                           degrade):
    """Shared non-function-calling capability gate (#51).

    Prints the same capability notes each command always printed. ``label`` is the
    command name (``chat``/``code``), and the two ``*_tail`` strings carry the
    per-profile wording. Returns ``(ok, rc)``: ``ok`` True means proceed with tools;
    ``ok`` False means the caller should surface ``rc`` -- ``None`` when the profile
    degrades to plain chat (``degrade=True``), else exit-code ``2``. An unverifiable
    (``None``) result prints a soft note and proceeds."""
    supported = supports_function_calling(models, model)
    if supported is False:
        print(
            f"{label}: model {model} does not support function calling; {degraded_tail}",
            file=sys.stderr,
        )
        return (False, None if degrade else 2)
    if supported is None:
        print(
            f"{label}: could not verify function-calling support for {model}; "
            f"{unverified_tail}",
            file=sys.stderr,
        )
    return (True, None)


def to_openai_tools(tools: List[Tool]) -> List[dict]:
    """Render tools as an OpenAI-compatible ``tools`` array for /chat/completions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


# --------------------------------------------------------------------------- #
# Cost ledger (#66): meter chat-completion spend across an agent run.
#
# Paid *tools* are already spend-gated (`check_spend` in `_mcp`); the model
# calls themselves were not. This ledger accumulates per-turn cost from the
# server-reported `usage` block and the catalog's per-1M-token pricing, so a
# session `--max-spend` can stop a runaway loop. Accounting is post-response
# (chat pricing is dynamic; there is no pre-call quote), so the gate fires
# *between* turns: once accumulated cost crosses the cap, no new paid turn
# starts and the loop forces a final answer (mirroring --max-tool-calls).
# --------------------------------------------------------------------------- #
def _usd_per_token(pricing, key) -> Optional[float]:
    """`pricing.<key>.usd` as a per-token rate (catalog prices are per 1M)."""
    if not isinstance(pricing, dict):
        return None
    node = pricing.get(key)
    if isinstance(node, dict) and isinstance(node.get("usd"), (int, float)):
        return float(node["usd"]) / 1_000_000.0
    return None


def _as_int(v) -> int:
    """A non-negative int from a usage field; 0 for None/garbage/negative.

    `bool` is an `int` subclass but is never a real token count, so it's garbage.
    """
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return int(v) if v > 0 else 0
    return 0


def _as_float(v) -> float:
    """A non-negative float from a stored cost field; 0.0 for None/garbage/negative.

    Mirrors :func:`_as_int` (bool is never a real cost); used when restoring a
    persisted ledger snapshot (#47) from a possibly hand-edited envelope.
    """
    if isinstance(v, bool):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else 0.0
    return 0.0


def _detail(usage: dict, section: str, key: str):
    """A nested usage sub-field (e.g. ``prompt_tokens_details.cached_tokens``).

    The ``*_details`` blocks are nullable in the API, so guard the middle level;
    returns None when the block is absent or not a dict.
    """
    block = usage.get(section)
    if isinstance(block, dict):
        return block.get(key)
    return None


#: The cache sub-fields we recognize inside ``prompt_tokens_details``, in priority
#: order. `reference/venice-openapi.yaml` documents exactly two (`cached_tokens`,
#: `cache_creation_input_tokens`); `cache_write_tokens` is OBSERVED on live
#: kimi/glm/deepseek responses but undocumented, so it is read as a fallback and
#: must never take precedence over the documented name. Deliberately NOT widened
#: to the Anthropic/DeepSeek-native top-level aliases (`cache_read_input_tokens`,
#: `prompt_cache_hit_tokens`): Venice never emits them, and an additive top-level
#: block would hit the clamp in `record()` -- truncating the write bucket to 0
#: while `CostLedger`'s "reported" flag swore it was measured, which is a NEW lie.
#: `_cache_tokens`'s three-state is the honest degradation for a rename (#98).
_CACHE_READ_KEYS = ("cached_tokens",)
_CACHE_WRITE_KEYS = ("cache_creation_input_tokens", "cache_write_tokens")


def _cache_tokens(usage: dict, keys) -> Optional[int]:
    """A cache token count from ``prompt_tokens_details``; None when unreported (#98).

    THREE-STATE ON PURPOSE, and that is the whole point of #98. `None` means "this
    response carried no such field at all", which is NOT the same as a reported zero:
    the spec marks `prompt_tokens_details` ``nullable: true`` and leaves it out of
    `required` (its own glm example ships ``prompt_tokens_details: null``), so
    coercing absence to 0 fabricates a "cache hit rate: 0.0%" that reads as a real
    measurement. A printed 0.0% must mean the provider said zero.

    Present-but-null and present-but-garbage both count as ABSENT -- a value we cannot
    read is not a measurement (mirrors :func:`_as_int`'s "a bool is never a token
    count" rule). :func:`_detail` is kept as-is for `reasoning_tokens`, which stays
    deliberately two-state: no incident, and no rate hangs off it.
    """
    for key in keys:
        v = _detail(usage, "prompt_tokens_details", key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return _as_int(v)
    return None


def _usage_dict(usage) -> Optional[dict]:
    """Normalize a `usage` block to a plain dict, or None when there isn't one.

    Split out of `record()` (#99) so that method can have exactly ONE return: the
    per-call trace row has to be appended even for the shapes that carry no tallies
    (`None`, an SDK object with no `usage`, a garbage value), and hanging an append
    off each of three early returns is the copy-into-every-exit-path shape that #86
    and #92 removed everywhere else. A call that happened, blocked for 40 seconds and
    came back with no usage block is exactly the row whose seconds matter most.
    """
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return usage if isinstance(usage, dict) else None


def _dump_raw_usage(usage) -> None:
    """Echo one API response's raw `usage` block to stderr when opted in (#98).

    The ledger aggregates, and an aggregate cannot answer "was the field even there?"
    after the fact -- nothing else persists the raw block, so diagnosing a cache
    regression used to mean re-running the incident live.

    Off unless $VENICE_USAGE_RAW is explicitly truthy -- a bare truthiness check would
    turn this ON for ``VENICE_USAGE_RAW=0``. STDERR only: `venice code --json` and every
    piped command must keep stdout machine-readable.

    Exactly ONE `print`, so each record stays an atomic line -- this fires on subagent
    and review ledgers too, which run under a ThreadPoolExecutor, and two writes would
    interleave into garbage under `--parallel`. A diagnostic must never take down the
    turn that produced it, hence the blanket serialization fallback.
    """
    if os.environ.get(_config.ENV_USAGE_RAW, "").strip().lower() not in (
        "1", "true", "yes", "on"
    ):
        return
    try:
        raw = usage.model_dump() if hasattr(usage, "model_dump") else usage
        line = json.dumps(raw, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - a diagnostic must never break the turn
        line = repr(usage)
    print(f"usage-raw: {line}", file=sys.stderr)


def format_duration(seconds) -> str:
    """A compact human duration: ``4.5s``, ``2m 14s``, ``1h 03m`` (#81).

    Sub-minute keeps a decimal -- a REPL turn is usefully "4.5s", not "4s". Above a
    minute the decimal is noise, and past an hour the seconds are. `_queue.progress_tick`
    keeps its own ``{:5.1f}s`` because the fixed width is what stops its carriage-return
    line jittering; a one-shot `venice code` run reporting "1247.3s" just fails to answer
    the question being asked. Garbage/negative -> "0.0s" (mirrors :func:`_as_float`: a bad
    value must never make a report unreadable).
    """
    s = _as_float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60):02d}m"


class CostLedger:
    """Accumulates estimated USD spend for one agent run.

    `max_spend` is the session cap (USD-equivalent; None = unmetered). The
    ledger is bound to a model's pricing on first use via :meth:`bind_pricing`;
    an unknown price means the turn's tokens are counted but not charged
    (degrade gracefully rather than hard-block on a missing price).
    """

    #: #99: rows of the per-call trace kept from the START of the session, frozen once
    #: full. This is the cold-start evidence -- the 08-03 collapse was diagnosable only
    #: because the very first calls were already at 0% -- and a ring buffer would drop
    #: exactly it after a long enough run.
    _CALLS_HEAD = 50
    #: #99: rows kept from the END, as a ring buffer. This is the state an operator is
    #: debugging *now*. Head+tail together are ~250 rows x ~130 bytes = ~32KB, which is
    #: small beside the message transcript already in the same session file.
    _CALLS_TAIL = 200

    def __init__(self, max_spend: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        # A non-positive cap means "uncapped" (mirrors --max-tool-calls 0).
        cap = float(max_spend) if max_spend is not None else None
        self.max_spend = cap if (cap is not None and cap > 0) else None
        # #52: an orthogonal cumulative *token* ceiling (prompt+completion), used by
        # per-subagent runs (`--subagent-max-tokens`). Same non-positive->None rule.
        tcap = int(max_tokens) if max_tokens is not None else None
        self.max_tokens = tcap if (tcap is not None and tcap > 0) else None
        self.total = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Cache buckets kept distinct -- they price differently and collapsing
        # them mis-costs cache-heavy long sessions (#75). Both are subsets of
        # `prompt_tokens`; `reasoning_tokens` is a subset of `completion_tokens`.
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.reasoning_tokens = 0
        self.unpriced = False  # saw a turn whose model price was unknown
        # #98: sticky "at least one recorded turn carried no recognized cache-read
        # (resp. cache-write) field", mirroring `unpriced`. These exist so a rendered
        # 0 can be told apart from an absent one -- the buckets above cannot say
        # "unknown", and `cache hit rate: 0.0%` was being emitted by byte-identical
        # code for a real miss and for a response with no cache block at all. They
        # drive DIFFERENT render sites (see `usage_report`): read-reported-but-
        # write-absent is the normal state for most Venice models, so folding them
        # into one marker would tag essentially every session.
        self.cache_read_unreported = False
        self.cache_write_unreported = False
        # #81: the operator's wall-clock. `elapsed_seconds` is BLOCKED time only --
        # the caller stamps the window and passes the delta, so the ledger never reads
        # a clock and every ledger test stays a pure-function assertion. `turns` counts
        # the windows, so `/usage` can show an average. A "turn" is one time the CLI
        # made you wait (one REPL `_do_turn`, one one-shot `code` run) -- NOT one API
        # call, which `record()` counts and which would make the average meaningless.
        self.elapsed_seconds = 0.0
        self.turns = 0
        # #82: per-tool execution time, name -> {"seconds": float, "calls": int}. Same
        # caller-stamps-the-clock contract as `elapsed_seconds` above, one level down:
        # `run_loop` brackets each `tool.invoke` and hands `record_tool` a delta.
        # A SUBSET of `elapsed_seconds`, never a partition of it -- the model wait,
        # `_compact`'s summary call (#101) and the out-of-loop turns are all wall time
        # that no tool owns. Under `--parallel` overlapping windows make the sum exceed
        # wall outright; `_tool_lines` labels that rather than hiding it.
        self.tools: Dict[str, dict] = {}
        # Serializes `record_tool`'s read-modify-write: under `--parallel` two
        # `venice_spawn` windows close on DIFFERENT pool workers and land on the SAME
        # key, and `row["calls"] += 1` is a load/add/store rather than one bytecode --
        # the same shape `_code._DISPATCH_LOCK` guards for its dispatch append.
        # Honest caveat, measured while writing the tests: CPython's GIL does not
        # actually lose one of these increments even at 16 threads x 20,000 iterations
        # with a 1ns switch interval, so no test can demonstrate the loss on this
        # interpreter (the test pins that the mutation happens under the lock instead).
        # The lock stays because GIL bytecode-boundary behaviour is not a language
        # guarantee and free-threaded builds drop it outright.
        self._tools_lock = threading.Lock()
        # #99: the per-API-call trace. The aggregates above cannot tell a cache that was
        # cold from call 1 apart from one that decayed apart from a compaction sawtooth --
        # all three average to the same low number, and the 08-03 AAR had to infer the
        # shape from the total being *exactly* 0 rather than merely low.
        #
        # HEAD + TAIL, not one list: a ring buffer loses the cold-start evidence (the
        # actual 08-03 failure) and a head-only cap goes blind to the state the operator is
        # debugging right now. Rows carry their own `n`, so the seam between the two is
        # self-describing -- an `n` jumping 50 -> 441 IS the drop marker, and no separate
        # counter has to be kept consistent with it.
        self._calls_head: List[dict] = []
        self._calls_tail = collections.deque(maxlen=self._CALLS_TAIL)
        # The TRUE count, which is also the source of each row's `n`. Deliberately not
        # `len()` of anything: it stays truthful across the cap and across a resume, and it
        # finally persists the number the `turns` comment above says this class counts but
        # never stored. Note "ledgered API calls", NOT "API calls": `_compact`'s summary
        # call (#101) never reaches `record()`, so the rows do not sum to the whole bill.
        # A `context_events` row is the marker that an unledgered call happened there.
        self.api_calls_total = 0
        # #99: prefix-affecting events (compaction today; #104's resume reseed later --
        # hence `kind`, and hence not naming this `compactions`). Uncapped on purpose:
        # a compaction is rare and is the highest-signal row in the trace.
        self.context_events: List[dict] = []
        # No lock here, unlike `tools` above, and the absence is deliberate rather than an
        # oversight: `record_tool` is called from pool workers, but `record()` only ever
        # runs on the thread that owns the ledger (parallel dispatch touches `on_tool`
        # alone, and every subagent gets its own fresh ledger).
        self._in = None          # per-token input rate (USD)
        self._out = None         # per-token output rate (USD)
        self._cache_in = None    # per-token cache-read rate (USD); None -> use _in
        self._cache_write = None  # per-token cache-write rate (USD); None -> use _in

    def bind_pricing(self, pricing) -> None:
        """Set the per-token rates from a catalog `model_spec.pricing` block.

        `cache_input`/`cache_write` are optional (present only for cache-capable
        models); left None they fall back to the plain input rate at cost time.
        """
        self._in = _usd_per_token(pricing, "input")
        self._out = _usd_per_token(pricing, "output")
        self._cache_in = _usd_per_token(pricing, "cache_input")
        self._cache_write = _usd_per_token(pricing, "cache_write")

    def to_dict(self) -> dict:
        """Serialize the running accumulators for cross-resume persistence (#47/#75).

        Only the tallies are stored -- the per-token rates and `max_spend` are
        re-derived from the catalog/cap at construction, so a resumed ledger keeps
        accruing at the *current* model's prices while carrying past totals forward.
        """
        return {
            "total": self.total,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "unpriced": self.unpriced,
            "elapsed_seconds": self.elapsed_seconds,  # #81
            "turns": self.turns,
            "cache_read_unreported": self.cache_read_unreported,    # #98
            "cache_write_unreported": self.cache_write_unreported,  # #98
            # #100: the one DERIVED key here, so a pipeline can alert on a cache
            # collapse without re-deriving it from two counters. `null` when the rate
            # is not knowable -- never a fabricated 0.0 (that is the #98 contract).
            # Rounded because `venice sessions show` prints this dict straight at a
            # human. Deliberately NOT read back by `restore()`; see the note there.
            "cache_hit_percent": self.cache_hit_percent(round_to=1),
            # #82: the SECOND derived key (`cache_hit_percent` is the first) -- the exact
            # number the run footers print, so a pipeline can alert on tool time without
            # re-summing the map below. Deliberately NOT read back by `restore()`.
            "tool_seconds": self.tool_seconds(),
            # #82: name -> {"seconds", "calls"}. Self-describing rather than a positional
            # pair, because this rides the `--json` envelope and the session file that
            # operators jq (`.usage.tools.shell.seconds`). Rows are COPIED so a caller
            # mutating the envelope cannot reach back into the live ledger. Sorted by
            # NAME, so two runs of the same shape produce a diffable envelope -- the
            # human block sorts by time instead (`_tool_lines`); different question,
            # different order. Kept LAST because `venice sessions show` prints this dict
            # as a one-line repr, and a nested map belongs after the readable scalars.
            "tools": {k: dict(self.tools[k]) for k in sorted(self.tools)},
            # #99: the TRUE ledgered-call count, which is also where each row's `n` comes
            # from. A scalar, so it sits with the readable ones ahead of the two lists.
            "api_calls_total": self.api_calls_total,
            # #99: the per-call trace and the prefix events, LAST for `tools`' reason one
            # key up -- `venice sessions show` renders this dict at a human, and the
            # nested collections belong after the scalars. Rows are COPIED (like `tools`)
            # so a caller mutating the envelope -- `code --json` hands it straight to
            # `json.dump` while the ledger is still live -- cannot reach back in here.
            "api_calls": [dict(r) for r in self.api_calls()],
            "context_events": [dict(e) for e in self.context_events],
        }

    def restore(self, d) -> None:
        """Seed the accumulators from a :meth:`to_dict` snapshot (resume, #47).

        Additive onto the current (freshly-priced) ledger so a session resumed
        mid-run keeps its cumulative token/cost totals. Tolerant of a partial or
        foreign dict (missing keys count as 0) so a hand-edited envelope can't crash
        the REPL. Also accepts the raw #75 usage field names as a fallback.
        """
        if not isinstance(d, dict):
            return
        self.total += _as_float(d.get("total"))
        self.prompt_tokens += _as_int(d.get("prompt_tokens"))
        self.completion_tokens += _as_int(d.get("completion_tokens"))
        self.cache_read_tokens += _as_int(
            d.get("cache_read_tokens", d.get("cached_tokens"))
        )
        self.cache_write_tokens += _as_int(
            d.get("cache_write_tokens", d.get("cache_creation_input_tokens"))
        )
        self.reasoning_tokens += _as_int(d.get("reasoning_tokens"))
        # #81: additive like the token tallies, so `--resume` reports the total time
        # this session has kept you waiting. Pre-#81 envelopes have neither key and
        # degrade to 0 -- no SESSION_VERSION bump, no `_session.py` change.
        self.elapsed_seconds += _as_float(d.get("elapsed_seconds"))
        self.turns += _as_int(d.get("turns"))
        if d.get("unpriced"):
            self.unpriced = True
        # #98: sticky-OR like `unpriced`, and falsy-by-default so a pre-#98 envelope
        # (which has neither key) claims "reported" rather than retroactively marking
        # every resumed session unknown. The inaccuracy has a one-turn lifetime: the
        # flags are sticky, so the resumed session's first `record()` sets the truth.
        if d.get("cache_read_unreported"):
            self.cache_read_unreported = True
        if d.get("cache_write_unreported"):
            self.cache_write_unreported = True
        # #100: `cache_hit_percent` is deliberately NOT restored. Every other field
        # above is additive because it is a tally; that one is DERIVED from the tallies
        # this method just seeded, so it recomputes itself for free -- and accumulating
        # it (the reflex the surrounding lines invite) would add two percentages.
        # #82: additive per tool NAME, like the token tallies, so `--resume` reports
        # where the whole session's tool time went and not just the latest leg. Tolerant
        # PER ROW, so one hand-edited entry cannot cost the others. A pre-#82 envelope
        # has no `tools` key and degrades to {} -- no SESSION_VERSION bump. No lock: this
        # runs single-threaded at construction, and taking it would imply otherwise.
        raw = d.get("tools")
        if isinstance(raw, dict):
            for name, row in raw.items():
                if not isinstance(row, dict):
                    continue
                key = str(name).strip()
                if not key:
                    continue
                cur = self.tools.setdefault(key, {"seconds": 0.0, "calls": 0})
                cur["seconds"] = round(cur["seconds"] + _as_float(row.get("seconds")), 3)
                cur["calls"] += _as_int(row.get("calls"))
        # `tool_seconds` is NOT restored, for the `cache_hit_percent` reason one field
        # over: it is DERIVED from the map the loop above just seeded, so it recomputes
        # for free, and accumulating it would double every resumed run's tool time.
        #
        # #99: a THIRD category, and the reason it is spelled out is that the two above
        # each warn against the other's reflex and neither one is right here. `api_calls`
        # and `context_events` are SEED-ONCE: they are lists, so the additive treatment
        # every tally above gets would CONCATENATE a second `restore()` into duplicate
        # rows, and the derived treatment would drop them -- which is not merely lossy but
        # DESTRUCTIVE, because the next `_autosave` overwrites `sess.usage` wholesale and
        # the previous leg's rows would be gone from disk. Seeding only into empty lists
        # makes the double call a no-op without needing a "have I restored yet" flag.
        if not self._calls_head and not self._calls_tail:
            raw_calls = d.get("api_calls")
            if isinstance(raw_calls, list):
                for row in raw_calls:
                    if not isinstance(row, dict):
                        continue  # tolerant PER ROW, like `tools` above
                    if len(self._calls_head) < self._CALLS_HEAD:
                        self._calls_head.append(dict(row))
                    else:
                        self._calls_tail.append(dict(row))
        if not self.context_events:
            raw_events = d.get("context_events")
            if isinstance(raw_events, list):
                self.context_events.extend(
                    dict(e) for e in raw_events if isinstance(e, dict)
                )
        # ADDITIVE, unlike the two lists it indexes: it is a tally like `turns`, and
        # keeping it additive is what makes a resumed session's next row continue at
        # `n = 51` instead of restarting at 1 and colliding with a restored row.
        self.api_calls_total += _as_int(d.get("api_calls_total"))
        # A hand-edited envelope can carry rows with no (or a truncated) total, which
        # would restart `n` inside a range the restored rows already occupy and make
        # `calls_dropped` negative. Floor it at what was actually seeded; the docstring
        # promise is that `n` is unique and ascending, and that has to survive a bad file.
        self.api_calls_total = max(
            self.api_calls_total, len(self._calls_head) + len(self._calls_tail)
        )

    def record_turn(self, seconds) -> None:
        """Add one blocked window's wall-clock and count the turn (#81).

        PURE by construction: the caller reads the clock and passes the delta, so this
        needs no clock-mocking to test and `record()` stays untouched. A garbage or
        negative delta counts as 0 seconds but still counts as a turn -- a monotonic
        clock cannot run backwards, so that guard only ever catches a caller bug, and
        silently dropping the turn would corrupt the average too. Rounded on the way in
        so the persisted envelope and the `--json` surface carry a tidy number.
        """
        self.elapsed_seconds = round(self.elapsed_seconds + _as_float(seconds), 3)
        self.turns += 1

    def record_tool(self, name, seconds) -> None:
        """Add one tool call's execution window to the per-tool aggregate (#82).

        PURE like :meth:`record_turn`: the caller brackets `tool.invoke` and passes the
        delta, so the ledger still never reads a clock and every test here stays a
        pure-function assertion.

        One CALL, not one invoke. A confirm-gated paid tool invokes TWICE (gated, then
        re-invoked with ``confirm=True``); `_run_one_call` sums both windows and flushes
        once, so `calls` counts what the model asked for while `seconds` counts what
        actually ran. Thread-safe -- see `_tools_lock`.

        Garbage or negative seconds count as 0 but still count the call, for
        `record_turn`'s reason: a monotonic clock cannot run backwards, so that only ever
        catches a caller bug, and dropping the call would corrupt the per-call average
        too. A blank name is dropped ENTIRELY, which is the one real difference: there is
        no row to put it on, and an anonymous line in the breakdown answers nothing.
        """
        key = str(name).strip() if name is not None else ""
        if not key:
            return
        secs = _as_float(seconds)
        with self._tools_lock:
            row = self.tools.get(key)
            if row is None:
                self.tools[key] = {"seconds": round(secs, 3), "calls": 1}
            else:
                row["seconds"] = round(row["seconds"] + secs, 3)
                row["calls"] += 1

    def tool_seconds(self) -> float:
        """Total measured tool-execution time (#82); 0.0 with nothing recorded."""
        return round(sum(r["seconds"] for r in self.tools.values()), 3)

    def tool_calls_total(self) -> int:
        """How many tool calls that time is spread across (#82).

        Not `tool_calls`: `calls_made`/`max_tool_calls` already crowd that vocabulary in
        `run_loop`, and a bare `tool_calls` reads like the OpenAI message field.
        """
        return sum(r["calls"] for r in self.tools.values())

    def _append_call(self, row: dict) -> None:
        """Land one trace row in the head/tail buffers (#99).

        `api_calls_total` is bumped FIRST and is what stamps `n`, so the ordinal is the
        call's true position in the session even after the head froze and the tail began
        overwriting. Head fills first; everything after it rolls through the deque.

        `n` is rebuilt into the FRONT of the row rather than assigned onto the end: these
        rows are read by eye in a session file, and the ordinal belongs at the left.
        """
        self.api_calls_total += 1
        row = dict(n=self.api_calls_total, **row)
        if len(self._calls_head) < self._CALLS_HEAD:
            self._calls_head.append(row)
        else:
            self._calls_tail.append(row)

    def api_calls(self) -> List[dict]:
        """The retained trace rows in call order: head, then tail (#99).

        A gap between the two is not an error and is not marked: consecutive rows whose
        `n` jumps say so themselves, which is why nothing stores a drop count.
        """
        return self._calls_head + list(self._calls_tail)

    def calls_dropped(self) -> int:
        """How many trace rows the cap discarded (#99); 0 when nothing was dropped."""
        return self.api_calls_total - len(self._calls_head) - len(self._calls_tail)

    def record_compaction(self, event: dict) -> None:
        """Log one prefix-affecting event, anchored to the trace (#99).

        `after_n` is the ordinal of the last call recorded BEFORE the event, so the
        measured post-compaction prompt size is simply the next row's `prompt_tokens` --
        which is why no `observed_tokens_after` is stored: the server does not report
        that number until the next call, and computing it here would be a guess wearing
        a measurement's name.
        """
        row = dict(event)
        # Rebuilt front-first for `_append_call`'s reason: these are read by eye in a
        # session file, and "what happened, and where" belongs left of the measurements.
        self.context_events.append(dict(
            kind=row.pop("kind", "compaction"),
            after_n=self.api_calls_total,
            **row,
        ))

    def record(self, usage, *, seconds=None) -> float:
        """Add one turn's `usage` (dict or SDK obj); return this turn's cost.

        Keeps the cache buckets distinct: cache-read, cache-write, and uncached
        input each price at their own rate, so a cache-heavy turn is costed
        correctly instead of collapsed to a flat input rate (#75). Both cache
        buckets are subsets of `prompt_tokens` in Venice's OpenAI-normalized
        usage shape, so uncached input is the remainder. With no cache tokens and
        no cache pricing this reduces exactly to the old `pt*in + ct*out`.

        A cache field the response never carried is recorded as *unreported* rather
        than as zero (#98) -- see :func:`_cache_tokens`. The arithmetic is unchanged
        by that: every shape that used to yield 0 still yields 0.

        NOT READ: the `/responses` endpoint's second usage shape (`input_tokens` /
        `input_tokens_details.cached_tokens`). The CLI never calls it today, but if it
        ever does this ledger silently zeroes rather than erroring.

        #99: appends one trace row on EVERY path, including the no-usage shapes -- see
        `_usage_dict`. `seconds` is the caller-stamped duration of the API call, keeping
        the same never-read-a-clock contract as `record_turn`/`record_tool`; None means
        the site did not bracket a window and renders as `n/a`, never as `0.0s`.
        """
        # FIRST, ahead of everything below: `record(None)` dumping `usage-raw: null`
        # is exactly the diagnostic that was missing -- it separates "the response had
        # no usage block" from "it had one with no cache fields" (#98).
        _dump_raw_usage(usage)
        usage = _usage_dict(usage)
        cost = 0.0
        # A response with no usage block still costs wall-clock and still moved the
        # prefix, so it gets a row -- with null tokens rather than a fabricated 0, the
        # #98 rule applied one level down.
        pt = ct = raw_read = raw_write = None
        if usage is not None:
            pt = _as_int(usage.get("prompt_tokens"))
            ct = _as_int(usage.get("completion_tokens"))
            # #98: None means the field was absent, which is not a reported zero. The
            # flags are what `usage_report` needs to refuse to print a fake 0.0%; the
            # counts fall back to 0 so the cost math below is untouched.
            raw_read = _cache_tokens(usage, _CACHE_READ_KEYS)
            raw_write = _cache_tokens(usage, _CACHE_WRITE_KEYS)
            if raw_read is None:
                self.cache_read_unreported = True
            if raw_write is None:
                self.cache_write_unreported = True
            cache_read = 0 if raw_read is None else raw_read
            cache_write = 0 if raw_write is None else raw_write
            reasoning = _as_int(
                _detail(usage, "completion_tokens_details", "reasoning_tokens")
            )
            # Clamp to subsets of prompt_tokens so a provider that reports the buckets
            # additively (rather than as a breakdown) can't drive uncached negative.
            cache_read = min(cache_read, pt)
            cache_write = min(cache_write, pt - cache_read)
            uncached = pt - cache_read - cache_write

            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.cache_read_tokens += cache_read
            self.cache_write_tokens += cache_write
            self.reasoning_tokens += min(reasoning, ct)
            # The row reports the CLAMPED buckets, so a row's numbers reconcile with the
            # aggregate they fed -- but only where the field existed at all: `raw_*` is
            # None-checked first so an absent field stays null through the clamp.
            raw_read = None if raw_read is None else cache_read
            raw_write = None if raw_write is None else cache_write

            if self._in is not None or self._out is not None:
                in_rate = self._in or 0.0
                read_rate = self._cache_in if self._cache_in is not None else in_rate
                write_rate = (self._cache_write if self._cache_write is not None
                              else in_rate)
                cost = (
                    uncached * in_rate
                    + cache_read * read_rate
                    + cache_write * write_rate
                    + ct * (self._out or 0.0)
                )
            else:
                self.unpriced = True
            self.total += cost
        self._append_call({
            "prompt_tokens": pt,
            "cache_read_tokens": raw_read,
            # Recorded although #99 only asked for cache-READ: a cache-WRITE spike at
            # call N is what separates a churning prefix from a cold start, i.e. two of
            # the three failures this trace exists to tell apart.
            "cache_write_tokens": raw_write,
            "completion_tokens": ct,
            "cost": round(cost, 6),
            # Rounded like `record_turn`'s, so the persisted envelope stays tidy. None
            # stays None -- an unstamped window is unknown, not instant.
            "seconds": None if seconds is None else round(_as_float(seconds), 3),
        })
        return cost

    def over(self) -> bool:
        """True when accumulated spend has reached/exceeded the cap."""
        return self.max_spend is not None and self.total >= self.max_spend

    def over_tokens(self) -> bool:
        """True when cumulative prompt+completion tokens reached/exceeded the cap.

        Orthogonal to :meth:`over` (which is USD-only): a per-subagent run is capped on
        tokens, not dollars (its LLM turns aren't charged against an external account by
        this mechanism -- see `_code.spawn_tool`/`scout_tool`). Counts raw tokens,
        cache-agnostic (both cache buckets are subsets of `prompt_tokens`).
        """
        return (
            self.max_tokens is not None
            and (self.prompt_tokens + self.completion_tokens) >= self.max_tokens
        )

    def cache_hit_percent(self, *, round_to: Optional[int] = None):
        """Cache-read hit rate as a percent 0-100, or None when unknowable (#98/#100).

        `None` means UNKNOWN and never zero -- the whole point of #98. Two ways to not
        know: nothing has been recorded (no input tokens to divide by), or every turn
        that was recorded carried no cache-read field at all. A caller that renders the
        None must say "n/a"; rendering it as 0.0 recreates the exact lie #98 removed.

        NOT a single source of truth for cache state, despite being the only place the
        rate is computed. It collapses "unrecorded" and "unreported" into one None, and
        it says nothing about the *write* bucket, so `usage_report`'s split line and
        `summary`'s fragment both still read the `*_unreported` flags for their wording.

        `round_to` is for the persisted/JSON surface (`to_dict`), where an unrounded
        89.10891089108911 lands in a session file and gets printed at a human by
        `venice sessions show`. The human renderers format with `:.1f` themselves.
        """
        if self.prompt_tokens == 0:
            return None
        if self.cache_read_unreported and self.cache_read_tokens == 0:
            return None
        pct = self.cache_read_tokens / self.prompt_tokens * 100.0
        return pct if round_to is None else round(pct, round_to)

    def _cache_fragment(self) -> str:
        """The one-line cache clause for :meth:`summary`, or "" when there is none (#100).

        Vocabulary is `usage_report`'s verbatim -- an operator can run `/cost` and
        `/usage` seconds apart, and two words for one state reads as two states.
        """
        pct = self.cache_hit_percent()
        if pct is None:
            # Distinguish "the provider never told us" (worth saying on a run footer --
            # it is the difference between a cache collapse and a blind spot) from
            # "nothing was recorded", which has nothing to report at all.
            return "cache n/a" if self.cache_read_unreported else ""
        if self.cache_read_unreported:
            return f"cache {pct:.1f}% hit [partially unreported]"
        return f"cache {pct:.1f}% hit"

    def summary(self, *, cache: bool = False) -> str:
        """A one-line human-readable total (for stderr / --json).

        `cache` opts into the #100 hit-rate clause. OFF by default because two of this
        method's call sites are stop-reason messages -- `run_loop`'s spend and token
        gates -- on unpriced subagent/review ledgers that are near-always cache-
        unreported, and "cache n/a" on `worker reached token cap 50,000` answers a
        question nobody asked there. The run footers and `/cost` opt in.
        """
        # Built once so the clause reaches BOTH branches below: the command-level tests
        # all render the unpriced one (the catalog fake carries no `pricing`), which is
        # exactly the branch a fragment appended to the priced path would miss.
        toks = f"tokens prompt={self.prompt_tokens} completion={self.completion_tokens}"
        frag = self._cache_fragment() if cache else ""
        if frag:
            # Inside the clause, not appended with " -- ": the footers render
            # `code: {duration} wall -- {summary()}`, where " -- " is THE top-level
            # field boundary (and is documented as one in the README).
            toks += f", {frag}"
        if self.unpriced and self.total == 0.0:
            return f"cost: (unpriced — model rate unknown) {toks}"
        s = f"cost: ${self.total:.4f}"
        if self.max_spend is not None:
            s += f" / cap ${self.max_spend:.2f}"
        s += f" ({toks})"
        if self.unpriced:
            s += " [partially unpriced]"
        return s

    def usage_report(self) -> str:
        """A multi-line token + cost breakdown for the REPL `/usage` command (#75).

        Keeps the cache buckets visible -- showing the uncached vs cache-read
        split is the whole point, since that split is what makes a long session's
        cost (and its affordability) legible. Mirrors `summary`'s unpriced
        handling; returns a one-line placeholder before any turn is recorded.
        """
        if self.prompt_tokens == 0 and self.completion_tokens == 0:
            # #99: `api_calls_total` and `context_events` join `turns` in this gate.
            # A run whose every response came back without a usage block has real rows
            # and real seconds and zero tokens, and a `/compact` before any turn has an
            # event and nothing else (its own summary call is unledgered, #101).
            # Reporting "(no usage recorded yet)" for either would hide the trace at
            # precisely the moment it is the only evidence there is.
            if not self.turns and not self.api_calls_total and not self.context_events:
                return "(no usage recorded yet)"
            # #81: time was spent but the provider reported no tokens -- a turn that
            # raised, or one aborted mid-flight. Report the clock honestly rather than
            # claiming nothing happened, but don't fabricate a 0-token cache breakdown
            # and a "cache hit rate: 0.0%" that would read as a real measurement.
            # #82: a turn that spent minutes in tools then raised before any usage came
            # back is exactly when the breakdown earns its keep -- wire BOTH sites.
            # #99: `_timing_line` gets the same `turns` gate it has on the main path --
            # reaching here with rows but no turn (a subagent ledger) would otherwise
            # print `over 0 turn(s)  (avg 0.0s)`, which is noise, not a measurement.
            return "\n".join(
                ["session usage:", "  (no tokens reported)"]
                + ([self._timing_line()] if self.turns else [])
                + self._tool_lines() + self._call_lines()  # #99: wire BOTH sites, as #82
            )
        uncached = self.prompt_tokens - self.cache_read_tokens - self.cache_write_tokens
        lines = ["session usage:"]
        # #98: the two flags govern DIFFERENT rows on purpose. Read-reported-but-
        # write-absent is the normal shape for most Venice models, so a shared marker
        # would decorate the hit rate of nearly every session with a warning about a
        # bucket the hit rate does not even use.
        if self.cache_read_unreported and self.cache_write_unreported:
            # Nothing about the split is known -- don't itemize a breakdown that is
            # entirely inferred, same reflex as "(no tokens reported)" above.
            split = "cache breakdown not reported"
        else:
            read_part = ("cache-read n/a" if self.cache_read_unreported
                         else f"{self.cache_read_tokens:,} cache-read")
            write_part = ("cache-write n/a" if self.cache_write_unreported
                          else f"{self.cache_write_tokens:,} cache-write")
            # NOTE: with a side unreported, `uncached` is an UPPER BOUND and the terms
            # no longer sum to the input total -- the n/a shows where the slack went.
            split = f"{uncached:,} uncached + {read_part} + {write_part}"
        lines.append(f"  input   {self.prompt_tokens:>10,} tok  ({split})")
        out = f"  output  {self.completion_tokens:>10,} tok"
        if self.reasoning_tokens:
            out += f"  (incl. {self.reasoning_tokens:,} reasoning)"
        lines.append(out)
        # #100: the rate itself comes from `cache_hit_percent` so `/usage`, `/cost` and
        # both run footers cannot drift apart on what "unknown" means.
        hit = self.cache_hit_percent()
        if hit is None:
            # Two distinct unknowns. `no input tokens` is only reachable past the
            # early-return above with completion tokens but no prompt tokens -- which
            # `restore()` can seed from a partial envelope, since (unlike `record`) it
            # does not clamp the cache buckets to the prompt total. Dividing by that
            # printed a 0.0% measured over nothing, which is the #98 lie one level down.
            why = ("no cache fields reported" if self.cache_read_unreported
                   else "no input tokens")
            lines.append(f"  cache hit rate: n/a ({why})")
        elif self.cache_read_unreported:
            # Some turns measured, some didn't: the rate is real but understated
            # against a prompt total that includes turns it could not see.
            lines.append(f"  cache hit rate: {hit:.1f}%  [partially unreported]")
        else:
            lines.append(f"  cache hit rate: {hit:.1f}%")
        if self.unpriced and self.total == 0.0:
            lines.append("  cost: (model rate unknown)")
        else:
            cost = f"  cost: ${self.total:.4f}"
            if self.max_spend is not None:
                cost += f" / cap ${self.max_spend:.2f}"
            if self.unpriced:
                cost += "  [partially unpriced]"
            lines.append(cost)
        if self.turns:  # #81
            lines.append(self._timing_line())
        lines.extend(self._tool_lines())  # #82 -- own gate; see `_tool_lines`
        lines.extend(self._call_lines())  # #99 -- likewise
        return "\n".join(lines)

    def _timing_line(self) -> str:
        """The `/usage` wall-clock row (#81).

        Only ever rendered when a window was actually stamped, so a ledger that has
        only seen `record()` -- `run_loop` in isolation, every per-subagent ledger --
        reports exactly what it reported before this existed.
        """
        avg = self.elapsed_seconds / self.turns if self.turns else 0.0
        return (f"  wall    {format_duration(self.elapsed_seconds):>10}  "
                f"over {self.turns} turn(s)  (avg {format_duration(avg)})")

    #: #82: rows in the `/usage` tools block before the tail folds into one `(+N more)`
    #: line. Bounded because an MCP server can attach dozens of tools, and a 40-row
    #: block buries the three names that actually ate the run.
    _TOOL_ROWS = 8

    def _tool_lines(self) -> List[str]:
        """The `/usage` per-tool breakdown (#82); ``[]`` when nothing was timed.

        Gated on the MAP, not on `self.turns` like `_timing_line`: a ledger that only
        ever saw `run_loop` (every per-subagent ledger) has real tool time and no turn,
        and the number is true there even though nothing renders it today. Coupling it
        to `turns` would be a false dependency that reads as intentional.

        Sorted by time DESCENDING -- the block exists to answer "what ate the run" --
        with the name as tiebreak, so two tools at the same duration cannot swap places
        between runs and turn a pinned test into a coin flip.
        """
        if not self.tools:
            return []
        total = self.tool_seconds()
        # `  tools   ` is 10 chars before the field, exactly like `  wall    ` above, so
        # the two durations line up in one column with no other coordination.
        head = (f"  tools   {format_duration(total):>10}  "
                f"across {self.tool_calls_total()} call(s)")
        # #82: under --parallel two subagent windows overlap in real time but BOTH land
        # in this total, so it can legitimately exceed the wall row it sits under. Say
        # which clock it is rather than letting the reader assume it's an arithmetic
        # bug; `_queue.progress_tick` sets the house precedent (locally measured `wall`
        # vs provider-reported `server`). The tolerance absorbs `record_turn`'s 3-dp
        # rounding -- a serial run's tools nest inside its turns and must never trip it.
        if self.turns and total > self.elapsed_seconds + 0.001:
            head += "  [concurrent -- exceeds wall]"
        rows = sorted(self.tools.items(), key=lambda kv: (-kv[1]["seconds"], kv[0]))
        shown, rest = rows[:self._TOOL_ROWS], rows[self._TOOL_ROWS:]
        more = f"(+{len(rest)} more)" if rest else ""
        # Width from the DATA: MCP tool names are unbounded, and a fixed column would
        # either truncate the name (the one thing the row is for) or wrap the block.
        w = max([12, len(more)] + [len(n) for n, _ in shown])
        lines = [head]
        for name, row in shown:
            lines.append(f"    {name:<{w}}  {format_duration(row['seconds']):>8}"
                         f"   {row['calls']} call(s)")
        if rest:
            # The residual carries its OWN seconds and calls, so the rows still
            # reconcile to the header. A truncated block whose parts do not add up is
            # the #98 lie in a new costume.
            lines.append(f"    {more:<{w}}  "
                         f"{format_duration(sum(r['seconds'] for _, r in rest)):>8}"
                         f"   {sum(r['calls'] for _, r in rest)} call(s)")
        return lines

    #: #99: trace rows shown from the head and the tail of the block before the middle
    #: folds into one elided line. 3+5 = 8, matching `_TOOL_ROWS` -- the head answers
    #: "was it cold from call 1", the tail answers "what is it doing now", and those are
    #: two of the three questions the trace exists for.
    _CALL_HEAD_ROWS = 3
    _CALL_TAIL_ROWS = 5

    @staticmethod
    def _call_row(label: str, w: int, pt, read, ct, secs) -> str:
        """One `/usage` trace line -- a call row or the elided-span summary (#99).

        `n/a` rather than `0` or `0.0s` for anything the response did not carry: an
        unstamped window is unknown, not instant, and an absent cache field is unknown,
        not a miss. That is #98's rule applied to the time and per-row dimensions.

        Both row kinds render through here so the elision line cannot drift out of the
        columns it sits in -- `_tool_lines` takes its label width from the data for the
        same reason one method up.
        """
        pct = "n/a" if (read is None or not pt) else f"{read / pt * 100:.0f}%"
        return (f"    {label:<{w}}  {'n/a' if pt is None else format(pt, ','):>9} in"
                f"  {pct:>4} cached  {'n/a' if ct is None else format(ct, ','):>7} out"
                f"  {'n/a' if secs is None else format_duration(secs):>8}")

    def _call_lines(self) -> List[str]:
        """The `/usage` per-API-call trace block (#99); ``[]`` when nothing was recorded.

        Gated on the LIST, not on `self.turns` -- `_tool_lines`' reasoning one method up:
        a ledger that only ever saw `run_loop` (every per-subagent ledger) has real rows
        and no turn, and coupling the two would be a false dependency.

        Head + tail with an elided middle, and the elision line carries its OWN totals so
        the rows still reconcile to the header. A truncated block whose parts do not add
        up is the #98 lie in a new costume.
        """
        rows = self.api_calls()
        if not rows:
            # A `/compact` before any turn leaves an event and no rows -- its own
            # summarization call is unledgered (#101). The event is the only record
            # that the prefix moved, so it renders on its own rather than vanishing
            # with the block that would have carried it.
            return [ln for ev in self.context_events for ln in self._event_lines(ev)]
        head_n = self._CALL_HEAD_ROWS
        tail_n = self._CALL_TAIL_ROWS
        secs = [r.get("seconds") for r in rows]
        timed = [s for s in secs if s is not None]
        head = (f"  calls   {format_duration(sum(timed)):>10}  "
                f"across {self.api_calls_total} API call(s)")
        # No silent caps: both the rows this render folded away and the rows the storage
        # cap never kept are named, because a block that quietly shows 8 of 300 reads as
        # "this is all of it".
        untimed = len(secs) - len(timed)
        if untimed:
            head += f"  [{untimed} untimed]"
        dropped = self.calls_dropped()
        if dropped:
            head += f"  [{dropped} row(s) dropped]"
        lines = [head]
        if len(rows) <= head_n + tail_n + 1:
            shown, elided, tail = rows, [], []
        else:
            shown, elided, tail = rows[:head_n], rows[head_n:-tail_n], rows[-tail_n:]
        elide_label = f"(+{len(elided)} elided)" if elided else ""
        # Width from the DATA, exactly like `_tool_lines`: the elision label is wider
        # than any `#N`, and a fixed column would leave it hanging out of the grid.
        w = max([4, len(elide_label)] + [len(f"#{r.get('n')}") for r in rows])

        def _emit(rs):
            for r in rs:
                lines.append(self._call_row(
                    f"#{r.get('n', '?')}", w, r.get("prompt_tokens"),
                    r.get("cache_read_tokens"), r.get("completion_tokens"),
                    r.get("seconds"),
                ))
                # Markers ALWAYS render, never elided -- they are rare and are the
                # highest-signal line in the block. One anchored inside the folded span
                # surfaces just after the elision line, where `after #N` keeps it placed.
                for ev in self.context_events:
                    if ev.get("after_n") == r.get("n"):
                        lines.extend(self._event_lines(ev))

        _emit(shown)
        if elided:
            # The elided span carries its OWN totals so the block still reconciles to the
            # header -- `_tool_lines`' `(+N more)` rule. Summed with `or 0` because a
            # no-usage row's fields are None, and the span's cache percentage is over the
            # input it could actually see.
            e_secs = [r.get("seconds") for r in elided if r.get("seconds") is not None]
            lines.append(self._call_row(
                elide_label, w,
                sum(r.get("prompt_tokens") or 0 for r in elided),
                sum(r.get("cache_read_tokens") or 0 for r in elided),
                sum(r.get("completion_tokens") or 0 for r in elided),
                sum(e_secs),
            ))
            for r in elided:
                for ev in self.context_events:
                    if ev.get("after_n") == r.get("n"):
                        lines.extend(self._event_lines(ev))
        _emit(tail)
        # An event anchored at call 0 (a compaction before any call was recorded) has no
        # row to hang off, so it would otherwise vanish from the block entirely.
        for ev in self.context_events:
            if not ev.get("after_n"):
                lines.extend(self._event_lines(ev))
        return lines

    @staticmethod
    def _event_lines(ev: dict) -> List[str]:
        """The one-or-two-line `/usage` marker for a context event (#99)."""
        out = [f"    -- compacted ({ev.get('trigger', 'auto')}) "
               f"after #{ev.get('after_n', '?')}: "
               f"{ev.get('messages_before', '?')} -> {ev.get('messages_after', '?')} msgs, "
               f"~{ev.get('est_tokens_before') or 0:,} -> "
               f"~{ev.get('est_tokens_after') or 0:,} tok est"]
        obs = ev.get("observed_tokens_before")
        if obs:
            # Its own line, and labelled "lower bound": the number is the PREVIOUS call's
            # prompt size, and the history grew after it was observed.
            out.append(f"       ({obs:,} tok measured before, lower bound)")
        return out

    def tools_fragment(self) -> str:
        """The run footers' tool-time clause (#82): ``" (2m 41s tools)"`` or ``""``.

        Rendered INSIDE the wall field -- ``code: 4m 12s wall (2m 41s tools) -- cost:``
        -- because `" -- "` is THE top-level field boundary and the README documents it
        as one. Same reflex as #100's cache clause going inside `summary`'s parens.

        Unlike `summary(cache=...)` this needs no opt-in flag: it is built by the two
        footers themselves rather than by `summary`, so `run_loop`'s spend/token
        stop-reason messages -- which call `summary()` on unpriced subagent ledgers --
        can never inherit this surface by accident.
        """
        if not self.tools:
            return ""
        total = self.tool_seconds()
        if self.turns and total > self.elapsed_seconds + 0.001:
            # A footer reading `4m 12s wall (6m 02s tools)` reads as an arithmetic bug
            # unless it says why. One word, still inside the parens.
            return f" ({format_duration(total)} tools, concurrent)"
        return f" ({format_duration(total)} tools)"


def _pricing_for(models, model_id):
    """The catalog `model_spec.pricing` block for `model_id`, or None."""
    for m in models or []:
        if isinstance(m, dict) and m.get("id") == model_id:
            spec = m.get("model_spec")
            if isinstance(spec, dict):
                return spec.get("pricing")
    return None


def _build_ledger(cap, models, model_id) -> CostLedger:
    """A CostLedger bound to `model_id`'s catalog pricing (cap may be None)."""
    ledger = CostLedger(max_spend=cap)
    pricing = _pricing_for(models, model_id)
    if pricing is not None:
        ledger.bind_pricing(pricing)
    return ledger


def ledger_from_args(args, models, model_id) -> Optional[CostLedger]:
    """The session CostLedger for a parsed-args namespace, or None when the run
    isn't spend-capped (#66).

    Enabled by ``--session-max-spend`` (or ``defaults.<cmd>.session_max_spend``)
    -- DISTINCT from ``--max-spend``, which is the *per-call* auto-approve cap
    for paid tools. Bound to the session model's catalog pricing; an unknown
    price degrades to token-counting without charging (the ledger still reports
    usage). `models` is the text catalog the command already fetched.
    """
    cap = getattr(args, "session_max_spend", None)
    if cap is None:
        return None
    return _build_ledger(cap, models, model_id)


def usage_ledger(args, models, model_id) -> CostLedger:
    """An always-on session ledger for the REPL's `/usage` + `/cost` (#75).

    Unlike :func:`ledger_from_args` (None unless the session is spend-capped),
    this always returns a priced ledger so `/usage` works in any interactive
    session. `--session-max-spend`, when set, still supplies the cap; an uncapped
    ledger meters usage without gating (`over()` is None-safe).
    """
    return _build_ledger(getattr(args, "session_max_spend", None), models, model_id)


def dispatch_map(tools: List[Tool]) -> Dict[str, Tool]:
    return {t.name: t for t in tools}


# --------------------------------------------------------------------------- #
# Web search (#77): one server-side Venice completion with `enable_web_search`, so
# the coding agent / scout can DISCOVER documentation -- not just fetch a URL it
# already knows (the `--browser` rail, #71). Rides the normal completion path (same
# key, same billing), so the per-agent tool-call budget bounds it. The `venice_web_search`
# rail Tool wrapper + `supportsWebSearch` model resolution live in `_code`; this module
# owns only the profile-agnostic completion helper.
# --------------------------------------------------------------------------- #
def _obj_to_dict(value) -> Optional[dict]:
    """A plain dict from a Venice SDK object (`model_dump`) or an already-dict value."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def _web_citations(venice_params) -> List[dict]:
    """Normalize `venice_parameters.web_search_citations` to `[{title,url[,date]}]`.

    Mirrors `chat._print_citations`: reads `title`/`url`/`date` (the API also carries
    `content`, dropped here to keep the handoff compact). URL-less items are skipped.
    """
    vp = _obj_to_dict(venice_params) or {}
    raw = vp.get("web_search_citations")
    if not isinstance(raw, list):
        return []
    cites: List[dict] = []
    for c in raw:
        cd = _obj_to_dict(c) or {}
        url = cd.get("url")
        if not url:
            continue
        cite = {"title": cd.get("title", ""), "url": url}
        if cd.get("date"):
            cite["date"] = cd["date"]
        cites.append(cite)
    return cites


def run_web_search(oai, model: str, query: str, *, mode: str = "on",
                   models=None) -> dict:
    """Make ONE Venice web-search completion and return its answer + citations (#77).

    Rides `/chat/completions` with `venice_parameters.enable_web_search` (`mode`: "on"
    forces search, "auto" leaves it to the model) + `enable_web_citations`, exactly as
    `venice chat --web-search` does. Returns
    `{"status":"ok","answer","citations":[{title,url[,date]}],"cost_estimate_usd","model"}`.
    `cost_estimate_usd` is a best-effort post-response estimate from the server `usage`
    block priced against the catalog (`None` when pricing is unknown -- web search is
    billed but rides the completion path, so the per-agent tool-call budget bounds it; no
    separate cap in v1). `openai.OpenAIError` propagates -- the Tool wrapper turns it into
    an error envelope.
    """
    query = (query or "").strip()
    if not query:
        return {"status": "error", "message": "web_search requires a non-empty 'query'"}
    _t0 = time.monotonic()  # #99: this ledger is a throwaway, but a documented exception
    resp = oai.chat.completions.create(  # costs more to reason about than the one line
        model=model,
        messages=[{"role": "user", "content": query}],
        extra_body={
            "venice_parameters": {
                "enable_web_search": mode,
                "enable_web_citations": True,
            }
        },
    )
    choices = getattr(resp, "choices", None) or []
    msg = getattr(choices[0], "message", None) if choices else None
    answer = (getattr(msg, "content", None) or "").strip() if msg is not None else ""
    citations = _web_citations(getattr(resp, "venice_parameters", None))
    led = _build_ledger(None, models, model)
    cost = led.record(getattr(resp, "usage", None), seconds=time.monotonic() - _t0)
    # Best-effort: report None (unknown) -- not $0.00 -- when we can't estimate, i.e. the
    # model price is unknown OR the response carried no usage tokens. A billed feature that
    # reports 0.0 reads as "free", which is worse than an honest "unknown".
    known = not led.unpriced and (led.prompt_tokens or led.completion_tokens)
    return {
        "status": "ok",
        "answer": answer,
        "citations": citations,
        "cost_estimate_usd": cost if known else None,
        "model": model,
    }


#: The `venice_web_search` rail tool name (#77). Named here beside the completion helper
#: and the SCOUT/SPAWN/MERGE names so the guards share one source of truth.
WEB_SEARCH_TOOL_NAME = "venice_web_search"


def supports_web_search(models, model_id) -> Optional[bool]:
    """Whether `model_id` advertises web search in the catalog (#77).

    True/False when the model is found and carries `supportsWebSearch`; None when it
    can't be determined (no catalog, model absent, or the field missing) -- treated as
    "unknown, attempt anyway", mirroring :func:`supports_function_calling`.
    """
    return _models.supports_capability(models, model_id, "supportsWebSearch")


def resolve_web_search_model(models, search_model, coding_model) -> Optional[str]:
    """Pick the model for a web-search completion (#77) -- no hardcoded id.

    Precedence: an explicit operator override (`--web-search-model` / config) is trusted
    as-is; else the coding model when it advertises `supportsWebSearch` (or the capability
    can't be determined -- attempt anyway); else the first catalog model that advertises
    it; else None (the caller surfaces an actionable error). Grounding the default in the
    live `/models` catalog avoids guessing a model id that may not exist.
    """
    if search_model:
        return search_model
    if supports_web_search(models, coding_model) is not False:
        return coding_model
    for m in models or []:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if mid and supports_web_search(models, mid) is True:
            return mid
    return None


# --------------------------------------------------------------------------- #
# JSON schemas for the built-in tools
#
# These mirror the parameter surface `venice mcp-serve` exposes (see
# `venice.mcp_server`), authored here as plain literals so nothing imports mcp.
# `confirm` / `max_spend` / `output_dir` are intentionally omitted (loop-injected).
# --------------------------------------------------------------------------- #
def _p(typ: str, desc: Optional[str] = None) -> dict:
    d = {"type": typ}
    if desc:
        d["description"] = desc
    return d


def _obj(props: dict, required: Optional[List[str]] = None) -> dict:
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


_IMAGE_SCHEMA = _obj(
    {
        "prompt": _p("string", "What to depict."),
        "model": _p("string", "Image model id (default: the catalog default)."),
        "variants": _p("integer", "How many images to generate, 1-4."),
        "format": _p("string", "Output format: png, webp, or jpeg."),
        "width": _p("integer"),
        "height": _p("integer"),
        "negative_prompt": _p("string"),
        "seed": _p("integer"),
        "cfg_scale": _p("number"),
        "steps": _p("integer"),
        "style_preset": _p("string"),
        "safe_mode": _p("boolean", "Blur adult/NSFW content. Defaults to on; set false to disable."),
        "hide_watermark": _p("boolean", "Omit the Venice watermark. Defaults to off; set true to hide it."),
    },
    required=["prompt"],
)

_TTS_SCHEMA = _obj(
    {
        "text": _p("string", "The text to speak."),
        "model": _p("string"),
        "voice": _p("string"),
        "format": _p("string", "Audio format, e.g. mp3, opus, wav."),
        "speed": _p("number", "0.25-4.0."),
    },
    required=["text"],
)

_BACKGROUND_PARAM = _p(
    "boolean",
    "Queue the render and return a job handle immediately instead of blocking "
    "(default false). When true, poll venice_job_status and fetch the file with "
    "venice_job_result using the returned queue_id, type, and model.",
)

_SFX_SCHEMA = _obj(
    {
        "prompt": _p("string", "What the sound effect should be."),
        "model": _p("string"),
        "duration": _p("integer", "Length in seconds."),
        "background": _BACKGROUND_PARAM,
    },
    required=["prompt"],
)

_MUSIC_SCHEMA = _obj(
    {
        "prompt": _p("string", "What the music/ambience should be."),
        "model": _p("string"),
        "duration": _p("integer", "Length in seconds."),
        "instrumental": _p("boolean", "Force an instrumental (no vocals)."),
        "lyrics": _p("string"),
        "speed": _p("number"),
        "background": _BACKGROUND_PARAM,
    },
    required=["prompt"],
)

_JOB_STATUS_SCHEMA = _obj(
    {
        "queue_id": _p("string", "The queue_id from a background venice_sfx/music/video call."),
        "type": _p("string", "sfx, music, or video -- the tool that started the job."),
        "model": _p("string", "The model id from the job handle."),
        "download_url": _p("string", "The download_url from the job handle (VPS video only)."),
    },
    required=["queue_id", "type", "model"],
)

_JOB_RESULT_SCHEMA = _obj(
    {
        "queue_id": _p("string", "The queue_id from a background venice_sfx/music/video call."),
        "type": _p("string", "sfx, music, or video -- the tool that started the job."),
        "model": _p("string", "The model id from the job handle."),
        "download_url": _p("string", "The download_url from the job handle (VPS video only)."),
        "max_wait": _p(
            "number",
            "Seconds to block-poll for the file (default 0 = one non-blocking "
            "attempt; returns status 'processing' if not ready yet). Capped at the "
            "render's server-side limit (300s audio, 900s video).",
        ),
    },
    required=["queue_id", "type", "model"],
)

_UPSCALE_SCHEMA = _obj(
    {
        "input_path": _p("string", "Path to a local image file to upscale."),
        "scale": _p("number", "Upscale factor, 1-4."),
        "enhance": _p("boolean"),
        "enhance_creativity": _p("number"),
        "enhance_prompt": _p("string"),
        "replication": _p("number"),
    },
    required=["input_path"],
)

_BG_REMOVE_SCHEMA = _obj(
    {
        "input_path": _p("string", "Path to a local image file."),
        "image_url": _p("string", "URL of an image (instead of input_path)."),
    },
)

_CHAT_SCHEMA = _obj(
    {
        "message": _p("string", "The message for the sub-completion."),
        "model": _p("string"),
        "system": _p("string"),
        "temperature": _p("number"),
        "max_tokens": _p("integer"),
        "web_search": _p("string", "One of auto, on, off."),
        "character": _p("string", "A Venice character Public ID slug."),
    },
    required=["message"],
)

#: The `venice_web_search` rail schema (#77). Deliberately minimal -- just the query.
#: The search mode and model are operator-controlled (flag/config), not model-facing, so
#: the model can't force search off or pick an arbitrary (possibly costly) model.
_WEB_SEARCH_SCHEMA = _obj(
    {
        "query": _p(
            "string",
            "What to look up on the web -- a question or search phrase. Returns a "
            "short answer plus the source URLs it cited.",
        ),
    },
    required=["query"],
)

_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Natural-language description of the code/text to find."),
        "k": _p("integer", "Number of results to return (default 8)."),
    },
    required=["query"],
)

_REINDEX_SCHEMA = _obj({})  # no parameters -- rebuilds the discovered .venice index

# Memory + task tools (#49). `scope` picks the tier (project rides the repo's
# .venice/, global travels with the agent); tasks are project-only (no scope).
_SCOPE_PROP = {
    "type": "string",
    "enum": ["project", "global"],
    "description": "Which memory tier: 'project' (default, rides the repo's .venice/ "
    "so subagents share it) or 'global' (user-global, travels with the agent).",
}
_TASK_STATUS_PROP = {
    "type": "string",
    "enum": list(_memory.TASK_STATUSES),
    "description": "Task status: pending, in_progress, or done.",
}
_MEMORY_WRITE_SCHEMA = _obj(
    {
        "name": _p("string", "Short slug id for the note (letters/digits/_.- only). "
                   "Reusing a name overwrites it."),
        "content": _p("string", "The note body to remember."),
        "scope": _SCOPE_PROP,
        "type": _p("string", "Optional kind, e.g. note/feedback/project/reference "
                   "(default: note)."),
        "description": _p("string", "Optional one-line summary shown in list/search."),
    },
    required=["name", "content"],
)
_MEMORY_READ_SCHEMA = _obj(
    {
        "name": _p("string", "The note's name."),
        "scope": _SCOPE_PROP,  # omit -> try project then global
    },
    required=["name"],
)
_MEMORY_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Substring to find in names/descriptions/bodies."),
        "scope": _SCOPE_PROP,  # omit -> search both tiers
    },
    required=["query"],
)
_MEMORY_LIST_SCHEMA = _obj({"scope": _SCOPE_PROP})  # omit -> both tiers; metadata only
_TASK_ADD_SCHEMA = _obj(
    {"text": _p("string", "What the task is.")},
    required=["text"],
)
_TASK_UPDATE_SCHEMA = _obj(
    {
        "id": _p("string", "The task id (from task_add/task_list)."),
        "status": _TASK_STATUS_PROP,
        "text": _p("string", "Optional new text for the task."),
    },
    required=["id"],
)
_TASK_LIST_SCHEMA = _obj({"status": _TASK_STATUS_PROP})  # omit -> all tasks

_MODELS_SCHEMA = _obj(
    {
        "type": {
            "type": "string",
            "enum": ["all", *MODEL_TYPES],
            "description": "Which catalog type to list model ids for "
            "(text, code, image, video, music, tts, embedding, upscale), "
            "or 'all' for a {type: [ids]} map.",
        },
    },
    required=["type"],
)

_MODEL_DETAILS_SCHEMA = _obj(
    {
        "model": _p("string", "The model id to describe (e.g. from venice_models)."),
    },
    required=["model"],
)

_VISION_SCHEMA = _obj(
    {
        "input_path": _p("string", "Path to a local image file to look at."),
        "image_url": _p("string", "URL of an image (instead of input_path)."),
        "prompt": _p(
            "string",
            "What to ask about the image (default: describe it in detail).",
        ),
        "model": _p(
            "string",
            "A vision-capable text model id (default: auto-picked from the catalog).",
        ),
        "max_tokens": _p("integer"),
    },
)

# Schema for a tool folded in ONLY via `only=` (e.g. `venice code --assets`), so it
# is not part of chat's default advertised set. Curated subset of
# `_mcp.image_edit_tool`; `confirm`/`max_spend`/`output_dir` omitted (loop-injected).
_IMAGE_EDIT_SCHEMA = _obj(
    {
        "prompt": _p("string", "Text directions for the edit, e.g. 'change the sky to a sunrise'."),
        "input_path": _p("string", "Path to a local base image to edit."),
        "image_url": _p("string", "URL of a base image (instead of input_path)."),
        "layer_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One or two local mask/overlay images (routes to /image/multi-edit).",
        },
        "model": _p("string", "Edit model id (default: the server picks one)."),
        "aspect_ratio": _p("string", "Output aspect ratio ('auto' infers from the input)."),
        "resolution": _p("string", "Output resolution tier, e.g. 1K/2K/4K."),
        "output_format": _p("string", "Output image format: png, jpeg, or webp."),
        "safe_mode": _p("boolean", "Blur adult/NSFW content. Defaults to on; set false to disable."),
    },
    required=["prompt"],
)

_VIDEO_SCHEMA = _obj(
    {
        "prompt": _p("string", "What the video should depict."),
        "model": _p("string", "Video model id (default: the catalog default)."),
        "duration": _p("string", "Clip length, e.g. '5s'."),
        "negative_prompt": _p("string"),
        "resolution": _p("string", "Output resolution tier, e.g. 720p/1080p."),
        "aspect_ratio": _p("string", "Output aspect ratio, e.g. 16:9."),
        "no_audio": _p("boolean", "Generate silent video (no soundtrack)."),
        "image_url": _p("string", "URL of a start/reference image (image-to-video)."),
        "end_image_url": _p("string", "URL of an end frame to interpolate toward."),
        "background": _BACKGROUND_PARAM,
    },
    required=["prompt"],
)

# Browser/web tools (#71). Rails like `shell`: the URL allow/deny policy is bound by the
# wiring, so `allow`/`deny` are DELIBERATELY absent from these schemas -- the model can't
# widen its own reach (mirrors how confirm/max_spend/output_dir are loop-injected).
_WEB_FETCH_SCHEMA = _obj(
    {
        "url": _p("string", "The http(s) URL to fetch."),
        "mode": _p("string", "text (default; HTML tags stripped) or html (raw)."),
        "max_bytes": _p("integer", "Cap on bytes downloaded."),
        "timeout": _p("integer", "Timeout in seconds."),
    },
    required=["url"],
)
_BROWSER_CAPTURE_SCHEMA = _obj(
    {
        "url": _p("string", "The http(s) URL to render."),
        "mode": _p(
            "string",
            "dom (default: post-JS HTML), text (DOM stripped to text), screenshot "
            "(writes a PNG, returns its path), or both. dom/text/both need a "
            "Chromium-family browser; Firefox is screenshot-only.",
        ),
        "wait_ms": _p("integer", "Milliseconds for JS to settle before capture."),
        "assert_contains": _p(
            "string",
            "Substring to check for in the rendered DOM; returns contains:true/false -- "
            "a deterministic 'did the JS land' check (dom/text/both modes).",
        ),
        "timeout": _p("integer", "Timeout in seconds."),
    },
    required=["url"],
)


class ToolSpec(NamedTuple):
    """One built-in tool's registry row (#50).

    The impl is stored by NAME and resolved via ``getattr(_mcp, impl)`` at
    :func:`builtin_tools` time, so a single source of truth wins and tests can patch
    ``_mcp.<impl>``. ``category`` (single, required) + optional ``tags`` are the
    capability axis the composition API (:func:`select`/:func:`tools_in`) reads.
    """

    name: str
    impl: str
    description: str
    parameters: dict
    paid: bool
    category: str
    tags: Tuple[str, ...] = ()


# The built-in venice tools. ``category`` reproduces the hand-maintained `only=`
# sets `code_tools` used to pass (see :func:`select`); it is ORTHOGONAL to the
# `_BUILTINS` vs `_CODE_ASSET_BUILTINS` split, which is the surface/advertisement
# axis (what `venice chat` shows by default).
_BUILTINS = [
    ToolSpec(
        "venice_image",
        "image_tool",
        "Generate 1-4 image variants from a text prompt via Venice /image/generate. "
        "Writes image file(s) and returns their paths (never inline blobs). Paid: "
        "over-cap calls need confirmation.",
        _IMAGE_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_tts",
        "tts_tool",
        "Synthesize speech from text via Venice /audio/speech. Writes an audio file "
        "and returns its path. Paid.",
        _TTS_SCHEMA,
        True,
        "audio",
    ),
    ToolSpec(
        "venice_sfx",
        "sfx_tool",
        "Generate a short sound effect via Venice's async audio queue (blocks with a "
        "capped wait). Writes an audio file and returns its path. Pass background=true "
        "to queue and return immediately, then fetch via venice_job_result. Paid.",
        _SFX_SCHEMA,
        True,
        "audio",
    ),
    ToolSpec(
        "venice_music",
        "music_tool",
        "Generate long-form music/ambience via Venice's async audio queue (blocks "
        "with a capped wait). Writes an audio file and returns its path. Pass "
        "background=true to queue and return immediately, then fetch via "
        "venice_job_result. Paid.",
        _MUSIC_SCHEMA,
        True,
        "audio",
    ),
    ToolSpec(
        "venice_upscale",
        "upscale_tool",
        "Upscale/enhance a local image (factor 1-4) via Venice /image/upscale. Writes "
        "the result and returns its path. Dynamic pricing, so it always needs "
        "confirmation.",
        _UPSCALE_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_bg_remove",
        "bg_remove_tool",
        "Remove an image's background via Venice /image/background-remove, returning a "
        "transparent PNG. Source is a local input_path OR an image_url. Dynamic "
        "pricing, so it always needs confirmation.",
        _BG_REMOVE_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_chat",
        "chat_tool",
        "Delegate a one-shot sub-completion to a Venice text model (optionally a "
        "different model or character) and return its reply text. Not spend-gated.",
        _CHAT_SCHEMA,
        False,
        "text",
    ),
    ToolSpec(
        "venice_models",
        "models_tool",
        "List available Venice model ids for a catalog type (text/code/image/video/"
        "music/tts/embedding/upscale, or 'all') via the free /models catalog. Use it "
        "to choose a valid `model` for the other venice_* tools instead of guessing. "
        "Read-only; not spend-gated.",
        _MODELS_SCHEMA,
        False,
        "catalog",
    ),
    ToolSpec(
        "venice_model_details",
        "model_details_tool",
        "Get one model's details: pricing (cost), capabilities (text models: "
        "supportsVision/supportsFunctionCalling/...), constraints (image/media "
        "models: aspectRatios, resolutions, qualities, promptCharacterLimit), and "
        "voices (TTS models: the valid voice ids for venice_tts) -- plus "
        "the full model_spec. Use it to budget input and confirm a model fits before "
        "using it. Read-only; not spend-gated.",
        _MODEL_DETAILS_SCHEMA,
        False,
        "catalog",
    ),
    ToolSpec(
        "venice_vision",
        "vision_tool",
        "Look at an image (a local input_path OR an image_url) with a vision-capable "
        "Venice text model and return what it sees as text. Optional prompt directs "
        "the question (default: a detailed description). Auto-picks a supportsVision "
        "model when model is omitted (see venice_model_details). Not spend-gated.",
        _VISION_SCHEMA,
        False,
        "vision",
    ),
    ToolSpec(
        "project_search",
        "search_tool",
        "Semantic search over the current project's local .venice index (built by "
        "`venice index`) for the chunks most relevant to a natural-language query. "
        "Returns file paths with line ranges and a short preview -- use it to locate "
        "code by meaning before reading files. Read-only; not spend-gated. Errors if "
        "no index exists yet. NOTE: results are a SNAPSHOT of the last index build; "
        "call reindex after editing files, or use grep for live matches.",
        _SEARCH_SCHEMA,
        False,
        "search",
    ),
    ToolSpec(
        "reindex",
        "reindex_tool",
        "Rebuild the project's .venice index so project_search reflects edits made "
        "this session (project_search is a snapshot; grep is live). Re-embeds only "
        "files whose contents changed, reusing the index's existing embedding "
        "backend. Takes no arguments. Paid (embeds changed files) -- always needs "
        "confirmation. Errors if no index exists yet (run `venice index` first).",
        _REINDEX_SCHEMA,
        True,
        "search",
    ),
    ToolSpec(
        "venice_job_status",
        "job_status_tool",
        "Peek at a backgrounded media render started with background=true on "
        "venice_sfx/venice_music/venice_video. Pass back the job handle's queue_id, "
        "type (sfx/music/video), and model. Returns processing/done/failed/not_found. "
        "Read-only, non-blocking; not spend-gated.",
        _JOB_STATUS_SCHEMA,
        False,
        "jobs",
    ),
    ToolSpec(
        "venice_job_result",
        "job_result_tool",
        "Fetch a backgrounded media render's file once ready (started with "
        "background=true). Pass back the job handle's queue_id, type, model (and "
        "download_url for VPS video). Writes the file and returns its path, or "
        "status 'processing' if not ready yet -- retry later. Free (charged at "
        "queue time); not spend-gated.",
        _JOB_RESULT_SCHEMA,
        False,
        "jobs",
    ),
]

# Extra paid tools NOT advertised by chat's default set. Folded in only when a
# caller passes `only=` (e.g. `venice code --assets`), so chat's default stays 8
# while `code_tools` can still select them by name.
_CODE_ASSET_BUILTINS = [
    ToolSpec(
        "venice_image_edit",
        "image_edit_tool",
        "Edit/inpaint an existing image via Venice /image/edit from a text prompt "
        "(base = a local input_path or an image_url; optional layer_paths route to "
        "/image/multi-edit for masks). Writes the result and returns its path. "
        "Dynamic pricing, so it always needs confirmation.",
        _IMAGE_EDIT_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_video",
        "video_tool",
        "Generate a short video via Venice's async video queue (blocks with a capped "
        "wait; can be slow). Optionally image-to-video from image_url. Writes an .mp4 "
        "and returns its path. Pass background=true to queue and return immediately, "
        "then fetch via venice_job_result. Dynamic pricing, so it always needs "
        "confirmation.",
        _VIDEO_SCHEMA,
        True,
        "video",
    ),
]


# --------------------------------------------------------------------------- #
# Composition API over the built-in registry (#50)
#
# `category` is the capability axis: it reproduces the hand-maintained `only=`
# name-sets `code_tools` used to pass, so a caller selects tools by capability
# instead of enumerating names. Read over the UNION of both registries so a
# `_CODE_ASSET_BUILTINS`-only tool (venice_image_edit/venice_video) is selectable
# by its category. This is orthogonal to the `_BUILTINS`/`_CODE_ASSET_BUILTINS`
# split, which stays the surface/advertisement axis for `builtin_tools(only=None)`.
# --------------------------------------------------------------------------- #
_REGISTRY = _BUILTINS + _CODE_ASSET_BUILTINS


def get(name: str) -> Optional[ToolSpec]:
    """The registry row for `name` (metadata only, no client), or None."""
    for spec in _REGISTRY:
        if spec.name == name:
            return spec
    return None


def list_categories() -> set:
    """Every category present in the built-in registry."""
    return {spec.category for spec in _REGISTRY}


def tools_in(category: str) -> set:
    """The names of registry tools in `category` (empty set if none)."""
    return {spec.name for spec in _REGISTRY if spec.category == category}


def select(categories=None, names=None, exclude=None) -> set:
    """A set of built-in tool names selected by capability.

    `categories` and/or `names` union into the selection (both None selects the
    whole registry); `exclude` (names or categories) is subtracted last. Unknown
    categories/names are simply ignored here -- the authoritative unknown-name guard
    stays in :func:`builtin_tools` (whose ValueError drives chat's exit 2), so this
    stays a pure name-set helper the `code_tools` call sites can compose with.
    """
    chosen = set()
    if categories is None and names is None:
        chosen = {spec.name for spec in _REGISTRY}
    else:
        if categories:
            for cat in categories:
                chosen |= tools_in(cat)
        if names:
            known = {spec.name for spec in _REGISTRY}
            chosen |= {n for n in names if n in known}
    if exclude:
        exclude = set(exclude)
        chosen -= exclude
        chosen -= {spec.name for spec in _REGISTRY if spec.category in exclude}
    return chosen


# Loop-controlled kwargs the model must never supply (stripped defensively).
_CONTROLLED = ("confirm", "max_spend", "output_dir")


def _clean(arguments) -> dict:
    if not isinstance(arguments, dict):
        return {}
    return {k: v for k, v in arguments.items() if k not in _CONTROLLED}


def _tool_section(name: str) -> str:
    """Config section for a tool: `venice_image` -> `image` (matches userconfig
    `_COMMAND_MAP` / the CLI command). Tools with no matching section (e.g.
    `venice_models`, `project_search`) simply resolve nothing."""
    return name[len("venice_"):] if name.startswith("venice_") else name


def _browser_args(arguments) -> dict:
    """Model-supplied browser-tool args with policy/loop-controlled keys stripped: the
    model must not set `allow`/`deny` (widen its URL policy) or the loop-controlled keys."""
    return {k: v for k, v in _clean(arguments).items() if k not in ("allow", "deny")}


def browser_tools(*, allow=(), deny=(), output_dir=None, config=None) -> List[Tool]:
    """The `web_fetch` + `browser_capture` rails (issue #71).

    The URL allow/deny policy is bound HERE (from the operator's config/flags), so the
    model can't widen it via tool arguments -- same discipline as the `shell` rail. Safe
    knobs still honor `defaults.browser.*` (#58), layered under the model's arguments.
    Both tools are free (no spend gate) and never require confirmation; the URL policy is
    the guard.
    """
    fetch_defaults = userconfig.config_defaults_for("browser", _mcp.web_fetch_tool, config)
    cap_defaults = userconfig.config_defaults_for("browser", _mcp.browser_capture_tool, config)

    def _web_fetch_invoke(arguments, *, confirm: bool = False):
        return _mcp.web_fetch_tool(
            allow=allow, deny=deny, **{**fetch_defaults, **_browser_args(arguments)})

    def _browser_capture_invoke(arguments, *, confirm: bool = False):
        return _mcp.browser_capture_tool(
            allow=allow, deny=deny, output_dir=output_dir,
            **{**cap_defaults, **_browser_args(arguments)})

    return [
        Tool(
            name="web_fetch",
            description=(
                "Fetch an http(s) URL with stdlib urllib and return its text (mode=text, "
                "default) or raw HTML (mode=html). Zero-dep; good for non-SPA pages. For "
                "JS-rendered pages use browser_capture. Read-only; not spend-gated. "
                "file://, the cloud metadata endpoint, and any host the operator denies "
                "are refused."
            ),
            parameters=_WEB_FETCH_SCHEMA,
            invoke=_web_fetch_invoke,
            paid=False,
            category="web",
            tags=("read", "network"),
        ),
        Tool(
            name="browser_capture",
            description=(
                "Headless-render an http(s) URL and return the post-JS DOM (mode=dom/text) "
                "and/or a screenshot PNG path (mode=screenshot/both) -- use it to verify a "
                "page's JS-injected content actually appeared. Pass assert_contains to "
                "check the DOM contains a substring (deterministic). DOM modes need a "
                "Chromium-family browser (Firefox is screenshot-only); reports 'no "
                "headless browser available' when none is installed. Read-only; not "
                "spend-gated."
            ),
            parameters=_BROWSER_CAPTURE_SCHEMA,
            invoke=_browser_capture_invoke,
            paid=False,
            category="web",
            tags=("read", "network"),
        ),
    ]


def memory_tools() -> List[Tool]:
    """The persistent memory + task rails (issue #49).

    Free, local, stdlib-only tools over the agent's own durable store (`_memory`):
    four `memory_*` tools (two-tier notes -- project rides `<root>/.venice/memory`,
    global rides `~/.config/venice/memory`) and three `task_*` tools (a project-only
    checklist). Like the shell/browser rails they are NOT in `_REGISTRY` (so they
    don't bloat chat's default advertised set) and are appended only when the caller
    opts in via `builtin_tools(memory=True)` / `code_tools(memory=True)` -- the #52
    planner enables them for a subagent the same way. Categories `memory`/`tasks`
    live on the built Tools for downstream iterators, not in the registry taxonomy.
    """
    def _free(impl):
        def invoke(arguments, *, confirm: bool = False):
            return impl(None, **_clean(arguments))
        return invoke

    return [
        Tool(
            name="memory_write",
            description=(
                "Save a durable note you can recall in a later step or session. "
                "scope='project' (default) rides the repo's .venice/ so subagents "
                "share it; scope='global' travels with you across projects. Reusing "
                "a name overwrites it."
            ),
            parameters=_MEMORY_WRITE_SCHEMA,
            invoke=_free(_mcp.memory_write_tool),
            paid=False,
            category="memory",
            tags=("write",),
        ),
        Tool(
            name="memory_read",
            description=(
                "Read one saved note by name (returns its body + metadata). Omit "
                "scope to try project then global."
            ),
            parameters=_MEMORY_READ_SCHEMA,
            invoke=_free(_mcp.memory_read_tool),
            paid=False,
            category="memory",
            tags=("read",),
        ),
        Tool(
            name="memory_search",
            description=(
                "Find saved notes by a plain substring over names/descriptions/"
                "bodies. Omit scope to search both tiers; each hit is tagged with "
                "its scope + a preview."
            ),
            parameters=_MEMORY_SEARCH_SCHEMA,
            invoke=_free(_mcp.memory_search_tool),
            paid=False,
            category="memory",
            tags=("read",),
        ),
        Tool(
            name="memory_list",
            description=(
                "List saved notes (names/types/descriptions/timestamps only, no "
                "bodies) -- the cheap index to decide what to memory_read. Omit "
                "scope to list both tiers."
            ),
            parameters=_MEMORY_LIST_SCHEMA,
            invoke=_free(_mcp.memory_list_tool),
            paid=False,
            category="memory",
            tags=("read",),
        ),
        Tool(
            name="task_add",
            description=(
                "Add a task to the project checklist (starts 'pending'). Use it to "
                "track multi-step work so progress survives across turns/resume."
            ),
            parameters=_TASK_ADD_SCHEMA,
            invoke=_free(_mcp.task_add_tool),
            paid=False,
            category="tasks",
            tags=("write",),
        ),
        Tool(
            name="task_update",
            description=(
                "Update a task by id: set status (pending/in_progress/done) and/or "
                "change its text. Mark a task in_progress when you start it and done "
                "when finished."
            ),
            parameters=_TASK_UPDATE_SCHEMA,
            invoke=_free(_mcp.task_update_tool),
            paid=False,
            category="tasks",
            tags=("write",),
        ),
        Tool(
            name="task_list",
            description=(
                "List the project's tasks (optionally filtered by status) to see "
                "what's left."
            ),
            parameters=_TASK_LIST_SCHEMA,
            invoke=_free(_mcp.task_list_tool),
            paid=False,
            category="tasks",
            tags=("read",),
        ),
    ]


def builtin_tools(
    client,
    *,
    max_spend: Optional[float] = None,
    output_dir: Optional[str] = None,
    only: Optional[set] = None,
    config: Optional[dict] = None,
    shell: bool = False,
    shell_root: Optional[str] = None,
    shell_allow=(),
    shell_deny=(),
    browser: bool = False,
    browser_allow=(),
    browser_deny=(),
    browser_output_dir: Optional[str] = None,
    memory: bool = False,
    exec_timeout: int = _exec.DEFAULT_EXEC_TIMEOUT,
) -> List[Tool]:
    """Build the in-process venice tools, bound to `client`.

    `max_spend`/`output_dir` are baked into the paid tools' closures; `confirm` is
    passed per-call by the loop. `only` restricts the set to the named tools (an
    unknown name raises ValueError so the caller can exit 2). With `only=None` the
    set is exactly `_BUILTINS` (chat's default); passing `only=` also makes the
    `_CODE_ASSET_BUILTINS` extras (e.g. `venice_image_edit`) selectable.

    `config` is a userconfig doc (issue #58): `defaults.<section>.*` values are
    layered UNDER the model's tool arguments, so an explicit tool arg still wins
    (precedence: model arg > config default > tool hardcoded default). Only keys
    in `userconfig._COMMAND_MAP[section]` (the #57 allow-list) that the tool
    function actually accepts are injected.

    `shell` (issue #33) appends a gated `shell` exec tool bound to `shell_root`
    (the same `_exec.run_cmd` rail `venice code`'s `run` uses), scoped by the
    `shell_allow`/`shell_deny` policy. It is added AFTER the `only` filter (it is a
    rail, not a venice API tool, so it isn't part of the selectable `_BUILTINS`
    set) and is never exposed via `mcp-serve`, which builds its own wrappers.

    `browser` (issue #71) likewise appends the `web_fetch`/`browser_capture` rails,
    scoped by the `browser_allow`/`browser_deny` URL policy (see `browser_tools`).

    `memory` (issue #49) appends the persistent memory + task rails (`memory_tools`):
    free, local notes (two tiers) + a project task list. Also a rail (added after the
    `only` filter, absent from `_BUILTINS`/`mcp-serve`).
    """

    def _config_defaults(section, impl) -> dict:
        # #58: shared with mcp-serve -- layer defaults.<section>.* under tool args.
        return userconfig.config_defaults_for(section, impl, config)

    def _make_paid(impl, section):
        defaults = _config_defaults(section, impl)

        def invoke(arguments, *, confirm: bool = False):
            return impl(
                client,
                confirm=confirm,
                max_spend=max_spend,
                output_dir=output_dir,
                **{**defaults, **_clean(arguments)},
            )

        return invoke

    def _make_free(impl, section):
        defaults = _config_defaults(section, impl)

        def invoke(arguments, *, confirm: bool = False):
            return impl(client, **{**defaults, **_clean(arguments)})

        return invoke

    source = _BUILTINS if only is None else _BUILTINS + _CODE_ASSET_BUILTINS
    tools = [
        Tool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            invoke=(
                _make_paid(getattr(_mcp, spec.impl), _tool_section(spec.name))
                if spec.paid
                else _make_free(getattr(_mcp, spec.impl), _tool_section(spec.name))
            ),
            paid=spec.paid,
            category=spec.category,
            tags=spec.tags,
        )
        for spec in source
    ]

    if only is not None:
        known = {t.name for t in tools}
        unknown = only - known
        if unknown:
            raise ValueError(
                "unknown tool(s): "
                + ", ".join(sorted(unknown))
                + "; available: "
                + ", ".join(sorted(known))
            )
        tools = [t for t in tools if t.name in only]

    if shell:
        root = shell_root or "."

        def _shell_invoke(arguments, *, confirm: bool = False):
            return _exec.run_cmd(
                root, confirm=confirm, exec_timeout=exec_timeout,
                allow=shell_allow, deny=shell_deny, **_clean(arguments),
            )

        tools.append(Tool(
            name="shell",
            description=(
                "Run a shell command (/bin/sh -c) with the working directory set to "
                f"{root}; returns exit code + captured output. Use for gh/git/curl/"
                "build/test automation. Requires confirmation. A command blocked by "
                "the operator's allow/deny policy is refused (see the error message)."
            ),
            parameters=_exec._RUN_SCHEMA,
            invoke=_shell_invoke,
            paid=True,
            category="exec",
            tags=("exec", "mutate"),
        ))

    if browser:
        tools.extend(browser_tools(
            allow=browser_allow, deny=browser_deny,
            output_dir=browser_output_dir, config=config,
        ))

    if memory:
        tools.extend(memory_tools())
    return tools


# --------------------------------------------------------------------------- #
# Capability guard
# --------------------------------------------------------------------------- #
def supports_function_calling(models, model_id) -> Optional[bool]:
    """Whether `model_id` advertises function calling in the catalog.

    True/False when the model is found and carries the (required) capability;
    None when it can't be determined (no catalog, model absent, or the field is
    missing) -- the caller then attempts the loop with a soft note.
    """
    return _models.supports_capability(models, model_id, "supportsFunctionCalling")


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def _assistant_dict(msg) -> dict:
    """Reconstruct an assistant turn for the message history (explicit, not
    model_dump()) so the follow-up tool messages carry the exact tool_call_ids."""
    d = {"role": "assistant", "content": (getattr(msg, "content", None) or "")}
    tcs = getattr(msg, "tool_calls", None) if msg is not None else None
    if tcs:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ]
    return d


# --------------------------------------------------------------------------- #
# Live progress (#54): a spinner while the model thinks + a line per tool call.
# All output is stderr and TTY-gated, so piped/`--json`/test runs stay silent.
# --------------------------------------------------------------------------- #
_SPIN_FRAMES = "|/-\\"


class _Spinner:
    """A tiny stderr spinner shown while awaiting the model.

    A no-op unless stderr is a TTY (so automation and the test-suite's StringIO
    stderr stay clean). Runs on a daemon thread; the line is cleared on exit.
    """

    def __init__(self, label: str = "working", *, enabled: bool = True):
        self._enabled = enabled and sys.stderr.isatty()
        self._label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if self._enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self):  # pragma: no cover - timing/thread, exercised via a fake TTY
        for frame in itertools.cycle(_SPIN_FRAMES):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r{frame} {self._label}… ")
            sys.stderr.flush()
            self._stop.wait(0.12)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            sys.stderr.write("\r\033[K")  # clear the spinner line
            sys.stderr.flush()
        return False


def _short_args(raw: str) -> str:
    """A compact, safe one-line summary of a tool call's arguments (never raises)."""
    try:
        args = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return ""
    if not isinstance(args, dict):
        return ""
    for key in ("path", "file", "command", "query", "prompt", "message", "pattern"):
        val = args.get(key)
        if isinstance(val, (str, int, float)):
            s = str(val).replace("\n", " ")
            return f"{key}={s[:57] + '...' if len(s) > 60 else s}"
    return ", ".join(sorted(args)[:3])


def _progress(text: str, *, enabled: bool) -> None:
    if enabled and sys.stderr.isatty():
        print(text, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Confirm gate (#55): a paid/side-effecting tool can prompt on a TTY. `a`/`all`
# accepts this call AND flips the run's gate to auto so nothing else prompts.
# --------------------------------------------------------------------------- #
def _prompt_yes() -> str:
    """Return "yes" (this call), "all" (this call + auto-accept the rest of the
    run), or "no". EOF -> "no"."""
    # #79: under attached Ctrl+C steering a SIGINT handler is installed around the loop;
    # restore the default handler for this confirm so Ctrl+C here aborts (as it always
    # did) rather than arming a steer the operator can't see while waiting to answer.
    from . import _steer
    try:
        with _steer.default_sigint():
            ans = input("Proceed? [y]es / [a]ll (accept rest) / [N]o ").strip().lower()
    except EOFError:
        return "no"
    if ans in ("a", "all"):
        return "all"
    if ans in ("y", "yes"):
        return "yes"
    return "no"


@contextlib.contextmanager
def _invoke_window(acc):
    """Bank one ``tool.invoke`` window into ``acc[0]`` (#82).

    A one-element list, mirroring `code._human_pause` -- the house pattern for "the
    caller owns the clock and the timing lives in a local rather than on an object".
    `CostLedger.record_tool` stays a pure accumulator; this is the caller side of that
    split.

    Banked in `finally`, so a tool that RAISED still reports the seconds it burned
    before raising. That is real waiting, and a crash loop that self-reports 0.0s hides
    precisely its own cost.
    """
    t = time.monotonic()
    try:
        yield
    finally:
        acc[0] += time.monotonic() - t


def _resolve_spend(tool: Tool, arguments: dict, result, gate: dict, *, window=None):
    """Hybrid gate: prompt on a TTY, else feed the block back to the model.

    `gate` is the run's mutable auto-accept holder (`{"auto": bool}`); answering
    `all` at the prompt sets ``gate["auto"] = True`` so subsequent paid calls in
    the same run skip the gate. Only reached for a paid tool that returned
    `confirmation_required` (which happens only while ``gate["auto"]`` is False).
    """
    # #82: `window` is `_run_one_call`'s one-element accumulator -- the re-invoke below
    # is the SAME tool call, so its seconds belong to the same row. None -> a throwaway,
    # so this stays callable standalone (a signature break is against house rules).
    if window is None:
        window = [0.0]
    if not tool.paid or gate["auto"]:
        return result
    if not (isinstance(result, dict) and result.get("status") == "confirmation_required"):
        return result
    message = result.get("message", f"{tool.name}: confirmation required")
    if sys.stdin.isatty():
        print(message, file=sys.stderr)
        ans = _prompt_yes()
        if ans in ("yes", "all"):
            if ans == "all":
                gate["auto"] = True
            try:
                with _invoke_window(window):  # #82: same call, same row
                    return tool.invoke(arguments, confirm=True)
            except Exception as e:  # pragma: no cover - impls shouldn't raise
                return {"status": "error", "message": f"{tool.name} failed: {e}"}
        print(f"{tool.name}: declined by user", file=sys.stderr)
    return result  # non-TTY or declined -> the model sees the gate and adapts


def _run_one_call(tc, dispatch: Dict[str, Tool], gate: dict, *, on_tool=None) -> dict:
    """Run one tool call and return the model-visible result dict.

    `on_tool` (#82) is an optional ``(name, seconds) -> None`` sink for the call's
    EXECUTION time -- `CostLedger.record_tool` when `run_loop` was given a ledger. A
    callback rather than the ledger itself for two reasons: this function's return value
    is `json.dumps`'d straight into the `tool` message the MODEL reads (see both
    callers), so a duration must never ride it; and the dispatch layer should not have
    to learn what a ledger is in order to say how long something took.

    What is timed, exactly:

    * only `tool.invoke`. The three validation returns below never entered a tool, so
      they are not tool time -- stamping them would land ~0.0s rows that deflate the
      per-call average, and a turn full of rejected calls would report suspiciously
      fast tools.
    * BOTH invokes of a confirm-gated paid tool, summed into one row and ONE call. The
      operator's read-time at the `Proceed?` prompt falls BETWEEN the two windows and is
      excluded for free -- `chat._finish` documents that wait as a known hole in the
      WALL clock; it is deliberately not a hole here.
    * a tool that raised (the window closes in `finally`).
    """
    tool = dispatch.get(tc.function.name)
    if tool is None:
        return {"status": "error", "message": f"unknown tool {tc.function.name!r}"}
    try:
        arguments = json.loads(tc.function.arguments or "{}")
    except (TypeError, ValueError) as e:
        return {"status": "error", "message": f"invalid JSON arguments: {e}"}
    if not isinstance(arguments, dict):
        return {"status": "error", "message": "tool arguments must be a JSON object"}
    # #82: opened BELOW the validation returns on purpose -- see the docstring.
    window = [0.0]
    try:
        try:
            with _invoke_window(window):
                result = tool.invoke(arguments, confirm=bool(gate["auto"]))
        except Exception as e:  # pragma: no cover - impls shouldn't raise
            return {"status": "error", "message": f"{tool.name} failed: {e}"}
        return _resolve_spend(tool, arguments, result, gate, window=window)
    finally:
        # One flush per CALL, after any confirm re-invoke, on every path out of here
        # including the raise above. `tool.name` rather than `tc.function.name`:
        # identical by construction of `dispatch_map`, but the tool's is canonical.
        if on_tool is not None:
            on_tool(tool.name, window[0])


def _dispatch_parallel(
    tool_calls,
    dispatch: Dict[str, Tool],
    gate: dict,
    messages: List[dict],
    *,
    calls_made: int,
    max_tool_calls: int,
    unlimited: bool,
    show: bool,
    on_tool=None,
) -> int:
    """Run one assistant turn's tool calls with subagent dispatches executed concurrently.

    The batched counterpart of :func:`run_loop`'s serial loop, used only under
    ``--parallel`` (#52). Calls in :data:`_PARALLELIZABLE` (``venice_scout``/
    ``venice_spawn``) run on a bounded thread pool; every other call runs serially. ALL
    loop bookkeeping stays here on the MAIN thread -- the pool workers only run the
    isolated nested ``tool.invoke`` (via :func:`_run_one_call`, which turns any exception
    into an error dict, so a worker never raises and can't poison the pool). Results are
    appended to ``messages`` in ORIGINAL ``tool_calls`` order (the OpenAI message
    contract: each ``tool`` message answers its assistant ``tool_calls`` entry), and the
    tool-call budget is honored exactly as the serial path does. Returns the updated
    ``calls_made``.

    ``on_tool`` (#82) is the ONE exception to the main-thread rule above: a tool's window
    closes where the tool ran, so a worker calls the sink directly.
    :meth:`CostLedger.record_tool` takes a lock for exactly this -- two ``venice_spawn``
    calls in one batch land on the same key from two threads.
    """
    n = len(tool_calls)
    # Budget allotment up front: the first `slots` calls (original order) run; the rest
    # are reported not-executed WITHOUT running -- identical outcome to the serial loop.
    slots = n if unlimited else max(0, max_tool_calls - calls_made)
    results: List[Optional[dict]] = [None] * n
    not_executed = {
        "status": "error",
        "message": "tool-call budget (--max-tool-calls) exhausted; not executed",
    }
    par_idx: List[int] = []
    ser_idx: List[int] = []
    for i, tc in enumerate(tool_calls):
        if i >= slots:
            results[i] = not_executed
        elif _is_parallelizable(tc):
            par_idx.append(i)
        else:
            ser_idx.append(i)

    # Announce the executable batch up front, in ORIGINAL order (deterministic; avoids
    # progress half-lines interleaving once workers start). stderr+TTY-gated -> a no-op
    # in tests/pipes/--json.
    for i in range(n):
        if i < slots:
            _progress(
                f"· {tool_calls[i].function.name} "
                f"{_short_args(tool_calls[i].function.arguments)}".rstrip(),
                enabled=show,
            )

    # Parallel batch: subagent calls run concurrently on a bounded pool.
    if par_idx:
        workers = min(_max_parallel(), len(par_idx))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="venice-subagent"
        ) as ex:
            futs = {
                ex.submit(_run_one_call, tool_calls[i], dispatch, gate,
                          on_tool=on_tool): i
                for i in par_idx
            }
            try:
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
            except BaseException:  # incl. KeyboardInterrupt on the main thread (#79)
                ex.shutdown(wait=False, cancel_futures=True)
                raise

    # Serial remainder, in original order (paid tools + the confirm gate stay unchanged).
    for i in ser_idx:
        results[i] = _run_one_call(tool_calls[i], dispatch, gate, on_tool=on_tool)

    # Commit on the main thread: append in ORIGINAL order, advance by the executed count.
    for i, tc in enumerate(tool_calls):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(results[i], default=str),
            }
        )
    return calls_made + min(slots, n)


def _emit_final(resp, json_out: bool) -> int:
    if json_out:
        json.dump(resp.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0
    content = ""
    if getattr(resp, "choices", None):
        content = resp.choices[0].message.content or ""
    print(content)
    # Reuse chat's citation printer; lazy import keeps this module chat-agnostic.
    from . import chat as _chat

    _chat._print_citations(getattr(resp, "venice_parameters", None))
    return 0


def run_loop(
    oai,
    model: str,
    messages: List[dict],
    base_kwargs: dict,
    tools: List[Tool],
    *,
    max_tool_calls: int,
    yes: bool,
    json_out: bool,
    budget: Optional[_compact.Budget] = None,
    ledger: Optional[CostLedger] = None,
    steer_drain: Optional[Callable[[], List[str]]] = None,
    parallel: bool = False,
) -> int:
    """Drive the function-calling loop until the model stops (or the cap is hit).

    `messages` is the persistent, mutable history (seeded with system+user).
    `base_kwargs` are per-turn generation kwargs (temperature/max_tokens/extra_body)
    re-applied on every create(); it must NOT contain `model`/`messages`. Non-streamed
    by design (tool-call deltas would need fragment reassembly; v1 buffers each turn).
    Only `openai.OpenAIError` from create() is fatal -- the caller maps it to an exit
    code; tool failures come back as dicts the model can recover from.

    `budget` (issue #48) enables auto-compaction: when given, each turn first
    records the previous response's `usage` and, once the prompt would exceed
    `budget.threshold_tokens`, summarizes the older prefix into one synthetic
    system message (keeping the system prompt + last `budget.keep_turns` turns
    verbatim). Compaction mutates `messages` in place and is best-effort: a
    failed summary call leaves the history alone.

    `ledger` (issue #66) meters chat-completion spend: each turn's `usage` is
    recorded against the session model's per-token rate, and once accumulated
    cost reaches `ledger.max_spend` the loop stops starting new paid turns and
    forces a final answer (the model wraps up with the history it has). The
    gate is post-response (chat has no pre-call quote), so it bounds *further*
    spend rather than preempting a turn already in flight.

    `steer_drain` (issue #78) enables mid-run steering: a callable returning any
    queued steering messages (from the session's file mailbox). It's polled at the
    top of each turn -- the natural checkpoint, after the previous turn's tool
    results were all appended -- and each message is appended as a tagged user turn
    so the model consumes it exactly as if the operator had typed it. Draining does
    NOT reset the spend/tool-call budgets (a steer is additive input, not a reset).
    """
    oai_tools = to_openai_tools(tools)
    dispatch = dispatch_map(tools)
    calls_made = 0
    gate = {"auto": bool(yes)}  # mutable so an `a`/`all` confirm flips the run to auto
    # #82: the per-tool timing sink, bound ONCE so both dispatch paths -- and every pool
    # worker under --parallel -- hand their windows to the same ledger. None when
    # unmetered, which keeps the two dispatch helpers ledger-free and makes every call
    # site a single `is not None`. `record_tool` is thread-safe; see its lock.
    on_tool = ledger.record_tool if ledger is not None else None
    # `--max-tool-calls 0` (or None) means unlimited -- run until the model stops
    # on its own (bounded in practice by the model's context window).
    unlimited = max_tool_calls is None or max_tool_calls <= 0
    show = not json_out  # progress feedback (further TTY-gated inside the helpers)
    if parallel:
        # Install the thread-local stdout router on the MAIN thread before any subagent
        # worker starts, so workers only ever push/pop a target and never race on install.
        _install_router()

    def _force_final(reason: str) -> int:
        print(reason, file=sys.stderr)
        _t0 = time.monotonic()
        with _Spinner("finishing", enabled=show):
            resp = oai.chat.completions.create(
                model=model,
                messages=messages,
                tools=oai_tools,
                tool_choice="none",
                **base_kwargs,
            )
        if ledger is not None:
            # #99: caller-stamped, like every other window in this file. NOT added to
            # `elapsed_seconds` -- that is `record_turn`'s job at the command level, and
            # stamping it here too would double-count the same wall time.
            ledger.record(getattr(resp, "usage", None),
                          seconds=time.monotonic() - _t0)
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        return _emit_final(resp, json_out)

    while True:
        # Mid-run steering (#78): drain any queued steers at the checkpoint boundary
        # (all prior tool results are appended, so a user turn here is contract-valid)
        # and consume them as tagged user turns before the next model call. Placed
        # before the spend gate so even a gate-forced final answer sees the steer.
        if steer_drain is not None:
            for _steer in steer_drain():
                messages.append({
                    "role": "user",
                    "content": "[steering message received mid-run]\n" + _steer,
                })
        # Spend gate (#66): don't start a new paid turn once the cap is hit.
        if ledger is not None and ledger.over():
            return _force_final(
                f"chat: reached --max-spend ({ledger.summary()}); "
                "requesting a final answer"
            )
        # Token gate (#52): a per-subagent cumulative-token ceiling, orthogonal to the USD
        # cap above. Only ever set on a disposable subagent ledger (the parent chat/REPL
        # ledger has max_tokens=None -> inert here). Post-turn like the spend gate, so it
        # bounds the *next* turn -- the crossing turn + this forced final both complete.
        if ledger is not None and ledger.over_tokens():
            return _force_final(
                f"code: worker reached token cap {ledger.max_tokens:,} "
                f"({ledger.summary()}); wrapping up"
            )
        _compact.maybe_compact(
            oai, model, messages, budget, base_kwargs,
            on_compact=lambda b, a: _progress(
                f"(auto-compacted history: {b} -> {a} messages)", enabled=show,
            ),
            ledger=ledger,  # #99: log the event; the summary call itself stays unmetered
        )
        _t0 = time.monotonic()
        with _Spinner("thinking", enabled=show):
            resp = oai.chat.completions.create(
                model=model,
                messages=messages,
                tools=oai_tools,
                tool_choice="auto",
                **base_kwargs,
            )
        if budget is not None:
            budget.observe(getattr(resp, "usage", None))
        if ledger is not None:
            # #99: see `_force_final` -- caller-stamped, and deliberately NOT folded into
            # `elapsed_seconds`. Unlike the per-tool windows these are strictly serial on
            # any one ledger (subagents each get their own), so their sum can never exceed
            # wall and `_call_lines` needs no `[concurrent]` marker.
            ledger.record(getattr(resp, "usage", None),
                          seconds=time.monotonic() - _t0)
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
        if not tool_calls:
            return _emit_final(resp, json_out)

        # Every tool_call in the turn must get a result (message-contract), even
        # ones past the budget -- those are reported not-executed rather than run.
        if parallel and any(_is_parallelizable(tc) for tc in tool_calls):
            # #52: run independent subagent dispatches concurrently. All bookkeeping
            # (result append in original order, budget) stays on the main thread.
            calls_made = _dispatch_parallel(
                tool_calls, dispatch, gate, messages,
                calls_made=calls_made, max_tool_calls=max_tool_calls,
                unlimited=unlimited, show=show, on_tool=on_tool,
            )
        else:
            for tc in tool_calls:
                if not unlimited and calls_made >= max_tool_calls:
                    result = {
                        "status": "error",
                        "message": "tool-call budget (--max-tool-calls) exhausted; "
                        "not executed",
                    }
                else:
                    _progress(
                        f"· {tc.function.name} {_short_args(tc.function.arguments)}".rstrip(),
                        enabled=show,
                    )
                    result = _run_one_call(tc, dispatch, gate, on_tool=on_tool)
                    calls_made += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(result, default=str),
                    }
                )

        if not unlimited and calls_made >= max_tool_calls:
            # The forced-final is the turn a long, over-budget run most needs
            # compacted -- it returns without re-entering the loop, so compact
            # here too or it ships the full history (#48).
            _compact.maybe_compact(
                oai, model, messages, budget, base_kwargs,
                on_compact=lambda b, a: _progress(
                    f"(auto-compacted history: {b} -> {a} messages)", enabled=show,
                ),
                ledger=ledger,  # #99
            )
            return _force_final(
                f"chat: reached --max-tool-calls ({max_tool_calls}); "
                "requesting a final answer"
            )


# --------------------------------------------------------------------------- #
# Scout subagent (#52 slice 1): a disposable, read-only "context firewall".
#
# The multi-agent epic's day-one primitive (see the operator's note on #52) is NOT
# a role-specialized worker but a *context firewall*: delegate "figure out X, report
# back concisely" to a subagent that runs :func:`run_loop` once on a FRESH message
# list with only read-only tools, then return just its conclusion. The planner's
# working context never sees the subagent's exploration (dozens of reads/greps), so
# it stays clean and lossless -- cheaper than compaction, which bounds pollution
# after the fact rather than preventing it.
#
# ``AgentProfile`` already framed this as "run the core with a profile + task"; this
# is the non-interactive core that framing pointed at. The executable read-only
# tool-builder + the ``venice_scout`` Tool wrapper live in ``_code`` (which owns the
# fs read tools); this module owns only the profile-agnostic core so it stays
# import-clean (never importing ``_code``/``code``).
# --------------------------------------------------------------------------- #
SCOUT_TOOL_NAME = "venice_scout"

SCOUT_SYSTEM = (
    "You are a SCOUT subagent: a disposable, read-only investigator spun up to "
    "answer one question for a coding agent, then discarded. You start from a fresh "
    "context and have ONLY read-only tools (read files, list directories, grep, "
    "read-only git, and -- when an index exists -- semantic search). You CANNOT and "
    "must NOT edit files, run commands, or make any change; if the task implies a "
    "change, investigate what the change would involve and report, do not attempt "
    "it.\n\n"
    "Investigate efficiently: prefer a few targeted reads/greps over broad sweeps, "
    "and stop as soon as you can answer. Your caller only sees your final report -- "
    "not your tool calls -- so the report must stand on its own.\n\n"
    "End with a report using EXACTLY these sections:\n"
    "FINDINGS: the direct answer, concrete (cite file paths / line numbers / "
    "symbols you actually saw).\n"
    "CONFIDENCE: high | medium | low, plus one clause on why.\n"
    "DEAD-ENDS: paths you tried that led nowhere (so the caller doesn't retry them); "
    "'none' if none.\n"
    "NOT CHECKED: what you did not verify or that was out of scope -- be honest "
    "about gaps.\n"
    "VERIFIED-LIVE vs HYPOTHETICAL: which claims you confirmed by reading actual "
    "files/output vs. inferred without checking.\n"
)

# The section headers SCOUT_SYSTEM mandates, in order. The single source of truth for
# :func:`_parse_sections` -- keep in lockstep with the prompt text above. Match the exact
# casing/punctuation the prompt uses (hyphen in ``DEAD-ENDS``/``VERIFIED-LIVE``, the space
# in ``NOT CHECKED``, lowercase `` vs ``); parsing is case-insensitive but the returned keys
# are these canonical strings.
SCOUT_SECTIONS = (
    "FINDINGS",
    "CONFIDENCE",
    "DEAD-ENDS",
    "NOT CHECKED",
    "VERIFIED-LIVE vs HYPOTHETICAL",
)


class _StdoutRouter:
    """Process-global ``sys.stdout`` proxy that routes writes to a per-thread target.

    Installed once (idempotently) as ``sys.stdout``. Each thread may push an in-memory
    target via :func:`_capture_stdout`; that thread's writes/attribute lookups route to
    it, while a thread with no target falls through to the real stdout captured at
    install time -- so an idle router is byte-for-byte transparent.

    This replaces the old global-swap capture (``old = sys.stdout; sys.stdout = buf``),
    which was not thread-safe: under ``--parallel`` (#52) several subagent threads each
    run a nested loop whose printed answer is firewalled by :func:`_capture_stdout`, and
    a global swap would interleave their output and corrupt the LIFO save/restore. Here
    each thread's target lives in a :class:`threading.local`, so concurrent captures
    never collide and the push/pop is per-thread nested-safe. The main-thread ``--json``
    capture in ``code`` keeps working unchanged (it pushes a target, reads it back, and
    the post-capture ``json.dump`` -- with no target -- routes to the real stdout).
    """

    def __init__(self, base):
        self._base = base
        self._local = threading.local()

    def _target(self):
        return getattr(self._local, "target", None) or self._base

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        return self._target().flush()

    def writelines(self, lines):
        return self._target().writelines(lines)

    def isatty(self):
        return self._target().isatty()

    def __getattr__(self, name):
        # encoding / errors / buffer / fileno / writable / newlines / ... -- delegate to
        # the active target. ``_base``/``_local`` live in ``__dict__`` so this never
        # recurses on them.
        return getattr(self._target(), name)

    def _push(self, buf):
        prev = getattr(self._local, "target", None)
        self._local.target = buf
        return prev

    def _pop(self, prev):
        self._local.target = prev


_ROUTER_LOCK = threading.Lock()


def _install_router():
    """Idempotently wrap ``sys.stdout`` in a :class:`_StdoutRouter`; return the router.

    Safe to call from any thread and any number of times -- the lock guards the one-time
    wrap so a concurrent first-install can't double-wrap.
    """
    with _ROUTER_LOCK:
        if not isinstance(sys.stdout, _StdoutRouter):
            sys.stdout = _StdoutRouter(sys.stdout)
        return sys.stdout


@contextlib.contextmanager
def _capture_stdout():
    """Route this thread's ``sys.stdout`` to an in-memory buffer for the block.

    Used by ``code`` (``--json`` capture) and by :func:`run_scout`/:func:`run_spawn` (to
    firewall a subagent's printed answer out of the planner's transcript). Thread-safe:
    installs the shared router if needed, then pushes/pops a per-thread target (nested-
    safe, LIFO) so concurrent subagent captures never collide. The router is never
    uninstalled -- it is transparent when no target is pushed.
    """
    router = _install_router()
    buf = io.StringIO()
    prev = router._push(buf)
    try:
        yield buf
    finally:
        router._pop(prev)


def _run_disposable(
    oai,
    model: str,
    task: str,
    tools: List[Tool],
    base_kwargs: dict,
    *,
    system: str,
    max_tool_calls: int,
    budget: Optional[_compact.Budget] = None,
    ledger: Optional[CostLedger] = None,
    focus: Optional[str] = None,
) -> dict:
    """Run one disposable subagent turn-loop on a FRESH context and return its report.

    The shared core behind :func:`run_scout` (read-only) and :func:`run_spawn`
    (write/paid-capable): seeds a fresh ``messages`` list (only ``system`` + ``task`` --
    nothing from the caller's context leaks in), drives :func:`run_loop` with the given
    ``tools`` under a stdout firewall (the printed final answer is captured and
    discarded), and returns ``{"status","report","tool_calls","truncated"}``. The report
    is recovered from the message tail -- the final assistant turn in both the natural-
    stop and cap-forced paths. ``openai.OpenAIError`` from the loop propagates to the
    caller (the Tool wrapper turns it into an error envelope).

    Capability-agnostic: the read-only-vs-write distinction and any self-spawn guard live
    in the two thin wrappers, not here. Runs with ``yes=True`` -- for the worker that is
    required (mutating tools are ``paid=True`` and would otherwise be blocked in a non-
    interactive parent); for the scout it is a no-op (all its tools are free).
    """
    task = (task or "").strip()
    if not task:
        return {"status": "error", "message": "subagent requires a non-empty task"}

    sys_prompt = system
    if focus:
        sys_prompt = f"{system}\nFocus hint (not a hard scope): {focus}\n"
    messages: List[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": task},
    ]
    with _capture_stdout() as buf:
        run_loop(
            oai, model, messages, base_kwargs, tools,
            max_tool_calls=max_tool_calls, yes=True, json_out=False,
            budget=budget, ledger=ledger,
        )
    report = (messages[-1].get("content") or "").strip() if messages else ""
    if not report:
        report = buf.getvalue().strip()

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    executed = [
        m for m in tool_msgs if "not executed" not in (m.get("content") or "")
    ]
    capped = bool(max_tool_calls) and max_tool_calls > 0
    truncated = bool(tool_msgs) and (
        len(executed) != len(tool_msgs)
        or (capped and len(executed) >= max_tool_calls)
    )
    return {
        "status": "ok",
        "report": report,
        "tool_calls": len(executed),
        "truncated": truncated,
    }


def _parse_sections(report: str, headers) -> Dict[str, str]:
    """Best-effort split of a subagent report into ``{canonical_header: body}``.

    The scout/spawn system prompts mandate fixed report sections (``SCOUT_SECTIONS`` /
    ``SPAWN_SECTIONS``), but the headers are prompt-enforced only -- a model may decorate
    them (``**OUTCOME:**``, ``### FINDINGS``) or drop one. This is a tolerant line scanner,
    not a strict parser: a line begins a section when, after stripping leading markdown
    noise, it equals a known header or starts with it followed by ``:`` or ``*`` (case-
    insensitively); the body is the remainder of that line plus every line up to the next
    header. It deliberately does NOT treat ``HEADER `` + prose as a header (no colon), so a
    body sentence that merely opens with a section word can't spuriously start a section.

    Returns only the sections actually found -- a missing section is simply absent
    (consumers use ``.get(...)``) and a report with no recognizable headers yields ``{}``.
    The caller always keeps the raw ``report`` string, so nothing is lost. First occurrence
    of a header wins; a later stray repeat is folded into the current section rather than
    clobbering the real one.
    """
    if not report:
        return {}
    # Longest header first so a header that is a prefix of another can't shadow it.
    ordered = sorted(headers, key=len, reverse=True)
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    buf: List[str] = []

    def _flush() -> None:
        if current is not None and current not in fields:
            fields[current] = "\n".join(buf).strip()

    for line in report.splitlines():
        stripped = line.strip().lstrip("#*->• \t").strip()
        up = stripped.upper()
        matched = None
        rest = ""
        for h in ordered:
            hu = h.upper()
            if up == hu or up.startswith(hu + ":") or up.startswith(hu + "*"):
                matched = h
                rest = stripped[len(h):].lstrip(" *:\t-—").rstrip()
                break
        if matched is not None:
            _flush()
            current = matched
            buf = [rest] if rest else []
        elif current is not None:
            buf.append(line)
    _flush()
    return fields


def run_scout(
    oai,
    model: str,
    task: str,
    tools: List[Tool],
    base_kwargs: dict,
    *,
    max_tool_calls: int,
    budget: Optional[_compact.Budget] = None,
    ledger: Optional[CostLedger] = None,
    focus: Optional[str] = None,
    system: str = SCOUT_SYSTEM,
) -> dict:
    """Run one disposable, read-only subagent turn-loop and return its report.

    A thin read-only wrapper over :func:`_run_disposable`. Invariant (fail loud): every
    tool must be ``paid=False`` and none may be a subagent tool -- a scout can never
    spend, mutate, or dispatch another subagent. Raising here is defense-in-depth behind
    the structural guarantee that ``_code.read_only_tools`` never builds such a tool.
    """
    bad = [t.name for t in tools if t.paid or t.name in _SUBAGENT_TOOL_NAMES]
    if bad:
        raise ValueError(
            "scout subagent tools must be read-only (paid=False) and must not "
            f"include a scout, spawn, merge, or review tool; got: {bad}"
        )
    out = _run_disposable(
        oai, model, task, tools, base_kwargs, system=system,
        max_tool_calls=max_tool_calls, budget=budget, ledger=ledger, focus=focus,
    )
    if out.get("status") == "ok":
        out["fields"] = _parse_sections(out.get("report", ""), SCOUT_SECTIONS)
    return out


# --------------------------------------------------------------------------- #
# Worker subagent (#52 slice 2): a disposable, WRITE/paid-capable role worker.
#
# Where the scout (slice 1) is a read-only context firewall, the worker is the same
# firewall for *doers*: the planner delegates a bounded implementation task ("implement
# X in file Y, report back"), the edit churn stays quarantined in the worker's fresh
# context, and the planner gets back a structured provenance report it can merge. The
# worker draws a category-scoped subset of the PARENT's already-built tools, so its
# writes inherit the #76 Roots protection (allow-minus-deny, fail loud outside it) and
# the shell allow/deny policy -- capability can never exceed what the operator granted
# the parent session. Containment is structural: Roots (writes) + shell policy (run) +
# category/tag filtering (blast radius) + max_tool_calls (turn bound). The one axis NOT
# yet bounded is paid *media* spend -- see the TODO in ``_code.spawn_tool``.
#
# The role->category presets + the ``venice_spawn`` Tool wrapper live in ``_code``; this
# module owns only the profile-agnostic core so it stays import-clean.
# --------------------------------------------------------------------------- #
SPAWN_TOOL_NAME = "venice_spawn"

#: The planner-harness merge tool (#52 planner slice). Named here beside the other
#: subagent tool names so :func:`run_spawn`'s recursion guard can reject it without
#: importing ``_code`` (which owns the executable Tool, built over the session's
#: dispatch record list). A worker must never merge -- merging is the planner's job.
MERGE_TOOL_NAME = "venice_merge"

#: The cold-context reviewer (#80 part 1a). Named here with the other subagent tool
#: names for one reason: every recursion guard in this module has to be able to reject
#: it without importing ``_code`` (which owns the executable Tool). Keeping all four
#: names co-located is what stops a guard silently forgetting one -- see
#: :func:`run_scout`, :func:`run_spawn` and :func:`run_review`, which each reject the
#: other three.
REVIEW_TOOL_NAME = "venice_review"

#: The four disposable-subagent tool names, as one set. Any of them inside a subagent's
#: toolset would mean nesting deeper than one level, which is the containment invariant
#: the #52 arc is built on. Guards reject ``_SUBAGENT_TOOL_NAMES - {their own}``.
_SUBAGENT_TOOL_NAMES = frozenset({
    SCOUT_TOOL_NAME, SPAWN_TOOL_NAME, MERGE_TOOL_NAME, REVIEW_TOOL_NAME,
})

#: Tool names that :func:`run_loop` may dispatch CONCURRENTLY under ``--parallel`` (#52).
#: Only the two disposable, fresh-context, side-effect-isolated subagent calls qualify --
#: ``venice_merge`` is deliberately EXCLUDED (it reads the shared ``dispatches`` list, and
#: a name-based allowlist keeps any future ``category="agent"`` tool serial until opted in).
#: ``venice_review`` (#80) would qualify -- it is disposable, fresh-context and
#: side-effect-free -- but a review is normally one terminal call rather than a batch, so
#: it stays serial in v1 rather than widening ``--parallel``'s semantics for no gain.
_PARALLELIZABLE = frozenset({SCOUT_TOOL_NAME, SPAWN_TOOL_NAME})

#: Upper bound on subagents dispatched concurrently in one turn. A small constant (not
#: ``ThreadPoolExecutor``'s cpu-based default) bounds simultaneous model connections; the
#: per-turn worker count is ``min(_MAX_PARALLEL, calls-in-the-batch)``. A ``--max-parallel``
#: knob is a deferred nice-to-have.
_MAX_PARALLEL = 4


def _is_parallelizable(tc) -> bool:
    """True if this tool call is a subagent dispatch safe to run concurrently."""
    return tc.function.name in _PARALLELIZABLE


def _max_parallel() -> int:
    return _MAX_PARALLEL


SPAWN_SYSTEM = (
    "You are a WORKER subagent: a disposable, role-scoped agent spun up to carry out "
    "ONE task for a coding agent (the planner), then discarded. You start from a fresh "
    "context and hold a scoped subset of the project's tools -- you CAN edit files and "
    "run commands within your grant. Writes are confined to the project's writable "
    "roots and fail loudly outside them; stay inside your task.\n\n"
    "Do exactly the task, nothing more -- don't wander into unrelated changes. Verify "
    "your work where you can (re-read a file you wrote, run the relevant test). Your "
    "caller only sees your final report -- not your tool calls -- so it must stand on "
    "its own and give the planner enough to merge your work with confidence.\n\n"
    "End with a report using EXACTLY these sections:\n"
    "OUTCOME: done | partial | blocked, plus one line on what you accomplished.\n"
    "CHANGES: files you wrote/edited (paths) and commands you ran -- concrete, so the "
    "planner can review them; 'none' if none.\n"
    "VERIFIED: what you confirmed live (re-read / ran) vs. what you assumed without "
    "checking -- be explicit which is which.\n"
    "FOLLOW-UPS: what remains or what the planner should do next; 'none' if none.\n"
    "BLOCKERS: anything that stopped you (a write blocked outside the writable root, a "
    "test you couldn't get passing); 'none' if none.\n"
)

# The section headers SPAWN_SYSTEM mandates, in order (see SCOUT_SECTIONS note).
SPAWN_SECTIONS = (
    "OUTCOME",
    "CHANGES",
    "VERIFIED",
    "FOLLOW-UPS",
    "BLOCKERS",
)


def run_spawn(
    oai,
    model: str,
    task: str,
    tools: List[Tool],
    base_kwargs: dict,
    *,
    max_tool_calls: int,
    budget: Optional[_compact.Budget] = None,
    ledger: Optional[CostLedger] = None,
    focus: Optional[str] = None,
    role: Optional[str] = None,
    system: str = SPAWN_SYSTEM,
) -> dict:
    """Run one disposable, write/paid-capable worker subagent and return its report.

    A thin wrapper over :func:`_run_disposable` that -- unlike :func:`run_scout` --
    ALLOWS paid/write tools (that is the point of a worker) but still rejects recursion:
    no tool may be a subagent tool, so nesting is capped at exactly one level (the
    planner scouts/spawns/reviews; a worker does none of them). A worker's containment is
    structural, not a confirm gate -- see the module note above and ``_code.spawn_tool``.
    """
    bad = [t.name for t in tools if t.name in _SUBAGENT_TOOL_NAMES]
    if bad:
        raise ValueError(
            "worker subagent tools must not include a spawn, scout, merge, or review "
            f"tool (no nested subagents; merging is the planner's job); got: {bad}"
        )
    if role:
        system = f"{system}\nYour role: {role}.\n"
    out = _run_disposable(
        oai, model, task, tools, base_kwargs, system=system,
        max_tool_calls=max_tool_calls, budget=budget, ledger=ledger, focus=focus,
    )
    if out.get("status") == "ok":
        out["fields"] = _parse_sections(out.get("report", ""), SPAWN_SECTIONS)
    return out


# --------------------------------------------------------------------------- #
# Reviewer subagent (#80 part 1a): a disposable, read-only, diff-scoped reviewer.
#
# Same firewall as the scout, aimed at a different question. The scout answers "what is
# true about this code?"; the reviewer answers "what is WRONG with this change?" -- and
# it must answer it without having written the change, which is the whole point. The
# evidence in aiforge#19: cold-context review of *merged* termforge work found 10
# confirmed bugs in v0.1.0 with tests green across 11 build configs. Independent eyes
# catch what breadth cannot.
#
# THE CONSTRAINT THIS PROMPT ENCODES: producing findings and certifying a diff are
# SEPARATE operations, and this subagent only ever holds the first. A reviewer that
# could record an approval would be useless as a gate, because the agent that wrote the
# code holds `apply_patch` and `shell` and would simply write the approval itself --
# not adversarially, just shortest-path-to-green. So the prompt says so out loud, the
# toolset is read-only, and `venice review` writes nothing to disk. Gating is #80 part 1b.
# --------------------------------------------------------------------------- #
REVIEW_SYSTEM = (
    "You are a REVIEWER subagent: a disposable, read-only code reviewer spun up to "
    "review ONE diff, then discarded. You did NOT write this code and you have no "
    "memory of how it came to be -- judge only what the diff shows. You start from a "
    "fresh context with ONLY read-only tools (read files, list directories, grep, "
    "read-only git). You CANNOT and must NOT edit files, run commands, apply a fix, "
    "or record any approval anywhere. Producing findings is your entire job; deciding "
    "whether the diff is acceptable is someone else's.\n\n"
    "The diff below is the review scope. Read beyond it only to judge what the diff "
    "does -- open a caller, a definition, a test -- and stop as soon as you can "
    "decide. Do not review code the diff does not touch.\n\n"
    "Report DEFECTS only: crashes, incorrect results, data loss, resource leaks, "
    "race conditions, unhandled errors, security holes, broken invariants, missing "
    "cleanup, off-by-one, and behaviour that contradicts the code's own stated "
    "contract. Do NOT report formatting, naming, style preferences, or refactors that "
    "are not defects. A short list of real defects is worth far more than a long list "
    "of opinions; if the diff is sound, say so.\n\n"
    "Every finding must be locatable and reproducible. If you cannot name a file, a "
    "line, and a concrete scenario that triggers it, you do not have a finding yet -- "
    "either verify it with a read, or leave it out and say so under NOT CHECKED.\n\n"
    "End with a report using EXACTLY these sections:\n"
    "SCOPE: the files you actually examined and how far past the diff you read.\n"
    "FINDINGS: one block per defect, most severe first, or the single word 'none'. "
    "Each block is exactly four lines, in this shape:\n"
    "  <path>:<line> [blocker|major|minor] one-line statement of the defect\n"
    "  WHY: the mechanism -- what the code does that is wrong.\n"
    "  REPRO: the concrete input, call, or state that triggers it, and what happens "
    "as a result. If you truly cannot name one, write 'REPRO: none -- <reason>'.\n"
    "  FIX: the smallest change that would resolve it.\n"
    "The path must be a path from the diff, exactly as it appears there, and the line "
    "must be a line number in the NEW version of that file.\n"
    "NOT CHECKED: what you did not verify, could not reach, or judged out of scope -- "
    "be honest about gaps.\n"
    "REVIEW: a final line that is exactly 'REVIEW: CLEAN' if you found no defects, or "
    "exactly 'REVIEW: FINDINGS' if you listed one or more.\n"
)

# The section headers REVIEW_SYSTEM mandates, in order (see SCOUT_SECTIONS note). Unlike
# the scout/spawn pairs this one is NOT honour-system: `test_review.py` asserts every
# entry here appears literally in REVIEW_SYSTEM, so the prompt and the parser cannot drift.
REVIEW_SECTIONS = (
    "SCOPE",
    "FINDINGS",
    "NOT CHECKED",
    "REVIEW",
)


def run_review(
    oai,
    model: str,
    task: str,
    tools: List[Tool],
    base_kwargs: dict,
    *,
    max_tool_calls: int,
    budget: Optional[_compact.Budget] = None,
    ledger: Optional[CostLedger] = None,
    focus: Optional[str] = None,
    system: str = REVIEW_SYSTEM,
) -> dict:
    """Run one disposable, read-only reviewer subagent and return its report.

    A thin read-only wrapper over :func:`_run_disposable`, shaped exactly like
    :func:`run_scout`. Invariant (fail loud): every tool must be ``paid=False`` and none
    may be a subagent tool -- a reviewer can never spend, mutate, or dispatch anything.
    That is defense-in-depth behind the structural guarantee that the reviewer is built
    from ``_code.read_only_tools``.

    The returned dict is ``_run_disposable``'s, plus ``fields`` parsed against
    :data:`REVIEW_SECTIONS`. It carries no verdict of its own and no notion of approval:
    turning a report into a pass/fail decision is the caller's job (``_review``), and
    turning a decision into a *gate* is nobody's job yet (#80 part 1b).
    """
    bad = [t.name for t in tools if t.paid or t.name in _SUBAGENT_TOOL_NAMES]
    if bad:
        raise ValueError(
            "reviewer subagent tools must be read-only (paid=False) and must not "
            f"include a scout, spawn, merge, or review tool; got: {bad}"
        )
    out = _run_disposable(
        oai, model, task, tools, base_kwargs, system=system,
        max_tool_calls=max_tool_calls, budget=budget, ledger=ledger, focus=focus,
    )
    if out.get("status") == "ok":
        out["fields"] = _parse_sections(out.get("report", ""), REVIEW_SECTIONS)
    return out
