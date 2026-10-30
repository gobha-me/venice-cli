"""Agent usage accounting, pricing, cache diagnostics, and session budgets.

Keep ledger state and its accounting helpers independent of tool dispatch and
execution-loop state. The historical names remain available from ``_agent``.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
from typing import NamedTuple, Optional

from .. import _numeric
from .. import config as _config

CACHE_GUARD_CHOICES = ("off", "warn", "stop")
CACHE_GUARD_MIN_PROMPT_TOKENS = 2_000


class CacheGuardEvent(NamedTuple):
    """One once-per-ledger reported-zero cache event (#105)."""

    action: str
    message: str


def _usd_per_token(pricing, key) -> Optional[float]:
    """`pricing.<key>.usd` as a per-token rate (catalog prices are per 1M)."""
    if not isinstance(pricing, dict):
        return None
    node = pricing.get(key)
    if isinstance(node, dict) and "usd" in node:
        try:
            return _numeric.non_negative_float(node["usd"]) / 1_000_000.0
        except ValueError:
            raise ValueError(f"invalid catalog pricing for {key}") from None
    return None


def _as_int(v) -> int:
    """A non-negative int from a usage field; 0 for None/garbage/negative.

    `bool` is an `int` subclass but is never a real token count, so it's garbage.
    """
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        try:
            return int(_numeric.non_negative_float(v)) if v > 0 else 0
        except (OverflowError, ValueError):
            return 0
    return 0


def _as_float(v) -> float:
    """A non-negative float from a stored cost field; 0.0 for None/garbage/negative.

    Mirrors :func:`_as_int` (bool is never a real cost); used when restoring a
    persisted ledger snapshot (#47) from a possibly hand-edited envelope.
    """
    if isinstance(v, bool):
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return _numeric.non_negative_float(v) if v > 0 else 0.0
        except ValueError:
            return 0.0
    return 0.0


_MODEL_USAGE_NUMERIC_FIELDS = (
    "total",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "api_calls_total",
)
_MODEL_USAGE_FLAG_FIELDS = (
    "unpriced",
    "cache_read_unreported",
    "cache_write_unreported",
)


def _new_model_usage_row() -> dict:
    """One model's main-loop usage counters (#104)."""
    return {
        "total": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls_total": 0,
        "unpriced": False,
        "cache_read_unreported": False,
        "cache_write_unreported": False,
    }


def _restore_model_usage_row(dest: dict, source) -> None:
    """Add a tolerant persisted model row into ``dest`` (#104)."""
    if not isinstance(source, dict):
        return
    dest["total"] += _as_float(source.get("total"))
    for key in _MODEL_USAGE_NUMERIC_FIELDS[1:]:
        dest[key] += _as_int(source.get(key))
    for key in _MODEL_USAGE_FLAG_FIELDS:
        if source.get(key):
            dest[key] = True


def _model_cache_hit_percent(row, *, round_to=None):
    """A model row's three-state cache hit percentage (#104)."""
    if not isinstance(row, dict):
        return None
    prompt = _as_int(row.get("prompt_tokens"))
    cached = _as_int(row.get("cache_read_tokens"))
    if prompt == 0:
        return None
    if row.get("cache_read_unreported") and cached == 0:
        return None
    pct = cached / prompt * 100.0
    return pct if round_to is None else round(pct, round_to)


def _public_model_usage_row(row) -> dict:
    """Copy one model row and add its derived cache percentage (#104)."""
    out = dict(row) if isinstance(row, dict) else _new_model_usage_row()
    out["cache_hit_percent"] = _model_cache_hit_percent(out, round_to=1)
    return out


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
        line = json.dumps(raw, sort_keys=True, default=str, allow_nan=False)
    except Exception:  # noqa: BLE001 - a diagnostic must never break the turn
        try:
            normalized = (
                {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else raw
            )
            line = json.dumps(normalized, default=str, allow_nan=False)
        except Exception:  # noqa: BLE001 - same diagnostic-only boundary
            line = "<usage is not valid strict JSON>"
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
                 max_tokens: Optional[int] = None, *, mirror=None,
                 cache_guard: str = "off", model_id=None, models=None):
        # A non-positive cap means "uncapped" (mirrors --max-tool-calls 0).
        cap = _numeric.finite_float(max_spend) if max_spend is not None else None
        self.max_spend = cap if (cap is not None and cap > 0) else None
        # #52: an orthogonal cumulative *token* ceiling (prompt+completion), used by
        # per-subagent runs (`--subagent-max-tokens`). Same non-positive->None rule.
        tcap = int(max_tokens) if max_tokens is not None else None
        self.max_tokens = tcap if (tcap is not None and tcap > 0) else None
        if cache_guard not in CACHE_GUARD_CHOICES:
            raise ValueError(
                "cache_guard must be one of: " + ", ".join(CACHE_GUARD_CHOICES)
            )
        # #105: code opts its PARENT ledger into this policy. Chat and disposable
        # subagent ledgers keep the constructor default (`off`), so fresh-prefix
        # off-loop calls cannot manufacture a cache-collapse warning. This is runtime
        # policy, not session data: a resumed process may warn again if the regression
        # is still live.
        self.cache_guard = cache_guard
        self._cache_guard_fired = False
        self._cache_guard_fired_models = set()
        # #104: production ledgers are bound to an exact model id and retain the
        # catalog so `/model` can atomically switch both routing and pricing. Bare
        # ledgers remain supported for focused tests and internal callers.
        self.model_id = None
        self._models = models
        self.models: dict[str, dict] = {}
        # Never restored: this is the current process/resume leg, not session history.
        self.current_run_models: dict[str, dict] = {}
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
        self.tools: dict[str, dict] = {}
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
        self._calls_head: list[dict] = []
        self._calls_tail = collections.deque(maxlen=self._CALLS_TAIL)
        # The TRUE count, which is also the source of each row's `n`. Deliberately not
        # `len()` of anything: it stays truthful across the cap and across a resume, and it
        # finally persists the number the `turns` comment above says this class counts but
        # never stored. Note "MAIN-LOOP API calls", NOT "API calls": `_compact`'s summary
        # call reaches `record()` as of #101, but with `bucket="compaction"`, which
        # deliberately appends no row and does not bump this counter -- so the trace's
        # rows still do not sum to the whole bill. `self.buckets` holds the remainder,
        # `billed_total()` adds the two, and the `calls` header names the difference.
        self.api_calls_total = 0
        # #99: prefix-affecting events (compaction today; #104's resume reseed later --
        # hence `kind`, and hence not naming this `compactions`). Uncapped on purpose:
        # a compaction is rare and is the highest-signal row in the trace.
        #
        # NOTE the two orthogonal axes that both nearly wanted the word "kind": this
        # one says WHAT HAPPENED TO THE PREFIX, while `record`'s `bucket=` below says
        # WHICH SPEND PARTITION a call belongs to. They are not the same question and
        # must not share a name.
        self.context_events: list[dict] = []
        # #101: off-loop spend, partitioned by bucket name -- "compaction", plus one per
        # subagent rail as of #117 ("scout"/"spawn"/"review"/"web_search"). name ->
        # {calls, cost, prompt_tokens, completion_tokens, seconds, unpriced, unreported}.
        #
        # A SEPARATE partition rather than more weight on the counters above, because
        # merging would corrupt the one signal the main-loop numbers exist to carry:
        # `_compact`'s summary call is a FRESH prefix that reads ~0% cached every time,
        # so folding it into `prompt_tokens`/`cache_read_tokens` manufactures a cache
        # cliff that never happened. For the same reason these calls get no trace row
        # and never touch `api_calls_total` -- see `record`.
        #
        # "cache buckets" (the `cache_read`/`cache_write` split above) is an unrelated
        # use of the word; it is never a `bucket=` argument and never a key in here.
        self.buckets: dict[str, dict] = {}
        # #117: this DID go unlocked, on the reasoning that `record()` only ever ran on
        # the thread owning the ledger. The rails slice ends that: a mirrored child
        # (below) writes its parent's row from whatever pool worker ran the subagent, and
        # under `--parallel` up to `_MAX_PARALLEL` scouts/spawns do it at once. Same
        # read-modify-write shape as `_tools_lock`, same GIL caveat -- see that comment.
        # Held around the row mutation ONLY (never the cost arithmetic), and never nested
        # with `_tools_lock`, so the two cannot deadlock.
        self._buckets_lock = threading.Lock()
        # #117: `(parent_ledger, bucket_name)`, or None for every main-loop ledger. Set on
        # a SUBAGENT ledger so each of its API calls is also banked into the parent's
        # off-loop bucket -- see `record`. Bound here rather than passed per call because
        # `record()` is the one chokepoint every rail's usage already flows through, so no
        # call site *can* forget it (the two rails that call the API outside `run_loop`,
        # `_review._retry_for_verdict` and `run_web_search`, are covered for free).
        if mirror is not None and mirror[0] is self:
            # Would double-count the ledger into its own bucket on every record.
            raise ValueError("CostLedger: a ledger cannot mirror into itself")
        self._mirror = mirror
        self._in = None          # per-token input rate (USD)
        self._out = None         # per-token output rate (USD)
        self._cache_in = None    # per-token cache-read rate (USD); None -> use _in
        self._cache_write = None  # per-token cache-write rate (USD); None -> use _in
        self.invalid_pricing = False
        if model_id is not None:
            self.bind_model(model_id, models)

    def bind_pricing(self, pricing) -> None:
        """Set the per-token rates from a catalog `model_spec.pricing` block.

        `cache_input`/`cache_write` are optional (present only for cache-capable
        models); left None they fall back to the plain input rate at cost time.
        """
        self.invalid_pricing = False
        try:
            self._in = _usd_per_token(pricing, "input")
            self._out = _usd_per_token(pricing, "output")
            self._cache_in = _usd_per_token(pricing, "cache_input")
            self._cache_write = _usd_per_token(pricing, "cache_write")
        except ValueError:
            self._in = self._out = self._cache_in = self._cache_write = None
            self.invalid_pricing = True

    def bind_model(self, model_id, models=None) -> None:
        """Select the active model and atomically rebind its catalog pricing (#104)."""
        if models is not None:
            self._models = models
        self.model_id = str(model_id) if model_id is not None else None
        # Always bind, including the missing-price case, so switching away from a
        # priced model cannot accidentally retain the previous model's rates.
        self.bind_pricing(_pricing_for(self._models, self.model_id))

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
            # #101: the THIRD derived key. `total` is main-loop-only so it keeps
            # reconciling with the token counters above it; this is the whole bill, so
            # a pipeline can alert on real spend without knowing which buckets exist.
            # Deliberately NOT read back by `restore()` -- `total` and `buckets` are
            # both restored additively below, and seeding this too would double-count.
            "billed_total": self.billed_total(),
            # #101: off-loop spend by bucket. Sorted by NAME like `tools` above, and
            # rows COPIED for the same reason -- `code --json` hands this envelope to
            # `json.dump` while the ledger is still live.
            "buckets": {k: dict(self.buckets[k]) for k in sorted(self.buckets)},
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
            # #104: additive public dimensions. `models` is cumulative session usage;
            # `current_run_models` begins empty after every restore.
            "models": {
                k: _public_model_usage_row(self.models[k]) for k in sorted(self.models)
            },
            "current_run_models": {
                k: _public_model_usage_row(self.current_run_models[k])
                for k in sorted(self.current_run_models)
            },
        }

    def restore(self, d, *, legacy_model=None) -> None:
        """Seed the accumulators from a :meth:`to_dict` snapshot (resume, #47).

        Additive onto the current (freshly-priced) ledger so a session resumed
        mid-run keeps its cumulative token/cost totals. Tolerant of a partial or
        foreign dict (missing keys count as 0) so a hand-edited envelope can't crash
        the REPL. Also accepts the raw #75 usage field names as a fallback.
        """
        if not isinstance(d, dict):
            return
        # A restore is the boundary between process legs. These diagnostics must never
        # inherit the previous leg, even if a caller reuses a ledger object.
        self.current_run_models.clear()
        self._cache_guard_fired_models.clear()
        self._cache_guard_fired = False
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
        # #101: additive per BUCKET NAME, like `tools` above -- but FIELD BY FIELD, and
        # that difference is not stylistic. `tools` rows are two numbers; a bucket row
        # carries `unpriced`, and a naive `cur[k] += v` loop would evaluate
        # `False + True` to the integer 1 and silently turn a flag into a count. No
        # crash, no test failure, a corrupt envelope. Numerics accumulate, `unpriced`
        # sticky-ORs like the top-level one, and anything else in the row is dropped.
        raw = d.get("buckets")
        if isinstance(raw, dict):
            for name, row in raw.items():
                if not isinstance(row, dict):
                    continue  # tolerant PER ROW, like `tools`
                key = str(name).strip()
                if not key:
                    continue
                cur = self.buckets.setdefault(key, self.new_bucket_row())
                cur["calls"] += _as_int(row.get("calls"))
                cur["cost"] += _as_float(row.get("cost"))
                cur["prompt_tokens"] += _as_int(row.get("prompt_tokens"))
                cur["completion_tokens"] += _as_int(row.get("completion_tokens"))
                cur["seconds"] = round(
                    cur["seconds"] + _as_float(row.get("seconds")), 3)
                # Both flags sticky-OR, like the top-level `unpriced` and the #98 pair:
                # one blind call in the bucket taints the bucket, and a later
                # well-reported one must not clear it.
                if row.get("unpriced"):
                    cur["unpriced"] = True
                if row.get("unreported"):
                    cur["unreported"] = True
        # `billed_total` is NOT restored, joining `cache_hit_percent` and `tool_seconds`
        # below: it is DERIVED from `total` and `buckets`, both of which the lines above
        # already seeded additively, so reading it back would count the whole bill twice.
        #
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
                    row = dict(row)
                    if legacy_model and not row.get("model"):
                        row["model"] = str(legacy_model)
                    if len(self._calls_head) < self._CALLS_HEAD:
                        self._calls_head.append(row)
                    else:
                        self._calls_tail.append(row)
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
        # #104: new envelopes carry exact model buckets. Older envelopes are attributed
        # to the session's SAVED model (passed by the REPL), never to a command-line
        # override selected for the new leg. If provenance is unavailable the aggregate
        # above remains authoritative and no model id is invented.
        raw_models = d.get("models")
        if isinstance(raw_models, dict):
            for model, row in raw_models.items():
                key = str(model).strip()
                if not key or not isinstance(row, dict):
                    continue
                _restore_model_usage_row(
                    self.models.setdefault(key, _new_model_usage_row()), row
                )
        elif legacy_model:
            _restore_model_usage_row(
                self.models.setdefault(str(legacy_model), _new_model_usage_row()),
                {
                    key: d.get(key)
                    for key in _MODEL_USAGE_NUMERIC_FIELDS + _MODEL_USAGE_FLAG_FIELDS
                },
            )
        # Deliberately ignore persisted `current_run_models`: a resume starts a fresh
        # diagnostic leg while the cumulative `models` map above survives.

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

    def api_calls(self) -> list[dict]:
        """The retained trace rows in call order: head, then tail (#99).

        A gap between the two is not an error and is not marked: consecutive rows whose
        `n` jumps say so themselves, which is why nothing stores a drop count.
        """
        return self._calls_head + list(self._calls_tail)

    def calls_dropped(self) -> int:
        """How many trace rows the cap discarded (#99); 0 when nothing was dropped."""
        return self.api_calls_total - len(self._calls_head) - len(self._calls_tail)

    def record_context_event(self, event: dict) -> None:
        """Log one context-affecting event, anchored to the trace (#99/#104).

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

    def record_compaction(self, event: dict) -> None:
        """Compatibility wrapper for the compaction worker's duck-typed hook."""
        self.record_context_event(dict(event, kind="compaction"))

    def record_bucket(self, name: str, *, cost: float = 0.0,
                      prompt_tokens: int = 0, completion_tokens: int = 0,
                      seconds=None, unpriced: bool = False,
                      unreported: bool = False) -> None:
        """Accumulate one off-loop API call into the `name` bucket (#101/#117).

        A PURE accumulator, like `record_turn`/`record_tool`: it takes numbers that have
        already been costed and never reads a `usage` block, so `record()` below remains
        the one and only place any API usage is parsed. That split is what lets a
        mirrored child (#117) hand its parent an already-priced row without the parent
        re-reading -- and re-mis-pricing, at the wrong model's rate -- the same usage.

        The ONE live writer of a bucket row: `record()`'s off-loop branch delegates here
        rather than keeping a second copy of these seven lines (`restore()` is the other
        writer, and `new_bucket_row` already exists because that duplication drifts).

        Callers must not hold `_buckets_lock`; this takes it, and takes it around the
        mutation only.
        """
        with self._buckets_lock:
            row = self.buckets.setdefault(name, self.new_bucket_row())
            row["calls"] += 1
            # #98's rule, one partition over: a response with no usage block leaves the
            # tokens and the cost UNKNOWN, not zero. The main loop can say so per-row
            # (the trace stores null) and dodges it in aggregate by having a
            # "(no tokens reported)" branch; a bucket has neither, so it needs its own
            # sticky marker or `0 in  0 out  $0.0000` would read as a measurement.
            if unreported:
                row["unreported"] = True
            row["cost"] += _as_float(cost)
            row["prompt_tokens"] += _as_int(prompt_tokens)
            row["completion_tokens"] += _as_int(completion_tokens)
            # Rounded on the way in like `record_turn`'s; an unstamped window adds 0.
            row["seconds"] = round(row["seconds"] + _as_float(seconds), 3)
            # Sticky-OR, mirroring the top-level `unpriced` -- one uncosted call in the
            # bucket taints the bucket's total, and a later priced one must not clear it.
            row["unpriced"] = row["unpriced"] or unpriced

    def bucket_cost(self, name=None) -> float:
        """Off-loop spend (#101): one bucket's cost, or every bucket's when `name` is None."""
        if name is not None:
            return _as_float(self.buckets.get(name, {}).get("cost", 0.0))
        # #117: snapshot under the lock. A mirrored child adds a NEW KEY on a pool
        # worker, and iterating a dict while another thread inserts into it raises
        # outright. Reads happen on the main thread while workers are joined today, so
        # this is insurance rather than a live fix -- but it is two lines.
        with self._buckets_lock:
            rows = list(self.buckets.values())
        return sum(_as_float(b.get("cost", 0.0)) for b in rows)

    def bucket_calls(self, name=None) -> int:
        """Off-loop API calls (#101) -- deliberately NOT part of `api_calls_total`."""
        if name is not None:
            return int(self.buckets.get(name, {}).get("calls", 0))
        with self._buckets_lock:      # #117: see `bucket_cost`
            rows = list(self.buckets.values())
        return sum(int(b.get("calls", 0)) for b in rows)

    @staticmethod
    def new_bucket_row() -> dict:
        """A zeroed off-loop bucket row (#101).

        ONE definition, used by `record()` and by `restore()`. Written out twice it
        would drift: seed a new field in `record` only, and a RESUMED ledger's rows
        silently lack the key while a live one has it -- a resume-only divergence with
        no crash and no failing test.
        """
        return {"calls": 0, "cost": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "seconds": 0.0,
                "unpriced": False, "unreported": False}

    @staticmethod
    def bucket_money(row) -> str:
        """How to render one bucket row's cost (#101), or "" when it is knowable-zero.

        Three ways this number can be absent and only one of them is a measurement, so
        they get three renderings. Shared with `venice sessions show`, which reads the
        same rows out of a session file -- a second hand-rolled formatter there was
        already printing a measured-looking `$0.0000` over an unpriced bucket.

        Tolerant of a hand-edited envelope, like `restore()` and unlike a bare
        `float()`: this also runs against JSON nobody validated.
        """
        row = row if isinstance(row, dict) else {}
        cost = _as_float(row.get("cost"))
        pt = _as_int(row.get("prompt_tokens"))
        ct = _as_int(row.get("completion_tokens"))
        if row.get("unpriced") and not cost:
            return "(unpriced)"          # the model had no catalog rate
        if row.get("unreported") and not (pt or ct):
            return "n/a"                 # the response carried no usage block
        if cost and f"${cost:.4f}" == "$0.0000":
            # Real spend that rounds away at 4dp -- cheap model, small prefix. Printing
            # `$0.0000` would be a measurement of nothing; `<$0.0001` is the truth.
            return "<$0.0001"
        return f"${cost:.4f}"

    def any_unreported(self) -> bool:
        """True when any bucket holds a call whose response carried no usage (#101).

        The reporting-axis twin of `any_unpriced`. `_bucket_lines` used to derive this
        per row as "unreported AND no tokens at all", which has no PARTIAL state: one
        usage-bearing call in the bucket hid the marker entirely and the undercount
        rendered as a complete measurement. The main token block has said
        `[partially unreported]` for this since #98; the partition needs to as well.
        """
        return any(bool(b.get("unreported")) for b in self.buckets.values())

    def any_unpriced(self) -> bool:
        """True when ANY recorded call had no catalog rate -- main loop or bucket (#101).

        Distinct from `self.unpriced`, which stays main-loop-only so `/usage`'s token
        block keeps meaning what it says. This is the question the RENDERED COST asks:
        "is the figure I am about to print the whole story?" A priced main loop beside
        an unpriced compaction still understates the bill by an unknown amount, and a
        wholly-unpriced ledger must say `unknown` rather than a fabricated `$0.0000`
        -- #98's rule, which `self.unpriced` alone can no longer answer for.
        """
        return self.unpriced or any(
            bool(b.get("unpriced")) for b in self.buckets.values()
        )

    def billed_total(self) -> float:
        """The whole bill (#101): main-loop spend plus every off-loop bucket.

        `total` stays main-loop-only so the token counters beside it keep reconciling
        with it. This is the number a human means by "what did this run cost", so it is
        what `over()` gates on and what `summary()` prints.
        """
        return self.total + self.bucket_cost()

    def _record_model_usage(self, *, usage_present: bool, cost: float,
                            prompt_tokens: int, completion_tokens: int,
                            cache_read_tokens: int, cache_write_tokens: int,
                            reasoning_tokens: int, cache_read_reported: bool,
                            cache_write_reported: bool, unpriced: bool) -> None:
        """Accumulate one main-loop call into both #104 model views."""
        if not self.model_id:
            return
        for rows in (self.models, self.current_run_models):
            row = rows.setdefault(self.model_id, _new_model_usage_row())
            row["api_calls_total"] += 1
            if not usage_present:
                continue
            row["total"] += cost
            row["prompt_tokens"] += prompt_tokens
            row["completion_tokens"] += completion_tokens
            row["cache_read_tokens"] += cache_read_tokens
            row["cache_write_tokens"] += cache_write_tokens
            row["reasoning_tokens"] += min(reasoning_tokens, completion_tokens)
            row["unpriced"] = row["unpriced"] or unpriced
            if not cache_read_reported:
                row["cache_read_unreported"] = True
            if not cache_write_reported:
                row["cache_write_unreported"] = True

    def record(self, usage, *, seconds=None, bucket=None) -> float:
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

        #101: `bucket` names an OFF-LOOP spend partition ("compaction"); None -- the
        default, and every main-loop caller -- is the behaviour above, unchanged. A
        bucketed call is costed by the same arithmetic and then accumulated ONLY into
        `self.buckets[bucket]`: no main token counters, no `unpriced`/cache-unreported
        flags, no trace row, no `api_calls_total`. That isolation is the point. The
        main-loop numbers exist to make a prefix-cache collapse legible, and
        `_compact`'s summary call is a fresh prefix that reads ~0% cached every single
        time -- averaged in, it fabricates exactly the cliff the trace exists to detect.

        This stays the ONE place any API usage is read. `bucket=` is a parameter rather
        than a second method for that reason: a `record_offloop()` sibling would be a
        second read site to keep in agreement with this one, which is how the
        bookkeeping #86/#92/#99 kept centralizing gets scattered again. `record_bucket`
        is not a counter-example -- it takes already-costed numbers and never parses a
        usage block, which is exactly why the #117 mirror below can route through it.

        #117: if this ledger was built with a `mirror`, the call is ALSO banked into the
        parent's bucket, at the tail. Everything above is unchanged by that -- a mirrored
        child keeps its own full main-loop accounting, which is what `over_tokens()` and
        so `--subagent-max-tokens` read.
        """
        # FIRST, ahead of everything below: `record(None)` dumping `usage-raw: null`
        # is exactly the diagnostic that was missing -- it separates "the response had
        # no usage block" from "it had one with no cache fields" (#98).
        _dump_raw_usage(usage)
        usage = _usage_dict(usage)
        cost = 0.0
        unpriced_turn = False
        # A response with no usage block still costs wall-clock and still moved the
        # prefix, so it gets a row -- with null tokens rather than a fabricated 0, the
        # #98 rule applied one level down.
        pt = ct = raw_read = raw_write = None
        cache_read = cache_write = reasoning = 0
        if usage is not None:
            pt = _as_int(usage.get("prompt_tokens"))
            ct = _as_int(usage.get("completion_tokens"))
            # #98: None means the field was absent, which is not a reported zero. The
            # flags are what `usage_report` needs to refuse to print a fake 0.0%; the
            # counts fall back to 0 so the cost math below is untouched.
            raw_read = _cache_tokens(usage, _CACHE_READ_KEYS)
            raw_write = _cache_tokens(usage, _CACHE_WRITE_KEYS)
            # #101: the flags stay MAIN-LOOP-ONLY. They drive `/usage`'s refusal to
            # print a cache rate, and an off-loop call (a fresh prefix by construction)
            # reporting no cache block says nothing about the conversation's cache.
            if bucket is None:
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

            if bucket is None:
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
                unpriced_turn = True
            if bucket is None:
                self.unpriced = unpriced_turn or self.unpriced
                self.total += cost
        if bucket is None:
            self._record_model_usage(
                usage_present=usage is not None,
                cost=cost,
                prompt_tokens=_as_int(pt),
                completion_tokens=_as_int(ct),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                reasoning_tokens=reasoning,
                cache_read_reported=raw_read is not None,
                cache_write_reported=raw_write is not None,
                unpriced=unpriced_turn,
            )
        # The two landing sites are structurally exclusive, and the method still has
        # exactly ONE return: an off-loop call must NOT get a trace row, and a main-loop
        # call must get one on every path including the no-usage shapes. An early return
        # in the branch above would have made "did every path land somewhere" a question
        # you have to read the whole method to answer.
        if bucket is not None:
            # #101: an off-loop call lands HERE and nowhere else -- no trace row, no
            # `api_calls_total` bump. Both omissions are load-bearing: `calls_dropped()`
            # is `api_calls_total - head - tail`, so counting a call that never appended
            # a row would make `/usage` report phantom dropped rows; and the trace is a
            # PREFIX-cache diagnostic that a fresh-prefix call would only pollute.
            self.record_bucket(
                bucket, cost=cost, prompt_tokens=_as_int(pt),
                completion_tokens=_as_int(ct), seconds=seconds,
                unpriced=unpriced_turn, unreported=usage is None,
            )
        else:
            call = {
                "prompt_tokens": pt,
                "cache_read_tokens": raw_read,
                # Recorded although #99 only asked for cache-READ: a cache-WRITE spike at
                # call N is what separates a churning prefix from a cold start, i.e. two
                # of the three failures this trace exists to tell apart.
                "cache_write_tokens": raw_write,
                "completion_tokens": ct,
                "cost": round(cost, 6),
                # Rounded like `record_turn`'s, so the persisted envelope stays tidy.
                # None stays None -- an unstamped window is unknown, not instant.
                "seconds": None if seconds is None else round(_as_float(seconds), 3),
            }
            if self.model_id:
                call["model"] = self.model_id
            self._append_call(call)
        # #117: a SUBAGENT ledger also banks this call into its parent's off-loop bucket.
        # Placed at the one chokepoint every rail's usage already flows through, so the
        # two rails that call the API outside `run_loop` (`_review._retry_for_verdict`,
        # `run_web_search`) are covered without either site knowing a parent exists.
        #
        # `record_bucket`, NOT `parent.record(usage, bucket=...)`, and all three reasons
        # are load-bearing:
        #   - PRICING. `cost` above was computed at THIS ledger's rate. `--review-model`
        #     and `--web-search-model` are deliberately a different model from the
        #     parent's, so re-costing the same usage over there would quietly bill the
        #     reviewer's tokens at the author's rate -- a fabricated number, which is the
        #     one thing this ledger refuses to print.
        #   - ONE RAW DUMP. `_dump_raw_usage` fires at the top of this method; routing
        #     through `record()` again would emit a second `usage-raw:` line for one API
        #     call, interleaved with other workers' lines under `--parallel`.
        #   - NO RECURSION. `record_bucket` is a leaf, so a mirror does not propagate to
        #     a grandparent. That is also the correct arithmetic: a grandparent bucket
        #     would double-count spend a parent bucket has already banked.
        # `usage is None` (not `pt is None`) is the unreported test, evaluated after
        # `_usage_dict` above, so an SDK object that normalized to nothing counts as
        # unreported rather than as a measured zero.
        if self._mirror is not None:
            parent, name = self._mirror
            parent.record_bucket(
                name, cost=cost, prompt_tokens=_as_int(pt),
                completion_tokens=_as_int(ct), seconds=seconds,
                unpriced=unpriced_turn, unreported=usage is None,
            )
        return cost

    def cache_guard_event(self, model: str) -> Optional[CacheGuardEvent]:
        """Return the first qualifying reported-zero cache event, else ``None``.

        The call trace is the authoritative three-state input (#98/#99): ``None``
        means the provider did not report a cache field and must never be treated as
        zero. The ordinal is the API-call ordinal, not a REPL/human turn -- one tool
        loop can make hundreds of calls, which is the failure #105 exists to stop.

        Only a valid ordinary-input + cache-input price pair makes the model eligible.
        Missing pricing cannot support either the "advertised" claim or the estimated
        missed discount. Invalid pricing is handled earlier by ``invalid_pricing``.
        """
        if self.cache_guard == "off":
            return None
        bound_model = self.model_id or model
        if self.model_id:
            if bound_model in self._cache_guard_fired_models:
                return None
            model_calls = _as_int(
                self.current_run_models.get(bound_model, {}).get("api_calls_total")
            )
        else:
            if self._cache_guard_fired:
                return None
            model_calls = self.api_calls_total
        rows = self.api_calls()
        if not rows or self._in is None or self._cache_in is None:
            return None
        row = rows[-1]
        prompt_tokens = row.get("prompt_tokens")
        if (
            model_calls < 2
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < CACHE_GUARD_MIN_PROMPT_TOKENS
            or row.get("cache_read_tokens") != 0
        ):
            return None

        if self.model_id:
            self._cache_guard_fired_models.add(bound_model)
        else:
            self._cache_guard_fired = True
        if self._cache_in == 0:
            advertised = "free cache input"
        elif self._in <= self._cache_in:
            advertised = "cache-input pricing"
        else:
            ratio = self._in / self._cache_in
            advertised = f"a {ratio:.1f}x cache discount"
        missed = max(self._in - self._cache_in, 0.0) * prompt_tokens
        amount = f"${missed:.4f}" if missed < 1 else f"${missed:.2f}"
        return CacheGuardEvent(
            self.cache_guard,
            f"cache: {bound_model} advertises {advertised} but returned 0 cached "
            f"tokens on API call {row['n']}; this run is paying full input price "
            f"(est. +{amount} so far)",
        )

    def over(self) -> bool:
        """True when accumulated spend has reached/exceeded the cap.

        #101: gates on `billed_total()`, so `--session-max-spend` counts off-loop spend
        (compaction today) and not just the main loop. Still monotonic -- both terms
        only grow -- so a gate that has tripped stays tripped.
        """
        return self.max_spend is not None and (
            self.invalid_pricing or self.billed_total() >= self.max_spend
        )

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
        row = self.current_run_models.get(self.model_id) if self.model_id else None
        if self.model_id and row is None:
            return ""
        pct = (_model_cache_hit_percent(row) if row is not None
               else self.cache_hit_percent())
        unreported = (bool(row.get("cache_read_unreported")) if row is not None
                      else self.cache_read_unreported)
        if pct is None:
            # Distinguish "the provider never told us" (worth saying on a run footer --
            # it is the difference between a cache collapse and a blind spot) from
            # "nothing was recorded", which has nothing to report at all.
            return "cache n/a" if unreported else ""
        if unreported:
            return f"cache {pct:.1f}% hit [partially unreported]"
        return f"cache {pct:.1f}% hit"

    @staticmethod
    def _model_usage_lines_for(title: str, rows: dict) -> list[str]:
        """Render one bounded per-model accounting section (#104)."""
        if not rows:
            return []
        lines = [f"  {title}:"]
        for model in sorted(rows):
            row = rows[model] if isinstance(rows[model], dict) else {}
            pct = _model_cache_hit_percent(row)
            if pct is None:
                cache = (
                    "n/a (no cache fields reported)"
                    if row.get("cache_read_unreported")
                    else "n/a (no input tokens)"
                )
            elif row.get("cache_read_unreported"):
                cache = f"{pct:.1f}% [partially unreported]"
            else:
                cache = f"{pct:.1f}%"
            cost = _as_float(row.get("total"))
            money = "unpriced" if row.get("unpriced") and not cost else f"${cost:.4f}"
            lines.append(
                f"    {model}: {_as_int(row.get('prompt_tokens')):,} in, "
                f"{_as_int(row.get('completion_tokens')):,} out, "
                f"cache hit rate: {cache}, {money}, "
                f"{_as_int(row.get('api_calls_total'))} call(s)"
            )
        return lines

    def _model_usage_lines(self) -> list[str]:
        """Cumulative and current-resume-leg model views (#104)."""
        return (
            self._model_usage_lines_for("models (session)", self.models)
            + self._model_usage_lines_for(
                "models (current run)", self.current_run_models
            )
        )

    def summary(self, *, cache: bool = False) -> str:
        """A one-line human-readable total (for stderr / --json).

        `cache` opts into the #100 hit-rate clause. OFF by default because two of this
        method's call sites are stop-reason messages -- `run_loop`'s spend and token
        gates -- which are reached with a subagent/review ledger as well as the parent
        one; those are unpriced and near-always cache-unreported, and "cache n/a" on
        `worker reached token cap 50,000` answers a question nobody asked there. The run
        footers and `/cost` opt in.

        #101: prints `billed_total()`, not `total`. It has to: `over()` gates on the
        billed number, and two of those call sites render the message announcing that
        gate -- printing a smaller figure beside `/ cap $5.00` would contradict itself
        on the one surface that cannot afford to. The `[incl. ...]` clause names the
        off-loop share, gated on there being one.

        The three sites that can receive a SUBAGENT ledger are byte-identical, and by
        structure rather than luck: `run_scout`/`run_spawn`/`run_review` never pass a
        `budget`, `maybe_compact` short-circuits without one, so a subagent ledger can
        never acquire a bucket.

        #117 was expected to delete that invariant and did NOT -- worth stating, because
        the issue predicted the opposite and the next reader will assume it landed. A
        mirrored subagent ledger writes its parent's `buckets`, never its own, so these
        messages still render `cost: $X (tokens ...)` with no clause. What DID change is
        that the parent now has more bucket NAMES, which is why the clause below is
        bounded instead of joining an unbounded list.
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
        # Each computed ONCE and reused below. Read twice, they are two chances for a
        # later edit to touch one and not its twin, and the result would be a footer
        # that contradicts itself (`$0.0000 ... [incl. $0.0031 compaction]`).
        billed = self.billed_total()
        unpriced = self.any_unpriced()
        off = self.bucket_cost()
        # #101: `billed`, not `total`, in the GUARD too, and as of #117 that is no longer
        # a precaution. The rails made mixed pricing REACHABLE: `--review-model` can be a
        # priced model while the author's is not, leaving `total == 0.0` beside a real
        # `review` bucket cost. Guarding on `total` would print
        # "(unpriced — model rate unknown)" straight over the top of money we did measure.
        # `any_unpriced()` still flags the half we could not price, below.
        if unpriced and billed == 0.0:
            return f"cost: (unpriced — model rate unknown) {toks}"
        s = f"cost: ${billed:.4f}"
        if self.max_spend is not None:
            s += f" / cap ${self.max_spend:.2f}"
        s += f" ({toks})"
        if unpriced:
            s += " [partially unpriced]"
        # #101: at the END, beside `[partially unpriced]`, rather than inside the parens
        # #100's cache clause uses -- those parens are labelled "tokens", and a dollar
        # figure is not one. This is also the only position that works on both branches.
        # Gated on there being off-loop spend, so every ledger that never compacted (and
        # every unpriced one) renders exactly as it did before.
        #
        # Gated on what will actually RENDER, not on the raw float: a cost below 5e-5 is
        # non-zero but formats as `$0.0000`, and `[incl. $0.0000 compaction]` names a
        # share of nothing -- #98's fabricated zero wearing a clause.
        if off > 0 and f"{off:.4f}" != "0.0000":
            names = sorted(
                k for k, v in self.buckets.items() if _as_float(v.get("cost")) > 0
            )
            # #117: BOUNDED. One or two names are the useful case and read well; five
            # (`compaction/review/scout/spawn/web_search`, reachable with
            # `--planner --review --web-search`) is 44 characters of clause on a footer
            # that is also two stop-reason messages, and it grows again with every future
            # bucket. Past two, defer to `/usage`, which breaks the same partition down
            # per row -- and reuse ITS word for the whole partition rather than coining a
            # third, so `[incl. $X off-loop]` and the `off-loop` block name one thing.
            label = "/".join(names) if len(names) <= 2 else "off-loop"
            s += f" [incl. ${off:.4f} {label}]"
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
            # event and an off-loop bucket but no main-loop tokens at all (#101).
            # Reporting "(no usage recorded yet)" for either would hide the trace at
            # precisely the moment it is the only evidence there is.
            # #101: `buckets` joins that gate. A `/compact` before any turn has an event
            # AND a real, priced off-loop call, and reporting "nothing yet" over the top
            # of money already spent is the same failure one clause up, with a price tag.
            if (not self.turns and not self.api_calls_total
                    and not self.context_events and not self.buckets):
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
            # #101: `_bucket_lines` wires into BOTH sites too -- the `#82`/`#99` rule,
            # third time. This branch is exactly where a `/compact`-before-any-turn
            # session lands, i.e. the one state where the bucket is the only money on
            # the page, so omitting it here would hide precisely what it exists to show.
            # #101: the cost line renders HERE too when there is off-loop money. A
            # `/compact` before the first turn reports no main-loop tokens but has
            # really spent, and `/cost` seconds later prints the figure AND the cap --
            # omitting them here is the two-surfaces-disagree failure `_cache_fragment`
            # warns about, in the one state where the cap is what you need to see.
            return "\n".join(
                ["session usage:", "  (no tokens reported)"]
                # ...but only when there is money to report. A run whose responses
                # simply carried no usage block has nothing to say here, and printing
                # `cost: $0.0000` for it would be the fabricated zero this branch was
                # built to avoid -- the line earns its place only once a bucket has
                # real spend that the token counters above cannot account for.
                + (self._cost_lines() if self.billed_total() > 0 else [])
                + ([self._timing_line()] if self.turns else [])
                + self._model_usage_lines()
                + self._tool_lines() + self._call_lines() + self._bucket_lines()
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
        if self.models:
            lines.append("  cache hit rate: see per-model breakdown")
        elif hit is None:
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
        lines.extend(self._cost_lines())
        lines.extend(self._model_usage_lines())
        if self.turns:  # #81
            lines.append(self._timing_line())
        lines.extend(self._tool_lines())  # #82 -- own gate; see `_tool_lines`
        lines.extend(self._call_lines())  # #99 -- likewise
        lines.extend(self._bucket_lines())  # #101 -- likewise
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

    def _tool_lines(self) -> list[str]:
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

    def _call_lines(self) -> list[str]:
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
            # summarization call is billed to a bucket, which appends no row (#101).
            # The event is the only record HERE that the prefix moved, so it renders on
            # its own rather than vanishing with the block that would have carried it;
            # the money is one block down, in `_bucket_lines`.
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
        # #101: same no-silent-caps rule. `api_calls_total` counts main-loop calls only,
        # and now that the ledger KNOWS about the off-loop ones, printing the smaller
        # number unqualified is a provable undercount. Named here as a residual, with
        # the detail in `_bucket_lines`, rather than folded into the count -- these
        # calls have no rows below, so a bigger number would not reconcile with them.
        off = self.bucket_calls()
        if off:
            head += f"  [+{off} off-loop]"
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

    def _cost_lines(self) -> list[str]:
        """`/usage`'s cost line (#101), shared by both of `usage_report`'s branches.

        Extracted because the no-tokens branch used to omit it entirely -- harmless
        while that state implied zero spend, wrong the moment a `/compact` before the
        first turn could land there with real money in a bucket. One definition means
        the two branches cannot disagree about the cap.

        The billed total here as in `summary`, for a sharper version of the same
        reason: `_cache_fragment` already notes an operator runs `/cost` and `/usage`
        seconds apart, so two numbers for one question read as two questions. The
        off-loop share is broken out by `_bucket_lines` rather than clause'd here,
        since this block has the room a one-liner does not.
        """
        billed = self.billed_total()
        unpriced = self.any_unpriced()
        if unpriced and billed == 0.0:
            return ["  cost: (model rate unknown)"]
        cost = f"  cost: ${billed:.4f}"
        if self.max_spend is not None:
            cost += f" / cap ${self.max_spend:.2f}"
        if unpriced:
            cost += "  [partially unpriced]"
        if self.any_unreported():
            cost += "  [partially unreported]"
        return [cost]

    def _bucket_lines(self) -> list[str]:
        """The `/usage` off-loop spend block (#101); ``[]`` when nothing was bucketed.

        These calls are deliberately absent from the `calls` trace above, so without
        this block the money would be in `to_dict()` and on no human surface at all.
        Sorted by NAME, like `to_dict`: `compaction` plus one row per subagent rail as
        of #117, so the stable order matters more than a by-cost ranking would.

        The seconds here are a SUBSET of `elapsed_seconds`, never a partition of it, and
        as of #117 they are not even disjoint from `self.tools`: a rail's API time sits
        inside the `venice_scout`/`venice_spawn`/`venice_review` tool window that
        contains it. Under `--parallel` several rails' windows also overlap each other,
        so the sum can exceed wall outright -- labelled below rather than hidden, exactly
        as `_tool_lines` labels the same state for tools.
        """
        if not self.buckets:
            return []
        w = max(len(k) for k in self.buckets)
        secs = sum(_as_float(b.get("seconds")) for b in self.buckets.values())
        # Leads with a right-aligned duration in the same column as `wall`/`tools`/
        # `calls`. The label is padded to the same 10 columns those three hard-code,
        # explicitly rather than by luck -- "off-loop" happens to be 8 characters, and
        # renaming it would otherwise slide the duration out of the shared grid.
        dur = "n/a" if (not secs and self.bucket_calls()) else format_duration(secs)
        head = (f"  {'off-loop':<8}{dur:>10}  "
                f"across {self.bucket_calls()} API call(s)  [not in the trace above]")
        # #117: same predicate and same wording as `_tool_lines`/`tools_fragment`. One
        # state must not grow two vocabularies across two blocks of one report -- an
        # operator reading `/usage` top to bottom would take them for two states.
        if self.turns and secs > self.elapsed_seconds + 0.001:
            head += "  [concurrent -- exceeds wall]"
        lines = [head]
        for name in sorted(self.buckets):
            row = self.buckets[name]
            pt = _as_int(row.get("prompt_tokens"))
            ct = _as_int(row.get("completion_tokens"))
            # PARTIALLY blind, not all-or-nothing: a bucket mixing one usage-bearing
            # call with one usage-less call has real numbers that are nonetheless an
            # undercount, and the earlier `unreported and not (pt or ct)` test hid the
            # marker the moment any call reported. Same shape as the main token block's
            # `[partially unreported]`, which has said this since #98.
            blind = bool(row.get("unreported"))
            mark = "" if not blind else ("  [unreported]" if not (pt or ct)
                                         else "  [partially unreported]")
            lines.append(
                f"    {name:<{w}}  {_as_int(row.get('calls'))} call(s)"
                f"  {('n/a' if blind and not pt else format(pt, ',')):>9} in"
                f"  {('n/a' if blind and not ct else format(ct, ',')):>7} out"
                f"  {self.bucket_money(row):>10}{mark}"
            )
        return lines

    @staticmethod
    def _event_lines(ev: dict) -> list[str]:
        """The one-or-two-line `/usage` marker for a context event (#99)."""
        if ev.get("kind") == "resume_reseed":
            return [
                "    -- system prompt reseeded on resume "
                f"after #{ev.get('after_n', '?')}; cache prefix is cold"
            ]
        out = [f"    -- compacted ({ev.get('trigger', 'auto')}) "
               f"after #{ev.get('after_n', '?')}: "
               f"{ev.get('messages_before', '?')} -> {ev.get('messages_after', '?')} msgs, "
               f"~{ev.get('est_tokens_before') or 0:,} -> "
               f"~{ev.get('est_tokens_after') or 0:,} tok est"]
        # #101: what the compaction COST to perform. "to summarize" is load-bearing --
        # a bare dollar figure beside "48 -> 13 msgs" reads as money saved, which is the
        # exact misreading that kept this key out of the event in #99.
        #
        # Gated on TRUTHINESS, not on the key being present: a static method cannot see
        # `self.unpriced`, and an unpriced session would otherwise render `$0.0000` --
        # #98's fabricated zero, one surface over. Pre-#101 envelopes have no key at all
        # and render exactly as they did.
        # COERCED, not formatted raw. `restore()` seeds `context_events` verbatim from
        # the session file (rows are copied, never validated field by field), so a
        # hand-edited or foreign-writer `"cost": "0.0031"` would make `:.4f` raise and
        # take `/usage` down with it -- on a resumed session, which is exactly when this
        # renders. Every sibling field on this line hedges the same way.
        cost = _as_float(ev.get("cost"))
        if cost and f"{cost:.4f}" != "0.0000":
            out[0] += f", ${cost:.4f} to summarize"
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
