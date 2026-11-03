"""Function-calling agent loop for `venice chat --tools` (issue #15).

`venice chat` is normally one-shot. With `--tools`, the model can invoke venice's
own endpoints as **in-process** function tools and the completion runs in a loop
(model -> tool_calls -> dispatch -> tool results -> repeat until it stops). This is
the self-contained-agent foundation for the vcoder epic (#25).

Import discipline: this module reuses the print-free `*_tool` primitives in
``commands._mcp`` but NEVER imports the ``mcp``/MCPServer SDK -- the whole point of
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
from typing import Callable, Dict, List, Optional, Tuple

from .. import userconfig

# Preserve the existing import surface while keeping static schemas out of the loop.
from . import (
    _compact,
    _cost_ledger,
    _exec,
    _mcp,
    _models,
    _openai,
    _shared,
    _tool_registry,
)
from . import upscale as _upscale

_video = _tool_registry._video
_p = _tool_registry._p
_obj = _tool_registry._obj
_IMAGE_SCHEMA = _tool_registry._IMAGE_SCHEMA
_TTS_SCHEMA = _tool_registry._TTS_SCHEMA
_BACKGROUND_PARAM = _tool_registry._BACKGROUND_PARAM
_SFX_SCHEMA = _tool_registry._SFX_SCHEMA
_MUSIC_SCHEMA = _tool_registry._MUSIC_SCHEMA
_JOB_STATUS_SCHEMA = _tool_registry._JOB_STATUS_SCHEMA
_JOB_RESULT_SCHEMA = _tool_registry._JOB_RESULT_SCHEMA
_UPSCALE_SCHEMA = _tool_registry._UPSCALE_SCHEMA
_BG_REMOVE_SCHEMA = _tool_registry._BG_REMOVE_SCHEMA
_CHAT_SCHEMA = _tool_registry._CHAT_SCHEMA
_WEB_SEARCH_SCHEMA = _tool_registry._WEB_SEARCH_SCHEMA
_SEARCH_SCHEMA = _tool_registry._SEARCH_SCHEMA
_REINDEX_SCHEMA = _tool_registry._REINDEX_SCHEMA
_SCOPE_PROP = _tool_registry._SCOPE_PROP
_TASK_STATUS_PROP = _tool_registry._TASK_STATUS_PROP
_MEMORY_WRITE_SCHEMA = _tool_registry._MEMORY_WRITE_SCHEMA
_MEMORY_READ_SCHEMA = _tool_registry._MEMORY_READ_SCHEMA
_MEMORY_SEARCH_SCHEMA = _tool_registry._MEMORY_SEARCH_SCHEMA
_MEMORY_LIST_SCHEMA = _tool_registry._MEMORY_LIST_SCHEMA
_TASK_ADD_SCHEMA = _tool_registry._TASK_ADD_SCHEMA
_TASK_UPDATE_SCHEMA = _tool_registry._TASK_UPDATE_SCHEMA
_TASK_LIST_SCHEMA = _tool_registry._TASK_LIST_SCHEMA
_MODELS_SCHEMA = _tool_registry._MODELS_SCHEMA
_MODEL_DETAILS_SCHEMA = _tool_registry._MODEL_DETAILS_SCHEMA
VISION_MODE_CHOICES = _tool_registry.VISION_MODE_CHOICES
_VISION_SCHEMA = _tool_registry._VISION_SCHEMA
_IMAGE_EDIT_SCHEMA = _tool_registry._IMAGE_EDIT_SCHEMA
_VIDEO_SCHEMA = _tool_registry._VIDEO_SCHEMA
_WEB_FETCH_SCHEMA = _tool_registry._WEB_FETCH_SCHEMA
_BROWSER_CAPTURE_SCHEMA = _tool_registry._BROWSER_CAPTURE_SCHEMA
ToolSpec = _tool_registry.ToolSpec
_BUILTINS = _tool_registry._BUILTINS
_CODE_ASSET_BUILTINS = _tool_registry._CODE_ASSET_BUILTINS
_REGISTRY = _tool_registry._REGISTRY
get = _tool_registry.get
list_categories = _tool_registry.list_categories
tools_in = _tool_registry.tools_in
select = _tool_registry.select

# --------------------------------------------------------------------------- #
# Tool descriptor + derived structures (pure functions of a list[Tool])
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tool:
    """One function tool the model may call.

    ``invoke(arguments, *, confirm=False)`` takes the model-supplied arguments
    object and normally returns a JSON-serializable result dict. A contextual
    in-process tool may receive loop-owned runtime facts and return a private
    :class:`_ToolOutcome`. ``paid`` marks tools whose result can be a
    ``confirmation_required`` gate.

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
    contextual: bool = False


@dataclass(frozen=True)
class _ToolRuntime:
    """Per-loop facts available to a contextual in-process tool.

    Tool schemas never expose this object.  It is minted by :func:`run_loop`, so
    a model cannot spoof the active frontend id or its catalog capabilities.
    """

    model: str
    models: Optional[List[dict]] = None


@dataclass(frozen=True)
class _ToolOutcome:
    """A model-visible tool result plus messages committed after its batch.

    Most tools keep returning a plain dict.  This private envelope exists for
    native vision, where the assistant's tool call must receive its ``tool``
    result before the loop can legally append the multimodal ``user`` message.
    """

    result: dict
    followups: Tuple[dict, ...] = ()


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
CACHE_GUARD_CHOICES = _cost_ledger.CACHE_GUARD_CHOICES
CACHE_GUARD_MIN_PROMPT_TOKENS = _cost_ledger.CACHE_GUARD_MIN_PROMPT_TOKENS
CacheGuardEvent = _cost_ledger.CacheGuardEvent
_usd_per_token = _cost_ledger._usd_per_token
_as_int = _cost_ledger._as_int
_as_float = _cost_ledger._as_float
_MODEL_USAGE_NUMERIC_FIELDS = _cost_ledger._MODEL_USAGE_NUMERIC_FIELDS
_MODEL_USAGE_FLAG_FIELDS = _cost_ledger._MODEL_USAGE_FLAG_FIELDS
_new_model_usage_row = _cost_ledger._new_model_usage_row
_restore_model_usage_row = _cost_ledger._restore_model_usage_row
_model_cache_hit_percent = _cost_ledger._model_cache_hit_percent
_public_model_usage_row = _cost_ledger._public_model_usage_row
_detail = _cost_ledger._detail
_CACHE_READ_KEYS = _cost_ledger._CACHE_READ_KEYS
_CACHE_WRITE_KEYS = _cost_ledger._CACHE_WRITE_KEYS
_cache_tokens = _cost_ledger._cache_tokens
_usage_dict = _cost_ledger._usage_dict
_dump_raw_usage = _cost_ledger._dump_raw_usage
format_duration = _cost_ledger.format_duration
CostLedger = _cost_ledger.CostLedger
_pricing_for = _cost_ledger._pricing_for


def _build_ledger(cap, models, model_id, *, max_tokens=None,
                  mirror=None, cache_guard="off") -> CostLedger:
    """A CostLedger bound to `model_id`'s catalog pricing (cap may be None).

    The ONLY caller of `bind_pricing`, deliberately: pricing a ledger is exactly
    "look this model up in the catalog", and a second site would be a second place
    to forget the lookup (which is how every rail ended up unpriced before #117).
    """
    return CostLedger(
        max_spend=cap, max_tokens=max_tokens, mirror=mirror,
        cache_guard=cache_guard, model_id=model_id, models=models,
    )


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
    return _build_ledger(
        getattr(args, "session_max_spend", None), models, model_id,
        cache_guard=getattr(args, "cache_guard", None) or "off",
    )


def subagent_ledger(models, model_id, *, max_tokens=None,
                    mirror=None) -> CostLedger:
    """The per-invoke ledger for one subagent rail (#117).

    ONE constructor for all four rails (scout/spawn/review/web search), so a fifth
    gets both halves for free and cannot ship with either missing:

    - PRICED against `model_id`'s own catalog entry. The rails used to build a bare
      `CostLedger(max_tokens=...)` on the reasoning that token counting needs no
      catalog -- true while the tokens went nowhere, false the moment they are billed
      to a parent. `--review-model` and `--web-search-model` are deliberately NOT the
      parent's model, so the rate has to be looked up per rail, here.
    - MIRRORED into the parent's off-loop bucket when a parent exists (`mirror`), and
      standalone when it does not -- `venice review` has no parent ledger and must
      keep working, so `mirror=None` is a supported state rather than an oversight.

    `max_tokens` is #52's per-worker ceiling and still meters this ledger's OWN
    main-loop counters, so `--subagent-max-tokens` is unchanged by the mirror.
    Uncapped in USD (`cap=None`): the parent's `--session-max-spend` is the money gate,
    and it now sees this spend through the bucket.
    """
    return _build_ledger(None, models, model_id, max_tokens=max_tokens, mirror=mirror)


def dispatch_map(tools: List[Tool]) -> Dict[str, Tool]:
    return {t.name: t for t in tools}


# --------------------------------------------------------------------------- #
# Web search (#77): one server-side Venice completion with `enable_web_search`, so
# the coding agent / scout can DISCOVER documentation and return citations. Rides the
# normal completion path (same
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
                   models=None, mirror=None) -> dict:
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
    # #117: still this call's OWN ledger -- priced against `model`, which is the resolved
    # web-search model and NOT necessarily the caller's. `mirror` (when the rail was built
    # with a parent) also banks the call into the parent's `web_search` bucket, which is
    # what finally puts these tokens on the books: pre-#117 only `cost_estimate_usd`
    # survived and every token count died here.
    led = subagent_ledger(models, model, mirror=mirror)
    cost = led.record(getattr(resp, "usage", None), seconds=time.monotonic() - _t0)
    # Best-effort: report None (unknown) -- not $0.00 -- when we can't estimate, i.e. the
    # model price is unknown OR the response carried no usage tokens. A billed feature that
    # reports 0.0 reads as "free", which is worse than an honest "unknown".
    #
    # NOT made redundant by the bucket above, and the two must both stay: this field is
    # MODEL-visible handoff provenance (the agent reads it to decide whether to search
    # again), the bucket is OPERATOR accounting. Deleting either to "de-duplicate" loses
    # a different audience.
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
# Runtime tool composition and loop-owned argument authority
# --------------------------------------------------------------------------- #
# Loop-controlled kwargs the model must never supply (stripped defensively).
_CONTROLLED = (
    "confirm", "max_spend", "output_dir", "path_authority", "download_url",
)


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
    controlled = ("allow", "deny", "private_hosts", "private_ranges")
    return {key: value for key, value in _clean(arguments).items() if key not in controlled}


def browser_tools(
    *, allow=(), deny=(), private_hosts=(), private_ranges=(), output_dir=None,
    config=None,
) -> List[Tool]:
    """Build browser tools with all network authority bound by the operator."""
    fetch_defaults = userconfig.config_defaults_for("browser", _mcp.web_fetch_tool, config)
    capture_defaults = userconfig.config_defaults_for(
        "browser", _mcp.browser_capture_tool, config
    )

    def invoke_fetch(arguments, *, confirm: bool = False):
        return _mcp.web_fetch_tool(
            allow=allow, deny=deny, private_hosts=private_hosts,
            private_ranges=private_ranges,
            **{**fetch_defaults, **_browser_args(arguments)},
        )

    def invoke_capture(arguments, *, confirm: bool = False):
        return _mcp.browser_capture_tool(
            allow=allow, deny=deny, private_hosts=private_hosts,
            private_ranges=private_ranges, output_dir=output_dir,
            **{**capture_defaults, **_browser_args(arguments)},
        )

    return [
        Tool(
            "web_fetch",
            "Fetch an HTTP(S) URL through a DNS-pinned egress boundary and return "
            "text or HTML. Redirects are revalidated; operator policy cannot be widened.",
            _WEB_FETCH_SCHEMA, invoke_fetch, paid=False, category="web",
            tags=("read", "network"),
        ),
        Tool(
            "browser_capture",
            "Render an HTTP(S) URL in sandboxed disposable Chromium through an "
            "enforcing proxy; return DOM/text and/or a screenshot path.",
            _BROWSER_CAPTURE_SCHEMA, invoke_capture, paid=False, category="web",
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
    browser_private_hosts=(),
    browser_private_ranges=(),
    browser_output_dir: Optional[str] = None,
    memory: bool = False,
    exec_timeout: int = _exec.DEFAULT_EXEC_TIMEOUT,
    media_path_authority=None,
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

    `browser` appends the pinned `web_fetch` and sandboxed `browser_capture` rails.

    `memory` (issue #49) appends the persistent memory + task rails (`memory_tools`):
    free, local notes (two tiers) + a project task list. Also a rail (added after the
    `only` filter, absent from `_BUILTINS`/`mcp-serve`).

    `media_path_authority` is loop-controlled and never advertised. It confines
    every model-supplied local media input to operator-authorized roots; chat
    defaults to the cwd captured here, while code supplies its mutable root set.
    """

    if media_path_authority is None:
        media_path_authority = _shared.MediaPathAuthority(os.getcwd())

    def _config_defaults(section, impl) -> dict:
        # #58: shared with mcp-serve -- layer defaults.<section>.* under tool args.
        return userconfig.config_defaults_for(section, impl, config)

    def _make_paid(impl, section, impl_name):
        defaults = _config_defaults(section, impl)
        retired_config = (
            _upscale.retired_config_keys(config) if section == "upscale" else ()
        )

        def invoke(arguments, *, confirm: bool = False):
            if section == "upscale":
                retired_args = tuple(
                    key for key in _upscale.RETIRED_CONFIG_KEYS if key in arguments
                )
                if retired_args:
                    names = ", ".join(retired_args)
                    return {
                        "status": "error",
                        "message": f"upscale: retired argument(s): {names}",
                    }
                if retired_config:
                    return {
                        "status": "error",
                        "message": _upscale.retired_config_message(retired_config),
                    }
            controlled = {
                "confirm": confirm,
                "max_spend": max_spend,
                "output_dir": output_dir,
            }
            if impl_name in {
                "image_tool", "upscale_tool", "bg_remove_tool", "video_tool",
                "image_edit_tool",
            }:
                controlled["path_authority"] = media_path_authority
            return impl(client, **controlled, **{**defaults, **_clean(arguments)})

        return invoke

    def _make_free(impl, section, impl_name):
        defaults = _config_defaults(section, impl)

        def invoke(arguments, *, confirm: bool = False, runtime=None):
            controlled = {}
            if impl_name == "vision_tool":
                controlled["path_authority"] = media_path_authority
                values = {**defaults, **_clean(arguments)}
                mode = values.pop("mode", "auto")
                if mode not in VISION_MODE_CHOICES:
                    return {
                        "status": "error",
                        "message": (
                            "vision: mode must be one of "
                            + ", ".join(VISION_MODE_CHOICES)
                        ),
                    }

                frontend = runtime.model if isinstance(runtime, _ToolRuntime) else None
                models = runtime.models if isinstance(runtime, _ToolRuntime) else None
                capability = (
                    _models.supports_capability(models, frontend, "supportsVision")
                    if frontend else None
                )
                native = mode == "native" or (mode == "auto" and capability is True)
                if mode == "native" and capability is not True:
                    state = "does not advertise" if capability is False else "has no known"
                    return {
                        "status": "error",
                        "message": (
                            f"vision: active model {frontend!r} {state} supportsVision; "
                            "use mode='delegate' or mode='auto'"
                        ),
                    }

                if native:
                    prepared = _mcp.prepare_vision_input(
                        values.get("input_path"),
                        image_url=values.get("image_url"),
                        path_authority=media_path_authority,
                    )
                    if prepared.get("status") != "ok":
                        return prepared
                    prompt = (
                        values.get("prompt") or _mcp.DEFAULT_VISION_PROMPT
                    ).strip()
                    followup = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": prepared["image_url"]},
                            },
                        ],
                    }
                    return _ToolOutcome(
                        {
                            "status": "ok",
                            "mode": "native",
                            "model": frontend,
                            "message": "image attached to the active frontend",
                        },
                        (followup,),
                    )

                result = impl(client, **controlled, **values)
                if isinstance(result, dict) and result.get("status") == "ok":
                    result = {**result, "mode": "delegate"}
                return result
            return impl(client, **controlled, **{**defaults, **_clean(arguments)})

        return invoke

    source = _BUILTINS if only is None else _BUILTINS + _CODE_ASSET_BUILTINS
    tools = [
        Tool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            invoke=(
                _make_paid(
                    getattr(_mcp, spec.impl), _tool_section(spec.name), spec.impl
                )
                if spec.paid
                else _make_free(
                    getattr(_mcp, spec.impl), _tool_section(spec.name), spec.impl
                )
            ),
            paid=spec.paid,
            category=spec.category,
            tags=spec.tags,
            contextual=(spec.impl == "vision_tool"),
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
            private_hosts=browser_private_hosts,
            private_ranges=browser_private_ranges,
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
_REASONING_FIELDS = ("reasoning_content", "reasoning_details", "reasoning")
_THOUGHT_SIGNATURE_FIELD = "thought_signature"


def _message_field(msg, name):
    """Read one response-message field from an SDK model or a plain mapping."""
    if isinstance(msg, dict):
        return msg.get(name)
    return getattr(msg, name, None) if msg is not None else None


def _request_value(value):
    """Turn nested SDK response values into request-safe Python values.

    ``reasoning_content`` is normally a string, but the compatible
    ``reasoning_details`` extension may contain SDK model objects.  Serialize only
    the selected value recursively instead of dumping the complete response message;
    the latter also carries response-only metadata rejected by request schemas.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_request_value(item) for item in value]
    if isinstance(value, tuple):
        return [_request_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _request_value(item) for key, item in value.items()}
    return value


def _assistant_dict(msg) -> dict:
    """Reconstruct the replayable subset of an assistant response.

    Keep content and exact tool-call ids plus one deliberately allowlisted reasoning
    extension.  Compatible providers use three aliases; when a response unexpectedly
    carries more than one, preserve the first in ``_REASONING_FIELDS`` order rather
    than sending conflicting thinking payloads back on the next request.  A Gemini
    ``thought_signature`` is independent of those aliases and must be passed back
    exactly when present.  Do not use a whole-message ``model_dump()``: response-only
    metadata can be rejected on replay.
    """
    d = {"role": "assistant", "content": (_message_field(msg, "content") or "")}
    for field in _REASONING_FIELDS:
        value = _message_field(msg, field)
        if value is not None:
            d[field] = _request_value(value)
            break
    signature = _message_field(msg, _THOUGHT_SIGNATURE_FIELD)
    if signature is not None:
        d[_THOUGHT_SIGNATURE_FIELD] = _request_value(signature)
    tcs = _message_field(msg, "tool_calls")
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


def _invoke_tool(tool: Tool, arguments: dict, *, confirm: bool, runtime=None):
    """Invoke one tool, adding loop-owned context only to contextual tools."""
    kwargs = {"confirm": confirm}
    if tool.contextual:
        kwargs["runtime"] = runtime
    return tool.invoke(arguments, **kwargs)


def _visible_tool_result(result) -> dict:
    return result.result if isinstance(result, _ToolOutcome) else result


def _tool_followups(result) -> Tuple[dict, ...]:
    return result.followups if isinstance(result, _ToolOutcome) else ()


def _vision_route_progress(name: str, result) -> Optional[str]:
    """Describe who inspected an image, without repeating its path or URL."""
    if name != "venice_vision":
        return None
    visible = _visible_tool_result(result)
    if not isinstance(visible, dict) or visible.get("status") != "ok":
        return None
    mode = visible.get("mode")
    model = visible.get("model")
    via = f" via {model}" if isinstance(model, str) and model else ""
    if mode == "native":
        return f"  ↳ vision: native{via} (image added to active model context)"
    if mode == "delegate":
        return f"  ↳ vision: delegated{via} (separate completion)"
    return None


def _resolve_spend(
    tool: Tool, arguments: dict, result, gate: dict, *, window=None, runtime=None
):
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
                    return _invoke_tool(
                        tool, arguments, confirm=True, runtime=runtime
                    )
            except Exception as e:  # pragma: no cover - impls shouldn't raise
                return {"status": "error", "message": f"{tool.name} failed: {e}"}
        print(f"{tool.name}: declined by user", file=sys.stderr)
    return result  # non-TTY or declined -> the model sees the gate and adapts


def _run_one_call(
    tc, dispatch: Dict[str, Tool], gate: dict, *, on_tool=None, runtime=None
):
    """Run one tool call and return its result dict or private outcome.

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
                result = _invoke_tool(
                    tool, arguments, confirm=bool(gate["auto"]), runtime=runtime
                )
        except Exception as e:  # pragma: no cover - impls shouldn't raise
            return {"status": "error", "message": f"{tool.name} failed: {e}"}
        return _resolve_spend(
            tool, arguments, result, gate, window=window, runtime=runtime
        )
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
    runtime=None,
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
                ex.submit(
                    _run_one_call, tool_calls[i], dispatch, gate,
                    on_tool=on_tool, runtime=runtime,
                ): i
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
        results[i] = _run_one_call(
            tool_calls[i], dispatch, gate, on_tool=on_tool, runtime=runtime
        )

    # Commit on the main thread: append in ORIGINAL order, advance by the executed count.
    for i, tc in enumerate(tool_calls):
        route = _vision_route_progress(tc.function.name, results[i])
        if route is not None:
            _progress(route, enabled=show)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(
                    _visible_tool_result(results[i]), default=str, allow_nan=False
                ),
            }
        )
    for result in results:
        messages.extend(_tool_followups(result))
    return calls_made + min(slots, n)


def _emit_final(resp, json_out: bool) -> int:
    if json_out:
        json.dump(
            resp.model_dump(), sys.stdout, indent=2, default=str, allow_nan=False
        )
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
    final_emitter: Optional[Callable[[object], int]] = None,
    models: Optional[List[dict]] = None,
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

    `final_emitter` is an internal output seam. By default the loop prints the final
    response according to `json_out`; callers that need to build an envelope only
    after their own accounting is finalized can capture the response instead.

    `models` is the current text catalog used only for loop-owned contextual policy
    such as native vision. It never rides the model-facing tool schema.
    """
    oai_tools = to_openai_tools(tools)
    dispatch = dispatch_map(tools)
    runtime = _ToolRuntime(model=model, models=models)
    if budget is not None and budget.protected_system_messages is None:
        budget.protected_system_messages = _compact.leading_system_count(messages)
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

    emit_final = (
        final_emitter
        if final_emitter is not None
        else lambda response: _emit_final(response, json_out)
    )

    def _force_final(reason: str) -> int:
        _compact.maybe_compact(
            oai, model, messages, budget, base_kwargs,
            on_compact=lambda b, a: _progress(
                f"(auto-compacted history: {b} -> {a} messages)", enabled=show,
            ),
            on_blocked=lambda why: print(
                f"auto-compaction refused: {why}", file=sys.stderr,
            ),
            on_fallback=lambda why: print(
                f"auto-compaction recovery: {why}", file=sys.stderr,
            ),
            ledger=ledger,
        )
        if budget is not None and budget.hard_blocked:
            return 2
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
        if budget is not None:
            budget.observe(getattr(resp, "usage", None), messages)
        if ledger is not None:
            # #99: caller-stamped, like every other window in this file. NOT added to
            # `elapsed_seconds` -- that is `record_turn`'s job at the command level, and
            # stamping it here too would double-count the same wall time.
            ledger.record(getattr(resp, "usage", None),
                          seconds=time.monotonic() - _t0)
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        return emit_final(resp)

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
        if ledger is not None and ledger.invalid_pricing:
            print(
                "chat: model catalog contains invalid non-finite pricing; "
                "refusing to start a paid turn",
                file=sys.stderr,
            )
            return 2
        if ledger is not None and ledger.over():
            return _force_final(
                f"reached --session-max-spend ({ledger.summary()}); "
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
            on_blocked=lambda reason: print(
                f"auto-compaction refused: {reason}", file=sys.stderr,
            ),
            on_fallback=lambda reason: print(
                f"auto-compaction recovery: {reason}", file=sys.stderr,
            ),
            ledger=ledger,  # #99: log the event; #101: and bill the summary call
        )
        if budget is not None and budget.hard_blocked:
            # Evidence preservation failed or even the newest retained turn is
            # transport-oversized. Do not follow a local refusal with the exact
            # request we already know is unsafe (the #241 HTTP 413 chain).
            return 2
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
            budget.observe(getattr(resp, "usage", None), messages)
        if ledger is not None:
            # #99: see `_force_final` -- caller-stamped, and deliberately NOT folded into
            # `elapsed_seconds`. Unlike the per-tool windows these are strictly serial on
            # any one ledger (subagents each get their own), so their sum can never exceed
            # wall and `_call_lines` needs no `[concurrent]` marker.
            ledger.record(getattr(resp, "usage", None),
                          seconds=time.monotonic() - _t0)
        cache_event = (
            ledger.cache_guard_event(model) if ledger is not None else None
        )
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
        if cache_event is not None and cache_event.action == "warn":
            print(cache_event.message, file=sys.stderr)
        if not tool_calls:
            if cache_event is not None and cache_event.action == "stop":
                # The model already supplied a final response. Buying another
                # completion would contradict the circuit breaker's purpose.
                print(
                    f"{cache_event.message}; response is already final",
                    file=sys.stderr,
                )
            return emit_final(resp)

        # Every tool_call in the turn must get a result (message-contract), even
        # ones past the budget -- those are reported not-executed rather than run.
        if parallel and any(_is_parallelizable(tc) for tc in tool_calls):
            # #52: run independent subagent dispatches concurrently. All bookkeeping
            # (result append in original order, budget) stays on the main thread.
            calls_made = _dispatch_parallel(
                tool_calls, dispatch, gate, messages,
                calls_made=calls_made, max_tool_calls=max_tool_calls,
                unlimited=unlimited, show=show, on_tool=on_tool,
                runtime=runtime,
            )
        else:
            followups = []
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
                    result = _run_one_call(
                        tc, dispatch, gate, on_tool=on_tool, runtime=runtime
                    )
                    calls_made += 1
                route = _vision_route_progress(tc.function.name, result)
                if route is not None:
                    _progress(route, enabled=show)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(
                            _visible_tool_result(result),
                            default=str,
                            allow_nan=False,
                        ),
                    }
                )
                followups.extend(_tool_followups(result))
            messages.extend(followups)

        if cache_event is not None and cache_event.action == "stop":
            # The assistant tool-call message must receive one result per call before
            # another assistant message is legal. Once that contract is satisfied,
            # reuse the existing no-tools finalization seam; do not invent a second
            # teardown path for the cache gate.
            return _force_final(
                f"{cache_event.message}; requesting a final answer"
            )

        if not unlimited and calls_made >= max_tool_calls:
            # The forced-final is the turn a long, over-budget run most needs
            # compacted. `_force_final` owns that preflight so every forced-final
            # path (tool cap, spend cap, token cap, cache guard) has the same byte
            # safety rail and no path attempts the summary twice.
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
    role: Optional[str] = None,
) -> dict:
    """Run one disposable subagent turn-loop on a FRESH context and return its report.

    The shared core behind :func:`run_scout` (read-only) and :func:`run_spawn`
    (write/paid-capable): seeds a fresh ``messages`` list (only static ``system`` + one
    user task -- nothing from the caller's context leaks in), drives :func:`run_loop`
    with the given ``tools`` under a stdout firewall (the printed final answer is
    captured and discarded), and returns
    ``{"status","report","tool_calls","truncated"}``. Dynamic focus/role guidance rides
    in that user turn so repeated rail invocations keep a byte-identical cacheable
    system prefix. The report is recovered from the message tail -- the final assistant
    turn in both the natural-stop and cap-forced paths. ``openai.OpenAIError`` from the
    loop propagates to the caller (the Tool wrapper turns it into an error envelope).

    Capability-agnostic: the read-only-vs-write distinction and any self-spawn guard live
    in the two thin wrappers, not here. Runs with ``yes=True`` -- for the worker that is
    required (mutating tools are ``paid=True`` and would otherwise be blocked in a non-
    interactive parent); for the scout it is a no-op (all its tools are free).
    """
    task = (task or "").strip()
    if not task:
        return {"status": "error", "message": "subagent requires a non-empty task"}

    guidance = []
    if role:
        guidance.append(f"Your role: {role}.")
    if focus:
        guidance.append(f"Focus hint (not a hard scope): {focus}")
    user_task = "\n".join(guidance) + f"\n\n{task}" if guidance else task
    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_task},
    ]
    # #128: this is a fresh conversation, not a continuation of the parent. Replace
    # (rather than inherit) its routing key; every call inside this disposable loop
    # then shares affinity, while two workers cannot collide on one backend identity.
    child_kwargs = _openai.with_prompt_cache_key(base_kwargs)
    with _capture_stdout() as buf:
        run_loop(
            oai, model, messages, child_kwargs, tools,
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
    out = _run_disposable(
        oai, model, task, tools, base_kwargs, system=system,
        max_tool_calls=max_tool_calls, budget=budget, ledger=ledger, focus=focus,
        role=role,
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
