"""`venice code` -- the vcoder coding agent (issue #30).

Wraps the built-in coding toolset (`commands._code`, #29) and the function-calling
loop (`_agent`, #15) in a coding-oriented harness with an explicit **plan ->
acceptance -> run** workflow, so it serves a human at a terminal, a shell script, or
a controlling LLM identically:

1. **Plan.** One model turn with ``tool_choice="none"`` (no side effects) emits a
   numbered plan + acceptance criteria.
2. **Acceptance boundary**, crossable three ways: an interactive prompt on a TTY
   (``a``uto / ``s``tep / ``e``dit / ``N``o); the ``--auto``/``--manual`` flags; or
   ``--plan-only`` (print the plan and exit, letting a caller approve out of band).
   Non-TTY with neither ``--auto`` nor ``--plan-only`` aborts (exit 2) -- side effects
   never run unattended without an explicit opt-in.
3. **Execute.** :func:`_agent.run_loop` with the accepted plan seeded in; autonomous
   (``--auto`` -> every tool auto-approved) or manual (per-step confirm gate on the
   ``paid=True`` write/edit/run tools).
4. **Acceptance check.** A final ``tool_choice="none"`` turn reports each criterion
   met/unmet and ends with an ``ACCEPTANCE: PASS``/``FAIL`` verdict. The parse is
   format-tolerant and re-prompts once for the verdict line if the first reply lacks
   it; ``--json`` emits the verdict structured. The exit code reflects it: 0 = pass
   (or check skipped), 1 = fail, 10 = verdict unparseable even after the re-prompt
   (the work may still be complete).

Unlike ``venice chat --tools`` (which degrades to plain chat), ``venice code`` errors
out on a non-tool-calling model -- coding without tools is pointless.

Import discipline mirrors `chat`: the `openai` SDK is lazy-imported; the coding
engine is stdlib-only + mcp-free. Reuses `_openai`/`_models`/`_agent` and, for the
`-i` REPL, `_repl.run` (with an injected coding tools session + gen kwargs).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from typing import List, Optional

from .. import _numeric, auth, config, userconfig
from ..client import build_client_from_auth
from . import (_agent, _browser, _code, _compact, _mailbox, _models, _openai,
               _repl, _review, _session, _steer)

_DEFAULT_MAX_TOOL_CALLS = 25


def _budget_for(args) -> Optional[_compact.Budget]:
    """The auto-compact budget for a run, or None when it isn't opted into (#48).

    Enabled by `--auto-compact` / `defaults.code.auto_compact`; threshold and
    keep-turns fall back to the `_compact` defaults when unset. Thin alias for
    the shared builder so every surface opts in identically.
    """
    return _compact.budget_from_args(args)

CODING_SYSTEM_PROMPT = """\
You are vcoder, an autonomous coding agent working inside a single project directory.

Project root: {root}
Available tools: {tools}

Guidelines:
- File paths are relative to the active project root (above); writes outside the \
writable roots fail loudly. If your work spans repositories, attach the other repo with \
attach_root -- it registers the root and switches the active directory so relative paths \
and run/git follow it -- rather than writing a path into the wrong repo.
- Explore before you change: use read_file, list_dir, grep (and project_search when \
available) to understand the code first.
- Prefer edit_file for small, targeted changes; use write_file for new files or full \
rewrites. Match the surrounding code's style.
- Use run to run tests, builds, or git mutations. run, write_file, and edit_file \
change the project and may require the user's confirmation before they execute.
- Make minimal, correct changes and verify them (run the tests or relevant command) \
when practical.
- Keep your final message a concise summary: what you changed and how you verified it."""

# The planner-harness overlay (#52 planner slice), appended to the coding system
# prompt by --planner. The workflow is prompt-mandated (the model decides what to
# decompose and when to dispatch -- run_loop stays the only loop); the structure
# around it is harness-enforced: task tools persist the checklist (#49), every
# scout/spawn dispatch is recorded for venice_merge, and task_id links the two.
PLANNER_PROTOCOL = """\

You are running as a PLANNER: decompose, dispatch, track, and MERGE.
1. DECOMPOSE: split the task into small self-contained units and task_add each one \
BEFORE dispatching anything.
2. DISPATCH serially, one unit at a time: task_update it in_progress; use \
venice_scout first when you need facts; delegate the work with venice_spawn, passing \
the unit's task_id. The subagent cannot see this conversation -- its task text must \
stand alone.
3. TRACK: when the report returns, task_update the unit done (or leave it \
in_progress with the blocker recorded in its text) before dispatching the next one. \
Never two dispatches in flight for one task.
4. MERGE (mandatory): after the last unit, call venice_merge for the consolidated \
rollup, resolve its warnings (re-dispatch, fix inline, or record a follow-up), and \
end your final message with a 'MERGE SUMMARY:' section -- what shipped, per-unit \
outcome, unresolved blockers/follow-ups.
Do trivial glue work yourself; dispatch anything multi-file or self-contained."""

# Appended AFTER PLANNER_PROTOCOL only under `--planner --parallel` (#52). It RELAXES
# step 2's "one unit at a time" for units that are provably independent, while keeping
# every other rule (task_add-first, dependent units serial, one dispatch per task, the
# mandatory MERGE). Kept a separate overlay so the default planner prompt -- and its
# test pins -- stay byte-identical when --parallel is off.
PLANNER_PARALLEL_OVERLAY = """

PARALLEL DISPATCH IS ENABLED, relaxing step 2's "one unit at a time" for INDEPENDENT \
units. You MAY emit several venice_scout/venice_spawn calls in a SINGLE turn; they run \
concurrently and their reports return together. Rules that still hold:
- task_add EVERY unit FIRST (step 1) before dispatching anything.
- Dispatch units together ONLY when they are truly INDEPENDENT: none needs another's \
output and no two touch the same files. Dependent units stay SERIAL -- dispatch the \
prerequisite, wait for its report, then dispatch what depends on it.
- Never two dispatches in flight for the SAME task_id.
- Mark every unit in a batch in_progress before dispatching it, and task_update each \
one done (or record its blocker) from the returned reports before starting the next batch.
- MERGE is unchanged and still mandatory: after the LAST unit call venice_merge, resolve \
its warnings, and end with a 'MERGE SUMMARY:' section.
Prefer a concurrent batch of independent units over strict one-at-a-time when it is safe."""

_PLAN_INSTRUCTION = (
    "Before doing anything, output a short numbered plan of the steps you will take, "
    "followed by an 'Acceptance criteria:' section listing concrete, checkable "
    "conditions for success. Do not use any tools yet -- just output the plan."
)
_PROCEED_MSG = (
    "The plan is accepted. Implement it now using the tools. When finished, briefly "
    "summarize what you changed and how you verified it."
)
_VERIFY_MSG = (
    "Now check the acceptance criteria from your plan against what you actually did. "
    "For each criterion, state MET or NOT MET in one line. Then output a final line "
    "that is exactly 'ACCEPTANCE: PASS' if every criterion is met, or "
    "'ACCEPTANCE: FAIL' otherwise."
)
_VERIFY_RETRY_MSG = (
    "Your reply did not end with the required verdict line. Reply with nothing but a "
    "single line that is exactly 'ACCEPTANCE: PASS' if every acceptance criterion is "
    "met, or 'ACCEPTANCE: FAIL' otherwise."
)

_VERDICT_RE = re.compile(r"ACCEPTANCE:\s*(PASS|FAIL)", re.IGNORECASE)


def _parse_verdict(report: Optional[str]) -> Optional[str]:
    """'pass' / 'fail' from the last ACCEPTANCE sentinel in the report, or None if
    no recognizable verdict is present (case/whitespace/markdown tolerant)."""
    m = _VERDICT_RE.findall(report or "")
    return m[-1].lower() if m else None


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "code",
        help="Coding agent: plan, then edit/run a project with Venice models.",
        description=(
            "Run a coding agent (vcoder) over a project directory. It proposes a "
            "plan, waits for your acceptance, then reads/edits files and runs "
            "commands using built-in, path-sandboxed tools. Autonomous with --auto, "
            "or step-by-step (confirming each change) by default on a terminal."
        ),
    )
    p.add_argument(
        "task", nargs="?",
        help="What to do. Use '-' or pipe stdin to read the task from stdin; omit "
        "on a terminal to start an interactive session.",
    )
    p.add_argument(
        "--root", default=None, metavar="DIR",
        help="Project directory the agent is sandboxed to (default: current "
        "directory; also $VENICE_CODE_ROOT).",
    )
    p.add_argument(
        "--model", "-m", default=None,
        help="Text model id (must support function calling).",
    )
    p.add_argument(
        "--system", "-s", default=None,
        help="Extra project-specific instructions appended to the coding prompt.",
    )
    p.add_argument("--temperature", "-t", type=_numeric.finite_float, default=None)
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    p.add_argument(
        "--json", action="store_true",
        help="Emit a JSON envelope (plan, final summary, acceptance) to stdout.",
    )

    grp = p.add_argument_group("Plan / run")
    mode = grp.add_mutually_exclusive_group()
    mode.add_argument(
        "--auto", action="store_true", default=None,
        help="Accept the plan and run autonomously (auto-approve every tool call). "
        "Required to run unattended (no terminal).",
    )
    mode.add_argument(
        "--manual", action="store_true", default=None,
        help="Accept the plan and run with per-step confirmation (default on a "
        "terminal).",
    )
    grp.add_argument("--yes", "-y", action="store_true", default=None,
                     help="Alias for --auto.")
    grp.add_argument(
        "--plan-only", action="store_true", dest="plan_only", default=False,
        help="Print the plan and exit without executing (for review/automation).",
    )
    grp.add_argument(
        "--no-plan", action="store_true", dest="no_plan", default=False,
        help="Skip the planning turn and execute directly.",
    )
    grp.add_argument(
        "--no-verify", action="store_true", dest="no_verify", default=False,
        help="Skip the post-run acceptance-criteria check.",
    )
    grp.add_argument(
        "--max-tool-calls", type=int, default=None, dest="max_tool_calls",
        metavar="N",
        help=f"Cap tool invocations before forcing a final answer (default: "
        f"{_DEFAULT_MAX_TOOL_CALLS}; 0 = unlimited, run until the model stops).",
    )
    grp.add_argument(
        "--exec-timeout", type=int, default=None, dest="exec_timeout",
        metavar="SECS",
        help=f"Timeout for run/git commands (default: {_code.DEFAULT_EXEC_TIMEOUT}).",
    )
    grp.add_argument(
        "--shell-allow", action="append", dest="shell_allow", default=None,
        metavar="CMD",
        help="Restrict the `run` tool to these commands (repeatable; globs ok on the "
        "leading token; a non-empty allowlist also requires a single simple command). "
        "Adds to the config `shell.allow` list, shared with `venice chat --shell` (#33).",
    )
    grp.add_argument(
        "--shell-deny", action="append", dest="shell_deny", default=None,
        metavar="PATTERN",
        help="Refuse `run` commands matching these globs (repeatable; matched on the "
        "whole line and each token; always enforced, wins over allow). Adds to config "
        "`shell.deny`.",
    )
    grp.add_argument(
        "--allow-root", action="append", dest="allow_root", default=None,
        metavar="DIR",
        help="Additional directory the file tools may read AND write, beyond the "
        "startup root (repeatable; for sessions that span repos). The agent can also "
        "attach one at runtime with the attach_root tool. Adds to config `roots.allow` (#76).",
    )
    grp.add_argument(
        "--deny-root", action="append", dest="deny_root", default=None,
        metavar="DIR",
        help="Directory excluded from writes (readable if under an allowed root; "
        "deny wins). Repeatable. Adds to config `roots.deny` (#76).",
    )
    grp.add_argument(
        "--assets", action="store_true", dest="assets", default=None,
        help="Also expose the in-process asset-generation tools (venice_image, "
        "image_edit, sfx, music, tts, upscale, bg_remove, video) so the agent can "
        "create images/audio/video in the project. Paid: each confirms per call "
        "unless --auto.",
    )
    grp.add_argument(
        "--browser", action="store_true", dest="browser", default=None,
        help="Reserved browser rail flag. Temporarily unavailable for security; "
        "fails closed before any API or network access.",
    )
    grp.add_argument(
        "--browser-allow", action="append", dest="browser_allow", default=None,
        metavar="HOST",
        help="Retained browser.allow config compatibility option; inert while the "
        "browser rail is security-disabled.",
    )
    grp.add_argument(
        "--browser-deny", action="append", dest="browser_deny", default=None,
        metavar="PATTERN",
        help="Retained browser.deny config compatibility option; inert while the "
        "browser rail is security-disabled.",
    )
    grp.add_argument(
        "--memory", action="store_true", dest="memory", default=None,
        help="Add persistent memory + task tools (memory_write/read/search/list, "
        "task_add/update/list) so the agent keeps durable notes and a checklist "
        "across turns/sessions -- the shared state a #52 planner hands to subagents. "
        "Project notes ride <root>/.venice/memory; global notes "
        "~/.config/venice/memory ($VENICE_MEMORY_DIR). Inspect with `venice memory` (#49).",
    )
    grp.add_argument(
        "--scout", action="store_true", default=None, dest="scout",
        help="Expose venice_scout: delegate a read-only investigation to a "
        "disposable subagent with a FRESH context and only read tools. It returns a "
        "structured report (findings/confidence/dead-ends/not-checked/verified) so "
        "heavy exploration doesn't pollute this session -- a context firewall, not a "
        "role-specialized worker. Read-only: the scout can't edit or run (#52).",
    )
    grp.add_argument(
        "--spawn", action="store_true", default=None, dest="spawn",
        help="Expose venice_spawn: delegate a bounded task to a disposable WORKER "
        "subagent with a FRESH context and a role-scoped subset of your tools. Unlike "
        "the scout it CAN edit/run (role 'code') or generate media (role 'asset', with "
        "--assets); returns a structured report (outcome/changes/verified/follow-ups/"
        "blockers). Writes stay inside your writable roots (fail loud outside) and it "
        "can't spawn further subagents or widen roots (#52).",
    )
    grp.add_argument(
        "--spawn-max-spend", type=_numeric.finite_float, default=None,
        dest="spawn_max_spend",
        metavar="USD",
        help="Per-worker USD cap on the cumulative estimated media spend of an 'asset' "
        "venice_spawn worker (default $2.00; <= 0 disables). A worker runs auto-approved, "
        "so this bounds its media blast radius in dollars; once reached, further paid "
        "media calls are refused and the worker wraps up. Config: defaults.code."
        "spawn_max_spend (#52).",
    )
    grp.add_argument(
        "--subagent-max-tokens", type=int, default=None, dest="subagent_max_tokens",
        metavar="N",
        help="Per-subagent cap on the cumulative prompt+completion tokens a venice_scout "
        "OR venice_spawn subagent spends across its turns (default off/uncapped; <= 0 "
        "disables). Once crossed the subagent is asked for a final answer and wraps up; "
        "its report carries the token count. This is a cumulative-usage ceiling, NOT a "
        "context-window size limit (re-sent history makes prompt tokens grow super-"
        "linearly), and is distinct from --max-tokens (per-turn output). Config: "
        "defaults.code.subagent_max_tokens (#52).",
    )
    grp.add_argument(
        "--review", action="store_true", default=None, dest="review",
        help="Expose venice_review: hand the current diff to a COLD-CONTEXT reviewer "
        "-- a disposable subagent with a FRESH context, only read-only tools, and (where "
        "the catalog allows) a different model, so it did not write the code it judges. "
        "It returns defects with file:line, severity and a repro; the agent fixes them "
        "before handing work back. FINDINGS ONLY: it cannot approve, certify, or record "
        "anything, and a review is not a merge gate (#80). Capped at "
        f"{_review.REVIEW_MAX_INVOCATIONS} reviews per session. Config: "
        "defaults.code.review.",
    )
    grp.add_argument(
        "--review-model", default=None, dest="review_model", metavar="MODEL",
        help="Model for --review. Default: a function-calling model from a DIFFERENT "
        "family than the coding model, so the reviewer's blind spots are not the "
        "author's; falls back to the coding model with a warning when the catalog "
        "offers no alternative. Config: defaults.code.review_model (#80).",
    )
    grp.add_argument(
        "--review-rounds", type=int, default=None, dest="review_rounds", metavar="N",
        help=f"Passes venice_review makes over the same diff (default "
        f"{_review.REVIEW_TOOL_ROUNDS}, max {_review.REVIEW_HARD_ROUNDS}); each is a "
        f"full model run. Config: defaults.code.review_rounds (#80).",
    )
    grp.add_argument(
        "--planner", action="store_true", default=None, dest="planner",
        help="Planner harness: implies --scout --spawn --memory, mandates the "
        "decompose -> task_add -> dispatch -> task_update -> merge protocol in the "
        "system prompt, records every scout/spawn dispatch, and exposes venice_merge "
        "-- a consolidated rollup of all dispatch reports, the task checklist, and "
        "structural warnings (merge is first-class, not prose). With --json the "
        "envelope carries the same rollup under 'planner'. Serial dispatch unless "
        "--parallel is also set. Config: defaults.code.planner (#52).",
    )
    grp.add_argument(
        "--parallel", action="store_true", default=None, dest="parallel",
        help="Dispatch INDEPENDENT scout/spawn subagents CONCURRENTLY (bounded pool) "
        "instead of one at a time -- so a planner's independent units overlap in "
        "wall-clock. Opt-in; serial otherwise. Only affects venice_scout/venice_spawn "
        "calls (everything else stays serial) and is inert without a subagent rail; "
        "best paired with --planner. Config: defaults.code.parallel (#52).",
    )
    grp.add_argument(
        "--web-search", action="store_true", default=None, dest="web_search",
        help="Expose venice_web_search: DISCOVER documentation on the web (a Venice "
        "web-search completion returning an answer + cited URLs). The planner and (with "
        "--scout) a read-only 'docs scout' can use it; spawn WORKERS cannot (injection "
        "blast radius). Billed; bounded by the tool-call budget. Config: "
        "defaults.code.web_search (#77).",
    )
    grp.add_argument(
        "--web-search-model", default=None, dest="web_search_model", metavar="MODEL",
        help="Model for --web-search (must advertise supportsWebSearch). Default: the "
        "coding --model if capable, else the first web-search-capable model in the "
        "catalog. Config: defaults.code.web_search_model (#77).",
    )
    grp.add_argument(
        "--auto-compact", action="store_true", default=None, dest="auto_compact",
        help="Summarize older history once it crosses the token budget, so long "
        "runs stay within the context window (#48; costs a summarization call).",
    )
    grp.add_argument(
        "--session-max-spend", type=_numeric.finite_float, default=None, metavar="USD",
        dest="session_max_spend",
        help="Cap total chat-completion spend for this run (#66): meters the "
        "model's calls from server token usage and stops starting new turns at "
        "the cap. Distinct from --max-spend (the per-call asset-tool cap).",
    )
    grp.add_argument(
        "--cache-guard", choices=_agent.CACHE_GUARD_CHOICES, default=None,
        dest="cache_guard",
        help="React when a cache-priced model explicitly reports 0 cached tokens "
        "after the cold first API call: off, warn (default), or stop and request "
        "a final answer. Config: defaults.code.cache_guard (#105).",
    )
    grp.add_argument(
        "--compact-threshold", type=int, default=None, dest="compact_threshold",
        metavar="TOKENS",
        help="Auto-compact once the prompt passes this many tokens "
        f"(default {_compact.DEFAULT_THRESHOLD_TOKENS}).",
    )
    grp.add_argument(
        "--compact-keep-turns", type=int, default=None, dest="compact_keep_turns",
        metavar="N",
        help="Turns kept verbatim when compacting "
        f"(default {_compact.DEFAULT_KEEP_TURNS}); older ones are summarized.",
    )

    it = p.add_argument_group("Interactive")
    it.add_argument(
        "--interactive", "-i", action="store_true", default=False,
        help="Interactive coding REPL (also entered with no task on a terminal). "
        "Tools are on; changes confirm per step unless --auto.",
    )
    it.add_argument(
        "--resume", default=None, metavar="ID|FILE",
        help="Resume a saved session by id (see `venice sessions ls`) or a "
        "transcript JSON file, interactively (#47).",
    )
    it.add_argument(
        "--continue", "-c", action="store_true", default=None, dest="cont",
        help="Resume the most recent code session (#47).",
    )
    it.add_argument(
        "--ephemeral", "--no-save", action="store_true", default=None,
        dest="ephemeral",
        help="Do not auto-save this session to ~/.config/venice/sessions/ (#47).",
    )
    p.set_defaults(handler=_run)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_task(args) -> Optional[str]:
    task = args.task
    if task == "-" or (task is None and not sys.stdin.isatty()):
        data = sys.stdin.read().strip()
        return data or None
    return task


def _gen_kwargs(args) -> dict:
    kw: dict = {}
    if args.temperature is not None:
        kw["temperature"] = args.temperature
    if args.max_tokens is not None:
        kw["max_tokens"] = args.max_tokens
    return kw


def _system_prompt(args, root: str, tools: List[_agent.Tool]) -> str:
    base = CODING_SYSTEM_PROMPT.format(root=root, tools=_code.tool_names(tools))
    if getattr(args, "planner", None):  # #52 planner slice: the harness protocol
        base += PLANNER_PROTOCOL
        if getattr(args, "parallel", None):  # #52: permit concurrent independent dispatch
            base += PLANNER_PARALLEL_OVERLAY
    if args.system:
        base += "\n\nProject-specific instructions:\n" + args.system
    return base


def _autonomous(args) -> bool:
    return bool(args.auto or args.yes)


#: The ``venice code`` profile (#51): the coding agent over the shared agent core --
#: always-on fs/exec/vcs tools (injected as a prebuilt session), the root-aware
#: coding system prompt (re-seeded on resume), a larger tool-call budget, the
#: plan/accept/verify harness, and a hard error (not a degrade) on a
#: non-function-calling model.
PROFILE = _agent.AgentProfile(
    name="code",
    label="venice code",
    build_gen_kwargs=_gen_kwargs,
    build_system=_system_prompt,
    default_max_tool_calls=_DEFAULT_MAX_TOOL_CALLS,
    plan_mode=True,
    degrade_to_chat=False,
    system_reseed=True,
    injects_tools_session=True,
)


def _finish(ledger, t0, human, *, json_out: bool) -> None:
    """Close the run's wall-clock window, once, and print the footer (#81).

    A one-shot `venice code` reported neither time nor cost when it finished. This
    is that report: one stderr line, on every exit that spent a model call --
    including the Ctrl+C and API-error paths, which are exactly the runs whose cost
    you most want to see and the ones a happy-path-only footer would hide.

    stderr, not stdout: the final answer is the deliverable and `venice code | ...`
    must stay clean. Not TTY-gated either -- this is a single line in the same class
    as "code: aborted" and the acceptance report, not a redraw surface like
    `_Spinner`, so gating it would only hide the number from logs and pipelines.

    Idempotent via `ledger.turns`: the one-shot ledger is built fresh (in `_run` as of
    #117, and handed down) and never `restore()`d on this path, so a non-zero turn count
    means "already stamped". That coupling is load-bearing -- if a future `--resume` ever
    seeds this ledger from a prior session's usage, this guard silently stops stamping
    and needs replacing with an explicit flag. #117's hoist deliberately did NOT add a
    restore here; only the REPL restores, on its own path.

    Subagent rails do not disturb this: a mirrored child banks into `buckets` via
    `record_bucket`, which never touches `turns`.
    """
    if ledger is None or ledger.turns:
        return
    ledger.record_turn(time.monotonic() - t0 - human[0])
    if not json_out:
        # cache=True (#100): a prompt-cache collapse is a silent 3-5x cost event, and
        # this footer is the only surface a one-shot run puts in front of an operator.
        # tools_fragment (#82) sits INSIDE the wall field, because " -- " is the
        # top-level field boundary the README documents. It reads `elapsed_seconds` to
        # decide whether to say "concurrent", which is why it must render AFTER the
        # `record_turn` above -- moving the stamp below this print would make the
        # marker fire on every run.
        print(f"code: {_agent.format_duration(ledger.elapsed_seconds)} wall"
              f"{ledger.tools_fragment()} -- {ledger.summary(cache=True)}",
              file=sys.stderr)


@contextlib.contextmanager
def _human_pause(acc):
    """Bank the seconds a prompt spent waiting on the operator (#81).

    `acc` is a one-element list used as a mutable accumulator (the run's timing lives
    in locals, not an object). Banked on the way out even if the block raises -- an
    EOFError at the edit prompt is still time you were not waiting on the CLI.
    """
    t = time.monotonic()
    try:
        yield
    finally:
        acc[0] += time.monotonic() - t


def _no_tool_turn(oai, model, messages, gen_kwargs, oai_tools, *, ledger=None) -> dict:
    """One completion with tools advertised but ``tool_choice="none"`` (no side
    effects) -- used for the plan turn and the acceptance-check turn.

    #81: these run OUTSIDE `run_loop`, so nothing else records their usage. They
    carry the whole transcript as prompt, which makes them among the largest calls
    in a run -- a reported cost that skipped them would understate it badly.
    """
    _t0 = time.monotonic()
    resp = oai.chat.completions.create(
        model=model, messages=messages, tools=oai_tools, tool_choice="none",
        **gen_kwargs,
    )
    if ledger is not None:
        # #99: the highest-value bracket in the change. These turns carry the whole
        # transcript (see above), so they are the largest rows in the trace -- a trace
        # whose biggest calls read `n/a` would be worse than no trace at all.
        ledger.record(getattr(resp, "usage", None), seconds=time.monotonic() - _t0)
    msg = resp.choices[0].message if getattr(resp, "choices", None) else None
    return _agent._assistant_dict(msg)


# Promoted to `_agent` (#52): the scout subagent firewalls its stdout the same way,
# and `_agent` must not import `code`. Kept here as an alias so `_run_oneshot`'s
# `--json` capture (and any other callers) keep working unchanged.
_capture_stdout = _agent._capture_stdout


@contextlib.contextmanager
def _code_session(tools):
    """A trivial tools-session for `_repl.run` (no external servers to hold open)."""
    yield tools, None


def _decide_mode(args) -> str:
    """Resolve the run mode from flags/TTY without prompting.

    Returns one of: ``manual`` / ``auto`` / ``prompt`` (ask on a TTY) /
    ``abort_usage`` (non-TTY with no mode flag -> fail safe).
    """
    if args.manual:  # explicit --manual wins over any config-filled auto default
        return "manual"
    if args.auto or args.yes:
        return "auto"
    if sys.stdin.isatty():
        return "prompt"
    return "abort_usage"


def _prompt_accept(*, no_plan: bool = False) -> str:
    opts = "[a]uto / [s]tep / [N]o" if no_plan else "[a]uto / [s]tep / [e]dit / [N]o"
    while True:
        try:
            ans = input(f"Accept and run? {opts}: ").strip().lower()
        except EOFError:
            return "abort"
        if ans in ("a", "auto"):
            return "auto"
        if ans in ("s", "step", "m", "manual"):
            return "manual"
        if ans in ("e", "edit") and not no_plan:
            return "edit"
        if ans in ("", "n", "no"):
            return "abort"
        print("Please answer a, s, e, or n.", file=sys.stderr)


def _model_provenance(args, dest: str) -> dict:
    """Describe whether an auxiliary model came from a flag, config, or auto-pick."""
    sources = getattr(args, "_config_sources", None)
    config_key = sources.get(dest) if isinstance(sources, dict) else None
    if config_key:
        return {"source": "config", "config_key": config_key}
    if getattr(args, dest, None) is not None:
        return {"source": "flag"}
    return {"source": "auto"}


def _config_model_recovery(provenance: dict) -> None:
    """Point a config-sourced auxiliary-model failure at its durable fix."""
    key = provenance.get("config_key")
    if key:
        print(
            f"code: that model came from {key}; update it or run: "
            f"venice config unset {key}",
            file=sys.stderr,
        )


def _record_auxiliary_model(args, dest: str, role: str, flag: str,
                            model: str) -> dict:
    """Build the public selection row and announce it before any paid call."""
    row = {"id": model, **_model_provenance(args, dest)}
    if row["source"] == "config":
        source = f"config {row['config_key']}"
    elif row["source"] == "flag":
        source = f"flag {flag}"
    else:
        source = "auto"
    print(f"code: {role} model: {model} (source: {source})", file=sys.stderr)
    return row


def _resolve_auxiliary_models(args, models, author_model: str):
    """Resolve enabled review/web-search models before a paid completion (#103).

    Returns ``(state, rc)``. ``state['resolved_models']`` is the stable public
    provenance map; the other keys are the already-validated values consumed by
    the tool factories. Disabled rails are intentionally ignored, including stale
    defaults for their otherwise-unused model flags.
    """
    state = {
        "resolved_models": {},
        "review_model": None,
        "review_decorrelated": False,
        "web_search_model": None,
    }

    if bool(getattr(args, "review", None)):
        provenance = _model_provenance(args, "review_model")
        picked, decorrelated = _review.resolve_reviewer_model(
            models, getattr(args, "review_model", None), author_model,
        )
        review_model, rc = _models.resolve_model(
            picked, models, label="code", noun="review model",
        )
        if rc is not None:
            _config_model_recovery(provenance)
            return None, rc
        ok, rc = _agent.check_function_calling(
            models, review_model, label="code",
            degraded_tail=(
                "venice code --review needs a tool-calling model "
                "(pass --review-model)."
            ),
            unverified_tail="attempting the review anyway",
            degrade=False,
        )
        if not ok:
            _config_model_recovery(provenance)
            return None, rc
        state["review_model"] = review_model
        state["review_decorrelated"] = decorrelated
        state["resolved_models"]["review"] = _record_auxiliary_model(
            args, "review_model", "review", "--review-model", review_model,
        )

    if bool(getattr(args, "web_search", None)):
        provenance = _model_provenance(args, "web_search_model")
        picked = _agent.resolve_web_search_model(
            models, getattr(args, "web_search_model", None), author_model,
        )
        if not picked:
            print(
                "code: no web-search-capable model available; pass "
                "--web-search-model (or set defaults.code.web_search_model)",
                file=sys.stderr,
            )
            _config_model_recovery(provenance)
            return None, 2
        web_model, rc = _models.resolve_model(
            picked, models, label="code", noun="web-search model",
        )
        if rc is not None:
            _config_model_recovery(provenance)
            return None, rc
        if _agent.supports_web_search(models, web_model) is False:
            print(
                f"code: web-search model {web_model!r} does not advertise "
                "supportsWebSearch",
                file=sys.stderr,
            )
            _config_model_recovery(provenance)
            return None, 2
        state["web_search_model"] = web_model
        state["resolved_models"]["web_search"] = _record_auxiliary_model(
            args, "web_search_model", "web-search", "--web-search-model", web_model,
        )

    return state, None


def _emit_plan_only(args, root, task, plan_text, *, usage=None,
                    resolved_models=None) -> int:
    if args.json:
        doc = {
            "root": root, "task": task, "plan": plan_text, "mode": "plan_only",
            "resolved_models": dict(resolved_models or {}),
        }
        if usage is not None:
            doc["usage"] = usage  # #81: a plan turn is a real call and a real wait
        json.dump(doc, sys.stdout, indent=2, default=str, allow_nan=False)
        sys.stdout.write("\n")
    else:
        print(plan_text)  # the plan is the deliverable -> stdout
    return 0


def _show_plan(plan_text) -> None:
    print("\n=== Proposed plan ===", file=sys.stderr)
    print(plan_text, file=sys.stderr)
    print("=====================", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _run(args) -> int:
    # Resolve a resumed session (#47) BEFORE apply_defaults so restored settings
    # outrank config defaults (both fill None dests; the session runs first).
    try:
        session = _session.resolve_from_args(args, "code")
    except _session.SessionError as e:
        print(f"code: {e}", file=sys.stderr)
        return 2
    _session.apply_to_args(args, session, "code")
    userconfig.apply_defaults(args, "code")
    userconfig.apply_literals(args, cache_guard="warn")
    if getattr(args, "browser", None):
        print(f"code: {_browser.UNAVAILABLE_MESSAGE}", file=sys.stderr)
        return 2

    # Faithful root restore: an explicit --root/$VENICE_CODE_ROOT still wins, else a
    # resumed session re-sandboxes to where it left off (tools + system prompt rebind
    # to this root below), else the cwd.
    root = os.path.realpath(
        args.root or os.environ.get(config.ENV_CODE_ROOT)
        or (session.root if session else None) or os.getcwd()
    )
    if not os.path.isdir(root):
        print(f"code: not a directory: {root}", file=sys.stderr)
        return 2

    task = _resolve_task(args)
    interactive = _agent.wants_interactive(args, task)
    if not interactive and not task:
        print("code: no task (pass an argument or pipe stdin)", file=sys.stderr)
        return 2

    openai = _openai.import_openai("code")
    if openai is None:
        return 2
    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    models = _models.catalog(client, "text")
    model, rc = _models.resolve_model(
        args.model, models, label="code", noun="text model",
        config_key="defaults.code.model",
    )
    if rc is not None:
        return rc

    ok, rc = _agent.check_function_calling(
        models, model, label=PROFILE.name,
        degraded_tail="venice code needs a tool-calling model (pass --model).",
        unverified_tail="attempting anyway",
        degrade=PROFILE.degrade_to_chat,
    )
    if not ok:
        return rc  # degrade_to_chat is False for code -> rc == 2

    auxiliary, rc = _resolve_auxiliary_models(args, models, model)
    if rc is not None:
        return rc
    resolved_models = auxiliary["resolved_models"]
    review_model = auxiliary["review_model"]
    decorrelated = auxiliary["review_decorrelated"]
    web_search_model = auxiliary["web_search_model"]

    oai = _openai.build_openai(openai, client)
    doc = userconfig.load_config()  # #58 tool defaults + #33 shell policy
    pol = userconfig.shell_policy(doc)
    shell_allow = list(pol["allow"]) + list(getattr(args, "shell_allow", None) or [])
    shell_deny = list(pol["deny"]) + list(getattr(args, "shell_deny", None) or [])
    bpol = userconfig.browser_policy(doc)  # #71 retained compatibility config
    browser_allow = list(bpol["allow"]) + list(getattr(args, "browser_allow", None) or [])
    browser_deny = list(bpol["deny"]) + list(getattr(args, "browser_deny", None) or [])
    rpol = userconfig.roots_policy(doc)  # #76 extra writable / read-only roots
    allow_root = list(rpol["allow"]) + list(getattr(args, "allow_root", None) or [])
    deny_root = list(rpol["deny"]) + list(getattr(args, "deny_root", None) or [])
    # #52 planner slice: --planner implies the three rails it orchestrates (there are
    # no --no-scout/--no-spawn/--no-memory flags, so nothing can conflict -- like
    # --assets). Must precede code_tools (reads memory).
    planner = bool(getattr(args, "planner", None))
    if planner:
        args.scout = args.spawn = args.memory = True
    tools = _code.code_tools(
        root, client,
        exec_timeout=args.exec_timeout or _code.DEFAULT_EXEC_TIMEOUT,
        include_search=True,
        assets=bool(args.assets),
        config=doc,  # #58: honor defaults.<cmd>.* in tools
        shell_allow=shell_allow,  # #33: `run` honors the shared allow/deny policy
        shell_deny=shell_deny,
        allow_root=allow_root,  # #76: extra writable roots
        deny_root=deny_root,
        browser=bool(getattr(args, "browser", None)),  # #71
        browser_allow=browser_allow,
        browser_deny=browser_deny,
        memory=bool(getattr(args, "memory", None)),  # #49
    )
    # gen_kwargs is built BEFORE the scout tool (its nested loop needs these per-turn
    # kwargs) and BEFORE build_system (so the coding prompt's tool list can advertise
    # venice_scout). `_gen_kwargs` reads only args.temperature/max_tokens -- no
    # dependency on `tools`, so the reorder is safe.
    gen_kwargs = PROFILE.build_gen_kwargs(args)
    # #128: Venice can route requests carrying one stable prompt_cache_key to the
    # same cache-bearing backend. Restore the saved key when one exists; old sessions
    # receive a key on first resume, and fresh/ephemeral runs mint one before the plan
    # turn so plan -> execute -> verify all share affinity. It rides in extra_body for
    # compatibility with the project's openai>=1.40 floor and is persisted automatically
    # with the session's generation kwargs.
    saved_cache_key = (
        _openai.prompt_cache_key(session.gen_kwargs) if session is not None else None
    )
    gen_kwargs = _openai.with_prompt_cache_key(gen_kwargs, saved_cache_key)
    # #52 planner slice: the session's shared dispatch record list. scout/spawn append
    # every launched dispatch to it; venice_merge (and the --json envelope) roll it up.
    dispatches = [] if planner else None
    # #117: the session ledger, HOISTED above the factory block below because every rail
    # needs it at factory time -- a subagent mirrors its usage into this object's off-loop
    # buckets, and the tools are built long before `_run_oneshot`/`_repl.run` would have
    # constructed one. It reads only `args.session_max_spend`, `models` and `model`, all
    # resolved well above here, so the hoist inverts no dependency.
    #
    # There must be exactly ONE of these per run: both entry points below ADOPT this
    # object rather than building their own, and a second ledger would silently orphan
    # every rail's spend with no error and no failing assertion. `_finish`'s
    # already-stamped guard also assumes this ledger reaches `_run_oneshot` un-restored
    # (see its docstring) -- true here, since only the REPL restores, on its own path.
    ledger = _agent.usage_ledger(args, models, model)
    # #77: opt-in web-discovery rail. Built once (root-independent) and shared between the
    # parent tool list and the scout's read-only inner set (a "docs scout"); workers never
    # get it (category "web" is in no spawn role). `models` is in scope from the guard above.
    ws_tool = None
    if bool(getattr(args, "web_search", None)):
        ws_tool = _code.web_search_tool(
            oai, model, models=models,
            search_model=web_search_model,
            parent_ledger=ledger,  # #117: bills the `web_search` bucket
        )
    # #52: per-subagent cumulative-token ceiling (None = uncapped); applies to BOTH the
    # read-only scout and the write/paid worker -- token burn is universal to both.
    subagent_max_tokens = getattr(args, "subagent_max_tokens", None)
    if bool(getattr(args, "scout", None)):  # #52: opt-in read-only scout subagent
        tools.append(_code.scout_tool(oai, model, root, client, gen_kwargs,
                                      include_search=True, web_tool=ws_tool,
                                      max_tokens=subagent_max_tokens,
                                      dispatches=dispatches,
                                      models=models, parent_ledger=ledger))  # #117
    if bool(getattr(args, "spawn", None)):  # #52 slice 2: write-capable worker subagent
        # Passes the live `tools` list: the worker draws a role-scoped subset of these
        # (the agent category -- scout/spawn -- is filtered out, so no nested subagents).
        # `spawn_max_spend` caps an 'asset' worker's cumulative media USD (#52 spend slice).
        tools.append(_code.spawn_tool(oai, model, gen_kwargs, tools,
                                      max_spend=getattr(args, "spawn_max_spend", None),
                                      max_tokens=subagent_max_tokens,
                                      dispatches=dispatches,
                                      models=models, parent_ledger=ledger))  # #117
    if bool(getattr(args, "review", None)):  # #80 part 1a: cold-context reviewer rail
        # The reviewer's model is resolved ONCE here and never advertised in the tool
        # schema (mirroring web_search_tool), so the agent cannot escalate itself onto
        # a costlier model. A reviewer from a different family than the author is the
        # point of the rail -- same-family is allowed but warned about, because
        # refusing would make --review useless on a single-model catalog.
        if not decorrelated:
            print(f"code: --review will use {review_model}, the same model family "
                  "that is authoring -- blind spots are correlated. Pass "
                  "--review-model to decorrelate.", file=sys.stderr)
        tools.append(_review.review_tool(
            oai, review_model, root, client, gen_kwargs, include_search=True,
            default_rounds=getattr(args, "review_rounds", None)
            or _review.REVIEW_TOOL_ROUNDS,
            max_tokens=subagent_max_tokens,
            exec_timeout=args.exec_timeout or _code.DEFAULT_EXEC_TIMEOUT,
            decorrelated=decorrelated,
            # #117: `review_model`, not `model` -- the reviewer is deliberately a
            # different (often costlier) model, and its bucket must be priced as such.
            models=models, parent_ledger=ledger,
        ))
    if ws_tool is not None:  # #77: parent (planner included) gets web discovery directly
        tools.append(ws_tool)
    if planner:
        tools.append(_code.merge_tool(dispatches))
    system = PROFILE.build_system(args, root, tools)

    roots_note = ""  # #76: surface extra writable / read-only roots in the banner
    if allow_root:
        roots_note += f" -- also writable: {', '.join(allow_root)}"
    if deny_root:
        roots_note += f" -- read-only: {', '.join(deny_root)}"
    if interactive:
        args.system = system
        args.yes = _autonomous(args)  # drive the REPL's per-turn gate
        print(
            f"code: sandboxed to {root}{roots_note} -- tools: {_code.tool_names(tools)}",
            file=sys.stderr,
        )
        return _repl.run(
            args, oai, openai, client, models, model, initial=task,
            tools_session=_code_session(tools), gen_kwargs=gen_kwargs,
            label=PROFILE.label, max_tool_calls=PROFILE.default_max_tool_calls,
            session=session, ephemeral=bool(getattr(args, "ephemeral", None)),
            root=root, system_reseed=PROFILE.system_reseed,
            ledger=ledger,  # #117: the rails already hold this one
            resolved_models=resolved_models,
        )

    return _run_oneshot(args, oai, openai, model, tools, system, gen_kwargs, root, task,
                        models, dispatches=dispatches,
                        ephemeral=bool(getattr(args, "ephemeral", None)),
                        ledger=ledger, resolved_models=resolved_models)  # #117


def _run_oneshot(args, oai, openai, model, tools, system, gen_kwargs, root, task,
                 models=None, *, dispatches=None, ephemeral=False, ledger=None,
                 resolved_models=None) -> int:
    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    oai_tools = _agent.to_openai_tools(tools)
    plan_text = None
    mode = None
    # #75/#81: an ALWAYS-ON ledger, not the cap-gated `ledger_from_args`, which returns
    # None unless --session-max-spend is set. A default `venice code` run therefore
    # metered nothing at all: no cost to report and an empty `usage` blob in every
    # session file it wrote. `--session-max-spend` still supplies the cap when given;
    # uncapped, this meters without gating (`over()`/`over_tokens()` are both None-safe).
    # Hoisted above the plan block so the plan turn -- a real API call, and a real wait --
    # is inside the accounting rather than outside it.
    #
    # #117: `_run` hoisted it further still (the rails need it at factory time) and passes
    # it in. Building a second one here would leave every rail mirroring into an orphan.
    # The fallback keeps this function callable on its own, as the tests do.
    if ledger is None:
        ledger = _agent.usage_ledger(args, models, model)
    t0 = time.monotonic()   # #81: the whole run is the window; see `_finish`
    human = [0.0]           # seconds spent waiting on *you* at a prompt, excluded

    if not args.no_plan:
        # Decide the run mode from flags/TTY up front so the non-TTY fail-safe
        # aborts *before* spending a plan turn (unless --plan-only, which only
        # prints a read-only plan and is safe unattended).
        mode_decision = _decide_mode(args)
        if mode_decision == "abort_usage" and not args.plan_only:
            print(
                "code: refusing to run unattended without --auto "
                "(or use --plan-only to just print the plan)",
                file=sys.stderr,
            )
            return 2
        while True:
            # Keep the request that produced the plan in the durable transcript.  A
            # temporary list here makes the assistant plan appear without the user
            # instruction it answered and breaks exact-prefix reuse on every later
            # request (including re-plans).
            messages.append({"role": "user", "content": _PLAN_INSTRUCTION})
            try:
                # Inside the re-plan loop: each `edit` revision buys another plan turn,
                # so recording per call (not once) is what makes the total honest.
                plan_message = _no_tool_turn(oai, model, messages, gen_kwargs,
                                             oai_tools, ledger=ledger)
                plan_text = plan_message.get("content") or ""
            except openai.OpenAIError as e:
                _finish(ledger, t0, human, json_out=args.json)
                return _openai.status_to_exit(openai, e, "code")
            messages.append(plan_message)

            if args.plan_only:
                _finish(ledger, t0, human, json_out=args.json)
                return _emit_plan_only(args, root, task, plan_text,
                                       usage=ledger.to_dict(),
                                       resolved_models=resolved_models)
            if not args.json:
                _show_plan(plan_text)

            decision = mode_decision
            if decision == "prompt":
                # #81: the plan gate and the edit prompt are the operator thinking, not
                # the CLI working. The ticket's metric is explicitly "how long it kept
                # me waiting", so bank these pauses and subtract them in `_finish`.
                with _human_pause(human):
                    decision = _prompt_accept()
            if decision == "edit":
                try:
                    with _human_pause(human):
                        fb = input("Describe the change to the plan (blank to cancel): ").strip()
                except EOFError:
                    fb = ""
                if not fb:
                    print("code: aborted", file=sys.stderr)
                    _finish(ledger, t0, human, json_out=args.json)
                    return 1
                messages.append({"role": "user", "content": "Revise the plan: " + fb})
                continue
            if decision == "abort":
                print("code: plan not accepted; aborting", file=sys.stderr)
                _finish(ledger, t0, human, json_out=args.json)
                return 1
            mode = decision
            break
    else:
        if args.plan_only:
            print("code: --plan-only and --no-plan are mutually exclusive",
                  file=sys.stderr)
            return 2
        decision = _decide_mode(args)
        if decision == "prompt":
            with _human_pause(human):   # #81: your thinking, not the CLI's working
                decision = _prompt_accept(no_plan=True)
        if decision == "abort":
            print("code: aborted", file=sys.stderr)
            return 1
        if decision == "abort_usage":
            print("code: refusing to run unattended without --auto", file=sys.stderr)
            return 2
        mode = decision

    # --- Execute ---
    messages.append({"role": "user", "content": _PROCEED_MSG})
    yes = mode == "auto"
    max_calls = (
        args.max_tool_calls if args.max_tool_calls is not None
        else PROFILE.default_max_tool_calls
    )
    # Mid-run steering (#78): persist this run as a session NOW -- before the loop --
    # so `sessions send <id|latest>` can target it while it runs, and so it's
    # resumable/inspectable afterwards. A fresh session is always minted (never the
    # resumed one, whose transcript we must not clobber); --ephemeral opts out and
    # leaves the run unsteerable, matching the REPL's persist-unless-ephemeral rule.
    active = None
    if not ephemeral:
        active = _session.new_session(
            "code", label=PROFILE.label, model=model, system=system,
            gen_kwargs=gen_kwargs, root=root, max_tool_calls=max_calls,
            messages=messages, resolved_models=resolved_models,
        )
        active.messages = messages  # share the live list so saves capture the transcript
        try:
            _session.save(active)   # create the file so `latest` resolves during the run
        except OSError as e:
            print(f"code: session save failed ({e}); run will not be steerable",
                  file=sys.stderr)
            active = None
    final_text = None
    budget = _budget_for(args)
    # #79: attached Ctrl+C steering. On an interactive tty, wrap the loop so the first
    # Ctrl+C pauses at the next checkpoint and prompts for a steering line (fed through
    # the #78 drain path); a second Ctrl+C aborts. Off a tty or in --json this yields the
    # plain #78 mailbox drain and installs no handler, so detached steering is unchanged.
    sid = active.id if active is not None else None
    steer_enabled = sys.stdin.isatty() and not args.json
    parallel = bool(getattr(args, "parallel", None))  # #52: concurrent subagent dispatch
    try:
        with _steer.pause_and_steer(sid, enabled=steer_enabled) as steer_drain:
            if args.json:
                with _capture_stdout() as buf:
                    _agent.run_loop(oai, model, messages, gen_kwargs, tools,
                                    max_tool_calls=max_calls, yes=yes, json_out=False,
                                    budget=budget, ledger=ledger, steer_drain=steer_drain,
                                    parallel=parallel)
                final_text = buf.getvalue().strip()
            else:
                _agent.run_loop(oai, model, messages, gen_kwargs, tools,
                                max_tool_calls=max_calls, yes=yes, json_out=False,
                                budget=budget, ledger=ledger, steer_drain=steer_drain,
                                parallel=parallel)
    except openai.OpenAIError as e:
        _finish(ledger, t0, human, json_out=args.json)
        return _openai.status_to_exit(openai, e, "code")
    except KeyboardInterrupt:
        # #79: abort (a 2nd Ctrl+C, or Ctrl+C at the steer prompt). Save the partial
        # transcript so the run stays inspectable/resumable -- side effects already ran --
        # then exit 130 (the documented Ctrl-C code) instead of an uncaught traceback.
        print("\ncode: aborted", file=sys.stderr)
        # Stamp BEFORE the snapshot below, or the persisted usage carries 0 seconds
        # for a run the operator just sat through.
        _finish(ledger, t0, human, json_out=args.json)
        if active is not None:
            if ledger is not None:
                try:
                    active.usage = ledger.to_dict()
                except Exception:
                    pass
            try:
                _session.save(active)
            except OSError:
                pass
        return 130

    # --- Acceptance check ---
    verdict = None          # None = skipped; else 'pass' | 'fail' | 'unknown'
    report = None
    if not args.no_verify and not args.no_plan:
        messages.append({"role": "user", "content": _VERIFY_MSG})
        try:
            report_message = _no_tool_turn(oai, model, messages, gen_kwargs, oai_tools,
                                           ledger=ledger)
            report = report_message.get("content") or ""
            parsed = _parse_verdict(report)
            if parsed is None:      # re-prompt ONCE for the exact verdict line
                messages.append(report_message)
                messages.append({"role": "user", "content": _VERIFY_RETRY_MSG})
                retry_message = _no_tool_turn(
                    oai, model, messages, gen_kwargs, oai_tools, ledger=ledger,
                )
                retry = retry_message.get("content") or ""
                report = f"{report}\n{retry}" if report else retry
                parsed = _parse_verdict(retry)
        except openai.OpenAIError as e:
            _finish(ledger, t0, human, json_out=args.json)
            return _openai.status_to_exit(openai, e, "code")
        verdict = parsed or "unknown"
        if not args.json:
            print("\n=== Acceptance check ===", file=sys.stderr)
            print(report, file=sys.stderr)
        if verdict == "unknown":
            print("code: could not parse an ACCEPTANCE verdict from the model "
                  "(work may be complete) -- exiting 10", file=sys.stderr)

    # Mid-run steering (#78): a steer that landed after the loop exited (a final
    # turn with no tool calls, or a cap-forced final) was never drained. v1 does not
    # re-loop -- it surfaces the leftovers so they aren't silently lost, then persists
    # the finished session so it's inspectable/resumable.
    unprocessed = _mailbox.drain(active.id) if active is not None else []
    if unprocessed and not args.json:
        print(f"\ncode: {len(unprocessed)} steering message(s) arrived after the run "
              "finished and were not processed:", file=sys.stderr)
        for _u in unprocessed:
            print(f"  - {_u.splitlines()[0][:200] if _u.strip() else '(empty)'}",
                  file=sys.stderr)
    # #81: stamp before the snapshot and before the envelope, so the persisted usage
    # and the --json surface both carry the run's wall-clock.
    _finish(ledger, t0, human, json_out=args.json)
    if active is not None:
        if ledger is not None:
            try:
                active.usage = ledger.to_dict()
            except Exception:
                pass
        try:
            _session.save(active)  # final transcript (+ usage) for `sessions`/`--resume`
        except OSError:
            pass

    if args.json:
        envelope = {
            "root": root, "task": task, "plan": plan_text, "mode": mode,
            "final": final_text,
            "resolved_models": dict(resolved_models or {}),
            # #81: `to_dict()` verbatim, so `venice code --json | jq .usage` and
            # `jq .usage <session>.json` are the same shape and cannot drift apart.
            # Numbers here, `format_duration` only on the human line. FOUR surfaces
            # share that shape, not two -- `--plan-only --json` via `_emit_plan_only`
            # and the Ctrl+C session save are the other pair, so a key added to
            # `to_dict()` (e.g. #100's `cache_hit_percent`) appears on all four.
            "usage": ledger.to_dict(),
        }
        if report is not None:
            envelope["acceptance"] = {
                "verdict": verdict,                                    # pass|fail|unknown
                "passed": {"pass": True, "fail": False}.get(verdict),  # None when unknown
                "report": report,
            }
        if dispatches is not None:  # #52 planner slice: the rollup, structurally --
            # callers get it even if the model skipped the venice_merge call.
            envelope["planner"] = _code.merge_summary(dispatches)
        if unprocessed:  # #78: steers that arrived post-run (not fed to the model)
            envelope["unprocessed_steering"] = unprocessed
        json.dump(envelope, sys.stdout, indent=2, default=str, allow_nan=False)
        sys.stdout.write("\n")

    return {None: 0, "pass": 0, "fail": 1, "unknown": 10}[verdict]
