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
from typing import List, Optional

from .. import auth, config, userconfig
from ..client import build_client_from_auth
from . import _agent, _code, _compact, _mailbox, _models, _openai, _repl, _session

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
    p.add_argument("--temperature", "-t", type=float, default=None)
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
        help="Expose web_fetch + browser_capture tools so the agent can fetch a URL "
        "and headless-render a page (screenshot / post-JS DOM) to verify its own work. "
        "http/https only; the cloud metadata endpoint is always blocked; scope hosts "
        "with --browser-allow/--browser-deny or the config `browser` section (#71).",
    )
    grp.add_argument(
        "--browser-allow", action="append", dest="browser_allow", default=None,
        metavar="HOST",
        help="Allow only these hosts for the browser tools (repeatable; globs ok, "
        "matched on the URL host). Adds to the config browser.allow list.",
    )
    grp.add_argument(
        "--browser-deny", action="append", dest="browser_deny", default=None,
        metavar="PATTERN",
        help="Refuse URLs whose host or full URL matches these globs (repeatable, "
        "always enforced, wins over --browser-allow). Adds to config browser.deny.",
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
        "--spawn-max-spend", type=float, default=None, dest="spawn_max_spend",
        metavar="USD",
        help="Per-worker USD cap on the cumulative estimated media spend of an 'asset' "
        "venice_spawn worker (default $2.00; <= 0 disables). A worker runs auto-approved, "
        "so this bounds its media blast radius in dollars; once reached, further paid "
        "media calls are refused and the worker wraps up. Config: defaults.code."
        "spawn_max_spend (#52).",
    )
    grp.add_argument(
        "--planner", action="store_true", default=None, dest="planner",
        help="Planner harness: implies --scout --spawn --memory, mandates the "
        "decompose -> task_add -> dispatch -> task_update -> merge protocol in the "
        "system prompt, records every scout/spawn dispatch, and exposes venice_merge "
        "-- a consolidated rollup of all dispatch reports, the task checklist, and "
        "structural warnings (merge is first-class, not prose). With --json the "
        "envelope carries the same rollup under 'planner'. Serial dispatch only. "
        "Config: defaults.code.planner (#52).",
    )
    grp.add_argument(
        "--web-search", action="store_true", default=None, dest="web_search",
        help="Expose venice_web_search: DISCOVER documentation on the web (a Venice "
        "web-search completion returning an answer + cited URLs). Pairs with --browser "
        "to then fetch a cited page under the browser.* URL policy. The planner and (with "
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
        "--session-max-spend", type=float, default=None, metavar="USD",
        dest="session_max_spend",
        help="Cap total chat-completion spend for this run (#66): meters the "
        "model's calls from server token usage and stops starting new turns at "
        "the cap. Distinct from --max-spend (the per-call asset-tool cap).",
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


def _no_tool_turn(oai, model, messages, gen_kwargs, oai_tools) -> str:
    """One completion with tools advertised but ``tool_choice="none"`` (no side
    effects) -- used for the plan turn and the acceptance-check turn."""
    resp = oai.chat.completions.create(
        model=model, messages=messages, tools=oai_tools, tool_choice="none",
        **gen_kwargs,
    )
    if getattr(resp, "choices", None):
        return resp.choices[0].message.content or ""
    return ""


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


def _emit_plan_only(args, root, task, plan_text) -> int:
    if args.json:
        json.dump(
            {"root": root, "task": task, "plan": plan_text, "mode": "plan_only"},
            sys.stdout, indent=2, default=str,
        )
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
        args.model, models, label="code", noun="text model"
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

    oai = _openai.build_openai(openai, client)
    doc = userconfig.load_config()  # #58 tool defaults + #33 shell policy
    pol = userconfig.shell_policy(doc)
    shell_allow = list(pol["allow"]) + list(getattr(args, "shell_allow", None) or [])
    shell_deny = list(pol["deny"]) + list(getattr(args, "shell_deny", None) or [])
    bpol = userconfig.browser_policy(doc)  # #71 URL allow/deny policy
    browser_allow = list(bpol["allow"]) + list(getattr(args, "browser_allow", None) or [])
    browser_deny = list(bpol["deny"]) + list(getattr(args, "browser_deny", None) or [])
    rpol = userconfig.roots_policy(doc)  # #76 extra writable / read-only roots
    allow_root = list(rpol["allow"]) + list(getattr(args, "allow_root", None) or [])
    deny_root = list(rpol["deny"]) + list(getattr(args, "deny_root", None) or [])
    # #52 planner slice: --planner implies the three rails it orchestrates (there are
    # no --no-scout/--no-spawn/--no-memory flags, so nothing can conflict -- the same
    # one-flag bundling as --browser/--assets). Must precede code_tools (reads memory).
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
    # #52 planner slice: the session's shared dispatch record list. scout/spawn append
    # every launched dispatch to it; venice_merge (and the --json envelope) roll it up.
    dispatches = [] if planner else None
    # #77: opt-in web-discovery rail. Built once (root-independent) and shared between the
    # parent tool list and the scout's read-only inner set (a "docs scout"); workers never
    # get it (category "web" is in no spawn role). `models` is in scope from the guard above.
    ws_tool = None
    if bool(getattr(args, "web_search", None)):
        ws_tool = _code.web_search_tool(
            oai, model, models=models,
            search_model=getattr(args, "web_search_model", None),
        )
    if bool(getattr(args, "scout", None)):  # #52: opt-in read-only scout subagent
        tools.append(_code.scout_tool(oai, model, root, client, gen_kwargs,
                                      include_search=True, web_tool=ws_tool,
                                      dispatches=dispatches))
    if bool(getattr(args, "spawn", None)):  # #52 slice 2: write-capable worker subagent
        # Passes the live `tools` list: the worker draws a role-scoped subset of these
        # (the agent category -- scout/spawn -- is filtered out, so no nested subagents).
        # `spawn_max_spend` caps an 'asset' worker's cumulative media USD (#52 spend slice).
        tools.append(_code.spawn_tool(oai, model, gen_kwargs, tools,
                                      max_spend=getattr(args, "spawn_max_spend", None),
                                      dispatches=dispatches))
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
        )

    return _run_oneshot(args, oai, openai, model, tools, system, gen_kwargs, root, task,
                        models, dispatches=dispatches,
                        ephemeral=bool(getattr(args, "ephemeral", None)))


def _run_oneshot(args, oai, openai, model, tools, system, gen_kwargs, root, task,
                 models=None, *, dispatches=None, ephemeral=False) -> int:
    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    oai_tools = _agent.to_openai_tools(tools)
    plan_text = None
    mode = None

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
            plan_messages = messages + [{"role": "user", "content": _PLAN_INSTRUCTION}]
            try:
                plan_text = _no_tool_turn(oai, model, plan_messages, gen_kwargs, oai_tools)
            except openai.OpenAIError as e:
                return _openai.status_to_exit(openai, e, "code")
            messages.append({"role": "assistant", "content": plan_text})

            if args.plan_only:
                return _emit_plan_only(args, root, task, plan_text)
            if not args.json:
                _show_plan(plan_text)

            decision = mode_decision
            if decision == "prompt":
                decision = _prompt_accept()
            if decision == "edit":
                try:
                    fb = input("Describe the change to the plan (blank to cancel): ").strip()
                except EOFError:
                    fb = ""
                if not fb:
                    print("code: aborted", file=sys.stderr)
                    return 1
                messages.append({"role": "user", "content": "Revise the plan: " + fb})
                continue
            if decision == "abort":
                print("code: plan not accepted; aborting", file=sys.stderr)
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
            messages=messages,
        )
        active.messages = messages  # share the live list so saves capture the transcript
        try:
            _session.save(active)   # create the file so `latest` resolves during the run
        except OSError as e:
            print(f"code: session save failed ({e}); run will not be steerable",
                  file=sys.stderr)
            active = None
    steer_drain = (
        (lambda sid=active.id: _mailbox.drain(sid)) if active is not None else None
    )
    final_text = None
    budget = _budget_for(args)
    ledger = _agent.ledger_from_args(args, models, model)  # #66 spend cap
    try:
        if args.json:
            with _capture_stdout() as buf:
                _agent.run_loop(oai, model, messages, gen_kwargs, tools,
                                max_tool_calls=max_calls, yes=yes, json_out=False,
                                budget=budget, ledger=ledger, steer_drain=steer_drain)
            final_text = buf.getvalue().strip()
        else:
            _agent.run_loop(oai, model, messages, gen_kwargs, tools,
                            max_tool_calls=max_calls, yes=yes, json_out=False,
                            budget=budget, ledger=ledger, steer_drain=steer_drain)
    except openai.OpenAIError as e:
        return _openai.status_to_exit(openai, e, "code")

    # --- Acceptance check ---
    verdict = None          # None = skipped; else 'pass' | 'fail' | 'unknown'
    report = None
    if not args.no_verify and not args.no_plan:
        messages.append({"role": "user", "content": _VERIFY_MSG})
        try:
            report = _no_tool_turn(oai, model, messages, gen_kwargs, oai_tools)
            parsed = _parse_verdict(report)
            if parsed is None:      # re-prompt ONCE for the exact verdict line
                messages.append({"role": "assistant", "content": report})
                messages.append({"role": "user", "content": _VERIFY_RETRY_MSG})
                retry = _no_tool_turn(oai, model, messages, gen_kwargs, oai_tools)
                report = f"{report}\n{retry}" if report else retry
                parsed = _parse_verdict(retry)
        except openai.OpenAIError as e:
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
        json.dump(envelope, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    return {None: 0, "pass": 0, "fail": 1, "unknown": 10}[verdict]
