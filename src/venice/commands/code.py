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
import io
import json
import os
import re
import sys
from typing import List, Optional

from .. import auth, config, userconfig
from ..client import build_client_from_auth
from . import _agent, _code, _models, _openai, _repl

_DEFAULT_MAX_TOOL_CALLS = 25

CODING_SYSTEM_PROMPT = """\
You are vcoder, an autonomous coding agent working inside a single project directory.

Project root: {root}
Available tools: {tools}

Guidelines:
- All file paths are relative to the project root; you cannot read or write files \
outside it.
- Explore before you change: use read_file, list_dir, grep (and project_search when \
available) to understand the code first.
- Prefer edit_file for small, targeted changes; use write_file for new files or full \
rewrites. Match the surrounding code's style.
- Use run to run tests, builds, or git mutations. run, write_file, and edit_file \
change the project and may require the user's confirmation before they execute.
- Make minimal, correct changes and verify them (run the tests or relevant command) \
when practical.
- Keep your final message a concise summary: what you changed and how you verified it."""

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
        f"{_DEFAULT_MAX_TOOL_CALLS}).",
    )
    grp.add_argument(
        "--exec-timeout", type=int, default=None, dest="exec_timeout",
        metavar="SECS",
        help=f"Timeout for run/git commands (default: {_code.DEFAULT_EXEC_TIMEOUT}).",
    )

    it = p.add_argument_group("Interactive")
    it.add_argument(
        "--interactive", "-i", action="store_true", default=False,
        help="Interactive coding REPL (also entered with no task on a terminal). "
        "Tools are on; changes confirm per step unless --auto.",
    )
    it.add_argument(
        "--resume", default=None, metavar="FILE",
        help="Resume a saved transcript JSON interactively (pairs with /save).",
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
    if args.system:
        base += "\n\nProject-specific instructions:\n" + args.system
    return base


def _autonomous(args) -> bool:
    return bool(args.auto or args.yes)


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


@contextlib.contextmanager
def _capture_stdout():
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


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
    userconfig.apply_defaults(args, "code")

    root = os.path.realpath(
        args.root or os.environ.get(config.ENV_CODE_ROOT) or os.getcwd()
    )
    if not os.path.isdir(root):
        print(f"code: not a directory: {root}", file=sys.stderr)
        return 2

    task = _resolve_task(args)
    interactive = bool(args.interactive or args.resume) or (
        task is None and sys.stdin.isatty()
    )
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

    supported = _agent.supports_function_calling(models, model)
    if supported is False:
        print(
            f"code: model {model} does not support function calling; venice code "
            "needs a tool-calling model (pass --model).",
            file=sys.stderr,
        )
        return 2
    if supported is None:
        print(
            f"code: could not verify function-calling support for {model}; "
            "attempting anyway",
            file=sys.stderr,
        )

    oai = _openai.build_openai(openai, client)
    tools = _code.code_tools(
        root, client,
        exec_timeout=args.exec_timeout or _code.DEFAULT_EXEC_TIMEOUT,
        include_search=True,
    )
    system = _system_prompt(args, root, tools)
    gen_kwargs = _gen_kwargs(args)

    if interactive:
        args.system = system
        args.yes = _autonomous(args)  # drive the REPL's per-turn gate
        print(
            f"code: sandboxed to {root} -- tools: {_code.tool_names(tools)}",
            file=sys.stderr,
        )
        return _repl.run(
            args, oai, openai, client, models, model, initial=task,
            tools_session=_code_session(tools), gen_kwargs=gen_kwargs,
            label="venice code",
        )

    return _run_oneshot(args, oai, openai, model, tools, system, gen_kwargs, root, task)


def _run_oneshot(args, oai, openai, model, tools, system, gen_kwargs, root, task) -> int:
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
    max_calls = args.max_tool_calls or _DEFAULT_MAX_TOOL_CALLS
    final_text = None
    try:
        if args.json:
            with _capture_stdout() as buf:
                _agent.run_loop(oai, model, messages, gen_kwargs, tools,
                                max_tool_calls=max_calls, yes=yes, json_out=False)
            final_text = buf.getvalue().strip()
        else:
            _agent.run_loop(oai, model, messages, gen_kwargs, tools,
                            max_tool_calls=max_calls, yes=yes, json_out=False)
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
        json.dump(envelope, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    return {None: 0, "pass": 0, "fail": 1, "unknown": 10}[verdict]
