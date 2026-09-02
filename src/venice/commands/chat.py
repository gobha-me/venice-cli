"""`venice chat` -- one-shot /chat/completions.

Built on the official `openai` SDK (Venice is OpenAI-compatible; the SDK is
lazy-imported so the rest of the stdlib-only CLI works without it). Venice's
own chat extensions -- web search, characters, thinking control, etc. -- are
surfaced as flags and passed through `extra_body={"venice_parameters": ...}`.

The free `/models?type=text` catalog GET (via the lean urllib client) is used to
validate `--model` and resolve a default before the paid completion call --
mirrors the guard pattern in `venice music`.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .. import _numeric, auth, userconfig
from ..client import build_client_from_auth
from . import (
    _agent, _compact, _mcp, _mcp_client, _models, _openai, _persona,
    _repl, _session,
)

# Named so `defaults.chat.web_search` can be validated against the same set the
# CLI enforces, rather than only argparse seeing it. (#57)
WEB_SEARCH_CHOICES = ("auto", "on", "off")


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "chat",
        help="One-shot chat completion (/chat/completions).",
        description=(
            "Send a single message to a Venice text model and print the reply. "
            "Reads the message from the argument, or from stdin when it is '-' "
            "or piped. Supports Venice extensions: web search, characters, and "
            "reasoning-model thinking control."
        ),
    )
    p.add_argument(
        "message",
        nargs="?",
        help="User message. Use '-' (or pipe stdin) to read from stdin.",
    )
    p.add_argument("--system", "-s", default=None, help="Optional system prompt.")
    p.add_argument(
        "--persona",
        default=None,
        metavar="NAME",
        help=(
            "Load a saved system prompt from ~/.config/venice/personas/<name>.md "
            "at launch. Ignored if --system is given. Switch mid-session with "
            "/persona."
        ),
    )
    p.add_argument(
        "--model",
        "-m",
        default=None,
        help="Text model id (default: the catalog's 'default'-trait model).",
    )
    p.add_argument("--temperature", "-t", type=_numeric.finite_float, default=None)
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")

    stream_grp = p.add_mutually_exclusive_group()
    stream_grp.add_argument(
        "--stream", dest="stream", action="store_true", default=True,
        help="Stream the reply incrementally (default).",
    )
    stream_grp.add_argument(
        "--no-stream", dest="stream", action="store_false",
        help="Wait for the full reply, then print it.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print the raw response object (forces --no-stream).",
    )

    # --- Interactive multi-turn / REPL (#22) ---
    it = p.add_argument_group("Interactive")
    it.add_argument(
        "--interactive", "-i", action="store_true", dest="interactive",
        default=False,
        help="Multi-turn REPL: hold a conversation across turns (also entered "
        "automatically when no message is given on a terminal). Slash-commands: "
        "/system /model /reset /save /exit.",
    )
    it.add_argument(
        "--resume", default=None, metavar="ID|FILE", dest="resume",
        help="Resume a saved session by id (see `venice sessions ls`) or load a "
        "transcript JSON file, continuing it interactively (#47).",
    )
    it.add_argument(
        "--continue", "-c", action="store_true", default=None, dest="cont",
        help="Resume the most recent chat session (#47).",
    )
    it.add_argument(
        "--ephemeral", "--no-save", action="store_true", default=None,
        dest="ephemeral",
        help="Do not auto-save this session to ~/.config/venice/sessions/ (#47).",
    )
    it.add_argument(
        "--auto-compact", action="store_true", default=None, dest="auto_compact",
        help="Summarize older history once it crosses the token budget, so long "
        "sessions stay within the context window (#48; costs a summarization "
        "call). Manual: /compact in the REPL.",
    )
    it.add_argument(
        "--compact-threshold", type=int, default=None, dest="compact_threshold",
        metavar="TOKENS",
        help="Auto-compact once the prompt passes this many tokens "
        f"(default {_compact.DEFAULT_THRESHOLD_TOKENS}).",
    )
    it.add_argument(
        "--compact-keep-turns", type=int, default=None, dest="compact_keep_turns",
        metavar="N",
        help="Turns kept verbatim when compacting "
        f"(default {_compact.DEFAULT_KEEP_TURNS}); older ones are summarized.",
    )

    # --- Venice extensions -> venice_parameters ---
    ext = p.add_argument_group("Venice extensions")
    ext.add_argument(
        "--web-search", choices=WEB_SEARCH_CHOICES, default=None,
        dest="web_search", help="Enable Venice web search (default: off).",
    )
    ext.add_argument(
        "--web-citations", action="store_true", dest="web_citations",
        help="Ask the model to cite web sources (with --web-search).",
    )
    ext.add_argument(
        "--web-scraping", action="store_true", dest="web_scraping",
        help="Scrape URLs in the message via Firecrawl.",
    )
    ext.add_argument(
        "--character", default=None, metavar="SLUG",
        help="Use a public Venice character (its Public ID slug).",
    )
    ext.add_argument(
        "--no-venice-system-prompt", action="store_true",
        dest="no_venice_system_prompt",
        help="Omit Venice's supplied system prompt (default: included).",
    )
    ext.add_argument(
        "--strip-thinking", action="store_true", dest="strip_thinking",
        help="Strip <think> blocks from reasoning-model output.",
    )
    ext.add_argument(
        "--no-thinking", action="store_true", dest="no_thinking",
        help="Disable thinking on supported reasoning models.",
    )
    ext.add_argument(
        "--x-search", action="store_true", dest="x_search",
        help="Enable xAI web+X search (extra ~$0.01/search; grok models).",
    )

    # --- Agent / tool calling (#15) ---
    ag = p.add_argument_group("Agent / tools")
    ag.add_argument(
        "--tools", "--agent", action="store_true", dest="tools", default=None,
        help="Let the model call venice's own tools (image/tts/sfx/music/upscale/"
        "bg-remove/chat) in a loop. Requires a function-calling model; degrades to "
        "plain chat otherwise. Implies non-streamed output for now.",
    )
    ag.add_argument(
        "--tool", action="append", dest="tool", default=None, metavar="NAME",
        help="Restrict the tool set to this tool (repeatable). Default: all seven.",
    )
    ag.add_argument(
        "--max-tool-calls", type=int, default=None, dest="max_tool_calls",
        metavar="N", help="Cap tool invocations before forcing a final answer "
        "(default: 8; 0 = unlimited, run until the model stops).",
    )
    ag.add_argument(
        "--max-spend", type=_numeric.non_negative_float, default=None, metavar="USD",
        help="Per-call auto-approve cap for paid tools (default: $0.10 / "
        "$VENICE_MCP_MAX_SPEND). Over-cap calls prompt on a TTY.",
    )
    ag.add_argument(
        "--session-max-spend", type=_numeric.finite_float, default=None, metavar="USD",
        dest="session_max_spend",
        help="Cap total chat-completion spend for this session (#66). Meters the "
        "model's own calls (not just paid tools) from server token usage; at the "
        "cap the agent stops starting new turns and wraps up. Distinct from "
        "--max-spend (the per-call tool cap).",
    )
    ag.add_argument("--yes", "-y", action="store_true", default=None,
                    help="Auto-approve every paid tool call and every side-effecting "
                    "external MCP tool (skips the confirm gate).")
    ag.add_argument("--output", "-o", type=Path, default=None,
                    help="Directory tools write generated files to. Default: cwd.")
    ag.add_argument(
        "--shell", "--exec", action="store_true", dest="shell", default=None,
        help="Add a gated `shell` tool (/bin/sh -c in the cwd) so the agent can run "
        "gh/git/curl/etc. Implies --tools. Confirms per command unless --yes; scope "
        "it with --shell-allow/--shell-deny or the config `shell` section (#33).",
    )
    ag.add_argument(
        "--shell-allow", action="append", dest="shell_allow", default=None,
        metavar="CMD",
        help="Allow only these commands for --shell (repeatable; globs ok, matched "
        "on the leading token). A non-empty allowlist also requires a single simple "
        "command (no operators/pipes/redirects). Adds to the config shell.allow list.",
    )
    ag.add_argument(
        "--shell-deny", action="append", dest="shell_deny", default=None,
        metavar="PATTERN",
        help="Refuse commands matching these globs (repeatable; matched on the whole "
        "line and each token, always enforced, wins over --shell-allow). Adds to "
        "config shell.deny.",
    )
    ag.add_argument(
        "--shell-unrestricted", action="store_true", dest="shell_unrestricted",
        default=None,
        help="Acknowledge running --shell with an empty allowlist under --yes "
        "(auto-approved arbitrary shell). Required for that combination.",
    )
    ag.add_argument(
        "--browser", action="store_true", dest="browser", default=None,
        help="Add DNS-pinned web_fetch and sandboxed browser_capture tools. Implies --tools.",
    )
    ag.add_argument(
        "--browser-allow", action="append", dest="browser_allow", default=None,
        metavar="HOST",
        help="Restrict browser destinations to matching host globs (repeatable).",
    )
    ag.add_argument(
        "--browser-deny", action="append", dest="browser_deny", default=None,
        metavar="PATTERN",
        help="Deny matching browser hosts or URLs; repeatable and deny wins.",
    )
    ag.add_argument(
        "--browser-private-host", action="append", dest="browser_private_host",
        default=None, metavar="HOST",
        help="Explicitly authorize this exact private/loopback hostname (repeatable; "
        "also requires --browser-private-range).",
    )
    ag.add_argument(
        "--browser-private-range", action="append", dest="browser_private_range",
        default=None, metavar="CIDR",
        help="Authorize a loopback, RFC1918, or IPv6 ULA CIDR (repeatable; also "
        "requires --browser-private-host).",
    )
    ag.add_argument(
        "--memory", action="store_true", dest="memory", default=None,
        help="Add persistent memory + task tools (memory_write/read/search/list, "
        "task_add/update/list) so the agent keeps durable notes and a checklist "
        "across turns/sessions. Implies --tools. Project notes ride <cwd>/.venice/, "
        "global notes ~/.config/venice/memory ($VENICE_MEMORY_DIR); see `venice "
        "memory` to inspect them (#49).",
    )

    # --- External MCP servers (#21) ---
    mc = p.add_argument_group("External MCP tools")
    mc.add_argument(
        "--mcp", action="append", dest="mcp", default=None, metavar="NAME",
        help="Attach a registered MCP server's tools (repeatable). Register servers "
        'first with `venice config add`. Implies the agent loop. Needs the [mcp] '
        'extra: pip install "venice-cli[mcp]". Side-effecting (non-read-only) tools '
        "prompt for confirmation unless --yes.",
    )
    mc.add_argument(
        "--no-mcp", action="store_true", dest="no_mcp", default=False,
        help="Attach no MCP servers, overriding any configured defaults.chat.mcp.",
    )
    p.set_defaults(handler=_run)


def _resolve_message(args) -> Optional[str]:
    """Positional message, or stdin when '-' / piped. None if nothing given."""
    msg = args.message
    if msg == "-" or (msg is None and not sys.stdin.isatty()):
        data = sys.stdin.read().strip()
        return data or None
    return msg


def _venice_parameters(args) -> dict:
    """Assemble venice_parameters from the extension flags (only set keys)."""
    vp: dict = {}
    if args.web_search is not None:
        vp["enable_web_search"] = args.web_search
    if args.web_citations:
        vp["enable_web_citations"] = True
    if args.web_scraping:
        vp["enable_web_scraping"] = True
    if args.character:
        vp["character_slug"] = args.character
    if args.no_venice_system_prompt:
        vp["include_venice_system_prompt"] = False
    if args.strip_thinking:
        vp["strip_thinking_response"] = True
    if args.no_thinking:
        vp["disable_thinking"] = True
    if args.x_search:
        vp["enable_x_search"] = True
    return vp


def _gen_kwargs(args) -> dict:
    """Per-turn generation kwargs (temperature/max_tokens/venice_parameters).

    No `model`/`messages` -- those are supplied per call. Shared by the one-shot
    path (`_build_kwargs`) and the interactive REPL, which re-applies these on
    every turn against a persistent message history.
    """
    kwargs: dict = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    vp = _venice_parameters(args)
    if vp:
        kwargs["extra_body"] = {"venice_parameters": vp}
    return kwargs


def _system_for(args, root=None, tools=None) -> Optional[str]:
    """chat's system prompt is just the user-supplied ``--system``/``--persona``
    string (verbatim, no template). ``root``/``tools`` are ignored -- they are part
    of the shared :class:`_agent.AgentProfile.build_system` contract that code's
    root-aware prompt needs."""
    return args.system


#: The ``venice chat`` profile (#51): plain chat over the shared agent core -- no
#: coding template, no plan/accept/verify, degrades to tool-less chat on a
#: non-function-calling model, and lets the REPL derive tools from ``args``.
PROFILE = _agent.AgentProfile(
    name="chat",
    label="venice chat",
    build_gen_kwargs=_gen_kwargs,
    build_system=_system_for,
    default_max_tool_calls=8,
    plan_mode=False,
    degrade_to_chat=True,
    system_reseed=False,
    injects_tools_session=False,
)


def _build_kwargs(args, model: str, message: str) -> dict:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": message})
    return {"model": model, "messages": messages, **_gen_kwargs(args)}


def _as_dict(value) -> Optional[dict]:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return None


def _print_citations(venice_params) -> None:
    vp = _as_dict(venice_params)
    if not vp:
        return
    cites = vp.get("web_search_citations")
    if not isinstance(cites, list) or not cites:
        return
    print("\nSources:", file=sys.stderr)
    for i, c in enumerate(cites, 1):
        cd = _as_dict(c) or {}
        title = cd.get("title", "")
        url = cd.get("url", "")
        date = cd.get("date")
        line = f"  [{i}] {title} -- {url}"
        if date:
            line += f" ({date})"
        print(line, file=sys.stderr)


def _print_usage(usage) -> None:
    u = _as_dict(usage)
    if not u:
        return
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens")
    tt = u.get("total_tokens")
    if tt is not None:
        print(f"usage: prompt={pt} completion={ct} total={tt}", file=sys.stderr)


def _finish(ledger, t0, *, quiet: bool = False) -> None:
    """Close the tool-loop run's wall-clock window, once, and print the footer (#86).

    A one-shot `venice chat --tools` reported neither time nor cost: the ledger was
    `ledger_from_args`, i.e. None unless the run was spend-capped. This is the same
    report #81 gave `venice code`, on every exit that spent a model call -- including
    the Ctrl+C and API-error paths, which are exactly the runs whose cost you most
    want to see and the ones a happy-path-only footer would hide.

    stderr, not stdout: the answer is the deliverable and `venice chat --tools ... |`
    must stay clean. Not TTY-gated -- one line in the same class as "chat: attached N
    MCP tool(s)", not a redraw surface like `_Spinner`, so gating it would only hide
    the number from logs and pipelines.

    Two deliberate divergences from `code._finish`:

    * No `human` accumulator. Chat has no plan/edit prompt to subtract, and the
      paid-tool confirm gate lives inside `run_loop` where this can't see it -- the
      same hole `code` has, so not a regression.
    * `quiet=True` stamps without printing when a successful `--json` run is about to
      carry this ledger in its envelope. Error and Ctrl+C paths still call the normal
      form from `finally`, so a failed run with no envelope does not lose its cost.

    Idempotent via `ledger.turns`: a non-zero turn count means "already stamped".
    Sound here because a one-shot ledger is never `restore()`d -- `_agent.wants_interactive`
    routes every `--resume`/`--continue`/`-i` run to `_repl.run`, so this function only
    ever sees a fresh ledger. If that routing changes, this guard silently stops
    stamping and needs replacing with an explicit flag.
    """
    if ledger is None or ledger.turns:
        return
    ledger.record_turn(time.monotonic() - t0)
    if quiet:
        return
    # cache=True (#100): the human footer and the JSON ledger must report the same
    # aggregate. The same goes for tools_fragment (#82). It reads `elapsed_seconds`
    # for its "concurrent" check, so it must render after the `record_turn` above.
    print(f"chat: {_agent.format_duration(ledger.elapsed_seconds)} wall"
          f"{ledger.tools_fragment()} -- {ledger.summary(cache=True)}", file=sys.stderr)


def _agent_json_envelope(resp, ledger) -> dict:
    """The aggregate machine contract for one `chat --tools` run (#88)."""
    content = ""
    if getattr(resp, "choices", None):
        content = resp.choices[0].message.content or ""
    return {
        "final": content,
        "usage": ledger.to_dict(),
        "venice_parameters": _as_dict(getattr(resp, "venice_parameters", None)),
    }


def _run(args) -> int:
    # Resolve a resumed session (#47) BEFORE apply_defaults so restored settings
    # (model/temperature/max_tokens/max_tool_calls) outrank config defaults: both
    # only fill None dests, and the session runs first -> explicit flag > session > config.
    try:
        session = _session.resolve_from_args(args, "chat")
    except _session.SessionError as e:
        print(f"chat: {e}", file=sys.stderr)
        return 2
    _session.apply_to_args(args, session, "chat")
    userconfig.apply_defaults(args, "chat")
    # These are tools, so each opt-in implies the agent loop.
    if (getattr(args, "shell", None) or getattr(args, "memory", None)
            or getattr(args, "browser", None)) \
            and not getattr(args, "tools", None):
        args.tools = True
    # A startup persona (--persona or defaults.chat.persona) seeds the same lever
    # both one-shot and REPL modes read -- args.system -- so it flows through
    # _build_kwargs and _seed_messages unchanged. An explicit system prompt
    # (--system / defaults.chat.system) wins.
    if getattr(args, "persona", None) and args.system is None:
        try:
            args.system = _persona.load(args.persona)
        except _persona.PersonaError as e:
            print(f"chat: {e}", file=sys.stderr)
            return 2
    message = _resolve_message(args)
    interactive = _agent.wants_interactive(args, message)
    if not interactive and not message:
        print("chat: no message (pass an argument or pipe stdin)", file=sys.stderr)
        return 2

    openai = _openai.import_openai("chat")
    if openai is None:
        return 2

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    models = _models.catalog(client, "text")
    model, rc = _models.resolve_model(
        args.model, models, label="chat", noun="text model",
        config_key="defaults.chat.model",
    )
    if rc is not None:
        return rc

    oai = _openai.build_openai(openai, client)

    if interactive:
        return _repl.run(args, oai, openai, client, models, model, initial=message,
                         session=session, ephemeral=bool(getattr(args, "ephemeral", None)))

    kwargs = _build_kwargs(args, model, message)

    if getattr(args, "tools", None) or _requested_mcp_servers(args):
        rc = _run_agent(args, oai, openai, client, models, model, kwargs)
        if rc is not None:
            return rc
        # else: model can't do function calling -> fall through to plain chat

    stream = args.stream and not args.json
    try:
        if stream:
            return _run_stream(oai, kwargs)
        return _run_once(oai, kwargs, args.json)
    except openai.OpenAIError as e:
        return _openai.status_to_exit(openai, e, "chat")
    except KeyboardInterrupt:
        notice = "chat: aborted"
        if stream:
            notice += " (partial output may appear above)"
        print(f"\n{notice}", file=sys.stderr)
        return 130


def _tools_for(args, client, models, model):
    """Resolve the built-in tool list for `model` (shared by one-shot + REPL).

    Returns ``(tools, None)`` on success; ``(None, None)`` when the model can't do
    function calling (caller degrades to plain chat); ``(None, 2)`` when the
    requested ``--tool`` subset is invalid. Prints the same capability notes the
    one-shot path always did.
    """
    ok, rc = _agent.check_function_calling(
        models, model, label=PROFILE.name,
        degraded_tail="running without tools",
        unverified_tail="attempting tools",
        degrade=PROFILE.degrade_to_chat,
    )
    if not ok:
        return None, rc  # degrade_to_chat -> (None, None): caller falls through to plain chat
    doc = userconfig.load_config()  # #58 tool defaults + #33 shell policy
    shell_on = bool(getattr(args, "shell", None))
    shell_allow, shell_deny = (), ()
    if shell_on:
        pol = userconfig.shell_policy(doc)
        shell_allow = list(pol["allow"]) + list(getattr(args, "shell_allow", None) or [])
        shell_deny = list(pol["deny"]) + list(getattr(args, "shell_deny", None) or [])
        # Loud-unrestricted guard (#33): auto-approving arbitrary shell must be
        # explicit -- an empty allowlist under --yes needs --shell-unrestricted.
        if (not shell_allow and getattr(args, "yes", None)
                and not getattr(args, "shell_unrestricted", None)):
            print(
                "chat: refusing to enable an unrestricted shell with --yes; pass "
                "--shell-unrestricted to confirm, or scope it with --shell-allow "
                "(or the config shell.allow list).",
                file=sys.stderr,
            )
            return None, 2
    browser_on = bool(getattr(args, "browser", None))
    browser_allow, browser_deny = (), ()
    browser_private_hosts, browser_private_ranges = (), ()
    if browser_on:
        bpol = userconfig.browser_policy(doc)  # #71 URL allow/deny policy
        browser_allow = list(bpol["allow"]) + list(getattr(args, "browser_allow", None) or [])
        browser_deny = list(bpol["deny"]) + list(getattr(args, "browser_deny", None) or [])
        browser_private_hosts = list(bpol["private_hosts"]) + list(
            getattr(args, "browser_private_host", None) or [])
        browser_private_ranges = list(bpol["private_ranges"]) + list(
            getattr(args, "browser_private_range", None) or [])
    try:
        tools = _agent.builtin_tools(
            client,
            max_spend=args.max_spend,
            output_dir=str(args.output) if args.output else None,
            only=set(args.tool) if args.tool else None,
            config=doc,  # #58: honor defaults.<cmd>.* in tools
            shell=shell_on,
            shell_root=os.getcwd(),
            shell_allow=shell_allow,
            shell_deny=shell_deny,
            browser=browser_on,
            browser_allow=browser_allow,
            browser_deny=browser_deny,
            browser_private_hosts=browser_private_hosts,
            browser_private_ranges=browser_private_ranges,
            browser_output_dir=str(args.output) if args.output else None,
            memory=bool(getattr(args, "memory", None)),
        )
    except ValueError as e:
        print(f"chat: {e}", file=sys.stderr)
        return None, 2
    return tools, None


def _requested_mcp_servers(args) -> list:
    """Server names to attach: none if --no-mcp, else --mcp NAMEs (or the config
    default that apply_defaults already filled onto args.mcp)."""
    if getattr(args, "no_mcp", False):
        return []
    return list(getattr(args, "mcp", None) or [])


@contextlib.contextmanager
def _tools_session(args, client, models, model):
    """Yield ``(tools, rc)`` = built-in tools plus any external MCP tools, holding
    the MCP servers open for the whole ``with`` block. Shared by the one-shot agent
    path and the REPL so wiring MCP once covers both.

    ``(tools, None)`` on success; ``(None, None)`` to degrade to plain chat (model
    can't do function calling); ``(None, 2)`` for a bad ``--tool`` subset, a missing
    ``[mcp]`` extra, or an unknown ``--mcp`` server. MCP is never opened on the
    degrade path.
    """
    tools, rc = _tools_for(args, client, models, model)
    if tools is None:
        yield None, rc
        return

    requested = _requested_mcp_servers(args)
    if not requested:
        yield tools, None  # base path untouched -- no probe, no [mcp] extra needed
        return

    mcp = _mcp.import_mcp("chat --mcp")
    if mcp is None:
        yield None, 2
        return
    try:
        specs = _mcp_client.resolve_specs(requested, userconfig.load_config())
    except ValueError as e:
        print(f"chat: {e}", file=sys.stderr)
        yield None, 2
        return

    # Enter attach() via an ExitStack so setup failure -> rc 2 while a failure in
    # the with-body (run_loop) still propagates and still tears the servers down.
    stack = contextlib.ExitStack()
    try:
        remote = stack.enter_context(_mcp_client.attach(specs))
    except Exception as e:  # noqa: BLE001 - a server that won't start
        print(f"chat: could not attach MCP server(s): {e}", file=sys.stderr)
        yield None, 2
        return
    with stack:
        if remote:
            print(
                f"chat: attached {len(remote)} MCP tool(s) from "
                f"{', '.join(n for n, _ in specs)}",
                file=sys.stderr,
            )
        yield tools + remote, None


def _run_agent(args, oai, openai, client, models, model, kwargs) -> Optional[int]:
    """Run the tool-calling loop. Returns an exit code, or None to signal the
    caller to fall through to plain (non-tool) chat because the model can't do
    function calling."""
    # Outside the `with` on purpose: attaching a cold MCP server is real time the
    # operator spent waiting on us, and `_finish` is inert until a ledger exists.
    t0 = time.monotonic()
    ledger = None
    final_response = []

    def _capture_final(resp) -> int:
        final_response.append(resp)
        return 0

    try:
        with _tools_session(args, client, models, model) as (tools, rc):
            if tools is None:
                return rc  # None -> degrade to plain chat; 2 -> invalid subset / MCP error

            if args.stream and not args.json:
                print("chat: tools imply non-streamed output for now", file=sys.stderr)

            messages = kwargs.pop("messages")
            kwargs.pop("model", None)
            # Minted here, past the `tools is None` guard, so every non-spending exit
            # (bad --tool subset, MCP failure, degrade) returns while it's still None
            # and `_finish` no-ops. Suppression is structural, not a list of exits to
            # remember. #86: was `ledger_from_args` -> None unless --session-max-spend,
            # so a default run metered nothing. `usage_ledger` is always on and adds no
            # gating: `_build_ledger` leaves max_tokens None and both `over()` and
            # `over_tokens()` short-circuit on a None cap. --session-max-spend still caps.
            ledger = _agent.usage_ledger(args, models, model)  # #66 spend cap, #86 metering
            result = _agent.run_loop(
                oai, model, messages, kwargs, tools,
                max_tool_calls=(args.max_tool_calls if args.max_tool_calls is not None
                                else PROFILE.default_max_tool_calls),
                yes=bool(args.yes),
                json_out=args.json,
                budget=_compact.budget_from_args(args),  # #48 auto-compact parity
                ledger=ledger,
                final_emitter=_capture_final if args.json else None,
                models=models,
            )
            if args.json and result == 0:
                if len(final_response) != 1:
                    print("chat: final response was not captured", file=sys.stderr)
                    return 2
                # The aggregate must be stamped before serialization; otherwise its
                # elapsed_seconds and turns fields would still describe an open run.
                _finish(ledger, t0, quiet=True)
                json.dump(
                    _agent_json_envelope(final_response[0], ledger),
                    sys.stdout,
                    indent=2,
                    default=str,
                    allow_nan=False,
                )
                sys.stdout.write("\n")
            return result
    except openai.OpenAIError as e:
        return _openai.status_to_exit(openai, e, "chat")
    except KeyboardInterrupt:
        # chat was the last command with no handler at all -- Ctrl+C gave a traceback.
        # 130 is what CPython already exits with; this just makes it deliberate, and
        # lets the `finally` cost the turns you did pay for before killing the run.
        print("\nchat: aborted", file=sys.stderr)
        return 130
    finally:
        # After the error handlers so their message lands first. A successful JSON run
        # stamps before its envelope above; idempotence makes this a no-op there. Error
        # and Ctrl+C paths have no envelope, so they still report their partial cost.
        _finish(ledger, t0)


def _run_once(oai, kwargs: dict, as_json: bool) -> int:
    resp = oai.chat.completions.create(**kwargs)
    if as_json:
        json.dump(
            resp.model_dump(), sys.stdout, indent=2, default=str, allow_nan=False
        )
        sys.stdout.write("\n")
        return 0
    content = ""
    if resp.choices:
        content = resp.choices[0].message.content or ""
    print(content)
    _print_citations(getattr(resp, "venice_parameters", None))
    _print_usage(getattr(resp, "usage", None))
    return 0


def _consume_stream(stream) -> str:
    """Write streamed deltas to stdout as they arrive; return the full reply text.

    Citations/usage are printed to stderr. Shared by the one-shot `_run_stream`
    and the REPL turn helper, which needs the accumulated text to append the
    assistant turn to its persistent history.
    """
    text, _usage = _consume_stream_full(stream)
    return text


def _consume_stream_full(stream):
    """Like `_consume_stream`, but also return the stream's final `usage`.

    The REPL's auto-compact budget (#48) observes the server-reported prompt
    token count; printing behavior is identical to `_consume_stream`.
    """
    message, usage = _consume_stream_message_full(stream)
    return message["content"], usage


def _consume_stream_message_full(stream):
    """Consume a stream and retain its replayable assistant-message extensions.

    The ordinary one-shot surface still returns/prints only visible content.  The
    multi-turn REPL uses this richer result so Kimi-style reasoning deltas survive in
    history and the next request can replay the complete assistant turn.
    """
    citations = None
    usage = None
    parts: list = []
    reasoning_parts = {field: [] for field in _agent._REASONING_FIELDS}
    thought_signature_parts: list = []
    for chunk in stream:
        vp = getattr(chunk, "venice_parameters", None)
        if vp is not None and citations is None:
            citations = vp
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
                parts.append(piece)
            for field in _agent._REASONING_FIELDS:
                value = getattr(delta, field, None)
                if value is not None:
                    reasoning_parts[field].append(_agent._request_value(value))
            signature = getattr(delta, _agent._THOUGHT_SIGNATURE_FIELD, None)
            if signature is not None:
                thought_signature_parts.append(_agent._request_value(signature))
    if parts:
        sys.stdout.write("\n")
    _print_citations(citations)
    _print_usage(usage)
    message = {"role": "assistant", "content": "".join(parts)}
    for field in _agent._REASONING_FIELDS:
        values = reasoning_parts[field]
        if not values:
            continue
        if all(isinstance(value, str) for value in values):
            message[field] = "".join(values)
        elif all(isinstance(value, list) for value in values):
            message[field] = [item for value in values for item in value]
        else:
            message[field] = values[0] if len(values) == 1 else values
        break
    if thought_signature_parts:
        if all(isinstance(value, str) for value in thought_signature_parts):
            message[_agent._THOUGHT_SIGNATURE_FIELD] = "".join(
                thought_signature_parts
            )
        else:
            message[_agent._THOUGHT_SIGNATURE_FIELD] = (
                thought_signature_parts[0]
                if len(thought_signature_parts) == 1
                else thought_signature_parts
            )
    return message, usage


def _run_stream(oai, kwargs: dict) -> int:
    kwargs = dict(kwargs)
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    _consume_stream(oai.chat.completions.create(**kwargs))
    return 0
