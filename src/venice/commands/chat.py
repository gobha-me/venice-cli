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
import sys
from pathlib import Path
from typing import Optional

from .. import auth, userconfig
from ..client import build_client_from_auth
from . import _agent, _mcp, _mcp_client, _models, _openai, _repl


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
        "--model",
        "-m",
        default=None,
        help="Text model id (default: the catalog's 'default'-trait model).",
    )
    p.add_argument("--temperature", "-t", type=float, default=None)
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
        "--resume", default=None, metavar="FILE", dest="resume",
        help="Load a saved transcript JSON and continue it interactively "
        "(pairs with /save).",
    )

    # --- Venice extensions -> venice_parameters ---
    ext = p.add_argument_group("Venice extensions")
    ext.add_argument(
        "--web-search", choices=("auto", "on", "off"), default=None,
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
        "(default: 8).",
    )
    ag.add_argument(
        "--max-spend", type=float, default=None, metavar="USD",
        help="Per-call auto-approve cap for paid tools (default: $0.10 / "
        "$VENICE_MCP_MAX_SPEND). Over-cap calls prompt on a TTY.",
    )
    ag.add_argument("--yes", "-y", action="store_true", default=None,
                    help="Auto-approve every paid tool call and every side-effecting "
                    "external MCP tool (skips the confirm gate).")
    ag.add_argument("--output", "-o", type=Path, default=None,
                    help="Directory tools write generated files to. Default: cwd.")

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


def _is_interactive(args, message) -> bool:
    """REPL when explicitly requested (`-i` / `--resume`) or when there is no
    message and stdin is an interactive terminal. A piped or `-` message is
    always one-shot."""
    if getattr(args, "interactive", False) or getattr(args, "resume", None):
        return True
    return message is None and sys.stdin.isatty()


def _run(args) -> int:
    userconfig.apply_defaults(args, "chat")
    message = _resolve_message(args)
    interactive = _is_interactive(args, message)
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
        args.model, models, label="chat", noun="text model"
    )
    if rc is not None:
        return rc

    oai = _openai.build_openai(openai, client)

    if interactive:
        return _repl.run(args, oai, openai, client, models, model, initial=message)

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


def _tools_for(args, client, models, model):
    """Resolve the built-in tool list for `model` (shared by one-shot + REPL).

    Returns ``(tools, None)`` on success; ``(None, None)`` when the model can't do
    function calling (caller degrades to plain chat); ``(None, 2)`` when the
    requested ``--tool`` subset is invalid. Prints the same capability notes the
    one-shot path always did.
    """
    supported = _agent.supports_function_calling(models, model)
    if supported is False:
        print(
            f"chat: model {model} does not support function calling; "
            "running without tools",
            file=sys.stderr,
        )
        return None, None
    if supported is None:
        print(
            f"chat: could not verify function-calling support for {model}; "
            "attempting tools",
            file=sys.stderr,
        )
    try:
        tools = _agent.builtin_tools(
            client,
            max_spend=args.max_spend,
            output_dir=str(args.output) if args.output else None,
            only=set(args.tool) if args.tool else None,
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
    with _tools_session(args, client, models, model) as (tools, rc):
        if tools is None:
            return rc  # None -> degrade to plain chat; 2 -> invalid subset / MCP error

        if args.stream and not args.json:
            print("chat: tools imply non-streamed output for now", file=sys.stderr)

        messages = kwargs.pop("messages")
        kwargs.pop("model", None)
        try:
            return _agent.run_loop(
                oai, model, messages, kwargs, tools,
                max_tool_calls=(args.max_tool_calls or 8),
                yes=bool(args.yes),
                json_out=args.json,
            )
        except openai.OpenAIError as e:
            return _openai.status_to_exit(openai, e, "chat")


def _run_once(oai, kwargs: dict, as_json: bool) -> int:
    resp = oai.chat.completions.create(**kwargs)
    if as_json:
        json.dump(resp.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0
    content = ""
    if resp.choices:
        content = resp.choices[0].message.content or ""
    print(content)
    _print_citations(getattr(resp, "venice_parameters", None))
    return 0


def _consume_stream(stream) -> str:
    """Write streamed deltas to stdout as they arrive; return the full reply text.

    Citations/usage are printed to stderr. Shared by the one-shot `_run_stream`
    and the REPL turn helper, which needs the accumulated text to append the
    assistant turn to its persistent history.
    """
    citations = None
    usage = None
    parts: list = []
    for chunk in stream:
        vp = getattr(chunk, "venice_parameters", None)
        if vp is not None and citations is None:
            citations = vp
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices:
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                sys.stdout.write(piece)
                sys.stdout.flush()
                parts.append(piece)
    if parts:
        sys.stdout.write("\n")
    _print_citations(citations)
    _print_usage(usage)
    return "".join(parts)


def _run_stream(oai, kwargs: dict) -> int:
    kwargs = dict(kwargs)
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    _consume_stream(oai.chat.completions.create(**kwargs))
    return 0
