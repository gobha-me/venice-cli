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

import json
import sys
from typing import Optional

from .. import auth, userconfig
from ..client import build_client_from_auth
from . import _models, _openai


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


def _build_kwargs(args, model: str, message: str) -> dict:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": message})

    kwargs: dict = {"model": model, "messages": messages}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    vp = _venice_parameters(args)
    if vp:
        kwargs["extra_body"] = {"venice_parameters": vp}
    return kwargs


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


def _run(args) -> int:
    userconfig.apply_defaults(args, "chat")
    message = _resolve_message(args)
    if not message:
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
    kwargs = _build_kwargs(args, model, message)

    stream = args.stream and not args.json
    try:
        if stream:
            return _run_stream(oai, kwargs)
        return _run_once(oai, kwargs, args.json)
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


def _run_stream(oai, kwargs: dict) -> int:
    kwargs = dict(kwargs)
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    stream = oai.chat.completions.create(**kwargs)

    citations = None
    usage = None
    wrote_any = False
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
                wrote_any = True
    if wrote_any:
        sys.stdout.write("\n")
    _print_citations(citations)
    _print_usage(usage)
    return 0
