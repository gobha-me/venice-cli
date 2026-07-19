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

import json
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import _mcp


# --------------------------------------------------------------------------- #
# Tool descriptor + derived structures (pure functions of a list[Tool])
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tool:
    """One function tool the model may call.

    ``invoke(arguments, *, confirm=False) -> dict`` takes the model-supplied
    arguments object and returns a JSON-serializable result dict. ``paid`` marks
    tools whose result can be a ``confirmation_required`` gate.
    """

    name: str
    description: str
    parameters: dict
    invoke: Callable[..., dict]
    paid: bool = False


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


def dispatch_map(tools: List[Tool]) -> Dict[str, Tool]:
    return {t.name: t for t in tools}


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

_SFX_SCHEMA = _obj(
    {
        "prompt": _p("string", "What the sound effect should be."),
        "model": _p("string"),
        "duration": _p("integer", "Length in seconds."),
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
    },
    required=["prompt"],
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

_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Natural-language description of the code/text to find."),
        "k": _p("integer", "Number of results to return (default 8)."),
    },
    required=["query"],
)

# (tool name, `_mcp` impl attribute, description, schema, paid). The impl is
# stored by NAME and resolved via getattr(_mcp, ...) at builtin_tools() time, so a
# single source of truth wins and tests can patch `_mcp.<impl>`.
_BUILTINS = [
    (
        "venice_image",
        "image_tool",
        "Generate 1-4 image variants from a text prompt via Venice /image/generate. "
        "Writes image file(s) and returns their paths (never inline blobs). Paid: "
        "over-cap calls need confirmation.",
        _IMAGE_SCHEMA,
        True,
    ),
    (
        "venice_tts",
        "tts_tool",
        "Synthesize speech from text via Venice /audio/speech. Writes an audio file "
        "and returns its path. Paid.",
        _TTS_SCHEMA,
        True,
    ),
    (
        "venice_sfx",
        "sfx_tool",
        "Generate a short sound effect via Venice's async audio queue (blocks with a "
        "capped wait). Writes an audio file and returns its path. Paid.",
        _SFX_SCHEMA,
        True,
    ),
    (
        "venice_music",
        "music_tool",
        "Generate long-form music/ambience via Venice's async audio queue (blocks "
        "with a capped wait). Writes an audio file and returns its path. Paid.",
        _MUSIC_SCHEMA,
        True,
    ),
    (
        "venice_upscale",
        "upscale_tool",
        "Upscale/enhance a local image (factor 1-4) via Venice /image/upscale. Writes "
        "the result and returns its path. Dynamic pricing, so it always needs "
        "confirmation.",
        _UPSCALE_SCHEMA,
        True,
    ),
    (
        "venice_bg_remove",
        "bg_remove_tool",
        "Remove an image's background via Venice /image/background-remove, returning a "
        "transparent PNG. Source is a local input_path OR an image_url. Dynamic "
        "pricing, so it always needs confirmation.",
        _BG_REMOVE_SCHEMA,
        True,
    ),
    (
        "venice_chat",
        "chat_tool",
        "Delegate a one-shot sub-completion to a Venice text model (optionally a "
        "different model or character) and return its reply text. Not spend-gated.",
        _CHAT_SCHEMA,
        False,
    ),
    (
        "project_search",
        "search_tool",
        "Semantic search over the current project's local .venice index (built by "
        "`venice index`) for the chunks most relevant to a natural-language query. "
        "Returns file paths with line ranges and a short preview -- use it to locate "
        "code by meaning before reading files. Read-only; not spend-gated. Errors if "
        "no index exists yet.",
        _SEARCH_SCHEMA,
        False,
    ),
]

# Loop-controlled kwargs the model must never supply (stripped defensively).
_CONTROLLED = ("confirm", "max_spend", "output_dir")


def _clean(arguments) -> dict:
    if not isinstance(arguments, dict):
        return {}
    return {k: v for k, v in arguments.items() if k not in _CONTROLLED}


def builtin_tools(
    client,
    *,
    max_spend: Optional[float] = None,
    output_dir: Optional[str] = None,
    only: Optional[set] = None,
) -> List[Tool]:
    """Build the 7 in-process venice tools, bound to `client`.

    `max_spend`/`output_dir` are baked into the paid tools' closures; `confirm` is
    passed per-call by the loop. `only` restricts the set to the named tools (an
    unknown name raises ValueError so the caller can exit 2).
    """

    def _make_paid(impl):
        def invoke(arguments, *, confirm: bool = False):
            return impl(
                client,
                confirm=confirm,
                max_spend=max_spend,
                output_dir=output_dir,
                **_clean(arguments),
            )

        return invoke

    def _make_free(impl):
        def invoke(arguments, *, confirm: bool = False):
            return impl(client, **_clean(arguments))

        return invoke

    tools = [
        Tool(
            name=name,
            description=desc,
            parameters=schema,
            invoke=(
                _make_paid(getattr(_mcp, impl_name))
                if paid
                else _make_free(getattr(_mcp, impl_name))
            ),
            paid=paid,
        )
        for (name, impl_name, desc, schema, paid) in _BUILTINS
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
    if not models:
        return None
    for m in models:
        if not isinstance(m, dict) or m.get("id") != model_id:
            continue
        spec = m.get("model_spec") or {}
        caps = spec.get("capabilities")
        if not isinstance(caps, dict):
            return None
        norm = {str(k).lower().replace("_", ""): v for k, v in caps.items()}
        val = norm.get("supportsfunctioncalling")
        return bool(val) if val is not None else None
    return None


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


def _prompt_yes() -> bool:
    try:
        ans = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    return ans in ("y", "yes")


def _resolve_spend(tool: Tool, arguments: dict, result, *, yes: bool):
    """Hybrid gate: prompt on a TTY, else feed the block back to the model.

    Only reached for a paid tool that returned `confirmation_required` (which can
    only happen when `yes` is False, since `confirm=True` bypasses the gate).
    """
    if not tool.paid or yes:
        return result
    if not (isinstance(result, dict) and result.get("status") == "confirmation_required"):
        return result
    message = result.get("message", f"{tool.name}: confirmation required")
    if sys.stdin.isatty():
        print(message, file=sys.stderr)
        if _prompt_yes():
            try:
                return tool.invoke(arguments, confirm=True)
            except Exception as e:  # pragma: no cover - impls shouldn't raise
                return {"status": "error", "message": f"{tool.name} failed: {e}"}
        print(f"{tool.name}: declined by user", file=sys.stderr)
    return result  # non-TTY or declined -> the model sees the gate and adapts


def _run_one_call(tc, dispatch: Dict[str, Tool], *, yes: bool) -> dict:
    tool = dispatch.get(tc.function.name)
    if tool is None:
        return {"status": "error", "message": f"unknown tool {tc.function.name!r}"}
    try:
        arguments = json.loads(tc.function.arguments or "{}")
    except (TypeError, ValueError) as e:
        return {"status": "error", "message": f"invalid JSON arguments: {e}"}
    if not isinstance(arguments, dict):
        return {"status": "error", "message": "tool arguments must be a JSON object"}
    try:
        result = tool.invoke(arguments, confirm=bool(yes))
    except Exception as e:  # pragma: no cover - impls shouldn't raise
        return {"status": "error", "message": f"{tool.name} failed: {e}"}
    return _resolve_spend(tool, arguments, result, yes=yes)


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
) -> int:
    """Drive the function-calling loop until the model stops (or the cap is hit).

    `messages` is the persistent, mutable history (seeded with system+user).
    `base_kwargs` are per-turn generation kwargs (temperature/max_tokens/extra_body)
    re-applied on every create(); it must NOT contain `model`/`messages`. Non-streamed
    by design (tool-call deltas would need fragment reassembly; v1 buffers each turn).
    Only `openai.OpenAIError` from create() is fatal -- the caller maps it to an exit
    code; tool failures come back as dicts the model can recover from.
    """
    oai_tools = to_openai_tools(tools)
    dispatch = dispatch_map(tools)
    calls_made = 0

    while True:
        resp = oai.chat.completions.create(
            model=model,
            messages=messages,
            tools=oai_tools,
            tool_choice="auto",
            **base_kwargs,
        )
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
        if not tool_calls:
            return _emit_final(resp, json_out)

        # Every tool_call in the turn must get a result (message-contract), even
        # ones past the budget -- those are reported not-executed rather than run.
        for tc in tool_calls:
            if calls_made >= max_tool_calls:
                result = {
                    "status": "error",
                    "message": "tool-call budget (--max-tool-calls) exhausted; "
                    "not executed",
                }
            else:
                result = _run_one_call(tc, dispatch, yes=yes)
                calls_made += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result, default=str),
                }
            )

        if calls_made >= max_tool_calls:
            print(
                f"chat: reached --max-tool-calls ({max_tool_calls}); "
                "requesting a final answer",
                file=sys.stderr,
            )
            resp = oai.chat.completions.create(
                model=model,
                messages=messages,
                tools=oai_tools,
                tool_choice="none",
                **base_kwargs,
            )
            msg = resp.choices[0].message if getattr(resp, "choices", None) else None
            messages.append(_assistant_dict(msg))
            return _emit_final(resp, json_out)
