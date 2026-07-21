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

import inspect
import itertools
import json
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .. import userconfig
from . import _mcp
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


class CostLedger:
    """Accumulates estimated USD spend for one agent run.

    `max_spend` is the session cap (USD-equivalent; None = unmetered). The
    ledger is bound to a model's pricing on first use via :meth:`bind_pricing`;
    an unknown price means the turn's tokens are counted but not charged
    (degrade gracefully rather than hard-block on a missing price).
    """

    def __init__(self, max_spend: Optional[float] = None):
        # A non-positive cap means "uncapped" (mirrors --max-tool-calls 0).
        cap = float(max_spend) if max_spend is not None else None
        self.max_spend = cap if (cap is not None and cap > 0) else None
        self.total = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unpriced = False  # saw a turn whose model price was unknown
        self._in = None        # per-token input rate (USD)
        self._out = None       # per-token output rate (USD)

    def bind_pricing(self, pricing) -> None:
        """Set the per-token rates from a catalog `model_spec.pricing` block."""
        self._in = _usd_per_token(pricing, "input")
        self._out = _usd_per_token(pricing, "output")

    def record(self, usage) -> float:
        """Add one turn's `usage` (dict or SDK obj); return this turn's cost."""
        if usage is None:
            return 0.0
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            return 0.0
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        if not isinstance(pt, (int, float)):
            pt = 0
        if not isinstance(ct, (int, float)):
            ct = 0
        self.prompt_tokens += int(pt)
        self.completion_tokens += int(ct)
        cost = 0.0
        if self._in is not None or self._out is not None:
            cost = float(pt) * (self._in or 0.0) + float(ct) * (self._out or 0.0)
        else:
            self.unpriced = True
        self.total += cost
        return cost

    def over(self) -> bool:
        """True when accumulated spend has reached/exceeded the cap."""
        return self.max_spend is not None and self.total >= self.max_spend

    def summary(self) -> str:
        """A one-line human-readable total (for stderr / --json)."""
        if self.unpriced and self.total == 0.0:
            return (
                f"cost: (unpriced — model rate unknown) "
                f"tokens prompt={self.prompt_tokens} completion={self.completion_tokens}"
            )
        s = f"cost: ${self.total:.4f}"
        if self.max_spend is not None:
            s += f" / cap ${self.max_spend:.2f}"
        s += f" (tokens prompt={self.prompt_tokens} completion={self.completion_tokens})"
        if self.unpriced:
            s += " [partially unpriced]"
        return s


def _pricing_for(models, model_id):
    """The catalog `model_spec.pricing` block for `model_id`, or None."""
    for m in models or []:
        if isinstance(m, dict) and m.get("id") == model_id:
            spec = m.get("model_spec")
            if isinstance(spec, dict):
                return spec.get("pricing")
    return None


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
    ledger = CostLedger(max_spend=cap)
    pricing = _pricing_for(models, model_id)
    if pricing is not None:
        ledger.bind_pricing(pricing)
    return ledger


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

_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Natural-language description of the code/text to find."),
        "k": _p("integer", "Number of results to return (default 8)."),
    },
    required=["query"],
)

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
        "no_safe_mode": _p("boolean", "Disable safe mode (defaults to on)."),
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
        "capped wait). Writes an audio file and returns its path. Pass background=true "
        "to queue and return immediately, then fetch via venice_job_result. Paid.",
        _SFX_SCHEMA,
        True,
    ),
    (
        "venice_music",
        "music_tool",
        "Generate long-form music/ambience via Venice's async audio queue (blocks "
        "with a capped wait). Writes an audio file and returns its path. Pass "
        "background=true to queue and return immediately, then fetch via "
        "venice_job_result. Paid.",
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
        "venice_models",
        "models_tool",
        "List available Venice model ids for a catalog type (text/code/image/video/"
        "music/tts/embedding/upscale, or 'all') via the free /models catalog. Use it "
        "to choose a valid `model` for the other venice_* tools instead of guessing. "
        "Read-only; not spend-gated.",
        _MODELS_SCHEMA,
        False,
    ),
    (
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
    ),
    (
        "venice_vision",
        "vision_tool",
        "Look at an image (a local input_path OR an image_url) with a vision-capable "
        "Venice text model and return what it sees as text. Optional prompt directs "
        "the question (default: a detailed description). Auto-picks a supportsVision "
        "model when model is omitted (see venice_model_details). Not spend-gated.",
        _VISION_SCHEMA,
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
    (
        "venice_job_status",
        "job_status_tool",
        "Peek at a backgrounded media render started with background=true on "
        "venice_sfx/venice_music/venice_video. Pass back the job handle's queue_id, "
        "type (sfx/music/video), and model. Returns processing/done/failed/not_found. "
        "Read-only, non-blocking; not spend-gated.",
        _JOB_STATUS_SCHEMA,
        False,
    ),
    (
        "venice_job_result",
        "job_result_tool",
        "Fetch a backgrounded media render's file once ready (started with "
        "background=true). Pass back the job handle's queue_id, type, model (and "
        "download_url for VPS video). Writes the file and returns its path, or "
        "status 'processing' if not ready yet -- retry later. Free (charged at "
        "queue time); not spend-gated.",
        _JOB_RESULT_SCHEMA,
        False,
    ),
]

# Extra paid tools NOT advertised by chat's default set. Folded in only when a
# caller passes `only=` (e.g. `venice code --assets`), so chat's default stays 8
# while `code_tools` can still select them by name.
_CODE_ASSET_BUILTINS = [
    (
        "venice_image_edit",
        "image_edit_tool",
        "Edit/inpaint an existing image via Venice /image/edit from a text prompt "
        "(base = a local input_path or an image_url; optional layer_paths route to "
        "/image/multi-edit for masks). Writes the result and returns its path. "
        "Dynamic pricing, so it always needs confirmation.",
        _IMAGE_EDIT_SCHEMA,
        True,
    ),
    (
        "venice_video",
        "video_tool",
        "Generate a short video via Venice's async video queue (blocks with a capped "
        "wait; can be slow). Optionally image-to-video from image_url. Writes an .mp4 "
        "and returns its path. Pass background=true to queue and return immediately, "
        "then fetch via venice_job_result. Dynamic pricing, so it always needs "
        "confirmation.",
        _VIDEO_SCHEMA,
        True,
    ),
]

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


def builtin_tools(
    client,
    *,
    max_spend: Optional[float] = None,
    output_dir: Optional[str] = None,
    only: Optional[set] = None,
    config: Optional[dict] = None,
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
    """

    def _config_defaults(section, impl) -> dict:
        if config is None:
            return {}
        section_map = userconfig._COMMAND_MAP.get(section)
        if not section_map:
            return {}
        try:
            params = set(inspect.signature(impl).parameters)
        except (TypeError, ValueError):
            return {}
        out: dict = {}
        for key, (dest, coerce) in section_map.items():
            if dest not in params:
                continue  # tool doesn't take this preference
            raw = userconfig.resolve_default(section, key, config)
            if raw is None:
                continue
            try:
                out[dest] = coerce(raw)
            except (TypeError, ValueError):
                pass  # a bad config value shouldn't break tool building
        return out

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
            name=name,
            description=desc,
            parameters=schema,
            invoke=(
                _make_paid(getattr(_mcp, impl_name), _tool_section(name))
                if paid
                else _make_free(getattr(_mcp, impl_name), _tool_section(name))
            ),
            paid=paid,
        )
        for (name, impl_name, desc, schema, paid) in source
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
    try:
        ans = input("Proceed? [y]es / [a]ll (accept rest) / [N]o ").strip().lower()
    except EOFError:
        return "no"
    if ans in ("a", "all"):
        return "all"
    if ans in ("y", "yes"):
        return "yes"
    return "no"


def _resolve_spend(tool: Tool, arguments: dict, result, gate: dict):
    """Hybrid gate: prompt on a TTY, else feed the block back to the model.

    `gate` is the run's mutable auto-accept holder (`{"auto": bool}`); answering
    `all` at the prompt sets ``gate["auto"] = True`` so subsequent paid calls in
    the same run skip the gate. Only reached for a paid tool that returned
    `confirmation_required` (which happens only while ``gate["auto"]`` is False).
    """
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
                return tool.invoke(arguments, confirm=True)
            except Exception as e:  # pragma: no cover - impls shouldn't raise
                return {"status": "error", "message": f"{tool.name} failed: {e}"}
        print(f"{tool.name}: declined by user", file=sys.stderr)
    return result  # non-TTY or declined -> the model sees the gate and adapts


def _run_one_call(tc, dispatch: Dict[str, Tool], gate: dict) -> dict:
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
        result = tool.invoke(arguments, confirm=bool(gate["auto"]))
    except Exception as e:  # pragma: no cover - impls shouldn't raise
        return {"status": "error", "message": f"{tool.name} failed: {e}"}
    return _resolve_spend(tool, arguments, result, gate)


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
    """
    oai_tools = to_openai_tools(tools)
    dispatch = dispatch_map(tools)
    calls_made = 0
    gate = {"auto": bool(yes)}  # mutable so an `a`/`all` confirm flips the run to auto
    # `--max-tool-calls 0` (or None) means unlimited -- run until the model stops
    # on its own (bounded in practice by the model's context window).
    unlimited = max_tool_calls is None or max_tool_calls <= 0
    show = not json_out  # progress feedback (further TTY-gated inside the helpers)

    def _force_final(reason: str) -> int:
        print(reason, file=sys.stderr)
        with _Spinner("finishing", enabled=show):
            resp = oai.chat.completions.create(
                model=model,
                messages=messages,
                tools=oai_tools,
                tool_choice="none",
                **base_kwargs,
            )
        if ledger is not None:
            ledger.record(getattr(resp, "usage", None))
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        return _emit_final(resp, json_out)

    while True:
        # Spend gate (#66): don't start a new paid turn once the cap is hit.
        if ledger is not None and ledger.over():
            return _force_final(
                f"chat: reached --max-spend ({ledger.summary()}); "
                "requesting a final answer"
            )
        _compact.maybe_compact(
            oai, model, messages, budget, base_kwargs,
            on_compact=lambda b, a: _progress(
                f"(auto-compacted history: {b} -> {a} messages)", enabled=show,
            ),
        )
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
            ledger.record(getattr(resp, "usage", None))
        msg = resp.choices[0].message if getattr(resp, "choices", None) else None
        messages.append(_assistant_dict(msg))
        tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
        if not tool_calls:
            return _emit_final(resp, json_out)

        # Every tool_call in the turn must get a result (message-contract), even
        # ones past the budget -- those are reported not-executed rather than run.
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
                result = _run_one_call(tc, dispatch, gate)
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
            )
            return _force_final(
                f"chat: reached --max-tool-calls ({max_tool_calls}); "
                "requesting a final answer"
            )
