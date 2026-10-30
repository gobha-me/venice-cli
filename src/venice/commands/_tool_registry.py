"""Static agent tool schemas and capability registry.

No MCP SDK, client, loop state, or HTTP calls belong here. The loop binds each
ToolSpec implementation name at runtime and owns confirmation/spend authority.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from . import _memory
from . import image as _image
from . import upscale as _upscale
from . import video as _video
from .models import MODEL_TYPES


def _p(typ: str, desc: Optional[str] = None) -> dict:
    """Describe one primitive tool argument."""
    d = {"type": typ}
    if desc:
        d["description"] = desc
    return d


def _obj(props: dict, required: Optional[list[str]] = None) -> dict:
    """Describe an object schema with explicit required fields."""
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
        "style_references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image": _p(
                        "string",
                        "Authorized local path, HTTP(S) URL, data URL, or raw base64.",
                    ),
                    "strength": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "description": "Optional model-supported strength.",
                    },
                },
                "required": ["image"],
                "additionalProperties": False,
            },
            "description": (
                "Style references, bounded by the selected live model catalog."
            ),
        },
        "aspect_ratio": _p("string"),
        "resolution": _p("string"),
        "embed_exif_metadata": _p("boolean"),
        "transparent_background": _p(
            "boolean", "Request a transparent background from supported models."
        ),
        "lora_strength": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "LoRA strength.",
        },
        "quality": {
            "type": "string",
            "enum": list(_image.QUALITY_CHOICES),
            "description": "Model-supported quality.",
        },
        "enable_web_search": _p(
            "boolean", "Allow supported models to use paid web search."
        ),
        "disable_prompt_optimization_thinking": _p("boolean"),
        "enhance_prompt": _p(
            "boolean", "Rewrite the prompt before generation; may add cost."
        ),
        "safe_mode": _p("boolean", "Blur adult/NSFW content. Defaults to on; set false to disable."),
        "hide_watermark": _p("boolean", "Omit the Venice watermark. Defaults to off; set true to hide it."),
    },
    required=["prompt"],
)

_TTS_SCHEMA = _obj(
    {
        "text": _p("string", "The text to speak."),
        "model": _p("string", "Live TTS catalog model id; defaults to tts-kokoro."),
        "voice": _p("string"),
        "format": _p(
            "string",
            "Model-specific audio format. Omit to use the model's catalog default.",
        ),
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
    },
    required=["queue_id", "type", "model"],
)

_JOB_RESULT_SCHEMA = _obj(
    {
        "queue_id": _p("string", "The queue_id from a background venice_sfx/music/video call."),
        "type": _p("string", "sfx, music, or video -- the tool that started the job."),
        "model": _p("string", "The model id from the job handle."),
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
        "scale": {
            "type": "number",
            "enum": list(_upscale.SCALES),
            "description": "Upscale factor: 2 or 4.",
        },
        "creativity": {
            "type": "number",
            "minimum": _upscale.MIN_CREATIVITY,
            "maximum": _upscale.MAX_CREATIVITY,
            "description": "Detail/texture creativity (server default 0.01).",
        },
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

#: The `venice_web_search` rail schema (#77). Deliberately minimal -- just the query.
#: The search mode and model are operator-controlled (flag/config), not model-facing, so
#: the model can't force search off or pick an arbitrary (possibly costly) model.
_WEB_SEARCH_SCHEMA = _obj(
    {
        "query": _p(
            "string",
            "What to look up on the web -- a question or search phrase. Returns a "
            "short answer plus the source URLs it cited.",
        ),
    },
    required=["query"],
)

_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Natural-language description of the code/text to find."),
        "k": _p("integer", "Number of results to return (default 8)."),
    },
    required=["query"],
)

_REINDEX_SCHEMA = _obj({})  # no parameters -- rebuilds the discovered .venice index

# Memory + task tools (#49). `scope` picks the tier (project rides the repo's
# .venice/, global travels with the agent); tasks are project-only (no scope).
_SCOPE_PROP = {
    "type": "string",
    "enum": ["project", "global"],
    "description": "Which memory tier: 'project' (default, rides the repo's .venice/ "
    "so subagents share it) or 'global' (user-global, travels with the agent).",
}
_TASK_STATUS_PROP = {
    "type": "string",
    "enum": list(_memory.TASK_STATUSES),
    "description": "Task status: pending, in_progress, or done.",
}
_MEMORY_WRITE_SCHEMA = _obj(
    {
        "name": _p("string", "Short slug id for the note (letters/digits/_.- only). "
                   "Reusing a name overwrites it."),
        "content": _p("string", "The note body to remember."),
        "scope": _SCOPE_PROP,
        "type": _p("string", "Optional kind, e.g. note/feedback/project/reference "
                   "(default: note)."),
        "description": _p("string", "Optional one-line summary shown in list/search."),
    },
    required=["name", "content"],
)
_MEMORY_READ_SCHEMA = _obj(
    {
        "name": _p("string", "The note's name."),
        "scope": _SCOPE_PROP,  # omit -> try project then global
    },
    required=["name"],
)
_MEMORY_SEARCH_SCHEMA = _obj(
    {
        "query": _p("string", "Substring to find in names/descriptions/bodies."),
        "scope": _SCOPE_PROP,  # omit -> search both tiers
    },
    required=["query"],
)
_MEMORY_LIST_SCHEMA = _obj({"scope": _SCOPE_PROP})  # omit -> both tiers; metadata only
_TASK_ADD_SCHEMA = _obj(
    {"text": _p("string", "What the task is.")},
    required=["text"],
)
_TASK_UPDATE_SCHEMA = _obj(
    {
        "id": _p("string", "The task id (from task_add/task_list)."),
        "status": _TASK_STATUS_PROP,
        "text": _p("string", "Optional new text for the task."),
    },
    required=["id"],
)
_TASK_LIST_SCHEMA = _obj({"status": _TASK_STATUS_PROP})  # omit -> all tasks

_MODELS_SCHEMA = _obj(
    {
        "type": {
            "type": "string",
            "enum": ["all", *MODEL_TYPES],
            "description": "Which catalog type to list model ids for "
            "(text, code, image, video, music, tts, embedding, upscale, asr, "
            "inpaint, decision), "
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

VISION_MODE_CHOICES = ("auto", "native", "delegate")

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
            "A delegated vision model id (used by delegate mode or auto fallback).",
        ),
        "max_tokens": _p("integer", "Maximum tokens for a delegated vision reply."),
        "mode": {
            "type": "string",
            "enum": list(VISION_MODE_CHOICES),
            "description": (
                "auto uses the active frontend when it advertises supportsVision, "
                "otherwise delegates; native requires a known vision-capable "
                "frontend; delegate always uses a separate vision model."
            ),
        },
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
            "description": (
                "Local mask/overlay images (routes to /image/multi-edit); "
                "the live model catalog supplies the count limit."
            ),
        },
        "model": _p("string", "Edit model id (default: the server picks one)."),
        "aspect_ratio": _p("string", "Output aspect ratio ('auto' infers from the input)."),
        "resolution": _p("string", "Output resolution tier, e.g. 1K/2K/4K."),
        "output_format": _p("string", "Output image format: png, jpeg, or webp."),
        "quality": {
            "type": "string",
            "enum": list(_image.QUALITY_CHOICES),
            "description": "Multi-edit quality.",
        },
        "disable_prompt_optimization_thinking": _p("boolean"),
        "enhance_prompt": _p("boolean", "Rewrite the edit prompt before generation."),
        "transparent_background": _p(
            "boolean", "Request a transparent background from supported models."
        ),
        "safe_mode": _p("boolean", "Blur adult/NSFW content. Defaults to on; set false to disable."),
    },
    required=["prompt"],
)

_VIDEO_SCHEMA = _obj(
    {
        "prompt": {
            "type": "string",
            "maxLength": _video.MAX_PROMPT,
            "description": "What the video should depict.",
        },
        "model": _p("string", "Video model id (default: the catalog default)."),
        "duration": _p("string", "Clip length, e.g. '5s'."),
        "negative_prompt": {"type": "string", "maxLength": _video.MAX_PROMPT},
        "resolution": _p("string", "Output resolution tier, e.g. 720p/1080p."),
        "aspect_ratio": _p("string", "Output aspect ratio, e.g. 16:9."),
        "no_audio": _p("boolean", "Generate silent video (no soundtrack)."),
        "image_url": _p("string", "Authorized local path or URL of a start image."),
        "image_strength": {
            "type": "number", "minimum": 0, "maximum": 1,
            "description": "Model-dependent image conditioning strength.",
        },
        "end_image_url": _p("string", "Authorized local path or URL of an end frame."),
        "video_url": _p("string", "Authorized local path or URL of an input video."),
        "audio_url": _p("string", "Authorized local path or URL of input audio."),
        "reference_image_urls": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _video.REF_IMAGE_MAX,
            "description": "Authorized local paths or URLs of reference images.",
        },
        "reference_video_urls": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _video.REF_VIDEO_MAX,
            "description": "Authorized local paths or URLs of reference videos.",
        },
        "reference_audio_urls": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _video.REF_AUDIO_MAX,
            "description": "Authorized local paths or URLs of reference audio clips.",
        },
        "scene_image_urls": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _video.SCENE_MAX,
            "description": "Authorized local paths or URLs of scene images.",
        },
        "reference_video_duration": _p(
            "number", "Aggregate reference-video duration used for the quote."
        ),
        "background": _BACKGROUND_PARAM,
    },
    required=["prompt"],
)

# Browser/web tools (#71). Rails like `shell`: the URL allow/deny policy is bound by the
# wiring, so `allow`/`deny` are DELIBERATELY absent from these schemas -- the model can't
# widen its own reach (mirrors how confirm/max_spend/output_dir are loop-injected).
_WEB_FETCH_SCHEMA = _obj(
    {
        "url": _p("string", "The http(s) URL to fetch."),
        "mode": _p("string", "text (default; HTML tags stripped) or html (raw)."),
        "max_bytes": _p("integer", "Cap on bytes downloaded."),
        "timeout": _p("integer", "Timeout in seconds."),
    },
    required=["url"],
)
_BROWSER_CAPTURE_SCHEMA = _obj(
    {
        "url": _p("string", "The http(s) URL to render."),
        "mode": _p(
            "string",
            "dom (default: post-JS HTML), text (DOM stripped to text), screenshot "
            "(writes a PNG, returns its path), or both. dom/text/both need a "
            "Chromium-family browser; Firefox is screenshot-only.",
        ),
        "wait_ms": _p("integer", "Milliseconds for JS to settle before capture."),
        "assert_contains": _p(
            "string",
            "Substring to check for in the rendered DOM; returns contains:true/false -- "
            "a deterministic 'did the JS land' check (dom/text/both modes).",
        ),
        "timeout": _p("integer", "Timeout in seconds."),
    },
    required=["url"],
)


class ToolSpec(NamedTuple):
    """One built-in tool's registry row (#50).

    The impl is stored by NAME and resolved via ``getattr(_mcp, impl)`` at
    :func:`builtin_tools` time, so a single source of truth wins and tests can patch
    ``_mcp.<impl>``. ``category`` (single, required) + optional ``tags`` are the
    capability axis the composition API (:func:`select`/:func:`tools_in`) reads.
    """

    name: str
    impl: str
    description: str
    parameters: dict
    paid: bool
    category: str
    tags: tuple[str, ...] = ()


# The built-in venice tools. ``category`` reproduces the hand-maintained `only=`
# sets `code_tools` used to pass (see :func:`select`); it is ORTHOGONAL to the
# `_BUILTINS` vs `_CODE_ASSET_BUILTINS` split, which is the surface/advertisement
# axis (what `venice chat` shows by default).
_BUILTINS = [
    ToolSpec(
        "venice_image",
        "image_tool",
        "Generate 1-4 image variants from a text prompt via Venice /image/generate. "
        "Writes image file(s) and returns their paths (never inline blobs). Paid: "
        "over-cap calls need confirmation.",
        _IMAGE_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_tts",
        "tts_tool",
        "Synthesize speech from text via Venice /audio/speech. Writes an audio file "
        "and returns its path. Paid.",
        _TTS_SCHEMA,
        True,
        "audio",
    ),
    ToolSpec(
        "venice_sfx",
        "sfx_tool",
        "Generate a short sound effect via Venice's async audio queue (blocks with a "
        "capped wait). Writes an audio file and returns its path. Pass background=true "
        "to queue and return immediately, then fetch via venice_job_result. Paid.",
        _SFX_SCHEMA,
        True,
        "audio",
    ),
    ToolSpec(
        "venice_music",
        "music_tool",
        "Generate long-form music/ambience via Venice's async audio queue (blocks "
        "with a capped wait). Writes an audio file and returns its path. Pass "
        "background=true to queue and return immediately, then fetch via "
        "venice_job_result. Paid.",
        _MUSIC_SCHEMA,
        True,
        "audio",
    ),
    ToolSpec(
        "venice_upscale",
        "upscale_tool",
        "Upscale a local image (factor 2 or 4) via Venice /image/upscale. Writes "
        "the result and returns its path. Dynamic pricing, so it always needs "
        "confirmation.",
        _UPSCALE_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_bg_remove",
        "bg_remove_tool",
        "Remove an image's background via Venice /image/background-remove, returning a "
        "transparent PNG. Source is a local input_path OR an image_url. Dynamic "
        "pricing, so it always needs confirmation.",
        _BG_REMOVE_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_chat",
        "chat_tool",
        "Delegate a one-shot sub-completion to a Venice text model (optionally a "
        "different model or character) and return its reply text. Not spend-gated.",
        _CHAT_SCHEMA,
        False,
        "text",
    ),
    ToolSpec(
        "venice_models",
        "models_tool",
        "List available Venice model ids for a catalog type (text/code/image/video/"
        "music/tts/embedding/upscale/asr/inpaint/decision, or 'all') via the free "
        "/models catalog. Use it "
        "to choose a valid `model` for the other venice_* tools instead of guessing. "
        "Read-only; not spend-gated.",
        _MODELS_SCHEMA,
        False,
        "catalog",
    ),
    ToolSpec(
        "venice_model_details",
        "model_details_tool",
        "Get one model's details: pricing (cost), capabilities (text models: "
        "supportsVision/supportsFunctionCalling/...), constraints (image/media "
        "models: aspectRatios, resolutions, qualities, promptCharacterLimit), and "
        "voices and formats (TTS models: valid voice ids plus supported/default "
        "formats for venice_tts) -- plus "
        "the full model_spec. Use it to budget input and confirm a model fits before "
        "using it. Read-only; not spend-gated.",
        _MODEL_DETAILS_SCHEMA,
        False,
        "catalog",
    ),
    ToolSpec(
        "venice_vision",
        "vision_tool",
        "Look at an image (a local input_path OR an image_url). mode=auto (default) "
        "attaches it to the active frontend when that model advertises supportsVision "
        "and otherwise delegates to a vision model; mode=native or mode=delegate "
        "forces either path. Optional prompt directs the question. Not spend-gated.",
        _VISION_SCHEMA,
        False,
        "vision",
    ),
    ToolSpec(
        "project_search",
        "search_tool",
        "Semantic search over the current project's local .venice index (built by "
        "`venice index`) for the chunks most relevant to a natural-language query. "
        "Returns file paths with line ranges and a short preview -- use it to locate "
        "code by meaning before reading files. Read-only; not spend-gated. Errors if "
        "no index exists yet. NOTE: results are a SNAPSHOT of the last index build; "
        "call reindex after editing files, or use grep for live matches.",
        _SEARCH_SCHEMA,
        False,
        "search",
    ),
    ToolSpec(
        "reindex",
        "reindex_tool",
        "Rebuild the project's .venice index so project_search reflects edits made "
        "this session (project_search is a snapshot; grep is live). Re-embeds only "
        "files whose contents changed, reusing the index's existing embedding "
        "backend. Takes no arguments. Paid (embeds changed files) -- always needs "
        "confirmation. Errors if no index exists yet (run `venice index` first).",
        _REINDEX_SCHEMA,
        True,
        "search",
    ),
    ToolSpec(
        "venice_job_status",
        "job_status_tool",
        "Peek at a backgrounded media render started with background=true on "
        "venice_sfx/venice_music/venice_video. Pass back the job handle's queue_id, "
        "type (sfx/music/video), and model. Returns processing/done/failed/not_found. "
        "Read-only, non-blocking; not spend-gated.",
        _JOB_STATUS_SCHEMA,
        False,
        "jobs",
    ),
    ToolSpec(
        "venice_job_result",
        "job_result_tool",
        "Fetch a backgrounded media render's file once ready (started with "
        "background=true). Pass back the job handle's queue_id, type, and model; "
        "private VPS retrieval metadata is resolved locally. Writes the file and "
        "returns its path, or "
        "status 'processing' if not ready yet -- retry later. Free (charged at "
        "queue time); not spend-gated.",
        _JOB_RESULT_SCHEMA,
        False,
        "jobs",
    ),
]

# Extra paid tools NOT advertised by chat's default set. Folded in only when a
# caller passes `only=` (e.g. `venice code --assets`), so chat's default stays 8
# while `code_tools` can still select them by name.
_CODE_ASSET_BUILTINS = [
    ToolSpec(
        "venice_image_edit",
        "image_edit_tool",
        "Edit/inpaint an existing image via Venice /image/edit from a text prompt "
        "(base = a local input_path or an image_url; optional layer_paths route to "
        "/image/multi-edit for masks). Writes the result and returns its path. "
        "Dynamic pricing, so it always needs confirmation.",
        _IMAGE_EDIT_SCHEMA,
        True,
        "image",
    ),
    ToolSpec(
        "venice_video",
        "video_tool",
        "Generate a short video via Venice's async video queue (blocks with a capped "
        "wait; can be slow). Optionally image-to-video from image_url. Writes an .mp4 "
        "and returns its path. Pass background=true to queue and return immediately, "
        "then fetch via venice_job_result. Dynamic pricing, so it always needs "
        "confirmation.",
        _VIDEO_SCHEMA,
        True,
        "video",
    ),
]


# --------------------------------------------------------------------------- #
# Composition API over the built-in registry (#50)
#
# `category` is the capability axis: it reproduces the hand-maintained `only=`
# name-sets `code_tools` used to pass, so a caller selects tools by capability
# instead of enumerating names. Read over the UNION of both registries so a
# `_CODE_ASSET_BUILTINS`-only tool (venice_image_edit/venice_video) is selectable
# by its category. This is orthogonal to the `_BUILTINS`/`_CODE_ASSET_BUILTINS`
# split, which stays the surface/advertisement axis for `builtin_tools(only=None)`.
# --------------------------------------------------------------------------- #
_REGISTRY = _BUILTINS + _CODE_ASSET_BUILTINS


def get(name: str) -> Optional[ToolSpec]:
    """The registry row for `name` (metadata only, no client), or None."""
    for spec in _REGISTRY:
        if spec.name == name:
            return spec
    return None


def list_categories() -> set:
    """Every category present in the built-in registry."""
    return {spec.category for spec in _REGISTRY}


def tools_in(category: str) -> set:
    """The names of registry tools in `category` (empty set if none)."""
    return {spec.name for spec in _REGISTRY if spec.category == category}


def select(categories=None, names=None, exclude=None) -> set:
    """A set of built-in tool names selected by capability.

    `categories` and/or `names` union into the selection (both None selects the
    whole registry); `exclude` (names or categories) is subtracted last. Unknown
    categories/names are simply ignored here -- the authoritative unknown-name guard
    stays in :func:`builtin_tools` (whose ValueError drives chat's exit 2), so this
    stays a pure name-set helper the `code_tools` call sites can compose with.
    """
    chosen = set()
    if categories is None and names is None:
        chosen = {spec.name for spec in _REGISTRY}
    else:
        if categories:
            for cat in categories:
                chosen |= tools_in(cat)
        if names:
            known = {spec.name for spec in _REGISTRY}
            chosen |= {n for n in names if n in known}
    if exclude:
        exclude = set(exclude)
        chosen -= exclude
        chosen -= {spec.name for spec in _REGISTRY if spec.category in exclude}
    return chosen
