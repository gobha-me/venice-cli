"""MCPServer wiring for `venice mcp-serve` -- the ONLY module that imports the mcp SDK.

Imported lazily from `commands.mcp_serve._run`, and only after the `[mcp]` extra is
confirmed present, so the base (stdlib-only) install and Python 3.9 never load it.
Most registered `venice_*` tools are thin, typed wrappers that delegate 1:1 to the
matching `commands._mcp.*_tool` implementation. The vision wrapper additionally
selects MCP-native content, but reuses the same pure input/delegation helpers. The
wrappers carry MCP schemas and LLM-facing docstrings; the impls carry print-free,
unit-tested logic.

Do NOT add `from __future__ import annotations` here: MCPServer builds each tool's
input schema via typing.get_type_hints, so the annotations must stay concrete
(`typing.Optional[int]`, not stringized `int | None`).
"""
import json
import os
from typing import Annotated, List, Literal, Optional

from mcp import types
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import userconfig
from .commands import _mcp, _shared, upscale as _upscale


def _merged(defaults: dict, host: dict) -> dict:
    """Layer config `defaults` UNDER host-supplied args (#58): an explicit (non-None)
    host value wins; where the host omitted an arg (MCPServer fills the wrapper's None
    default) the config default applies; keys the wrapper never exposes (e.g. image
    safe_mode/hide_watermark) come purely from config. Same precedence as
    `_agent._make_paid`'s `{**defaults, **_clean(arguments)}`."""
    return {**defaults, **{k: v for k, v in host.items() if v is not None}}


def _vision_text_result(result: dict) -> types.CallToolResult:
    """Convert the existing delegated result envelope to one MCP text block."""
    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(result))],
        is_error=result.get("status") == "error",
    )


def _native_vision_result(input_path, prompt, path_authority) -> types.CallToolResult:
    """Return one authorized local image as prompt text plus MCP ImageContent."""
    prepared = _mcp.prepare_vision_input(
        input_path, path_authority=path_authority
    )
    if prepared.get("status") != "ok":
        return _vision_text_result(prepared)

    try:
        header, data = prepared["image_url"].split(",", 1)
        mime, encoding = header[len("data:"):].split(";", 1)
        if (
            not header.startswith("data:")
            or encoding != "base64"
            or not mime.startswith("image/")
            or not data
        ):
            raise ValueError
    except (AttributeError, ValueError):
        return _vision_text_result({
            "status": "error",
            "message": "vision: authorized image could not be encoded",
        })
    return types.CallToolResult(content=[
        types.TextContent(text=(prompt or _mcp.DEFAULT_VISION_PROMPT).strip()),
        types.ImageContent(data=data, mime_type=mime),
    ])


def build_server(client, doc=None, root=None, host_image_content=False) -> MCPServer:
    """Build an MCPServer exposing venice tools, all bound to `client`.

    `doc` is a userconfig document (issue #58): `defaults.<section>.*` values are
    layered UNDER each host-supplied tool arg, so an explicit arg still wins
    (precedence: host arg > config default > tool hardcoded default) -- the same
    contract `venice chat`/`code` already honor. `doc=None` loads the config file.
    `host_image_content` is an operator declaration fixed at server startup; an
    absent declaration never permits an ImageContent result.
    """
    server = MCPServer("venice")
    path_authority = _shared.MediaPathAuthority(
        os.path.realpath(root or os.getcwd())
    )
    if doc is None:
        doc = userconfig.load_config()
    _defaults = {
        "image": userconfig.config_defaults_for("image", _mcp.image_tool, doc),
        "tts": userconfig.config_defaults_for("tts", _mcp.tts_tool, doc),
        "sfx": userconfig.config_defaults_for("sfx", _mcp.sfx_tool, doc),
        "music": userconfig.config_defaults_for("music", _mcp.music_tool, doc),
        "upscale": userconfig.config_defaults_for("upscale", _mcp.upscale_tool, doc),
        "bg_remove": userconfig.config_defaults_for("bg_remove", _mcp.bg_remove_tool, doc),
        "video": userconfig.config_defaults_for("video", _mcp.video_tool, doc),
        "image_edit": userconfig.config_defaults_for("image_edit", _mcp.image_edit_tool, doc),
        "chat": userconfig.config_defaults_for("chat", _mcp.chat_tool, doc),
        "vision": userconfig.config_defaults_for("vision", _mcp.vision_tool, doc),
    }
    _retired_upscale_config = _upscale.retired_config_keys(doc)

    @server.tool()
    def venice_image(
        prompt: str,
        model: Optional[str] = None,
        variants: Optional[int] = None,
        format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        steps: Optional[int] = None,
        style_preset: Optional[str] = None,
        style_references: Optional[List[dict]] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        embed_exif_metadata: Optional[bool] = None,
        lora_strength: Optional[int] = None,
        quality: Optional[str] = None,
        enable_web_search: Optional[bool] = None,
        disable_prompt_optimization_thinking: Optional[bool] = None,
        enhance_prompt: Optional[bool] = None,
        safe_mode: Optional[bool] = None,
        hide_watermark: Optional[bool] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate 1-4 image variants from a text prompt via Venice /image/generate.
        Writes PNG/WebP/JPEG file(s) and returns their paths (never inline blobs).
        safe_mode blurs adult content; hide_watermark removes the Venice watermark.
        Omitted args fall back to defaults.image.* in the user's config, then to the
        built-ins (model=venice-sd35, format=png, variants=1); variants multiplies
        the cost 1-4x, so omit it unless more than one image is wanted. Paid: the cost
        is estimated up front; if it is over the auto-approve cap the call returns
        status=confirmation_required and you must re-call with confirm=true."""
        return _mcp.image_tool(
            client, prompt,
            **_merged(_defaults["image"], dict(
                model=model, variants=variants, format=format,
                width=width, height=height, negative_prompt=negative_prompt,
                seed=seed, cfg_scale=cfg_scale, steps=steps,
                style_preset=style_preset, style_references=style_references,
                aspect_ratio=aspect_ratio, resolution=resolution,
                embed_exif_metadata=embed_exif_metadata,
                lora_strength=lora_strength, quality=quality,
                enable_web_search=enable_web_search,
                disable_prompt_optimization_thinking=(
                    disable_prompt_optimization_thinking
                ),
                enhance_prompt=enhance_prompt, safe_mode=safe_mode,
                hide_watermark=hide_watermark, output_dir=output_dir,
                confirm=confirm, max_spend=max_spend,
                path_authority=path_authority,
            )),
        )

    @server.tool()
    def venice_tts(
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        format: Optional[str] = None,
        speed: Optional[float] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Synthesize speech from text via Venice /audio/speech. Writes an audio file
        and returns its path. Paid: cost is estimated per character; over-cap calls
        need confirm=true. Omitted args fall back to defaults.tts.* in the user's
        config, then to model=tts-kokoro and the selected model's live catalog
        format default."""
        return _mcp.tts_tool(
            client, text,
            **_merged(_defaults["tts"], dict(
                model=model, voice=voice, format=format, speed=speed,
                output_dir=output_dir, confirm=confirm, max_spend=max_spend,
            )),
        )

    @server.tool()
    def venice_sfx(
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate a short sound effect via Venice's async audio queue (blocks with a
        capped wait until ready). Writes an audio file and returns its path. Paid: a
        quote is fetched first; over-cap quotes need confirm=true. Omitted args fall
        back to defaults.sfx.* in the user's config, then to the built-ins
        (model=elevenlabs-sound-effects-v2, duration=5)."""
        return _mcp.sfx_tool(
            client, prompt,
            **_merged(_defaults["sfx"], dict(
                model=model, duration=duration, output_dir=output_dir,
                confirm=confirm, max_spend=max_spend,
            )),
        )

    @server.tool()
    def venice_music(
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        instrumental: Optional[bool] = None,
        lyrics: Optional[str] = None,
        speed: Optional[float] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate long-form music/ambience (~60-90s) via Venice's async audio queue
        (blocks with a capped wait). Writes an audio file and returns its path. Paid:
        a quote is fetched first; over-cap quotes need confirm=true. An omitted model
        falls back to defaults.music.model, then to elevenlabs-music."""
        return _mcp.music_tool(
            client, prompt,
            **_merged(_defaults["music"], dict(
                model=model, duration=duration, instrumental=instrumental,
                lyrics=lyrics, speed=speed, output_dir=output_dir, confirm=confirm,
                max_spend=max_spend,
            )),
        )

    @server.tool()
    def venice_upscale(
        input_path: str,
        scale: Optional[Literal[2, 4]] = None,
        creativity: Optional[
            Annotated[float, Field(ge=_upscale.MIN_CREATIVITY,
                                   le=_upscale.MAX_CREATIVITY)]
        ] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Upscale a local image (factor 2 or 4) via Venice /image/upscale. Writes
        the result and returns its path. Pricing is dynamic (no up-front estimate), so
        this ALWAYS requires confirm=true. An omitted scale falls back to
        defaults.upscale.scale, then to 2.0."""
        if _retired_upscale_config:
            return {
                "status": "error",
                "message": _upscale.retired_config_message(_retired_upscale_config),
            }
        return _mcp.upscale_tool(
            client, input_path,
            **_merged(_defaults["upscale"], dict(
                scale=scale, creativity=creativity,
                output_dir=output_dir, confirm=confirm,
                max_spend=max_spend, path_authority=path_authority,
            )),
        )

    @server.tool()
    def venice_bg_remove(
        input_path: Optional[str] = None,
        image_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Remove an image's background via Venice /image/background-remove, returning a
        transparent PNG. Source is a local file (input_path) OR an image_url. Writes
        the result and returns its path. Dynamic pricing, so ALWAYS requires
        confirm=true."""
        return _mcp.bg_remove_tool(
            client, input_path,
            **_merged(_defaults["bg_remove"], dict(
                image_url=image_url, output_dir=output_dir,
                confirm=confirm, max_spend=max_spend,
                path_authority=path_authority,
            )),
        )

    @server.tool()
    def venice_chat(
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        web_search: Optional[str] = None,
        character: Optional[str] = None,
    ) -> dict:
        """One-shot chat completion via Venice /chat/completions; returns the reply
        text (and token usage when available). web_search is one of auto/on/off. Not
        spend-gated. Requires the [openai] extra."""
        return _mcp.chat_tool(
            client, message,
            **_merged(_defaults["chat"], dict(
                model=model, system=system, temperature=temperature,
                max_tokens=max_tokens, web_search=web_search, character=character,
            )),
        )

    @server.tool()
    def venice_vision(
        input_path: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[Literal["auto", "native", "delegate"]] = None,
    ) -> types.CallToolResult:
        """Inspect one image natively or through a delegated Venice vision model.

        Exactly one of input_path or image_url is required. mode=auto returns a
        validated local image as MCP ImageContent only when the operator started
        this stdio server with --host-image-content; otherwise it delegates and
        returns text. mode=native requires both that declaration and input_path.
        mode=delegate always delegates. model and max_tokens configure only the
        delegated path. Omitted args use defaults.vision.*, then mode=auto.
        """
        args = _merged(_defaults["vision"], dict(
            input_path=input_path, image_url=image_url, prompt=prompt,
            model=model, max_tokens=max_tokens, mode=mode,
        ))
        selected = args.pop("mode", None) or "auto"

        if selected == "native":
            if not host_image_content:
                return _vision_text_result({
                    "status": "error",
                    "message": (
                        "vision: native mode requires mcp-serve "
                        "--host-image-content"
                    ),
                })
            if args.get("image_url"):
                return _vision_text_result({
                    "status": "error",
                    "message": (
                        "vision: native MCP image content accepts only input_path; "
                        "use mode=delegate for image_url"
                    ),
                })
            return _native_vision_result(
                args.get("input_path"), args.get("prompt"), path_authority
            )

        if (
            selected == "auto"
            and host_image_content
            and args.get("input_path")
        ):
            return _native_vision_result(
                args.get("input_path"), args.get("prompt"), path_authority
            )

        return _vision_text_result(_mcp.vision_tool(
            client,
            **{**args, "path_authority": path_authority},
        ))

    @server.tool()
    def venice_video(
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        no_audio: Optional[bool] = None,
        image_url: Optional[str] = None,
        end_image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        audio_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        reference_video_urls: Optional[List[str]] = None,
        reference_audio_urls: Optional[List[str]] = None,
        scene_image_urls: Optional[List[str]] = None,
        reference_video_duration: Optional[float] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
        # #57 Class C2: Optional/None, NOT the concrete constant. `_merged` drops
        # only None, so a concrete default here is non-None on every call and
        # silently beats `defaults.video.max_wait` -- while the CLI path looks
        # perfectly correct. When neither host nor config supplies one,
        # `_mcp.video_tool`'s own signature default applies.
        max_wait: Optional[float] = None,
    ) -> dict:
        """Generate a video via Venice's async /video queue and return the file path.
        Text-to-video (prompt) plus optional image/reference conditioning: each *_url
        takes an http(s)/data URL or a local path. LONG-RUNNING -- blocks while polling
        up to max_wait seconds (a host may time out). Paid: a quote is fetched first;
        over-cap or dynamic quotes need confirm=true. Omitted args fall back to
        defaults.video.* in the user's config, then to the built-in duration=5s.
        Duration/resolution/aspect ratio are validated against the selected model's
        live catalog constraints before quote or spend."""
        return _mcp.video_tool(
            client, prompt,
            **_merged(_defaults["video"], dict(
                model=model, duration=duration,
                negative_prompt=negative_prompt, resolution=resolution,
                aspect_ratio=aspect_ratio, no_audio=no_audio, image_url=image_url,
                end_image_url=end_image_url, video_url=video_url, audio_url=audio_url,
                reference_image_urls=reference_image_urls,
                reference_video_urls=reference_video_urls,
                reference_audio_urls=reference_audio_urls,
                scene_image_urls=scene_image_urls,
                reference_video_duration=reference_video_duration,
                output_dir=output_dir, confirm=confirm, max_spend=max_spend,
                max_wait=max_wait, path_authority=path_authority,
            )),
        )

    @server.tool()
    def venice_image_edit(
        prompt: str,
        input_path: Optional[str] = None,
        image_url: Optional[str] = None,
        layer_paths: Optional[List[str]] = None,
        model: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        output_format: Optional[str] = None,
        quality: Optional[str] = None,
        disable_prompt_optimization_thinking: Optional[bool] = None,
        enhance_prompt: Optional[bool] = None,
        safe_mode: Optional[bool] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Edit/inpaint an image via Venice /image/edit and return the file path. Base
        image is a local input_path OR an image_url; repeatable layer_paths (masks/
        overlays) route to /image/multi-edit and use the model's live input limit.
        Pricing is dynamic (no up-front estimate),
        so this ALWAYS requires confirm=true."""
        return _mcp.image_edit_tool(
            client, prompt,
            **_merged(_defaults["image_edit"], dict(
                input_path=input_path, image_url=image_url,
                layer_paths=layer_paths, model=model, aspect_ratio=aspect_ratio,
                resolution=resolution, output_format=output_format,
                quality=quality,
                disable_prompt_optimization_thinking=(
                    disable_prompt_optimization_thinking
                ),
                enhance_prompt=enhance_prompt, safe_mode=safe_mode,
                output_dir=output_dir, confirm=confirm,
                max_spend=max_spend, path_authority=path_authority,
            )),
        )

    return server


def serve(client, doc=None, host_image_content=False) -> None:
    """Build the server and run it over stdio (blocks until the transport closes)."""
    build_server(
        client, doc=doc, host_image_content=host_image_content
    ).run(transport="stdio")
