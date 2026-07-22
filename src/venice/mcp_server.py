"""FastMCP wiring for `venice mcp-serve` -- the ONLY module that imports the mcp SDK.

Imported lazily from `commands.mcp_serve._run`, and only after the `[mcp]` extra is
confirmed present, so the base (stdlib-only) install and Python 3.9 never load it.
Each registered `venice_*` tool is a thin, typed wrapper that delegates 1:1 to the
matching `commands._mcp.*_tool` implementation -- the wrapper carries the MCP schema
and the LLM-facing docstring; the impl carries the (print-free, unit-tested) logic.

Do NOT add `from __future__ import annotations` here: FastMCP builds each tool's
input schema via typing.get_type_hints, so the annotations must stay concrete
(`typing.Optional[int]`, not stringized `int | None`).
"""
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from . import userconfig
from .commands import _mcp


def _merged(defaults: dict, host: dict) -> dict:
    """Layer config `defaults` UNDER host-supplied args (#58): an explicit (non-None)
    host value wins; where the host omitted an arg (FastMCP fills the wrapper's None
    default) the config default applies; keys the wrapper never exposes (e.g. image
    safe_mode/hide_watermark) come purely from config. Same precedence as
    `_agent._make_paid`'s `{**defaults, **_clean(arguments)}`."""
    return {**defaults, **{k: v for k, v in host.items() if v is not None}}


def build_server(client, doc=None) -> FastMCP:
    """Build a FastMCP server exposing venice tools, all bound to `client`.

    `doc` is a userconfig document (issue #58): `defaults.<section>.*` values are
    layered UNDER each host-supplied tool arg, so an explicit arg still wins
    (precedence: host arg > config default > tool hardcoded default) -- the same
    contract `venice chat`/`code` already honor. `doc=None` loads the config file.
    """
    server = FastMCP("venice")
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
    }

    @server.tool()
    def venice_image(
        prompt: str,
        model: str = _mcp._image.DEFAULT_IMAGE_MODEL,
        variants: int = 1,
        format: str = _mcp._image.DEFAULT_FORMAT,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        steps: Optional[int] = None,
        style_preset: Optional[str] = None,
        safe_mode: Optional[bool] = None,
        hide_watermark: Optional[bool] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate 1-4 image variants from a text prompt via Venice /image/generate.
        Writes PNG/WebP/JPEG file(s) and returns their paths (never inline blobs).
        safe_mode blurs adult content; hide_watermark removes the Venice watermark
        (both fall back to your config defaults.image.* when omitted). Paid: the cost
        is estimated up front; if it is over the auto-approve cap the call returns
        status=confirmation_required and you must re-call with confirm=true."""
        return _mcp.image_tool(
            client, prompt,
            **_merged(_defaults["image"], dict(
                model=model, variants=variants, format=format,
                width=width, height=height, negative_prompt=negative_prompt,
                seed=seed, cfg_scale=cfg_scale, steps=steps,
                style_preset=style_preset, safe_mode=safe_mode,
                hide_watermark=hide_watermark, output_dir=output_dir,
                confirm=confirm, max_spend=max_spend,
            )),
        )

    @server.tool()
    def venice_tts(
        text: str,
        model: str = _mcp._tts.DEFAULT_TTS_MODEL,
        voice: Optional[str] = None,
        format: str = _mcp._tts.DEFAULT_FORMAT,
        speed: Optional[float] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Synthesize speech from text via Venice /audio/speech. Writes an audio file
        and returns its path. Paid: cost is estimated per character; over-cap calls
        need confirm=true."""
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
        model: str = _mcp._sfx.DEFAULT_SFX_MODEL,
        duration: int = _mcp._sfx.DEFAULT_DURATION,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate a short sound effect via Venice's async audio queue (blocks with a
        capped wait until ready). Writes an audio file and returns its path. Paid: a
        quote is fetched first; over-cap quotes need confirm=true."""
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
        model: str = _mcp._music.DEFAULT_MUSIC_MODEL,
        duration: Optional[int] = None,
        instrumental: bool = False,
        lyrics: Optional[str] = None,
        speed: Optional[float] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate long-form music/ambience (~60-90s) via Venice's async audio queue
        (blocks with a capped wait). Writes an audio file and returns its path. Paid:
        a quote is fetched first; over-cap quotes need confirm=true."""
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
        scale: float = 2.0,
        enhance: bool = False,
        enhance_creativity: Optional[float] = None,
        enhance_prompt: Optional[str] = None,
        replication: Optional[float] = None,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Upscale/enhance a local image (factor 1-4) via Venice /image/upscale. Writes
        the result and returns its path. Pricing is dynamic (no up-front estimate), so
        this ALWAYS requires confirm=true."""
        return _mcp.upscale_tool(
            client, input_path,
            **_merged(_defaults["upscale"], dict(
                scale=scale, enhance=enhance,
                enhance_creativity=enhance_creativity, enhance_prompt=enhance_prompt,
                replication=replication, output_dir=output_dir, confirm=confirm,
                max_spend=max_spend,
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
    def venice_video(
        prompt: str,
        model: Optional[str] = None,
        duration: str = _mcp._video.DEFAULT_VIDEO_DURATION,
        negative_prompt: Optional[str] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        no_audio: bool = False,
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
        max_wait: float = _mcp._video.config.VIDEO_POLL_MAX_WAIT_SEC,
    ) -> dict:
        """Generate a video via Venice's async /video queue and return the file path.
        Text-to-video (prompt) plus optional image/reference conditioning: each *_url
        takes an http(s)/data URL or a local path. LONG-RUNNING -- blocks while polling
        up to max_wait seconds (a host may time out). Paid: a quote is fetched first;
        over-cap or dynamic quotes need confirm=true."""
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
                max_wait=max_wait,
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
        no_safe_mode: bool = False,
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Edit/inpaint an image via Venice /image/edit and return the file path. Base
        image is a local input_path OR an image_url; one or two layer_paths (masks/
        overlays) route to /image/multi-edit. Pricing is dynamic (no up-front estimate),
        so this ALWAYS requires confirm=true."""
        return _mcp.image_edit_tool(
            client, prompt,
            **_merged(_defaults["image_edit"], dict(
                input_path=input_path, image_url=image_url,
                layer_paths=layer_paths, model=model, aspect_ratio=aspect_ratio,
                resolution=resolution, output_format=output_format,
                no_safe_mode=no_safe_mode, output_dir=output_dir, confirm=confirm,
                max_spend=max_spend,
            )),
        )

    return server


def serve(client, doc=None) -> None:
    """Build the server and run it over stdio (blocks until the transport closes)."""
    build_server(client, doc=doc).run(transport="stdio")
