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
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .commands import _mcp


def build_server(client) -> FastMCP:
    """Build a FastMCP server exposing venice tools, all bound to `client`."""
    server = FastMCP("venice")

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
        output_dir: Optional[str] = None,
        confirm: bool = False,
        max_spend: Optional[float] = None,
    ) -> dict:
        """Generate 1-4 image variants from a text prompt via Venice /image/generate.
        Writes PNG/WebP/JPEG file(s) and returns their paths (never inline blobs).
        Paid: the cost is estimated up front; if it is over the auto-approve cap the
        call returns status=confirmation_required and you must re-call with
        confirm=true."""
        return _mcp.image_tool(
            client, prompt, model=model, variants=variants, format=format,
            width=width, height=height, negative_prompt=negative_prompt, seed=seed,
            cfg_scale=cfg_scale, steps=steps, style_preset=style_preset,
            output_dir=output_dir, confirm=confirm, max_spend=max_spend,
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
            client, text, model=model, voice=voice, format=format, speed=speed,
            output_dir=output_dir, confirm=confirm, max_spend=max_spend,
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
            client, prompt, model=model, duration=duration, output_dir=output_dir,
            confirm=confirm, max_spend=max_spend,
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
            client, prompt, model=model, duration=duration, instrumental=instrumental,
            lyrics=lyrics, speed=speed, output_dir=output_dir, confirm=confirm,
            max_spend=max_spend,
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
            client, input_path, scale=scale, enhance=enhance,
            enhance_creativity=enhance_creativity, enhance_prompt=enhance_prompt,
            replication=replication, output_dir=output_dir, confirm=confirm,
            max_spend=max_spend,
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
            client, input_path, image_url=image_url, output_dir=output_dir,
            confirm=confirm, max_spend=max_spend,
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
            client, message, model=model, system=system, temperature=temperature,
            max_tokens=max_tokens, web_search=web_search, character=character,
        )

    return server


def serve(client) -> None:
    """Build the server and run it over stdio (blocks until the transport closes)."""
    build_server(client).run(transport="stdio")
