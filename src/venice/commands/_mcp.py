"""MCP tool implementations for `venice mcp-serve` (issue #14, Direction A of #16).

Import-clean by design: this module must NOT import `mcp` (or `openai`) at module
scope, so `venice --help` and the base, stdlib-only install keep working, and the
tests here run on Python 3.9 where the `mcp` SDK cannot even be installed. The thin
FastMCP wiring lives in `venice.mcp_server`; everything with real logic lives here:
the lazy `import_mcp` probe, the spend gate, the output-dir resolver, and the seven
print-free `*_tool` functions the server delegates to.

CRITICAL invariant -- an MCP stdio server owns **stdout** for JSON-RPC framing, so
every function here is *print-free*: it composes the print-free client/command
primitives (`poll_retrieve`, `_build_body`, `_decode_images`, price lookups, ...),
writes any output file itself, and returns a structured dict. It never writes to
stdout and never raises to the host -- API/validation problems come back as
`{"status": "error", ...}`. (Reused helpers that write to *stderr* -- clamp/validate
warnings, progress ticks -- are fine; stderr is not the transport.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from ..client import VeniceAPIError
from . import _audio, _index, _models, _openai, _queue, _shared
from . import bg_remove as _bg
from . import chat as _chat
from . import image as _image
from . import image_edit as _image_edit
from . import models as _models_cmd
from . import music as _music
from . import sfx as _sfx
from . import tts as _tts
from . import upscale as _upscale
from . import video as _video


# ---- lazy SDK probe (mirrors _openai.import_openai) --------------------------

def import_mcp(label: str):
    """Import the mcp SDK lazily. None (after printing a hint) if absent.

    `label` names the command in the hint (e.g. "mcp-serve"). The hint notes the
    Python >=3.10 floor because the `[mcp]` extra is gated to 3.10+.
    """
    try:
        import mcp
    except ImportError:
        print(
            f"venice {label} needs the mcp package (Python >=3.10): "
            'pip install "venice-cli[mcp]" (or: pip install mcp)',
            file=sys.stderr,
        )
        return None
    return mcp


# ---- spend gate --------------------------------------------------------------

DEFAULT_MCP_MAX_SPEND = 0.10  # USD: auto-approve ceiling for a single tool call


def resolve_max_spend(max_spend: Optional[float]) -> float:
    """Cap precedence: explicit arg -> $VENICE_MCP_MAX_SPEND -> DEFAULT."""
    if max_spend is not None:
        try:
            return float(max_spend)
        except (TypeError, ValueError):
            pass
    env = os.environ.get("VENICE_MCP_MAX_SPEND")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return DEFAULT_MCP_MAX_SPEND


def check_spend(
    cost: Optional[float], *, confirm: bool, max_spend: Optional[float], label: str
) -> Optional[dict]:
    """Gate a paid call. None => proceed; dict => the host must re-call with confirm.

    Auto-approves a known cost that is <= the cap. Requires `confirm=true` when the
    cost is over the cap OR unknown (dynamic-priced upscale/bg-remove). Reuses
    `_shared.over_budget` for the comparison.
    """
    if confirm:
        return None
    cap = resolve_max_spend(max_spend)
    if cost is None or _shared.over_budget(cost, cap):
        shown = f"${cost:.4f}" if cost is not None else "unknown"
        return {
            "status": "confirmation_required",
            "estimated_cost_usd": cost,
            "max_spend_usd": cap,
            "message": (
                f"{label}: estimated cost {shown} is over the auto-approve cap of "
                f"${cap:.4f} (or could not be estimated). Re-call with confirm=true "
                "to proceed, or raise the cap via the max_spend argument / "
                "VENICE_MCP_MAX_SPEND."
            ),
        }
    return None


# ---- output + result helpers -------------------------------------------------

def resolve_output_dir(output_dir: Optional[str]) -> Path:
    """Where tools write files: arg -> $VENICE_MCP_OUTPUT_DIR -> cwd."""
    return Path(output_dir or os.environ.get("VENICE_MCP_OUTPUT_DIR") or os.getcwd())


def _write(path: Path, data: bytes) -> Optional[str]:
    """Write bytes to path (creating parents). Returns an error string, else None."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as e:
        return str(e)
    return None


def _err(message: str) -> dict:
    return {"status": "error", "message": message}


# ---- tools -------------------------------------------------------------------

def image_tool(
    client,
    prompt: str,
    *,
    model: str = _image.DEFAULT_IMAGE_MODEL,
    variants: int = 1,
    format: str = _image.DEFAULT_FORMAT,
    width: Optional[int] = None,
    height: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    style_preset: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    safe_mode: bool = True,
    hide_watermark: bool = False,
    output_dir: Optional[str] = None,
    confirm: bool = False,
    max_spend: Optional[float] = None,
) -> dict:
    """Generate 1-4 image variants via /image/generate; write files, return paths."""
    if not prompt or not prompt.strip():
        return _err("image: prompt is required")
    if not (_image.MIN_VARIANTS <= variants <= _image.MAX_VARIANTS):
        return _err(
            f"image: variants {variants} out of range "
            f"({_image.MIN_VARIANTS}-{_image.MAX_VARIANTS})"
        )
    if format not in _image.FORMATS:
        return _err(
            f"image: unknown format {format!r}; choose from {', '.join(_image.FORMATS)}"
        )

    prompt = prompt.strip()
    price = _image._fetch_image_price(client, model)
    cost = _image._estimate_cost(price, variants)
    gate = check_spend(cost, confirm=confirm, max_spend=max_spend, label="image")
    if gate is not None:
        return gate

    ns = SimpleNamespace(
        model=model, prompt=prompt, format=format, safe_mode=safe_mode,
        hide_watermark=hide_watermark, variants=variants, width=width, height=height,
        aspect_ratio=aspect_ratio, resolution=resolution, negative_prompt=negative_prompt,
        seed=seed, cfg_scale=cfg_scale, steps=steps, style_preset=style_preset,
        style_prefix=None,
    )
    body = _image._build_body(prompt, ns)
    try:
        doc = client.post_json("/image/generate", body)
    except VeniceAPIError as e:
        return _err(f"image failed: {e}")

    images = _image._decode_images(doc)
    if not images:
        return _err("image: server returned no images")

    out_dir = resolve_output_dir(output_dir)
    ext = _image.EXT_BY_FORMAT.get(format, ".bin")
    base = f"venice-image-{_image._short_id(prompt, model, seed)}"
    total = len(images)
    paths: List[str] = []
    for i, data in enumerate(images):
        p = _image._variant_path(out_dir, base, i + 1, total, ext)
        werr = _write(p, data)
        if werr:
            return _err(f"image: could not write {p}: {werr}")
        paths.append(str(p.resolve()))
    return {
        "status": "ok",
        "paths": paths,
        "count": total,
        "bytes": sum(len(d) for d in images),
        "model": model,
        "cost_estimate_usd": cost,
    }


def tts_tool(
    client,
    text: str,
    *,
    model: str = _tts.DEFAULT_TTS_MODEL,
    voice: Optional[str] = None,
    format: str = _tts.DEFAULT_FORMAT,
    speed: Optional[float] = None,
    output_dir: Optional[str] = None,
    confirm: bool = False,
    max_spend: Optional[float] = None,
) -> dict:
    """Synthesize speech via /audio/speech; write an audio file, return its path."""
    if not text or not text.strip():
        return _err("tts: text is required")
    if model not in _tts.TTS_MODELS:
        return _err(f"tts: unknown model {model!r}")
    if format not in _tts.FORMATS:
        return _err(
            f"tts: unknown format {format!r}; choose from {', '.join(_tts.FORMATS)}"
        )
    if speed is not None and not (0.25 <= speed <= 4.0):
        return _err(f"tts: speed {speed} out of range (0.25-4.0)")

    text = text.strip()
    price = _tts._fetch_tts_price_per_million(client, model)
    cost = _tts._estimate_cost(len(text), price)
    gate = check_spend(cost, confirm=confirm, max_spend=max_spend, label="tts")
    if gate is not None:
        return gate

    body: dict = {"input": text, "model": model, "response_format": format}
    if voice:
        body["voice"] = voice
    if speed is not None:
        body["speed"] = speed
    try:
        _status, _ctype, audio = client.request("POST", "/audio/speech", json_body=body)
    except VeniceAPIError as e:
        return _err(f"tts failed: {e}")
    if not audio:
        return _err("tts: server returned empty body")

    short = _tts._short_id(text, model, voice)
    out_path = _tts._resolve_output_path(resolve_output_dir(output_dir), short, format)
    werr = _write(out_path, audio)
    if werr:
        return _err(f"tts: could not write {out_path}: {werr}")
    return {
        "status": "ok",
        "path": str(out_path.resolve()),
        "bytes": len(audio),
        "model": model,
        "cost_estimate_usd": cost,
    }


def _queue_media(
    client,
    *,
    model: str,
    queue_body: dict,
    quote_value,
    confirm: bool,
    max_spend: Optional[float],
    label: str,
    name_prefix: str,
    output_dir: Optional[str],
    max_wait: float,
) -> dict:
    """Shared quote-gated queue -> poll -> save for sfx/music (print-free).

    The caller has already fetched the quote (`quote_value`) and built the
    `/audio/queue` body; here we gate on spend, queue, poll via the print-free
    `_audio.retrieve_bytes`, write the file, and best-effort `/audio/complete`.
    """
    gate = check_spend(quote_value, confirm=confirm, max_spend=max_spend, label=label)
    if gate is not None:
        return gate

    try:
        queued = client.post_json("/audio/queue", queue_body)
    except VeniceAPIError as e:
        return _err(f"{label} queue failed: {e}")
    queue_id = queued.get("queue_id") or queued.get("id") or ""
    if not queue_id:
        return _err(f"{label}: queue response missing queue_id")

    try:
        ctype, audio = _audio.retrieve_bytes(
            client, model, queue_id,
            poll_interval=_sfx.config.SFX_POLL_INTERVAL_SEC, max_wait=max_wait,
        )
    except VeniceAPIError as e:
        return _err(f"{label} retrieve failed: {e}")
    except TimeoutError as e:
        return _err(f"{label}: {e}; the job {queue_id} may still finish server-side")

    ext, _unknown = _audio.ext_for(ctype)
    out_path = _queue.resolve_output_path(
        resolve_output_dir(output_dir), queue_id, ext, prefix=name_prefix
    )
    werr = _write(out_path, audio)
    if werr:
        return _err(f"{label}: could not write {out_path}: {werr}")

    try:  # best-effort cleanup; the file is already saved
        client.post_json("/audio/complete", {"model": model, "queue_id": queue_id})
    except VeniceAPIError:
        pass
    return {
        "status": "ok",
        "path": str(out_path.resolve()),
        "bytes": len(audio),
        "model": model,
        "queue_id": queue_id,
        "cost_estimate_usd": quote_value,
    }


def sfx_tool(
    client,
    prompt: str,
    *,
    model: str = _sfx.DEFAULT_SFX_MODEL,
    duration: int = _sfx.DEFAULT_DURATION,
    output_dir: Optional[str] = None,
    confirm: bool = False,
    max_spend: Optional[float] = None,
    max_wait: float = _sfx.config.SFX_POLL_MAX_WAIT_SEC,
) -> dict:
    """Generate a sound effect via the async audio queue; write a file, return path."""
    if not prompt or not prompt.strip():
        return _err("sfx: prompt is required")
    if model not in _sfx.SFX_MODELS:
        return _err(f"sfx: unknown model {model!r}; choose from {', '.join(sorted(_sfx.SFX_MODELS))}")

    duration = _sfx._clamp_duration(model, duration)  # stderr warnings only
    try:
        quote = client.post_json(
            "/audio/quote", {"model": model, "duration_seconds": duration}
        )
    except VeniceAPIError as e:
        return _err(f"sfx quote rejected: {e}")
    quote_value = quote.get("quote", quote)

    return _queue_media(
        client,
        model=model,
        queue_body={"model": model, "prompt": prompt.strip(), "duration_seconds": duration},
        quote_value=quote_value,
        confirm=confirm,
        max_spend=max_spend,
        label="sfx",
        name_prefix="venice-sfx",
        output_dir=output_dir,
        max_wait=max_wait,
    )


def music_tool(
    client,
    prompt: str,
    *,
    model: str = _music.DEFAULT_MUSIC_MODEL,
    duration: Optional[int] = None,
    instrumental: bool = False,
    lyrics: Optional[str] = None,
    speed: Optional[float] = None,
    output_dir: Optional[str] = None,
    confirm: bool = False,
    max_spend: Optional[float] = None,
    max_wait: float = _music.config.SFX_POLL_MAX_WAIT_SEC,
) -> dict:
    """Generate long-form music/ambience via the async audio queue; return the path."""
    if not prompt or not prompt.strip():
        return _err("music: prompt is required")

    ns = SimpleNamespace(
        prompt=prompt.strip(), model=model, duration=duration,
        instrumental=instrumental, lyrics=lyrics, speed=speed,
    )
    spec = _music.fetch_music_spec(client, model)
    rc = _music._validate(ns, spec)  # stderr warnings only
    if rc is not None:
        return _err(f"music: request rejected by client-side validation (exit {rc})")

    quote_body = {"model": model}
    if duration is not None:
        quote_body["duration_seconds"] = duration
    try:
        quote = client.post_json("/audio/quote", quote_body)
    except VeniceAPIError as e:
        return _err(f"music quote rejected: {e}")
    quote_value = quote.get("quote", quote)

    queue_body: dict = {"model": model, "prompt": prompt.strip()}
    if duration is not None:
        queue_body["duration_seconds"] = duration
    if instrumental:
        queue_body["force_instrumental"] = True
    if lyrics:
        queue_body["lyrics_prompt"] = lyrics
    if speed is not None:
        queue_body["speed"] = speed

    return _queue_media(
        client,
        model=model,
        queue_body=queue_body,
        quote_value=quote_value,
        confirm=confirm,
        max_spend=max_spend,
        label="music",
        name_prefix="venice-music",
        output_dir=output_dir,
        max_wait=max_wait,
    )


def _binary_op_tool(
    client, *, endpoint: str, body: dict, out_path: Path, label: str,
    confirm: bool, max_spend: Optional[float],
) -> dict:
    """Shared dynamic-priced (always confirm) binary op for upscale/bg-remove.

    Cost is unknown up front, so `check_spend(None, ...)` forces `confirm=true`.
    Cannot reuse `_shared.post_binary_op` -- it prints the path to stdout.
    """
    gate = check_spend(None, confirm=confirm, max_spend=max_spend, label=label)
    if gate is not None:
        return gate
    try:
        _ctype, payload = client.post_for_bytes_or_json(endpoint, body)
    except VeniceAPIError as e:
        return _err(f"{label} failed: {e}")
    if not isinstance(payload, (bytes, bytearray)):
        return _err(f"{label}: unexpected non-image response from {endpoint}")
    werr = _write(out_path, bytes(payload))
    if werr:
        return _err(f"{label}: could not write {out_path}: {werr}")
    return {
        "status": "ok",
        "path": str(out_path.resolve()),
        "bytes": len(payload),
        "cost_estimate_usd": None,
    }


def upscale_tool(
    client,
    input_path: str,
    *,
    scale: float = 2.0,
    enhance: bool = False,
    enhance_creativity: Optional[float] = None,
    enhance_prompt: Optional[str] = None,
    replication: Optional[float] = None,
    output_dir: Optional[str] = None,
    confirm: bool = False,
    max_spend: Optional[float] = None,
) -> dict:
    """Upscale/enhance an image via /image/upscale (dynamic price -> needs confirm)."""
    inp = Path(input_path)
    ns = SimpleNamespace(
        input=inp, scale=scale, enhance=enhance,
        enhance_creativity=enhance_creativity, enhance_prompt=enhance_prompt,
        replication=replication,
    )
    rc = _upscale._validate(ns)  # stderr warnings only
    if rc is not None:
        return _err(f"upscale: invalid arguments (exit {rc})")

    image_b64 = _shared.encode_base64(inp)
    body = _upscale._build_body(ns, image_b64)
    out_path = resolve_output_dir(output_dir) / f"{inp.stem}-upscaled.png"
    return _binary_op_tool(
        client, endpoint=_upscale.ENDPOINT, body=body, out_path=out_path,
        label="upscale", confirm=confirm, max_spend=max_spend,
    )


def bg_remove_tool(
    client,
    input_path: Optional[str] = None,
    *,
    image_url: Optional[str] = None,
    output_dir: Optional[str] = None,
    confirm: bool = False,
    max_spend: Optional[float] = None,
) -> dict:
    """Remove an image background via /image/background-remove (dynamic -> confirm)."""
    inp = Path(input_path) if input_path else None
    ns = SimpleNamespace(input=inp, image_url=image_url)
    rc = _bg._validate(ns)  # stderr warnings only
    if rc is not None:
        return _err(f"bg-remove: invalid arguments (exit {rc})")

    body = _bg._build_body(ns)
    default_name = f"{inp.stem}-nobg.png" if inp is not None else _bg.URL_DEFAULT_NAME
    out_path = resolve_output_dir(output_dir) / default_name
    return _binary_op_tool(
        client, endpoint=_bg.ENDPOINT, body=body, out_path=out_path,
        label="bg-remove", confirm=confirm, max_spend=max_spend,
    )


def chat_tool(
    client,
    message: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    web_search: Optional[str] = None,
    character: Optional[str] = None,
) -> dict:
    """One-shot chat completion via /chat/completions; return the reply text.

    Cheap relative to media generation, so it is not spend-gated. Needs the
    `[openai]` extra (Venice is OpenAI-compatible); returns an error dict if absent.
    """
    if not message or not message.strip():
        return _err("chat: message is required")

    openai = _openai.import_openai("chat")  # stderr hint if missing
    if openai is None:
        return _err('chat: needs the openai package: pip install "venice-cli[openai]"')

    models = _models.catalog(client, "text")
    resolved, rc = _models.resolve_model(
        model, models, label="chat", noun="text model"
    )
    if rc is not None:
        return _err(f"chat: could not resolve model (exit {rc})")

    ns = SimpleNamespace(
        system=system, temperature=temperature, max_tokens=max_tokens,
        web_search=web_search, web_citations=False, web_scraping=False,
        character=character, no_venice_system_prompt=False, strip_thinking=False,
        no_thinking=False, x_search=False,
    )
    oai = _openai.build_openai(openai, client)
    kwargs = _chat._build_kwargs(ns, resolved, message.strip())
    try:
        resp = oai.chat.completions.create(**kwargs)
    except openai.OpenAIError as e:
        return _err(f"chat: API error: {e}")

    content = ""
    if getattr(resp, "choices", None):
        content = resp.choices[0].message.content or ""
    out = {"status": "ok", "content": content, "model": resolved}
    usage = _chat._as_dict(getattr(resp, "usage", None))
    if usage:
        out["usage"] = usage
    return out


def video_tool(
    client,
    prompt: str,
    *,
    model: Optional[str] = None,
    duration: str = _video.DEFAULT_VIDEO_DURATION,
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
    max_wait: float = _video.config.VIDEO_POLL_MAX_WAIT_SEC,
) -> dict:
    """Generate a video via Venice's async /video queue; write a file, return path.

    Text-to-video (prompt only) plus image/reference conditioning (#18): each
    `*_url` accepts an http(s)/data URL or a local path (encoded to a data URL).
    Long-running -- blocks while polling up to `max_wait`. Paid: a quote is
    fetched first; over-cap or dynamic quotes need confirm=true.
    """
    if not prompt or not prompt.strip():
        return _err("video: prompt is required")

    models = _models.catalog(client, "video")
    model, rc = _models.resolve_model(
        model, models, label="video", noun="video model"
    )
    if rc is not None:
        return _err(f"video: could not resolve model (exit {rc})")

    ns = SimpleNamespace(
        image=image_url, end_image=end_image_url, video=video_url,
        audio_input=audio_url, reference_image=reference_image_urls,
        reference_video=reference_video_urls, reference_audio=reference_audio_urls,
        scene_image=scene_image_urls, element=None,
        reference_video_duration=reference_video_duration,
        resolution=resolution, aspect_ratio=aspect_ratio, no_audio=no_audio,
    )
    quote_media, queue_media, rc = _video._collect_media(ns)  # stderr on bad media
    if rc is not None:
        return _err(f"video: invalid media input (exit {rc})")

    extra = _video._shared_params(ns)
    quote_body = {"model": model, "duration": duration}
    quote_body.update(extra)
    quote_body.update(quote_media)
    try:
        quote = client.post_json("/video/quote", quote_body)
    except VeniceAPIError as e:
        return _err(f"video quote rejected: {e}")
    quote_value = quote.get("quote", quote)

    gate = check_spend(quote_value, confirm=confirm, max_spend=max_spend, label="video")
    if gate is not None:
        return gate

    queue_body = {"model": model, "prompt": prompt.strip(), "duration": duration}
    queue_body.update(extra)
    if negative_prompt:
        queue_body["negative_prompt"] = negative_prompt
    queue_body.update(queue_media)
    try:
        queued = client.post_json("/video/queue", queue_body)
    except VeniceAPIError as e:
        return _err(f"video queue failed: {e}")
    queue_id = queued.get("queue_id") or queued.get("id") or ""
    if not queue_id:
        return _err("video: queue response missing queue_id")
    download_url = queued.get("download_url") or None

    try:
        ctype, data = _video.retrieve_bytes(
            client, model, queue_id,
            poll_interval=_video.config.VIDEO_POLL_INTERVAL_SEC,
            max_wait=max_wait, download_url=download_url,
        )
    except VeniceAPIError as e:
        return _err(f"video retrieve failed: {e}")
    except TimeoutError as e:
        return _err(f"video: {e}; the job {queue_id} may still finish server-side")
    except _video.NoVideoStream:
        return _err(f"video: job {queue_id} completed but returned no video stream")

    ext, _unknown = _queue.ext_for(ctype, _video.VIDEO_EXT_BY_CTYPE, default=".mp4")
    out_path = _queue.resolve_output_path(
        resolve_output_dir(output_dir), queue_id, ext, prefix="venice-video"
    )
    werr = _write(out_path, data)
    if werr:
        return _err(f"video: could not write {out_path}: {werr}")

    try:  # best-effort cleanup; the file is already saved
        client.post_json("/video/complete", {"model": model, "queue_id": queue_id})
    except VeniceAPIError:
        pass
    return {
        "status": "ok",
        "path": str(out_path.resolve()),
        "bytes": len(data),
        "model": model,
        "queue_id": queue_id,
        "cost_estimate_usd": quote_value,
    }


def image_edit_tool(
    client,
    prompt: str,
    *,
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
    """Edit/inpaint an image via /image/edit (dynamic price -> needs confirm).

    Base image is a local `input_path` or an `image_url`; one or two
    `layer_paths` (masks/overlays) route to /image/multi-edit. Writes the
    result and returns its path.
    """
    inp = Path(input_path) if input_path else None
    layers = [Path(p) for p in (layer_paths or [])]
    ns = SimpleNamespace(
        input=inp, image_url=image_url, prompt=prompt, layer=layers or None,
        model=model, aspect_ratio=aspect_ratio, resolution=resolution,
        output_format=output_format, no_safe_mode=no_safe_mode,
    )
    rc = _image_edit._validate(ns)  # stderr warnings only
    if rc is not None:
        return _err(f"image-edit: invalid arguments (exit {rc})")

    base_image = _shared.encode_base64(inp) if inp is not None else image_url
    layers_b64 = [_shared.encode_base64(p) for p in layers]
    endpoint, body = _image_edit._build_body(ns, base_image, layers_b64)

    ext = _image_edit.EXT_BY_FORMAT.get(output_format or "png", ".png")
    name = f"{inp.stem}-edit{ext}" if inp is not None else f"{_image_edit.URL_DEFAULT_STEM}{ext}"
    out_path = resolve_output_dir(output_dir) / name
    return _binary_op_tool(
        client, endpoint=endpoint, body=body, out_path=out_path,
        label="image-edit", confirm=confirm, max_spend=max_spend,
    )


def search_tool(client, query, *, k: int = 8) -> dict:
    """Semantic search over the project's local `.venice` index (issue #24).

    Cheap (one query embedding + an in-process cosine scan), so not spend-gated.
    Discovers the index by walking up from the process cwd; returns the top-`k`
    chunks as ``{path, start, end, score, preview}``. Never builds an index
    (indexing is slow/paid) -- if none exists it returns an error telling the
    caller to run `venice index` first. Requires the `[openai]` extra for the
    query embedding.
    """
    if not query or not str(query).strip():
        return _err("search: query is required")
    store_dir = _index.discover_store(None)
    if store_dir is None:
        return _err("search: no .venice index found; run `venice index` first")
    try:
        results = _index.search_index(store_dir, str(query), k=int(k or 8))
    except _index.IndexingError as e:
        return _err(str(e) or f"search: failed (exit {e.exit_code})")
    return {"status": "ok", "results": results, "count": len(results)}


def models_tool(client, *, type: str) -> dict:
    """List available Venice model ids for a catalog `type` (or "all").

    Wraps the free `/models?type=...` GET (`_models.catalog`) so an agent can
    discover valid model ids before choosing a `model` for the other tools.
    Read-only; not spend-gated. Returns a flat id list for one type, or a
    {type: [ids]} map for "all".
    """
    valid = _models_cmd.MODEL_TYPES
    if type != "all" and type not in valid:
        return _err(
            f"models: unknown type {type!r}; choose from "
            + ", ".join(("all", *valid))
        )
    types = valid if type == "all" else (type,)
    by_type = {}
    for t in types:
        cat = _models.catalog(client, t)
        if cat is None:
            return _err(f"models: /models catalog unavailable for type {t!r}")
        by_type[t] = [m["id"] for m in cat
                      if isinstance(m, dict) and m.get("id")]
    if type == "all":
        return {"status": "ok", "type": "all",
                "count": sum(len(v) for v in by_type.values()),
                "models": by_type}
    return {"status": "ok", "type": type,
            "count": len(by_type[type]), "models": by_type[type]}


def model_details_tool(client, *, model: str) -> dict:
    """Details for one model id: pricing (cost), plus capabilities and constraints
    so an agent can budget input and confirm a model fits before using it.

    `capabilities` (supportsVision/supportsFunctionCalling/... ) is populated only
    for text/LLM models. Image/media models expose their metadata under
    `constraints` (aspectRatios, resolutions, qualities, promptCharacterLimit).
    `voices` lists the voice ids available for TTS models (null otherwise), so the
    agent can pick a valid `voice` for venice_tts without guessing.
    The full `model_spec` is returned too so nothing is dropped. Read-only; not
    spend-gated. May scan the catalog by type to locate the id.
    """
    if not model or not str(model).strip():
        return _err("models: a model id is required")
    m = _models_cmd._find_model(client, str(model).strip())
    if m is None:
        return _err(f"models: no model with id {model!r}")
    spec = m.get("model_spec") if isinstance(m.get("model_spec"), dict) else {}
    constraints = spec.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    return {
        "status": "ok",
        "id": m.get("id"),
        "type": m.get("type"),
        "name": spec.get("name") or m.get("name"),
        "pricing": spec.get("pricing"),
        # Text/LLM models only; null for image/media models.
        "capabilities": spec.get("capabilities"),
        # Image/media metadata: aspectRatios, resolutions, qualities, etc.
        "constraints": spec.get("constraints"),
        "available_context_tokens": spec.get("availableContextTokens"),
        # promptCharacterLimit is nested under constraints; fall back to top level.
        "prompt_character_limit": (
            constraints.get("promptCharacterLimit")
            if constraints.get("promptCharacterLimit") is not None
            else spec.get("promptCharacterLimit")
        ),
        "traits": spec.get("traits"),
        # TTS voice-id list (flat list of strings); null for non-voice models.
        "voices": spec.get("voices"),
        # Full spec so nothing curated-away is lost (#59 / kimi-k3 feedback).
        "model_spec": spec,
    }
