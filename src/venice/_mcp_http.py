"""Remote MCP handler bodies with explicit storage and spend-policy context.

Registration and public SDK schemas stay in ``mcp_server``. Resolve the principal
at request time; never cache an owner on the shared server context. Reservations,
quotas, and job publication continue to go through ``RemoteMediaStore``.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mcp import types
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError
from starlette.responses import JSONResponse, Response, StreamingResponse

from . import _egress, remote_media
from ._mcp_auth import _byte_range, _file_chunks, _http_auth
from ._mcp_common import _merged, _vision_text_result
from .commands import _mcp, _shared, _video_jobs

REMOTE_DATA_URL_MAX_CHARS = 4 * 1024 * 1024
REMOTE_IMPORT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RemoteToolContext:
    """Server dependencies; principal identity is resolved for each operation."""

    client: object
    resolve_owner: Callable[[], str]
    issuer_url: str
    scopes: Sequence[str]
    chat_defaults: dict
    vision_defaults: dict
    media_defaults: dict
    media_store: remote_media.RemoteMediaStore | None
    media_path_authority: _shared.MediaPathAuthority | None
    remote_max_spend: float
    allow_dynamic_spend: bool


def _valid_remote_image_url(value):
    """Accept only network/data image URLs; never reinterpret text as a pod path."""
    if not isinstance(value, str) or not value or len(value) > REMOTE_DATA_URL_MAX_CHARS:
        return False
    if value.startswith("data:image/"):
        header, separator, payload = value.partition(",")
        return bool(separator and payload and header.endswith(";base64"))
    try:
        parsed = urllib.parse.urlsplit(value)
        parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme in ("http", "https")
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    )


def _media_descriptor(store, record) -> dict:
    return {
        "uri": store.resource_uri(record.id),
        "name": record.filename,
        "mime_type": record.mime_type,
        "bytes": record.size,
        "expires_at": record.expires_at,
    }


def _remote_result(result: dict, *, store=None, records=()) -> types.CallToolResult:
    """Return safe JSON metadata plus first-class MCP ResourceLink blocks."""
    clean = dict(result)
    clean.pop("path", None)
    clean.pop("paths", None)
    if store is not None and isinstance(clean.get("message"), str):
        clean["message"] = clean["message"].replace(
            str(store.root.resolve()), "[private media]"
        )
    if records:
        clean["media"] = [_media_descriptor(store, record) for record in records]
    content = [types.TextContent(text=json.dumps(clean, sort_keys=True))]
    for record in records:
        content.append(types.ResourceLink(
            name=record.filename,
            title=record.filename,
            uri=store.resource_uri(record.id),
            description=(
                "Principal-bound Venice media; expires automatically. Read with "
                "MCP resources/read or download over authenticated HTTPS."
            ),
            mimeType=record.mime_type,
            size=record.size,
        ))
    return types.CallToolResult(
        content=content,
        structuredContent=clean,
        isError=clean.get("status") == "error",
    )


def _media_error(message: str) -> types.CallToolResult:
    if isinstance(message, remote_media.MediaStoreError) and not isinstance(
        message,
        (
            remote_media.MediaNotFound,
            remote_media.MediaQuotaError,
            remote_media.MediaValidationError,
        ),
    ):
        message = "media storage is unavailable"
    return _remote_result({"status": "error", "message": str(message)})


def remote_chat(
    ctx: RemoteToolContext,
    message,
    model=None,
    system=None,
    temperature=None,
    max_tokens=None,
    web_search=None,
    character=None,
):
    """Run one Venice chat completion. This consumes API credits and has no
    pre-call spend quote. Omitted values use defaults.chat.*, then built-ins."""
    return _mcp.chat_tool(
        ctx.client,
        message,
        **_merged(ctx.chat_defaults, dict(
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            web_search=web_search,
            character=character,
        )),
    )


def remote_vision(
    ctx: RemoteToolContext,
    image_url,
    prompt=None,
    model=None,
    max_tokens=None,
):
    """Inspect one remote HTTP(S) or data image through a delegated Venice
    vision model. Server-local paths and native host content are unavailable."""
    if not _valid_remote_image_url(image_url):
        return _vision_text_result({
            "status": "error",
            "message": (
                "vision: image_url must be an HTTP(S) URL without credentials "
                "or a bounded base64 image data URL"
            ),
        })
    args = _merged(ctx.vision_defaults, dict(
        image_url=image_url, prompt=prompt, model=model, max_tokens=max_tokens,
    ))
    args.pop("mode", None)
    args.pop("input_path", None)
    if image_url is None:
        args.pop("image_url", None)
    return _vision_text_result(_mcp.vision_tool(
        ctx.client, input_path=None, mode="delegate", **args,
    ))


def remote_media_vision(
    ctx: RemoteToolContext,
    media_uri=None,
    image_url=None,
    prompt=None,
    model=None,
    max_tokens=None,
):
    """Inspect exactly one owner-bound media URI or bounded remote image URL."""
    if bool(media_uri) == bool(image_url):
        return _media_error(
            "vision: exactly one of media_uri or image_url is required"
        )
    input_path = None
    path_authority = None
    if media_uri:
        try:
            record = ctx.media_store.get_uri(ctx.resolve_owner(), media_uri)
        except remote_media.MediaStoreError as exc:
            return _media_error(exc)
        if record.kind != "image":
            return _media_error("vision: media_uri must identify an image")
        input_path = str(record.path)
        path_authority = _shared.MediaPathAuthority(str(ctx.media_store.blobs))
    elif not _valid_remote_image_url(image_url):
        return _media_error(
            "vision: image_url must be an HTTP(S) URL without credentials "
            "or a bounded base64 image data URL"
        )
    args = _merged(ctx.vision_defaults, dict(
        image_url=image_url, prompt=prompt, model=model, max_tokens=max_tokens,
    ))
    args.pop("mode", None)
    return _remote_result(_mcp.vision_tool(
        ctx.client, input_path=input_path, mode="delegate",
        path_authority=path_authority, **args,
    ), store=ctx.media_store)


def _remote_media_args(ctx: RemoteToolContext, section, supplied):
    args = dict(ctx.media_defaults[section])
    for name in (
        "output_dir", "confirm", "max_spend", "hard_max_spend",
        "require_confirmation", "background", "job_store", "path_authority",
        "input_path", "image_url", "layer_paths", "style_references",
        "end_image_url", "video_url", "audio_url",
        "reference_image_urls", "reference_video_urls",
        "reference_audio_urls", "scene_image_urls",
    ):
        args.pop(name, None)
    args.update({name: value for name, value in supplied.items() if value is not None})
    return args


def _owned(ctx: RemoteToolContext, uri, expected_kind=None):
    record = ctx.media_store.get_uri(ctx.resolve_owner(), uri)
    if expected_kind is not None and record.kind != expected_kind:
        raise remote_media.MediaValidationError(
            f"media URI must identify {expected_kind} content"
        )
    return record


def _stored_call(ctx: RemoteToolContext, kind, count, confirm, invoke):
    owner = ctx.resolve_owner()
    work_dir = ctx.media_store.new_work_dir()
    reservations = []
    committed = []
    try:
        if confirm:
            reservations = ctx.media_store.reserve(
                owner, remote_media.OUTPUT_MAX_BYTES[kind], count=count
            )
        result = invoke(work_dir)
        if result.get("status") != "ok":
            return _remote_result(result, store=ctx.media_store)
        if not confirm:
            return _media_error("paid media call proceeded without confirmation")
        paths = result.get("paths") or [result.get("path")]
        paths = [Path(value) for value in paths if value]
        if not paths or len(paths) > len(reservations):
            return _media_error("media generator returned an invalid output set")
        for index, path in enumerate(paths):
            try:
                path.resolve().relative_to(work_dir.resolve())
            except (OSError, ValueError):
                raise remote_media.MediaValidationError(
                    "media generator returned a path outside its private work directory"
                ) from None
            record = ctx.media_store.commit_file(
                owner, reservations[index], path,
                expected_kind=kind, filename=path.name,
            )
            committed.append(record)
            reservations[index] = ""
        return _remote_result(result, store=ctx.media_store, records=committed)
    except remote_media.MediaStoreError as exc:
        for record in committed:
            try:
                ctx.media_store.delete(owner, record.id)
            except remote_media.MediaStoreError:
                pass
        return _media_error(exc)
    finally:
        for reservation in reservations:
            if reservation:
                ctx.media_store.release(owner, reservation)
        shutil.rmtree(work_dir, ignore_errors=True)


def remote_media_import(ctx: RemoteToolContext, url):
    """Import a bounded public HTTPS URL or base64 image data URL.

    The resulting URI is owner-bound and can be used by other remote media
    tools. Prefer POST /media for larger input or when the host can upload.
    """
    owner = ctx.resolve_owner()
    staged = ctx.media_store.new_temp_path()
    declared_mime = None
    filename = None
    reservation = None
    try:
        if url.startswith("data:image/"):
            if len(url) > REMOTE_DATA_URL_MAX_CHARS:
                raise remote_media.MediaValidationError("image data URL is too large")
            header, separator, payload = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise remote_media.MediaValidationError(
                    "image data URL must use base64 encoding"
                )
            declared_mime = header[5:-7]
            try:
                data = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError):
                raise remote_media.MediaValidationError(
                    "image data URL has invalid base64"
                ) from None
            reservation = ctx.media_store.reserve(owner, max(1, len(data)))[0]
            with staged.open("wb") as handle:
                handle.write(data)
        else:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or parsed.username or parsed.password:
                raise remote_media.MediaValidationError(
                    "import URL must be public HTTPS without credentials"
                )
            _egress.validate_https_url(url)
            response = _egress.build_https_opener().open(
                urllib.request.Request(url, headers={"Accept": "image/*,audio/*,video/*"}),
                timeout=REMOTE_IMPORT_TIMEOUT_SECONDS,
            )
            try:
                declared_mime = response.headers.get("Content-Type")
                raw_length = response.headers.get("Content-Length")
                reserve_bytes = remote_media.INPUT_MAX_BYTES["video"]
                if raw_length is not None:
                    try:
                        reserve_bytes = int(raw_length)
                    except ValueError:
                        raise remote_media.MediaValidationError(
                            "import returned an invalid Content-Length"
                        ) from None
                    if not 0 < reserve_bytes <= remote_media.INPUT_MAX_BYTES["video"]:
                        raise remote_media.MediaValidationError(
                            "import exceeds the maximum media input size"
                        )
                reservation = ctx.media_store.reserve(owner, reserve_bytes)[0]
                filename = os.path.basename(
                    urllib.parse.unquote(urllib.parse.urlsplit(response.geturl()).path)
                ) or None
                remaining = reserve_bytes + 1
                with staged.open("wb") as handle:
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                if remaining == 0:
                    raise remote_media.MediaValidationError(
                        "import exceeded the maximum media input size"
                    )
            finally:
                response.close()
        record = ctx.media_store.put_input(
            owner, staged, declared_mime=declared_mime, filename=filename,
            reservation_id=reservation,
        )
        reservation = None
        return _remote_result(
            {"status": "ok"}, store=ctx.media_store, records=[record]
        )
    except (remote_media.MediaStoreError, OSError, urllib.error.URLError):
        return _media_error("media import failed validation or download policy")
    finally:
        if reservation:
            ctx.media_store.release(owner, reservation)
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


def remote_media_delete(ctx: RemoteToolContext, media_uri):
    """Delete one owner-bound media object before its normal expiry."""
    try:
        owner = ctx.resolve_owner()
        ctx.media_store.delete(owner, ctx.media_store.id_from_uri(media_uri))
        return {"status": "ok", "deleted": True}
    except remote_media.MediaStoreError:
        return {"status": "error", "message": "media resource not found"}


def remote_image(
    ctx: RemoteToolContext,
    prompt,
    model=None,
    variants=None,
    format=None,
    width=None,
    height=None,
    negative_prompt=None,
    seed=None,
    aspect_ratio=None,
    resolution=None,
    style_references=None,
    transparent_background=None,
    confirm=False,
):
    """Generate images into private storage. Every paid call requires confirm."""
    args = _remote_media_args(ctx, "image", dict(
        model=model, variants=variants, format=format, width=width, height=height,
        negative_prompt=negative_prompt, seed=seed, aspect_ratio=aspect_ratio,
        resolution=resolution, style_references=style_references,
        transparent_background=transparent_background,
    ))
    try:
        refs = []
        for reference in args.get("style_references") or []:
            item = dict(reference)
            item["image"] = str(_owned(ctx, item.get("image"), "image").path)
            refs.append(item)
        if refs:
            args["style_references"] = refs
    except (TypeError, remote_media.MediaStoreError) as exc:
        return _media_error(exc)
    try:
        count = int(args.get("variants", 1))
    except (TypeError, ValueError):
        return _media_error("image: variants must be an integer")
    return _stored_call(ctx, "image", count, confirm, lambda output: _mcp.image_tool(
        ctx.client, prompt, **args, output_dir=str(output), confirm=confirm,
        max_spend=ctx.remote_max_spend, hard_max_spend=ctx.remote_max_spend,
        require_confirmation=True, path_authority=ctx.media_path_authority,
    ))


def remote_tts(
    ctx: RemoteToolContext,
    text,
    model=None,
    voice=None,
    format=None,
    speed=None,
    confirm=False,
):
    """Synthesize speech into private storage. Every paid call requires confirm."""
    args = _remote_media_args(ctx, "tts", dict(
        model=model, voice=voice, format=format, speed=speed,
    ))
    return _stored_call(ctx, "audio", 1, confirm, lambda output: _mcp.tts_tool(
        ctx.client, text, **args, output_dir=str(output), confirm=confirm,
        max_spend=ctx.remote_max_spend, hard_max_spend=ctx.remote_max_spend,
        require_confirmation=True,
    ))


def _dynamic_disabled(ctx: RemoteToolContext, label):
    if ctx.allow_dynamic_spend:
        return None
    return _media_error(
        f"{label}: disabled because Venice does not provide an up-front "
        "price; the operator may explicitly enable dynamic spend"
    )


def remote_upscale(
    ctx: RemoteToolContext,
    media_uri,
    scale=None,
    creativity=None,
    confirm=False,
):
    """Upscale an owner-bound image. Disabled unless dynamic spend is enabled."""
    disabled = _dynamic_disabled(ctx, "upscale")
    if disabled:
        return disabled
    try:
        source = _owned(ctx, media_uri, "image")
    except remote_media.MediaStoreError as exc:
        return _media_error(exc)
    args = _remote_media_args(ctx, "upscale", dict(
        scale=scale, creativity=creativity,
    ))
    return _stored_call(ctx, "image", 1, confirm, lambda output: _mcp.upscale_tool(
        ctx.client, str(source.path), **args, output_dir=str(output),
        confirm=confirm, max_spend=ctx.remote_max_spend,
        hard_max_spend=ctx.remote_max_spend, require_confirmation=True,
        path_authority=ctx.media_path_authority,
    ))


def remote_bg_remove(ctx: RemoteToolContext, media_uri, confirm=False):
    """Remove a stored image background. Requires dynamic-spend opt-in."""
    disabled = _dynamic_disabled(ctx, "bg-remove")
    if disabled:
        return disabled
    try:
        source = _owned(ctx, media_uri, "image")
    except remote_media.MediaStoreError as exc:
        return _media_error(exc)
    args = _remote_media_args(ctx, "bg_remove", {})
    args.pop("image_url", None)
    return _stored_call(ctx, "image", 1, confirm, lambda output: _mcp.bg_remove_tool(
        ctx.client, str(source.path), **args, output_dir=str(output),
        confirm=confirm, max_spend=ctx.remote_max_spend,
        hard_max_spend=ctx.remote_max_spend, require_confirmation=True,
        path_authority=ctx.media_path_authority,
    ))


def remote_image_edit(
    ctx: RemoteToolContext,
    prompt,
    media_uri,
    layer_media_uris=None,
    model=None,
    aspect_ratio=None,
    resolution=None,
    output_format=None,
    quality=None,
    transparent_background=None,
    confirm=False,
):
    """Edit owner-bound images. Disabled unless dynamic spend is enabled."""
    disabled = _dynamic_disabled(ctx, "image-edit")
    if disabled:
        return disabled
    try:
        source = _owned(ctx, media_uri, "image")
        layers = [
            str(_owned(ctx, uri, "image").path) for uri in (layer_media_uris or [])
        ]
    except remote_media.MediaStoreError as exc:
        return _media_error(exc)
    args = _remote_media_args(ctx, "image_edit", dict(
        model=model, aspect_ratio=aspect_ratio, resolution=resolution,
        output_format=output_format, quality=quality,
        transparent_background=transparent_background,
    ))
    args.pop("image_url", None)
    return _stored_call(ctx, "image", 1, confirm, lambda output: _mcp.image_edit_tool(
        ctx.client, prompt, input_path=str(source.path), layer_paths=layers or None,
        **args, output_dir=str(output), confirm=confirm,
        max_spend=ctx.remote_max_spend, hard_max_spend=ctx.remote_max_spend,
        require_confirmation=True, path_authority=ctx.media_path_authority,
    ))


def _queued_call(ctx: RemoteToolContext, kind, confirm, invoke):
    owner = ctx.resolve_owner()
    reservation = None
    try:
        if confirm:
            ctx.media_store.check_job_capacity(owner)
            output_kind = "video" if kind == "video" else "audio"
            reservation = ctx.media_store.reserve(
                owner, remote_media.OUTPUT_MAX_BYTES[output_kind]
            )[0]
        result = invoke(_RemoteVideoJobStore(ctx.media_store, owner))
        if result.get("status") != "queued":
            return _remote_result(result, store=ctx.media_store)
        if not confirm or reservation is None:
            return _media_error("paid queue call proceeded without confirmation")
        job = ctx.media_store.create_job(
            owner, backend_id=result["queue_id"], kind=kind,
            model=result["model"], reservation_id=reservation,
            cost=result.get("cost_estimate_usd"),
        )
        reservation = None
        return _remote_result({
            "status": "queued", "job_id": job.id, "type": job.kind,
            "model": job.model, "cost_estimate_usd": job.cost,
            "expires_at": job.expires_at,
        })
    except remote_media.MediaStoreError as exc:
        return _media_error(exc)
    finally:
        if reservation:
            ctx.media_store.release(owner, reservation)


def remote_sfx(ctx: RemoteToolContext, prompt, model=None, duration=None, confirm=False):
    """Queue sound-effect generation. Use job_result with the returned job ID."""
    args = _remote_media_args(ctx, "sfx", dict(model=model, duration=duration))
    return _queued_call(ctx, "sfx", confirm, lambda _jobs: _mcp.sfx_tool(
        ctx.client, prompt, **args, confirm=confirm,
        max_spend=ctx.remote_max_spend, hard_max_spend=ctx.remote_max_spend,
        require_confirmation=True, background=True,
    ))


def remote_music(
    ctx: RemoteToolContext,
    prompt,
    model=None,
    duration=None,
    instrumental=None,
    lyrics=None,
    speed=None,
    confirm=False,
):
    """Queue music generation. Use job_result with the returned job ID."""
    args = _remote_media_args(ctx, "music", dict(
        model=model, duration=duration, instrumental=instrumental,
        lyrics=lyrics, speed=speed,
    ))
    return _queued_call(ctx, "music", confirm, lambda _jobs: _mcp.music_tool(
        ctx.client, prompt, **args, confirm=confirm,
        max_spend=ctx.remote_max_spend, hard_max_spend=ctx.remote_max_spend,
        require_confirmation=True, background=True,
    ))


def remote_video(
    ctx: RemoteToolContext,
    prompt,
    model=None,
    duration=None,
    negative_prompt=None,
    resolution=None,
    aspect_ratio=None,
    no_audio=None,
    image_media_uri=None,
    image_strength=None,
    end_image_media_uri=None,
    video_media_uri=None,
    audio_media_uri=None,
    reference_image_media_uris=None,
    reference_video_media_uris=None,
    reference_audio_media_uris=None,
    scene_image_media_uris=None,
    confirm=False,
):
    """Queue video generation using only owner-bound conditioning media."""
    try:
        one = lambda uri, kind: str(_owned(ctx, uri, kind).path) if uri else None
        many = lambda uris, kind: [one(uri, kind) for uri in uris] if uris else None
        inputs = dict(
            image_url=one(image_media_uri, "image"),
            end_image_url=one(end_image_media_uri, "image"),
            video_url=one(video_media_uri, "video"),
            audio_url=one(audio_media_uri, "audio"),
            reference_image_urls=many(reference_image_media_uris, "image"),
            reference_video_urls=many(reference_video_media_uris, "video"),
            reference_audio_urls=many(reference_audio_media_uris, "audio"),
            scene_image_urls=many(scene_image_media_uris, "image"),
        )
    except remote_media.MediaStoreError as exc:
        return _media_error(exc)
    args = _remote_media_args(ctx, "video", dict(
        model=model, duration=duration, negative_prompt=negative_prompt,
        resolution=resolution, aspect_ratio=aspect_ratio, no_audio=no_audio,
        image_strength=image_strength,
        **inputs,
    ))
    args.pop("max_wait", None)
    return _queued_call(ctx, "video", confirm, lambda jobs: _mcp.video_tool(
        ctx.client, prompt, **args, confirm=confirm,
        max_spend=ctx.remote_max_spend, hard_max_spend=ctx.remote_max_spend,
        require_confirmation=True, background=True,
        path_authority=ctx.media_path_authority, job_store=jobs,
    ))


def _finish_remote_job(ctx: RemoteToolContext, job_id):
    owner = ctx.resolve_owner()
    try:
        job = ctx.media_store.get_job(owner, job_id)
        if job.status == "ready":
            record = ctx.media_store.get(owner, job.resource_id)
            return _remote_result(
                {"status": "ok", "job_id": job.id},
                store=ctx.media_store, records=[record],
            )
        if job.status == "failed":
            return _media_error(job.error or "media job failed")
        work_dir = ctx.media_store.new_work_dir()
        try:
            result = _mcp.job_result_tool(
                ctx.client, queue_id=job.backend_id, type=job.kind,
                model=job.model, max_wait=0, output_dir=str(work_dir),
                job_store=_RemoteVideoJobStore(ctx.media_store, owner), complete=False,
            )
            if result.get("status") != "ok":
                safe = dict(result)
                safe.pop("queue_id", None)
                safe["job_id"] = job.id
                if result.get("status") in ("failed", "not_found"):
                    ctx.media_store.fail_job(owner, job.id, result.get("message", "job failed"))
                return _remote_result(safe, store=ctx.media_store)
            path = Path(result["path"])
            try:
                path.resolve().relative_to(work_dir.resolve())
            except (OSError, ValueError):
                raise remote_media.MediaValidationError(
                    "job returned a path outside its private work directory"
                ) from None
            output_kind = "video" if job.kind == "video" else "audio"
            record = ctx.media_store.commit_file(
                owner, job.reservation_id, path,
                expected_kind=output_kind, filename=path.name, job_id=job.id,
            )
            _mcp.complete_job(
                ctx.client, queue_id=job.backend_id, type=job.kind,
                model=job.model, job_store=_RemoteVideoJobStore(ctx.media_store, owner),
            )
            return _remote_result(
                {"status": "ok", "job_id": job.id, "model": job.model},
                store=ctx.media_store, records=[record],
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    except remote_media.MediaStoreError as exc:
        return _media_error(exc)


def remote_job_status(ctx: RemoteToolContext, job_id):
    """Non-blocking status probe; completed media is stored immediately."""
    return _finish_remote_job(ctx, job_id)


def remote_job_result(ctx: RemoteToolContext, job_id):
    """Non-blocking result fetch. Retry while status is processing."""
    return _finish_remote_job(ctx, job_id)


def remote_media_resource(ctx: RemoteToolContext, resource_id):
    try:
        record = ctx.media_store.get(ctx.resolve_owner(), resource_id)
    except remote_media.MediaNotFound as exc:
        raise ResourceNotFoundError("media resource not found") from exc
    except remote_media.MediaStoreError as exc:
        raise ResourceError("media storage is unavailable") from exc
    if record.size > ctx.media_store.mcp_read_max_bytes:
        raise ResourceError(
            "media exceeds the MCP inline read limit; use its authenticated HTTPS URI"
        )
    try:
        return record.path.read_bytes()
    except OSError as exc:
        raise ResourceError("media resource could not be read") from exc


async def upload_media(ctx: RemoteToolContext, request):
    owner, failure = _http_auth(request, ctx.issuer_url, ctx.scopes)
    if failure:
        return failure
    staged = ctx.media_store.new_temp_path()
    total = 0
    reservation = None
    try:
        declared_length = request.headers.get("content-length")
        reserve_bytes = remote_media.INPUT_MAX_BYTES["video"]
        if declared_length is not None:
            try:
                reserve_bytes = int(declared_length)
                if not 0 < reserve_bytes <= remote_media.INPUT_MAX_BYTES["video"]:
                    raise remote_media.MediaValidationError("upload is too large")
            except ValueError:
                raise remote_media.MediaValidationError(
                    "invalid Content-Length"
                ) from None
        reservation = ctx.media_store.reserve(owner, reserve_bytes)[0]
        with staged.open("wb") as handle:
            async for chunk in request.stream():
                total += len(chunk)
                if total > reserve_bytes:
                    raise remote_media.MediaValidationError("upload is too large")
                handle.write(chunk)
        record = ctx.media_store.put_input(
            owner, staged,
            declared_mime=request.headers.get("content-type"),
            filename=request.headers.get("x-venice-filename"),
            reservation_id=reservation,
        )
        reservation = None
        return JSONResponse(_media_descriptor(ctx.media_store, record), status_code=201)
    except remote_media.MediaValidationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except remote_media.MediaQuotaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    except remote_media.MediaStoreError:
        return JSONResponse({"error": "media storage unavailable"}, status_code=503)
    finally:
        if reservation:
            ctx.media_store.release(owner, reservation)
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


async def media_bytes(ctx: RemoteToolContext, request):
    owner, failure = _http_auth(request, ctx.issuer_url, ctx.scopes)
    if failure:
        return failure
    resource_id = request.path_params["resource_id"]
    try:
        record = ctx.media_store.get(owner, resource_id)
    except remote_media.MediaNotFound:
        return JSONResponse({"error": "media resource not found"}, status_code=404)
    except remote_media.MediaStoreError:
        return JSONResponse({"error": "media storage unavailable"}, status_code=503)
    if request.method == "DELETE":
        try:
            ctx.media_store.delete(owner, resource_id)
        except remote_media.MediaStoreError:
            return JSONResponse({"error": "media storage unavailable"}, status_code=503)
        return Response(status_code=204)
    try:
        start, end, partial = _byte_range(request.headers.get("range"), record.size)
    except (TypeError, ValueError):
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{record.size}"},
        )
    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Length": str(length),
        "Content-Disposition": (
            "attachment; filename*=UTF-8''" + urllib.parse.quote(record.filename)
        ),
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{record.size}"
    if request.method == "HEAD":
        return Response(
            status_code=206 if partial else 200,
            media_type=record.mime_type, headers=headers,
        )
    return StreamingResponse(
        _file_chunks(record.path, start, length),
        status_code=206 if partial else 200,
        media_type=record.mime_type, headers=headers,
    )


class _RemoteVideoJobStore:
    def __init__(self, store, owner):
        self.store = store
        self.owner = owner

    def remember(self, queue_id, model, url):
        try:
            self.store.remember_download_url(self.owner, queue_id, model, url)
        except remote_media.MediaStoreError as exc:
            raise _video_jobs.VideoJobStoreError(str(exc)) from None

    def lookup(self, queue_id, model):
        try:
            return self.store.lookup_download_url(self.owner, queue_id, model)
        except remote_media.MediaStoreError as exc:
            raise _video_jobs.VideoJobStoreError(str(exc)) from None

    def forget(self, queue_id, model):
        try:
            self.store.forget_download_url(self.owner, queue_id, model)
        except remote_media.MediaStoreError as exc:
            raise _video_jobs.VideoJobStoreError(str(exc)) from None
