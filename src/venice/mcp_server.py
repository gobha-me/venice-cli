"""Authenticated HTTP MCP tools and public server entry points.

Imported lazily from commands.mcp_serve. Local stdio wiring lives in
_mcp_stdio; OAuth and HTTP helpers live in _mcp_auth. The public imports here
preserve the existing build_server, serve, and JWKSJWTVerifier entry points.

Do NOT add `from __future__ import annotations` here: MCPServer builds each tool's
input schema via typing.get_type_hints, so the annotations must stay concrete
(`typing.Optional[int]`, not stringized `int | None`).
"""
import base64
import binascii
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Literal, Optional

from mcp import types
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from . import _egress, remote_media, userconfig
from .commands import _mcp, _shared, _video_jobs
from ._mcp_auth import JWKSJWTVerifier as JWKSJWTVerifier
from ._mcp_auth import _byte_range, _file_chunks, _http_auth
from ._mcp_common import _merged, _vision_text_result
from ._mcp_stdio import build_server as build_server, serve as serve


REMOTE_DATA_URL_MAX_CHARS = 4 * 1024 * 1024
REMOTE_IMPORT_TIMEOUT_SECONDS = 30


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


def _remote_owner(issuer_url: str) -> str:
    """Resolve the already-verified MCP request to a private storage partition."""
    token = get_access_token()
    if token is None:
        raise remote_media.MediaStoreError("authenticated principal is required")
    return remote_media.principal_key(
        issuer=issuer_url,
        subject=token.subject or "",
        client_id=token.client_id,
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


def build_http_server(
    client,
    *,
    public_url,
    issuer_url,
    jwks_url,
    audience,
    scopes,
    doc=None,
    token_verifier=None,
    media_dir=None,
    media_ttl_seconds=remote_media.DEFAULT_TTL_SECONDS,
    media_max_objects=remote_media.DEFAULT_MAX_OBJECTS,
    media_principal_max_bytes=remote_media.DEFAULT_PRINCIPAL_MAX_BYTES,
    media_global_max_bytes=remote_media.DEFAULT_GLOBAL_MAX_BYTES,
    media_max_pending_jobs=remote_media.DEFAULT_MAX_PENDING_JOBS,
    media_global_max_pending_jobs=remote_media.DEFAULT_GLOBAL_MAX_PENDING_JOBS,
    media_mcp_read_max_bytes=remote_media.DEFAULT_MCP_READ_MAX_BYTES,
    remote_max_spend=_mcp.DEFAULT_MCP_MAX_SPEND,
    allow_dynamic_spend=False,
) -> MCPServer:
    """Build the authenticated, path-independent remote MCP profile."""
    if doc is None:
        doc = userconfig.load_config()
    verifier = token_verifier or JWKSJWTVerifier(
        jwks_url=jwks_url,
        issuer=issuer_url,
        audience=audience,
        required_scopes=scopes,
    )
    server = MCPServer(
        "venice",
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=issuer_url,
            resource_server_url=public_url,
            required_scopes=list(scopes),
        ),
    )
    chat_defaults = userconfig.config_defaults_for("chat", _mcp.chat_tool, doc)
    vision_defaults = userconfig.config_defaults_for("vision", _mcp.vision_tool, doc)
    media_defaults = {
        name: userconfig.config_defaults_for(name, function, doc)
        for name, function in (
            ("image", _mcp.image_tool), ("tts", _mcp.tts_tool),
            ("sfx", _mcp.sfx_tool), ("music", _mcp.music_tool),
            ("upscale", _mcp.upscale_tool), ("bg_remove", _mcp.bg_remove_tool),
            ("video", _mcp.video_tool), ("image_edit", _mcp.image_edit_tool),
        )
    }
    remote_annotations = types.ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        open_world_hint=True,
    )
    media_store = None
    if media_dir:
        parsed_public = urllib.parse.urlsplit(public_url)
        media_store = remote_media.RemoteMediaStore(
            media_dir,
            public_origin=f"{parsed_public.scheme}://{parsed_public.netloc}",
            ttl_seconds=media_ttl_seconds,
            max_objects=media_max_objects,
            principal_max_bytes=media_principal_max_bytes,
            global_max_bytes=media_global_max_bytes,
            max_pending_jobs=media_max_pending_jobs,
            global_max_pending_jobs=media_global_max_pending_jobs,
            mcp_read_max_bytes=media_mcp_read_max_bytes,
        )

    @server.tool(
        name="venice_chat",
        title="Venice chat completion",
        annotations=remote_annotations,
    )
    def remote_chat(
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        web_search: Optional[str] = None,
        character: Optional[str] = None,
    ) -> dict:
        """Run one Venice chat completion. This consumes API credits and has no
        pre-call spend quote. Omitted values use defaults.chat.*, then built-ins."""
        return _mcp.chat_tool(
            client,
            message,
            **_merged(chat_defaults, dict(
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                web_search=web_search,
                character=character,
            )),
        )

    if media_store is None:
        @server.tool(
            name="venice_vision",
            title="Venice remote image analysis",
            annotations=remote_annotations,
        )
        def remote_vision(
            image_url: str,
            prompt: Optional[str] = None,
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
        ) -> types.CallToolResult:
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
            args = _merged(vision_defaults, dict(
                image_url=image_url, prompt=prompt, model=model, max_tokens=max_tokens,
            ))
            args.pop("mode", None)
            args.pop("input_path", None)
            if image_url is None:
                args.pop("image_url", None)
            return _vision_text_result(_mcp.vision_tool(
                client, input_path=None, mode="delegate", **args,
            ))
    else:
        @server.tool(
            name="venice_vision",
            title="Venice remote image analysis",
            annotations=remote_annotations,
        )
        def remote_media_vision(
            media_uri: Optional[str] = None,
            image_url: Optional[str] = None,
            prompt: Optional[str] = None,
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
        ) -> types.CallToolResult:
            """Inspect exactly one owner-bound media URI or bounded remote image URL."""
            if bool(media_uri) == bool(image_url):
                return _media_error(
                    "vision: exactly one of media_uri or image_url is required"
                )
            input_path = None
            path_authority = None
            if media_uri:
                try:
                    record = media_store.get_uri(_remote_owner(issuer_url), media_uri)
                except remote_media.MediaStoreError as exc:
                    return _media_error(exc)
                if record.kind != "image":
                    return _media_error("vision: media_uri must identify an image")
                input_path = str(record.path)
                path_authority = _shared.MediaPathAuthority(str(media_store.blobs))
            elif not _valid_remote_image_url(image_url):
                return _media_error(
                    "vision: image_url must be an HTTP(S) URL without credentials "
                    "or a bounded base64 image data URL"
                )
            args = _merged(vision_defaults, dict(
                image_url=image_url, prompt=prompt, model=model, max_tokens=max_tokens,
            ))
            args.pop("mode", None)
            return _remote_result(_mcp.vision_tool(
                client, input_path=input_path, mode="delegate",
                path_authority=path_authority, **args,
            ), store=media_store)

    if media_store is not None:
        media_path_authority = _shared.MediaPathAuthority(str(media_store.blobs))

        def _remote_media_args(section: str, supplied: dict):
            args = dict(media_defaults[section])
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

        def _owned(uri: str, expected_kind: Optional[str] = None):
            record = media_store.get_uri(_remote_owner(issuer_url), uri)
            if expected_kind is not None and record.kind != expected_kind:
                raise remote_media.MediaValidationError(
                    f"media URI must identify {expected_kind} content"
                )
            return record

        def _stored_call(kind: str, count: int, confirm: bool, invoke):
            owner = _remote_owner(issuer_url)
            work_dir = media_store.new_work_dir()
            reservations = []
            committed = []
            try:
                if confirm:
                    reservations = media_store.reserve(
                        owner, remote_media.OUTPUT_MAX_BYTES[kind], count=count
                    )
                result = invoke(work_dir)
                if result.get("status") != "ok":
                    return _remote_result(result, store=media_store)
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
                    record = media_store.commit_file(
                        owner, reservations[index], path,
                        expected_kind=kind, filename=path.name,
                    )
                    committed.append(record)
                    reservations[index] = ""
                return _remote_result(result, store=media_store, records=committed)
            except remote_media.MediaStoreError as exc:
                for record in committed:
                    try:
                        media_store.delete(owner, record.id)
                    except remote_media.MediaStoreError:
                        pass
                return _media_error(exc)
            finally:
                for reservation in reservations:
                    if reservation:
                        media_store.release(owner, reservation)
                shutil.rmtree(work_dir, ignore_errors=True)

        @server.tool(
            name="venice_media_import", title="Import media into private storage",
            annotations=remote_annotations,
        )
        def remote_media_import(url: str) -> types.CallToolResult:
            """Import a bounded public HTTPS URL or base64 image data URL.

            The resulting URI is owner-bound and can be used by other remote media
            tools. Prefer POST /media for larger input or when the host can upload.
            """
            owner = _remote_owner(issuer_url)
            staged = media_store.new_temp_path()
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
                    reservation = media_store.reserve(owner, max(1, len(data)))[0]
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
                        reservation = media_store.reserve(owner, reserve_bytes)[0]
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
                record = media_store.put_input(
                    owner, staged, declared_mime=declared_mime, filename=filename,
                    reservation_id=reservation,
                )
                reservation = None
                return _remote_result(
                    {"status": "ok"}, store=media_store, records=[record]
                )
            except (remote_media.MediaStoreError, OSError, urllib.error.URLError):
                return _media_error("media import failed validation or download policy")
            finally:
                if reservation:
                    media_store.release(owner, reservation)
                try:
                    staged.unlink()
                except FileNotFoundError:
                    pass

        @server.tool(
            name="venice_media_delete", title="Delete private media",
            annotations=types.ToolAnnotations(
                read_only_hint=False, destructive_hint=True, open_world_hint=False,
            ),
        )
        def remote_media_delete(media_uri: str) -> dict:
            """Delete one owner-bound media object before its normal expiry."""
            try:
                owner = _remote_owner(issuer_url)
                media_store.delete(owner, media_store.id_from_uri(media_uri))
                return {"status": "ok", "deleted": True}
            except remote_media.MediaStoreError:
                return {"status": "error", "message": "media resource not found"}

        @server.tool(
            name="venice_image", title="Generate stored images",
            annotations=remote_annotations,
        )
        def remote_image(
            prompt: str, model: Optional[str] = None,
            variants: Optional[int] = None, format: Optional[str] = None,
            width: Optional[int] = None, height: Optional[int] = None,
            negative_prompt: Optional[str] = None, seed: Optional[int] = None,
            aspect_ratio: Optional[str] = None, resolution: Optional[str] = None,
            style_references: Optional[List[dict]] = None,
            transparent_background: Optional[bool] = None,
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Generate images into private storage. Every paid call requires confirm."""
            args = _remote_media_args("image", dict(
                model=model, variants=variants, format=format, width=width, height=height,
                negative_prompt=negative_prompt, seed=seed, aspect_ratio=aspect_ratio,
                resolution=resolution, style_references=style_references,
                transparent_background=transparent_background,
            ))
            try:
                refs = []
                for reference in args.get("style_references") or []:
                    item = dict(reference)
                    item["image"] = str(_owned(item.get("image"), "image").path)
                    refs.append(item)
                if refs:
                    args["style_references"] = refs
            except (TypeError, remote_media.MediaStoreError) as exc:
                return _media_error(exc)
            try:
                count = int(args.get("variants", 1))
            except (TypeError, ValueError):
                return _media_error("image: variants must be an integer")
            return _stored_call("image", count, confirm, lambda output: _mcp.image_tool(
                client, prompt, **args, output_dir=str(output), confirm=confirm,
                max_spend=remote_max_spend, hard_max_spend=remote_max_spend,
                require_confirmation=True, path_authority=media_path_authority,
            ))

        @server.tool(
            name="venice_tts", title="Generate stored speech",
            annotations=remote_annotations,
        )
        def remote_tts(
            text: str, model: Optional[str] = None, voice: Optional[str] = None,
            format: Optional[str] = None, speed: Optional[float] = None,
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Synthesize speech into private storage. Every paid call requires confirm."""
            args = _remote_media_args("tts", dict(
                model=model, voice=voice, format=format, speed=speed,
            ))
            return _stored_call("audio", 1, confirm, lambda output: _mcp.tts_tool(
                client, text, **args, output_dir=str(output), confirm=confirm,
                max_spend=remote_max_spend, hard_max_spend=remote_max_spend,
                require_confirmation=True,
            ))

        def _dynamic_disabled(label: str):
            if allow_dynamic_spend:
                return None
            return _media_error(
                f"{label}: disabled because Venice does not provide an up-front "
                "price; the operator may explicitly enable dynamic spend"
            )

        @server.tool(
            name="venice_upscale", title="Upscale stored image",
            annotations=remote_annotations,
        )
        def remote_upscale(
            media_uri: str, scale: Optional[Literal[2, 4]] = None,
            creativity: Optional[float] = None, confirm: bool = False,
        ) -> types.CallToolResult:
            """Upscale an owner-bound image. Disabled unless dynamic spend is enabled."""
            disabled = _dynamic_disabled("upscale")
            if disabled:
                return disabled
            try:
                source = _owned(media_uri, "image")
            except remote_media.MediaStoreError as exc:
                return _media_error(exc)
            args = _remote_media_args("upscale", dict(
                scale=scale, creativity=creativity,
            ))
            return _stored_call("image", 1, confirm, lambda output: _mcp.upscale_tool(
                client, str(source.path), **args, output_dir=str(output),
                confirm=confirm, max_spend=remote_max_spend,
                hard_max_spend=remote_max_spend, require_confirmation=True,
                path_authority=media_path_authority,
            ))

        @server.tool(
            name="venice_bg_remove", title="Remove stored image background",
            annotations=remote_annotations,
        )
        def remote_bg_remove(
            media_uri: str, confirm: bool = False,
        ) -> types.CallToolResult:
            """Remove a stored image background. Requires dynamic-spend opt-in."""
            disabled = _dynamic_disabled("bg-remove")
            if disabled:
                return disabled
            try:
                source = _owned(media_uri, "image")
            except remote_media.MediaStoreError as exc:
                return _media_error(exc)
            args = _remote_media_args("bg_remove", {})
            args.pop("image_url", None)
            return _stored_call("image", 1, confirm, lambda output: _mcp.bg_remove_tool(
                client, str(source.path), **args, output_dir=str(output),
                confirm=confirm, max_spend=remote_max_spend,
                hard_max_spend=remote_max_spend, require_confirmation=True,
                path_authority=media_path_authority,
            ))

        @server.tool(
            name="venice_image_edit", title="Edit stored image",
            annotations=remote_annotations,
        )
        def remote_image_edit(
            prompt: str, media_uri: str, layer_media_uris: Optional[List[str]] = None,
            model: Optional[str] = None, aspect_ratio: Optional[str] = None,
            resolution: Optional[str] = None, output_format: Optional[str] = None,
            quality: Optional[str] = None,
            transparent_background: Optional[bool] = None, confirm: bool = False,
        ) -> types.CallToolResult:
            """Edit owner-bound images. Disabled unless dynamic spend is enabled."""
            disabled = _dynamic_disabled("image-edit")
            if disabled:
                return disabled
            try:
                source = _owned(media_uri, "image")
                layers = [
                    str(_owned(uri, "image").path) for uri in (layer_media_uris or [])
                ]
            except remote_media.MediaStoreError as exc:
                return _media_error(exc)
            args = _remote_media_args("image_edit", dict(
                model=model, aspect_ratio=aspect_ratio, resolution=resolution,
                output_format=output_format, quality=quality,
                transparent_background=transparent_background,
            ))
            args.pop("image_url", None)
            return _stored_call("image", 1, confirm, lambda output: _mcp.image_edit_tool(
                client, prompt, input_path=str(source.path), layer_paths=layers or None,
                **args, output_dir=str(output), confirm=confirm,
                max_spend=remote_max_spend, hard_max_spend=remote_max_spend,
                require_confirmation=True, path_authority=media_path_authority,
            ))

        class _RemoteVideoJobStore:
            def __init__(self, owner):
                self.owner = owner

            def remember(self, queue_id, model, url):
                try:
                    media_store.remember_download_url(self.owner, queue_id, model, url)
                except remote_media.MediaStoreError as exc:
                    raise _video_jobs.VideoJobStoreError(str(exc)) from None

            def lookup(self, queue_id, model):
                try:
                    return media_store.lookup_download_url(self.owner, queue_id, model)
                except remote_media.MediaStoreError as exc:
                    raise _video_jobs.VideoJobStoreError(str(exc)) from None

            def forget(self, queue_id, model):
                try:
                    media_store.forget_download_url(self.owner, queue_id, model)
                except remote_media.MediaStoreError as exc:
                    raise _video_jobs.VideoJobStoreError(str(exc)) from None

        def _queued_call(kind: str, confirm: bool, invoke):
            owner = _remote_owner(issuer_url)
            reservation = None
            try:
                if confirm:
                    media_store.check_job_capacity(owner)
                    output_kind = "video" if kind == "video" else "audio"
                    reservation = media_store.reserve(
                        owner, remote_media.OUTPUT_MAX_BYTES[output_kind]
                    )[0]
                result = invoke(_RemoteVideoJobStore(owner))
                if result.get("status") != "queued":
                    return _remote_result(result, store=media_store)
                if not confirm or reservation is None:
                    return _media_error("paid queue call proceeded without confirmation")
                job = media_store.create_job(
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
                    media_store.release(owner, reservation)

        @server.tool(
            name="venice_sfx", title="Queue stored sound effect",
            annotations=remote_annotations,
        )
        def remote_sfx(
            prompt: str, model: Optional[str] = None,
            duration: Optional[int] = None, confirm: bool = False,
        ) -> types.CallToolResult:
            """Queue sound-effect generation. Use job_result with the returned job ID."""
            args = _remote_media_args("sfx", dict(model=model, duration=duration))
            return _queued_call("sfx", confirm, lambda _jobs: _mcp.sfx_tool(
                client, prompt, **args, confirm=confirm,
                max_spend=remote_max_spend, hard_max_spend=remote_max_spend,
                require_confirmation=True, background=True,
            ))

        @server.tool(
            name="venice_music", title="Queue stored music",
            annotations=remote_annotations,
        )
        def remote_music(
            prompt: str, model: Optional[str] = None,
            duration: Optional[int] = None, instrumental: Optional[bool] = None,
            lyrics: Optional[str] = None, speed: Optional[float] = None,
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Queue music generation. Use job_result with the returned job ID."""
            args = _remote_media_args("music", dict(
                model=model, duration=duration, instrumental=instrumental,
                lyrics=lyrics, speed=speed,
            ))
            return _queued_call("music", confirm, lambda _jobs: _mcp.music_tool(
                client, prompt, **args, confirm=confirm,
                max_spend=remote_max_spend, hard_max_spend=remote_max_spend,
                require_confirmation=True, background=True,
            ))

        @server.tool(
            name="venice_video", title="Queue stored video",
            annotations=remote_annotations,
        )
        def remote_video(
            prompt: str, model: Optional[str] = None,
            duration: Optional[str] = None, negative_prompt: Optional[str] = None,
            resolution: Optional[str] = None, aspect_ratio: Optional[str] = None,
            no_audio: Optional[bool] = None,
            image_media_uri: Optional[str] = None,
            image_strength: Optional[float] = None,
            end_image_media_uri: Optional[str] = None,
            video_media_uri: Optional[str] = None,
            audio_media_uri: Optional[str] = None,
            reference_image_media_uris: Optional[List[str]] = None,
            reference_video_media_uris: Optional[List[str]] = None,
            reference_audio_media_uris: Optional[List[str]] = None,
            scene_image_media_uris: Optional[List[str]] = None,
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Queue video generation using only owner-bound conditioning media."""
            try:
                one = lambda uri, kind: str(_owned(uri, kind).path) if uri else None
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
            args = _remote_media_args("video", dict(
                model=model, duration=duration, negative_prompt=negative_prompt,
                resolution=resolution, aspect_ratio=aspect_ratio, no_audio=no_audio,
                image_strength=image_strength,
                **inputs,
            ))
            args.pop("max_wait", None)
            return _queued_call("video", confirm, lambda jobs: _mcp.video_tool(
                client, prompt, **args, confirm=confirm,
                max_spend=remote_max_spend, hard_max_spend=remote_max_spend,
                require_confirmation=True, background=True,
                path_authority=media_path_authority, job_store=jobs,
            ))

        def _finish_remote_job(job_id: str):
            owner = _remote_owner(issuer_url)
            try:
                job = media_store.get_job(owner, job_id)
                if job.status == "ready":
                    record = media_store.get(owner, job.resource_id)
                    return _remote_result(
                        {"status": "ok", "job_id": job.id},
                        store=media_store, records=[record],
                    )
                if job.status == "failed":
                    return _media_error(job.error or "media job failed")
                work_dir = media_store.new_work_dir()
                try:
                    result = _mcp.job_result_tool(
                        client, queue_id=job.backend_id, type=job.kind,
                        model=job.model, max_wait=0, output_dir=str(work_dir),
                        job_store=_RemoteVideoJobStore(owner), complete=False,
                    )
                    if result.get("status") != "ok":
                        safe = dict(result)
                        safe.pop("queue_id", None)
                        safe["job_id"] = job.id
                        if result.get("status") in ("failed", "not_found"):
                            media_store.fail_job(owner, job.id, result.get("message", "job failed"))
                        return _remote_result(safe, store=media_store)
                    path = Path(result["path"])
                    try:
                        path.resolve().relative_to(work_dir.resolve())
                    except (OSError, ValueError):
                        raise remote_media.MediaValidationError(
                            "job returned a path outside its private work directory"
                        ) from None
                    output_kind = "video" if job.kind == "video" else "audio"
                    record = media_store.commit_file(
                        owner, job.reservation_id, path,
                        expected_kind=output_kind, filename=path.name, job_id=job.id,
                    )
                    _mcp.complete_job(
                        client, queue_id=job.backend_id, type=job.kind,
                        model=job.model, job_store=_RemoteVideoJobStore(owner),
                    )
                    return _remote_result(
                        {"status": "ok", "job_id": job.id, "model": job.model},
                        store=media_store, records=[record],
                    )
                finally:
                    shutil.rmtree(work_dir, ignore_errors=True)
            except remote_media.MediaStoreError as exc:
                return _media_error(exc)

        @server.tool(
            name="venice_job_status", title="Check stored media job",
            annotations=types.ToolAnnotations(
                read_only_hint=True, destructive_hint=False, open_world_hint=True,
            ),
        )
        def remote_job_status(job_id: str) -> types.CallToolResult:
            """Non-blocking status probe; completed media is stored immediately."""
            return _finish_remote_job(job_id)

        @server.tool(
            name="venice_job_result", title="Fetch stored media job result",
            annotations=remote_annotations,
        )
        def remote_job_result(job_id: str) -> types.CallToolResult:
            """Non-blocking result fetch. Retry while status is processing."""
            return _finish_remote_job(job_id)

        resource_template = f"{media_store.public_origin}/media/{{resource_id}}"

        @server.resource(
            resource_template,
            name="venice_media",
            title="Venice private media",
            description="Read one authenticated, principal-bound media resource.",
            mime_type="application/octet-stream",
        )
        def remote_media_resource(resource_id: str) -> bytes:
            try:
                record = media_store.get(_remote_owner(issuer_url), resource_id)
            except remote_media.MediaNotFound as exc:
                raise ResourceNotFoundError("media resource not found") from exc
            except remote_media.MediaStoreError as exc:
                raise ResourceError("media storage is unavailable") from exc
            if record.size > media_store.mcp_read_max_bytes:
                raise ResourceError(
                    "media exceeds the MCP inline read limit; use its authenticated HTTPS URI"
                )
            try:
                return record.path.read_bytes()
            except OSError as exc:
                raise ResourceError("media resource could not be read") from exc

        @server.custom_route("/media", methods=["POST"], include_in_schema=False)
        async def upload_media(request: Request):
            owner, failure = _http_auth(request, issuer_url, scopes)
            if failure:
                return failure
            staged = media_store.new_temp_path()
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
                reservation = media_store.reserve(owner, reserve_bytes)[0]
                with staged.open("wb") as handle:
                    async for chunk in request.stream():
                        total += len(chunk)
                        if total > reserve_bytes:
                            raise remote_media.MediaValidationError("upload is too large")
                        handle.write(chunk)
                record = media_store.put_input(
                    owner, staged,
                    declared_mime=request.headers.get("content-type"),
                    filename=request.headers.get("x-venice-filename"),
                    reservation_id=reservation,
                )
                reservation = None
                return JSONResponse(_media_descriptor(media_store, record), status_code=201)
            except remote_media.MediaValidationError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except remote_media.MediaQuotaError as exc:
                return JSONResponse({"error": str(exc)}, status_code=413)
            except remote_media.MediaStoreError:
                return JSONResponse({"error": "media storage unavailable"}, status_code=503)
            finally:
                if reservation:
                    media_store.release(owner, reservation)
                try:
                    staged.unlink()
                except FileNotFoundError:
                    pass

        @server.custom_route(
            "/media/{resource_id}", methods=["GET", "HEAD", "DELETE"],
            include_in_schema=False,
        )
        async def media_bytes(request: Request):
            owner, failure = _http_auth(request, issuer_url, scopes)
            if failure:
                return failure
            resource_id = request.path_params["resource_id"]
            try:
                record = media_store.get(owner, resource_id)
            except remote_media.MediaNotFound:
                return JSONResponse({"error": "media resource not found"}, status_code=404)
            except remote_media.MediaStoreError:
                return JSONResponse({"error": "media storage unavailable"}, status_code=503)
            if request.method == "DELETE":
                try:
                    media_store.delete(owner, resource_id)
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

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(_request: Request):
        return JSONResponse({"status": "ok"})

    return server


def serve_http(
    client,
    *,
    host,
    port,
    public_url,
    issuer_url,
    jwks_url,
    audience,
    scopes,
    allowed_origins=(),
    doc=None,
    media_dir=None,
    media_ttl_seconds=remote_media.DEFAULT_TTL_SECONDS,
    media_max_objects=remote_media.DEFAULT_MAX_OBJECTS,
    media_principal_max_bytes=remote_media.DEFAULT_PRINCIPAL_MAX_BYTES,
    media_global_max_bytes=remote_media.DEFAULT_GLOBAL_MAX_BYTES,
    media_max_pending_jobs=remote_media.DEFAULT_MAX_PENDING_JOBS,
    media_global_max_pending_jobs=remote_media.DEFAULT_GLOBAL_MAX_PENDING_JOBS,
    media_mcp_read_max_bytes=remote_media.DEFAULT_MCP_READ_MAX_BYTES,
    remote_max_spend=_mcp.DEFAULT_MCP_MAX_SPEND,
    allow_dynamic_spend=False,
) -> None:
    """Run the authenticated remote profile over stateless Streamable HTTP."""
    allowed_host = urllib.parse.urlsplit(public_url).netloc
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[allowed_host],
        allowed_origins=list(allowed_origins),
    )
    build_http_server(
        client,
        doc=doc,
        public_url=public_url,
        issuer_url=issuer_url,
        jwks_url=jwks_url,
        audience=audience,
        scopes=scopes,
        media_dir=media_dir,
        media_ttl_seconds=media_ttl_seconds,
        media_max_objects=media_max_objects,
        media_principal_max_bytes=media_principal_max_bytes,
        media_global_max_bytes=media_global_max_bytes,
        media_max_pending_jobs=media_max_pending_jobs,
        media_global_max_pending_jobs=media_global_max_pending_jobs,
        media_mcp_read_max_bytes=media_mcp_read_max_bytes,
        remote_max_spend=remote_max_spend,
        allow_dynamic_spend=allow_dynamic_spend,
    ).run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )
