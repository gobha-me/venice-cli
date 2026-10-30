"""Authenticated HTTP MCP tools and public server entry points.

Imported lazily from commands.mcp_serve. Local stdio wiring lives in
_mcp_stdio; remote handler bodies live in _mcp_http and authentication in
_mcp_auth. The public imports here
preserve the existing build_server, serve, and JWKSJWTVerifier entry points.

Do NOT add `from __future__ import annotations` here: MCPServer builds each tool's
input schema via typing.get_type_hints, so the annotations must stay concrete
(`typing.Optional[int]`, not stringized `int | None`).
"""
import urllib.parse
from functools import partial
from typing import Literal, Optional

from mcp import types
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import _mcp_auth, _mcp_http, _mcp_stdio, remote_media, userconfig
from .commands import _mcp, _shared

JWKSJWTVerifier = _mcp_auth.JWKSJWTVerifier
build_server = _mcp_stdio.build_server
serve = _mcp_stdio.serve


REMOTE_DATA_URL_MAX_CHARS = _mcp_http.REMOTE_DATA_URL_MAX_CHARS
REMOTE_IMPORT_TIMEOUT_SECONDS = _mcp_http.REMOTE_IMPORT_TIMEOUT_SECONDS


_valid_remote_image_url = _mcp_http._valid_remote_image_url
_media_descriptor = _mcp_http._media_descriptor
_remote_result = _mcp_http._remote_result
_media_error = _mcp_http._media_error


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
    # Our verifier checks its configured JWT audience itself. SDK 2.1.1 has
    # no resource-validation switch; newer 2.x releases warn if it is omitted.
    auth_options = {}
    if "validate_token_resource" in AuthSettings.model_fields:
        auth_options["validate_token_resource"] = False
    server = MCPServer(
        "venice",
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=issuer_url,
            resource_server_url=public_url,
            required_scopes=list(scopes),
            **auth_options,
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

    tool_context = _mcp_http.RemoteToolContext(
        client=client,
        resolve_owner=partial(_remote_owner, issuer_url),
        issuer_url=issuer_url,
        scopes=scopes,
        chat_defaults=chat_defaults,
        vision_defaults=vision_defaults,
        media_defaults=media_defaults,
        media_store=media_store,
        media_path_authority=(
            _shared.MediaPathAuthority(str(media_store.blobs))
            if media_store is not None else None
        ),
        remote_max_spend=remote_max_spend,
        allow_dynamic_spend=allow_dynamic_spend,
    )

    @server.tool(
        name="venice_chat",
        title="Venice chat completion",
        annotations=remote_annotations,
    )
    def remote_chat(
        message: str,
        model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        system: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        temperature: Optional[float] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        max_tokens: Optional[int] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        web_search: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        character: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
    ) -> dict:
        """Run one Venice chat completion. This consumes API credits and has no
        pre-call spend quote. Omitted values use defaults.chat.*, then built-ins."""
        return _mcp_http.remote_chat(
            tool_context,
            message=message, model=model, system=system, temperature=temperature,
            max_tokens=max_tokens, web_search=web_search, character=character,
        )

    if media_store is None:
        @server.tool(
            name="venice_vision",
            title="Venice remote image analysis",
            annotations=remote_annotations,
        )
        def remote_vision(
            image_url: str,
            prompt: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            max_tokens: Optional[int] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        ) -> types.CallToolResult:
            """Inspect one remote HTTP(S) or data image through a delegated Venice
            vision model. Server-local paths and native host content are unavailable."""
            return _mcp_http.remote_vision(
                tool_context,
                image_url=image_url, prompt=prompt, model=model, max_tokens=max_tokens,
            )
    else:
        @server.tool(
            name="venice_vision",
            title="Venice remote image analysis",
            annotations=remote_annotations,
        )
        def remote_media_vision(
            media_uri: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            image_url: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            prompt: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            max_tokens: Optional[int] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        ) -> types.CallToolResult:
            """Inspect exactly one owner-bound media URI or bounded remote image URL."""
            return _mcp_http.remote_media_vision(
                tool_context,
                media_uri=media_uri, image_url=image_url, prompt=prompt, model=model,
                max_tokens=max_tokens,
            )

    if media_store is not None:
        @server.tool(
            name="venice_media_import", title="Import media into private storage",
            annotations=remote_annotations,
        )
        def remote_media_import(url: str) -> types.CallToolResult:
            """Import a bounded public HTTPS URL or base64 image data URL.

            The resulting URI is owner-bound and can be used by other remote media
            tools. Prefer POST /media for larger input or when the host can upload.
            """
            return _mcp_http.remote_media_import(
                tool_context,
                url=url,
            )

        @server.tool(
            name="venice_media_delete", title="Delete private media",
            annotations=types.ToolAnnotations(
                read_only_hint=False, destructive_hint=True, open_world_hint=False,
            ),
        )
        def remote_media_delete(media_uri: str) -> dict:
            """Delete one owner-bound media object before its normal expiry."""
            return _mcp_http.remote_media_delete(
                tool_context,
                media_uri=media_uri,
            )

        @server.tool(
            name="venice_image", title="Generate stored images",
            annotations=remote_annotations,
        )
        def remote_image(
            prompt: str, model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            variants: Optional[int] = None, format: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            width: Optional[int] = None, height: Optional[int] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            negative_prompt: Optional[str] = None, seed: Optional[int] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            aspect_ratio: Optional[str] = None, resolution: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            style_references: Optional[list[dict]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            transparent_background: Optional[bool] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Generate images into private storage. Every paid call requires confirm."""
            return _mcp_http.remote_image(
                tool_context,
                prompt=prompt, model=model, variants=variants, format=format, width=width,
                height=height, negative_prompt=negative_prompt, seed=seed,
                aspect_ratio=aspect_ratio, resolution=resolution,
                style_references=style_references,
                transparent_background=transparent_background, confirm=confirm,
            )

        @server.tool(
            name="venice_tts", title="Generate stored speech",
            annotations=remote_annotations,
        )
        def remote_tts(
            text: str, model: Optional[str] = None, voice: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            format: Optional[str] = None, speed: Optional[float] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Synthesize speech into private storage. Every paid call requires confirm."""
            return _mcp_http.remote_tts(
                tool_context,
                text=text, model=model, voice=voice, format=format, speed=speed,
                confirm=confirm,
            )


        @server.tool(
            name="venice_upscale", title="Upscale stored image",
            annotations=remote_annotations,
        )
        def remote_upscale(
            media_uri: str, scale: Optional[Literal[2, 4]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            creativity: Optional[float] = None, confirm: bool = False,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        ) -> types.CallToolResult:
            """Upscale an owner-bound image. Disabled unless dynamic spend is enabled."""
            return _mcp_http.remote_upscale(
                tool_context,
                media_uri=media_uri, scale=scale, creativity=creativity, confirm=confirm,
            )

        @server.tool(
            name="venice_bg_remove", title="Remove stored image background",
            annotations=remote_annotations,
        )
        def remote_bg_remove(
            media_uri: str, confirm: bool = False,
        ) -> types.CallToolResult:
            """Remove a stored image background. Requires dynamic-spend opt-in."""
            return _mcp_http.remote_bg_remove(
                tool_context,
                media_uri=media_uri, confirm=confirm,
            )

        @server.tool(
            name="venice_image_edit", title="Edit stored image",
            annotations=remote_annotations,
        )
        def remote_image_edit(
            prompt: str, media_uri: str, layer_media_uris: Optional[list[str]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            model: Optional[str] = None, aspect_ratio: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            resolution: Optional[str] = None, output_format: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            quality: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            transparent_background: Optional[bool] = None, confirm: bool = False,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        ) -> types.CallToolResult:
            """Edit owner-bound images. Disabled unless dynamic spend is enabled."""
            return _mcp_http.remote_image_edit(
                tool_context,
                prompt=prompt, media_uri=media_uri, layer_media_uris=layer_media_uris,
                model=model, aspect_ratio=aspect_ratio, resolution=resolution,
                output_format=output_format, quality=quality,
                transparent_background=transparent_background, confirm=confirm,
            )


        @server.tool(
            name="venice_sfx", title="Queue stored sound effect",
            annotations=remote_annotations,
        )
        def remote_sfx(
            prompt: str, model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            duration: Optional[int] = None, confirm: bool = False,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
        ) -> types.CallToolResult:
            """Queue sound-effect generation. Use job_result with the returned job ID."""
            return _mcp_http.remote_sfx(
                tool_context,
                prompt=prompt, model=model, duration=duration, confirm=confirm,
            )

        @server.tool(
            name="venice_music", title="Queue stored music",
            annotations=remote_annotations,
        )
        def remote_music(
            prompt: str, model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            duration: Optional[int] = None, instrumental: Optional[bool] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            lyrics: Optional[str] = None, speed: Optional[float] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Queue music generation. Use job_result with the returned job ID."""
            return _mcp_http.remote_music(
                tool_context,
                prompt=prompt, model=model, duration=duration, instrumental=instrumental,
                lyrics=lyrics, speed=speed, confirm=confirm,
            )

        @server.tool(
            name="venice_video", title="Queue stored video",
            annotations=remote_annotations,
        )
        def remote_video(
            prompt: str, model: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            duration: Optional[str] = None, negative_prompt: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            resolution: Optional[str] = None, aspect_ratio: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            no_audio: Optional[bool] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            image_media_uri: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            image_strength: Optional[float] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            end_image_media_uri: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            video_media_uri: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            audio_media_uri: Optional[str] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            reference_image_media_uris: Optional[list[str]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            reference_video_media_uris: Optional[list[str]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            reference_audio_media_uris: Optional[list[str]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            scene_image_media_uris: Optional[list[str]] = None,  # noqa: FA100  # reason: MCP SDK builds schemas from concrete annotations
            confirm: bool = False,
        ) -> types.CallToolResult:
            """Queue video generation using only owner-bound conditioning media."""
            return _mcp_http.remote_video(
                tool_context,
                prompt=prompt, model=model, duration=duration,
                negative_prompt=negative_prompt, resolution=resolution,
                aspect_ratio=aspect_ratio, no_audio=no_audio,
                image_media_uri=image_media_uri, image_strength=image_strength,
                end_image_media_uri=end_image_media_uri, video_media_uri=video_media_uri,
                audio_media_uri=audio_media_uri,
                reference_image_media_uris=reference_image_media_uris,
                reference_video_media_uris=reference_video_media_uris,
                reference_audio_media_uris=reference_audio_media_uris,
                scene_image_media_uris=scene_image_media_uris, confirm=confirm,
            )


        @server.tool(
            name="venice_job_status", title="Check stored media job",
            annotations=types.ToolAnnotations(
                read_only_hint=True, destructive_hint=False, open_world_hint=True,
            ),
        )
        def remote_job_status(job_id: str) -> types.CallToolResult:
            """Non-blocking status probe; completed media is stored immediately."""
            return _mcp_http.remote_job_status(
                tool_context,
                job_id=job_id,
            )

        @server.tool(
            name="venice_job_result", title="Fetch stored media job result",
            annotations=remote_annotations,
        )
        def remote_job_result(job_id: str) -> types.CallToolResult:
            """Non-blocking result fetch. Retry while status is processing."""
            return _mcp_http.remote_job_result(
                tool_context,
                job_id=job_id,
            )

        resource_template = f"{media_store.public_origin}/media/{{resource_id}}"

        @server.resource(
            resource_template,
            name="venice_media",
            title="Venice private media",
            description="Read one authenticated, principal-bound media resource.",
            mime_type="application/octet-stream",
        )
        def remote_media_resource(resource_id: str) -> bytes:
            return _mcp_http.remote_media_resource(
                tool_context,
                resource_id=resource_id,
            )

        @server.custom_route("/media", methods=["POST"], include_in_schema=False)
        async def upload_media(request: Request):
            return await _mcp_http.upload_media(
                tool_context,
                request=request,
            )

        @server.custom_route(
            "/media/{resource_id}", methods=["GET", "HEAD", "DELETE"],
            include_in_schema=False,
        )
        async def media_bytes(request: Request):
            return await _mcp_http.media_bytes(
                tool_context,
                request=request,
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
