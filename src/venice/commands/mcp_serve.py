"""`venice mcp-serve` -- run Venice tools over stdio or authenticated HTTP.

Direction A of the MCP epic (#16 / #14): venice is the *callee*. It speaks MCP over
stdio -- JSON-RPC frames on stdout -- exposing image/sfx/music/tts/upscale/bg-remove/
chat/vision as MCP tools, so a host (Claude Code, or the #15 host) calls them
instead of shelling out to the CLI.

The `mcp` SDK is imported lazily (behind the `[mcp]` extra, Python >=3.10) so the
base stdlib-only CLI and `venice --help` keep working without it -- the same
discipline `chat` uses for the openai SDK. Once the server starts, stdout belongs to
the JSON-RPC transport, so this command's own diagnostics go to stderr only.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass

from .. import _numeric, auth, remote_media, userconfig
from ..client import build_client_from_auth
from . import _mcp, _openai


_SCOPE_RE = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]{1,256}\Z")


class RemoteConfigError(ValueError):
    """The authenticated HTTP listener was not configured safely."""


@dataclass(frozen=True)
class RemoteConfig:
    public_url: str
    issuer_url: str
    jwks_url: str
    audience: str
    scopes: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    media_dir: str = ""
    media_ttl_seconds: int = remote_media.DEFAULT_TTL_SECONDS
    media_max_objects: int = remote_media.DEFAULT_MAX_OBJECTS
    media_principal_max_bytes: int = remote_media.DEFAULT_PRINCIPAL_MAX_BYTES
    media_global_max_bytes: int = remote_media.DEFAULT_GLOBAL_MAX_BYTES
    media_max_pending_jobs: int = remote_media.DEFAULT_MAX_PENDING_JOBS
    media_global_max_pending_jobs: int = remote_media.DEFAULT_GLOBAL_MAX_PENDING_JOBS
    media_mcp_read_max_bytes: int = remote_media.DEFAULT_MCP_READ_MAX_BYTES
    remote_max_spend: float = _mcp.DEFAULT_MCP_MAX_SPEND
    allow_dynamic_spend: bool = False


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return port


def _setting(args, name: str, env_name: str, environ) -> str:
    value = getattr(args, name, None)
    if value is None:
        value = environ.get(env_name)
    return str(value).strip() if value is not None else ""


def _https_url(value: str, label: str, *, endpoint_path=None) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        raise RemoteConfigError(f"{label} is not a valid URL") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RemoteConfigError(
            f"{label} must be an https URL without credentials, query, or fragment"
        )
    if endpoint_path is not None and parsed.path != endpoint_path:
        raise RemoteConfigError(f"{label} path must be {endpoint_path!r}")
    # Accessing .port above also rejects malformed/non-numeric ports. Keep the
    # original spelling otherwise: OAuth issuer comparison is exact.
    del port
    return value


def _origin(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        parsed.port
    except ValueError:
        raise RemoteConfigError("--allowed-origin is not a valid URL") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise RemoteConfigError(
            "--allowed-origin values must be https origins without a path, "
            "credentials, query, or fragment"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def _positive_setting(environ, name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise RemoteConfigError(f"{name} must be an integer") from None
    if value <= 0:
        raise RemoteConfigError(f"{name} must be greater than zero")
    return value


def _boolean_setting(environ, name: str, default: bool = False) -> bool:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    value = str(raw).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise RemoteConfigError(f"{name} must be true or false")


def resolve_remote_config(args, environ=None) -> RemoteConfig:
    """Resolve the fail-closed OAuth resource-server settings for HTTP mode."""
    environ = os.environ if environ is None else environ
    fields = {
        "public_url": _setting(args, "public_url", "VENICE_MCP_PUBLIC_URL", environ),
        "issuer_url": _setting(args, "oauth_issuer", "VENICE_MCP_OAUTH_ISSUER", environ),
        "jwks_url": _setting(args, "oauth_jwks_url", "VENICE_MCP_OAUTH_JWKS_URL", environ),
        "audience": _setting(args, "oauth_audience", "VENICE_MCP_OAUTH_AUDIENCE", environ),
    }
    missing = [name for name, value in fields.items() if not value]
    cli_names = {
        "public_url": "--public-url",
        "issuer_url": "--oauth-issuer",
        "jwks_url": "--oauth-jwks-url",
        "audience": "--oauth-audience",
    }
    if missing:
        raise RemoteConfigError(
            "HTTP mode requires " + ", ".join(cli_names[name] for name in missing)
        )

    cli_scopes = getattr(args, "oauth_scope", None)
    scopes = cli_scopes if cli_scopes else environ.get("VENICE_MCP_OAUTH_SCOPES", "").split()
    scopes = tuple(dict.fromkeys(str(scope).strip() for scope in scopes if str(scope).strip()))
    if not scopes:
        raise RemoteConfigError(
            "HTTP mode requires --oauth-scope (or VENICE_MCP_OAUTH_SCOPES)"
        )
    if not all(_SCOPE_RE.fullmatch(scope) for scope in scopes):
        raise RemoteConfigError(
            "OAuth scopes must be 1-256 character RFC 6749 scope tokens"
        )

    cli_origins = getattr(args, "allowed_origin", None)
    origins = (
        cli_origins
        if cli_origins
        else environ.get("VENICE_MCP_ALLOWED_ORIGINS", "").split()
    )
    allowed_origins = tuple(
        dict.fromkeys(_origin(str(value).strip()) for value in origins)
    )
    if len(allowed_origins) > 32:
        raise RemoteConfigError("at most 32 --allowed-origin values are accepted")
    media_dir = _setting(args, "media_dir", "VENICE_MCP_MEDIA_DIR", environ)
    if media_dir and not os.path.isabs(media_dir):
        raise RemoteConfigError("--media-dir must be an absolute dedicated directory")
    remote_max_spend = getattr(args, "remote_max_spend", None)
    if remote_max_spend is None:
        remote_max_spend = environ.get(
            "VENICE_MCP_REMOTE_MAX_SPEND", _mcp.DEFAULT_MCP_MAX_SPEND
        )
    try:
        remote_max_spend = _numeric.non_negative_float(remote_max_spend)
    except (TypeError, ValueError) as exc:
        raise RemoteConfigError(
            f"VENICE_MCP_REMOTE_MAX_SPEND is invalid: {exc}"
        ) from None
    allow_dynamic = bool(getattr(args, "allow_dynamic_spend", False))
    if not allow_dynamic:
        allow_dynamic = _boolean_setting(
            environ, "VENICE_MCP_REMOTE_ALLOW_DYNAMIC_SPEND"
        )
    if allow_dynamic and not media_dir:
        raise RemoteConfigError("--allow-dynamic-spend requires --media-dir")
    if allow_dynamic and remote_max_spend < 10.0:
        raise RemoteConfigError(
            "dynamic remote media spend requires --remote-max-spend of at least 10"
        )
    return RemoteConfig(
        public_url=_https_url(fields["public_url"], "--public-url", endpoint_path="/mcp"),
        issuer_url=_https_url(fields["issuer_url"], "--oauth-issuer"),
        jwks_url=_https_url(fields["jwks_url"], "--oauth-jwks-url"),
        audience=fields["audience"],
        scopes=scopes,
        allowed_origins=allowed_origins,
        media_dir=media_dir,
        media_ttl_seconds=_positive_setting(
            environ, "VENICE_MCP_MEDIA_TTL_SECONDS", remote_media.DEFAULT_TTL_SECONDS
        ),
        media_max_objects=_positive_setting(
            environ, "VENICE_MCP_MEDIA_MAX_OBJECTS", remote_media.DEFAULT_MAX_OBJECTS
        ),
        media_principal_max_bytes=_positive_setting(
            environ, "VENICE_MCP_MEDIA_PRINCIPAL_MAX_BYTES",
            remote_media.DEFAULT_PRINCIPAL_MAX_BYTES,
        ),
        media_global_max_bytes=_positive_setting(
            environ, "VENICE_MCP_MEDIA_GLOBAL_MAX_BYTES",
            remote_media.DEFAULT_GLOBAL_MAX_BYTES,
        ),
        media_max_pending_jobs=_positive_setting(
            environ, "VENICE_MCP_MEDIA_MAX_PENDING_JOBS",
            remote_media.DEFAULT_MAX_PENDING_JOBS,
        ),
        media_global_max_pending_jobs=_positive_setting(
            environ, "VENICE_MCP_MEDIA_GLOBAL_MAX_PENDING_JOBS",
            remote_media.DEFAULT_GLOBAL_MAX_PENDING_JOBS,
        ),
        media_mcp_read_max_bytes=_positive_setting(
            environ, "VENICE_MCP_MEDIA_MCP_READ_MAX_BYTES",
            remote_media.DEFAULT_MCP_READ_MAX_BYTES,
        ),
        remote_max_spend=remote_max_spend,
        allow_dynamic_spend=allow_dynamic,
    )


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "mcp-serve",
        help="Run Venice tools over stdio or authenticated Streamable HTTP.",
        description=(
            "Speaks MCP over stdio with all local tools, or authenticated "
            "Streamable HTTP with the remote-safe chat/vision profile. Needs "
            "the [mcp] extra "
            "(Python >=3.10): "
            'pip install "venice-cli[mcp]". Attach it with, e.g., '
            "`claude mcp add venice -- venice mcp-serve`. Spend on paid tools is "
            "gated: costs over VENICE_MCP_MAX_SPEND (default $0.10) need confirm=true. "
            "Generated files land in VENICE_MCP_OUTPUT_DIR (default: cwd)."
        ),
    )
    p.add_argument(
        "--host-image-content",
        action="store_true",
        help=(
            "declare that this stdio host delivers MCP ImageContent to a "
            "vision-capable frontend (default: text-only/delegated vision)"
        ),
    )
    p.add_argument(
        "--http", action="store_true",
        help="serve the remote-safe tool profile over authenticated Streamable HTTP",
    )
    p.add_argument("--host", help="HTTP bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=_port, help="HTTP bind port (default: 8000)")
    p.add_argument("--public-url", help="public HTTPS MCP endpoint (must end in /mcp)")
    p.add_argument("--oauth-issuer", help="exact HTTPS OAuth issuer URL")
    p.add_argument("--oauth-jwks-url", help="HTTPS JSON Web Key Set URL")
    p.add_argument("--oauth-audience", help="required JWT audience")
    p.add_argument(
        "--oauth-scope", action="append",
        help="required OAuth scope; repeat to require more than one",
    )
    p.add_argument(
        "--allowed-origin", action="append",
        help="allow one exact HTTPS browser Origin; repeat as needed",
    )
    p.add_argument(
        "--media-dir",
        help="enable principal-bound remote media using this private storage directory",
    )
    p.add_argument(
        "--remote-max-spend", type=_numeric.non_negative_float, metavar="USD",
        help="hard per-call ceiling for authenticated remote media (default 0.10)",
    )
    p.add_argument(
        "--allow-dynamic-spend", action="store_true", default=None,
        help="enable unknown-price remote media calls (requires max spend >= 10)",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    mcp = _mcp.import_mcp("mcp-serve")  # lazy probe -> None + stderr hint if absent
    if mcp is None:
        return 2

    remote = getattr(args, "http", False)
    remote_only = (
        "host", "port", "public_url", "oauth_issuer", "oauth_jwks_url",
        "oauth_audience", "oauth_scope", "allowed_origin",
        "media_dir", "remote_max_spend", "allow_dynamic_spend",
    )
    if not remote and any(getattr(args, name, None) is not None for name in remote_only):
        print(
            "venice mcp-serve: HTTP listener/OAuth flags require --http",
            file=sys.stderr,
        )
        return 2
    if remote and getattr(args, "host_image_content", False):
        print(
            "venice mcp-serve: --host-image-content is stdio-only; HTTP vision "
            "accepts remote image URLs and always delegates",
            file=sys.stderr,
        )
        return 2
    if remote and _openai.import_openai("mcp-serve --http") is None:
        return 2
    try:
        remote_config = resolve_remote_config(args) if remote else None
        # Fail fast on auth *before* stdout is handed to the JSON-RPC transport.
        client = build_client_from_auth()
    except (auth.AuthError, RemoteConfigError) as e:
        print(str(e), file=sys.stderr)
        return 2

    # Lazy: only import MCPServer/PyJWT after the optional SDK probe passes.
    from ..mcp_server import serve, serve_http

    doc = userconfig.load_config()  # #58: honor defaults.<section>.* in exposed tools

    try:
        if remote:
            assert remote_config is not None
            host = getattr(args, "host", None) or "127.0.0.1"
            port = getattr(args, "port", None) or 8000
            print(
                f"venice mcp-serve: starting authenticated HTTP server on "
                f"{host}:{port} (public endpoint {remote_config.public_url})",
                file=sys.stderr,
            )
            serve_http(
                client, doc=doc, host=host, port=port,
                public_url=remote_config.public_url,
                issuer_url=remote_config.issuer_url,
                jwks_url=remote_config.jwks_url,
                audience=remote_config.audience,
                scopes=remote_config.scopes,
                allowed_origins=remote_config.allowed_origins,
                media_dir=remote_config.media_dir or None,
                media_ttl_seconds=remote_config.media_ttl_seconds,
                media_max_objects=remote_config.media_max_objects,
                media_principal_max_bytes=remote_config.media_principal_max_bytes,
                media_global_max_bytes=remote_config.media_global_max_bytes,
                media_max_pending_jobs=remote_config.media_max_pending_jobs,
                media_global_max_pending_jobs=(
                    remote_config.media_global_max_pending_jobs
                ),
                media_mcp_read_max_bytes=remote_config.media_mcp_read_max_bytes,
                remote_max_spend=remote_config.remote_max_spend,
                allow_dynamic_spend=remote_config.allow_dynamic_spend,
            )
        else:
            print("venice mcp-serve: starting stdio MCP server (Ctrl-C to stop)",
                  file=sys.stderr)
            serve(
                client,
                doc=doc,
                host_image_content=getattr(args, "host_image_content", False),
            )
    except KeyboardInterrupt:
        return 130
    return 0
