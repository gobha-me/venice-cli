"""OAuth verification, principal checks, and HTTP media range streaming.

Keep transport authentication independent from tool registration. This module
requires the optional MCP SDK and is never imported by the base CLI dispatcher.
"""
from __future__ import annotations

from pathlib import Path

import anyio
import jwt
from jwt.exceptions import PyJWTError
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken, TokenVerifier
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import remote_media

REMOTE_TOKEN_MAX_BYTES = 16 * 1024
REMOTE_JWT_ALGORITHMS = ("RS256", "ES256", "EdDSA")


class JWKSJWTVerifier(TokenVerifier):
    """Verify bounded asymmetric OAuth access tokens against a cached JWKS."""

    def __init__(self, *, jwks_url, issuer, audience, required_scopes):
        self.issuer = issuer
        self.audience = audience
        self.required_scopes = frozenset(required_scopes)
        self._jwks = jwt.PyJWKClient(
            jwks_url,
            cache_keys=True,
            max_cached_keys=16,
            cache_jwk_set=True,
            lifespan=300,
            timeout=5,
        )

    @staticmethod
    def _scope_set(claims):
        raw = claims.get("scope")
        if raw is None:
            raw = claims.get("scp")
        if isinstance(raw, str):
            return frozenset(raw.split())
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            return frozenset(raw)
        return frozenset()

    def _verify(self, token):
        if not isinstance(token, str) or not token or len(token.encode()) > REMOTE_TOKEN_MAX_BYTES:
            return None
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")
            if algorithm not in REMOTE_JWT_ALGORITHMS or not header.get("kid"):
                return None
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                audience=self.audience,
                issuer=self.issuer,
                leeway=60,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except (PyJWTError, OSError, ValueError):
            # Every signature, claim, parse, and JWKS transport failure is the
            # same invalid credential to the caller. In particular, never log
            # the token or its untrusted claims here.
            return None

        subject = claims.get("sub")
        client_id = claims.get("client_id") or claims.get("azp")
        scopes = self._scope_set(claims)
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(client_id, str)
            or not client_id
            or not self.required_scopes.issubset(scopes)
        ):
            return None
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=sorted(scopes),
            expires_at=claims["exp"],
            resource=self.audience,
            subject=subject,
            claims={"iss": self.issuer},
        )

    async def verify_token(self, token):
        return await anyio.to_thread.run_sync(self._verify, token)


def _http_auth(request: Request, issuer_url: str, required_scopes):
    user = request.scope.get("user")
    if not isinstance(user, AuthenticatedUser):
        return None, JSONResponse(
            {"error": "invalid_token", "error_description": "Authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
    access = user.access_token
    if not set(required_scopes).issubset(access.scopes):
        return None, JSONResponse(
            {"error": "insufficient_scope"},
            status_code=403,
            headers={"WWW-Authenticate": 'Bearer error="insufficient_scope"'},
        )
    try:
        owner = remote_media.principal_key(
            issuer=issuer_url,
            subject=access.subject or "",
            client_id=access.client_id,
        )
    except remote_media.MediaStoreError:
        return None, JSONResponse(
            {"error": "invalid_token", "error_description": "Principal is incomplete"},
            status_code=401,
        )
    return owner, None


def _byte_range(value: str | None, size: int):
    """Parse one RFC 7233 byte range; multiple ranges are deliberately refused."""
    if not value:
        return 0, size - 1, False
    if not value.startswith("bytes=") or "," in value:
        raise ValueError
    raw = value[6:]
    start_raw, separator, end_raw = raw.partition("-")
    if not separator:
        raise ValueError
    if not start_raw:
        length = int(end_raw)
        if length <= 0:
            raise ValueError
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else size - 1
        if start < 0 or end < start or start >= size:
            raise ValueError
        end = min(end, size - 1)
    return start, end, True


def _file_chunks(path: Path, start: int, length: int):
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
