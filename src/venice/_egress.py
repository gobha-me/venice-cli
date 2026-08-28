"""Pinned public-Internet HTTPS transport for untrusted download URLs.

The policy lives at connection establishment, not at the URL-string seam: every
connection resolves its destination exactly once, rejects the whole answer set if
any address is not globally routable, and connects only to one of those validated
numeric addresses.  TLS still uses the original hostname for SNI and certificate
verification, and urllib supplies the original HTTP Host header.

This is intentionally stdlib-only.  Ambient proxy settings are disabled by the
opener factory because a proxy would move DNS and connection authority outside this
boundary.  Redirects are revalidated and bounded by :class:`GuardedRedirectHandler`.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

MAX_REDIRECTS = 5


class EgressPolicyError(OSError):
    """A URL or resolved destination is outside the public HTTPS policy."""


def safe_url(url: str) -> str:
    """Render a URL without userinfo, query credentials, or a fragment."""
    try:
        parts = urllib.parse.urlsplit(str(url or ""))
        host = parts.hostname or ""
        port = parts.port
    except (TypeError, ValueError):
        return "<invalid-url>"
    if not host:
        return "<invalid-url>"
    display_host = f"[{host}]" if ":" in host else host
    netloc = display_host + (f":{port}" if port is not None else "")
    return urllib.parse.urlunsplit(
        ((parts.scheme or "").lower(), netloc, parts.path or "/", "", "")
    )


def validate_https_url(url: str) -> Tuple[str, int]:
    """Return ``(hostname, port)`` for a structurally valid HTTPS URL."""
    value = str(url or "")
    if value != value.strip():
        raise EgressPolicyError("download URL contains leading or trailing whitespace")
    try:
        parts = urllib.parse.urlsplit(value)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise EgressPolicyError(f"invalid download URL: {safe_url(value)}") from None
    if (parts.scheme or "").lower() != "https":
        raise EgressPolicyError(
            f"blocked download URL scheme (HTTPS required): {safe_url(value)}"
        )
    if not host:
        raise EgressPolicyError(f"download URL has no host: {safe_url(value)}")
    if parts.username is not None or parts.password is not None:
        raise EgressPolicyError(
            f"download URL userinfo is not allowed: {safe_url(value)}"
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise EgressPolicyError("download URL contains control characters")
    return host, port or 443


def _public_ip(value: str):
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        raise EgressPolicyError("resolver returned an invalid IP address") from None
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if (
        not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise EgressPolicyError(
            f"blocked non-public download destination: {address.compressed}"
        )
    return address


def resolve_public(host: str, port: int) -> List[Tuple[int, int, int, tuple]]:
    """Resolve once and reject the complete answer set if any IP is non-public."""
    try:
        answers = socket.getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise EgressPolicyError(f"download DNS resolution failed for {host!r}: {e}") from None
    if not answers:
        raise EgressPolicyError(f"download DNS resolution returned no addresses for {host!r}")

    validated: List[Tuple[int, int, int, tuple]] = []
    seen = set()
    for family, socktype, proto, _canonname, sockaddr in answers:
        if family not in (socket.AF_INET, socket.AF_INET6):
            raise EgressPolicyError("resolver returned an unsupported address family")
        _public_ip(sockaddr[0])
        key = (family, socktype, proto, sockaddr)
        if key not in seen:
            seen.add(key)
            validated.append((family, socktype, proto, sockaddr))
    return validated


def _connect_public(host: str, port: int, timeout, source_address=None) -> socket.socket:
    answers = resolve_public(host, port)
    last_error: Optional[OSError] = None
    for family, socktype, proto, sockaddr in answers:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            # sockaddr came from the validated one-shot resolution above.  Passing
            # it directly prevents socket.connect from resolving the hostname again.
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            last_error = e
            sock.close()
    if last_error is not None:
        raise last_error
    raise EgressPolicyError(f"no usable public address for {host!r}")


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection whose TCP socket is pinned to a validated public address."""

    def connect(self) -> None:
        if self._tunnel_host:
            # The opener disables proxies, so a tunnel would indicate that the
            # boundary was bypassed by a future handler change.
            raise EgressPolicyError("proxy tunnels are disabled for downloads")
        raw = _connect_public(
            self.host, self.port, self.timeout, getattr(self, "source_address", None)
        )
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


class PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        def connection(host, **kwargs):
            return PinnedHTTPSConnection(host, **kwargs)

        return self.do_open(connection, req, context=self._context)


class GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            validate_https_url(newurl)
        except EgressPolicyError as e:
            raise urllib.error.HTTPError(
                safe_url(newurl), code, f"blocked redirect: {e}", headers, fp
            ) from None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_https_opener() -> urllib.request.OpenerDirector:
    """Build an opener whose only network path is the pinned HTTPS handler."""
    context = ssl.create_default_context()
    opener = urllib.request.OpenerDirector()
    # Construct the handler graph explicitly.  build_opener would silently add
    # HTTP, file, FTP, data, and environment-proxy handlers that this boundary
    # must not own.
    for handler in (
        urllib.request.UnknownHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        GuardedRedirectHandler(),
        PinnedHTTPSHandler(context=context),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener
