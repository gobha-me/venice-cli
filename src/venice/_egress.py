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

import fnmatch
import http.client
import ipaddress
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

MAX_REDIRECTS = 5


class EgressPolicyError(OSError):
    """A URL or resolved destination is outside the public HTTPS policy."""


_PRIVATE_AUTHORITY = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("::1/128"),
)
_METADATA_HOSTS = frozenset({
    "metadata.google.internal", "metadata.azure.internal",
})
_METADATA_ADDRESSES = frozenset({
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("169.254.170.2"),
    ipaddress.ip_address("100.100.100.200"),
    ipaddress.ip_address("168.63.129.16"),
    ipaddress.ip_address("fd00:ec2::254"),
})


def _canonical_host(host: str) -> str:
    value = str(host or "").rstrip(".").lower()
    if not value:
        raise EgressPolicyError("URL has no host")
    try:
        return value.encode("idna").decode("ascii")
    except UnicodeError:
        raise EgressPolicyError("URL host is not valid IDNA") from None


def _address(value: str):
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        raise EgressPolicyError("resolver returned an invalid IP address") from None
    if getattr(address, "ipv4_mapped", None) is not None:
        raise EgressPolicyError("IPv4-mapped IPv6 destinations are blocked")
    return address


def _hard_blocked(address) -> bool:
    return (
        address in _METADATA_ADDRESSES
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _private_authority_network(value: str):
    try:
        network = ipaddress.ip_network(str(value), strict=True)
    except ValueError:
        raise EgressPolicyError(f"invalid private browser range: {value!r}") from None
    if not any(
        network.version == parent.version and network.subnet_of(parent)
        for parent in _PRIVATE_AUTHORITY
    ):
        raise EgressPolicyError(
            f"private browser range is outside loopback, RFC1918, or IPv6 ULA: {network}"
        )
    return network


@dataclass(frozen=True)
class DestinationPolicy:
    """Immutable operator-owned network authority for an untrusted URL."""

    allow: Tuple[str, ...] = ()
    deny: Tuple[str, ...] = ()
    private_hosts: Tuple[str, ...] = ()
    private_ranges: Tuple[object, ...] = ()

    @classmethod
    def create(
        cls, *, allow=(), deny=(), private_hosts=(), private_ranges=()
    ) -> "DestinationPolicy":
        hosts = tuple(_canonical_host(host) for host in (private_hosts or ()))
        ranges = tuple(_private_authority_network(value) for value in (private_ranges or ()))
        return cls(
            tuple(str(value) for value in (allow or ())),
            tuple(str(value) for value in (deny or ())),
            hosts,
            ranges,
        )

    def endpoint(self, url: str, *, schemes=("http", "https")) -> Tuple[str, int, str]:
        value = str(url or "")
        if not value or value != value.strip():
            raise EgressPolicyError("URL is empty or contains surrounding whitespace")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise EgressPolicyError("URL contains control characters")
        try:
            parts = urllib.parse.urlsplit(value)
            port = parts.port
            raw_host = parts.hostname
        except ValueError:
            raise EgressPolicyError(f"invalid URL: {safe_url(value)}") from None
        scheme = (parts.scheme or "").lower()
        if scheme not in schemes:
            raise EgressPolicyError(
                f"blocked URL scheme {scheme or '(none)'!r}: {safe_url(value)}"
            )
        if parts.username is not None or parts.password is not None:
            raise EgressPolicyError(f"URL userinfo is not allowed: {safe_url(value)}")
        host = _canonical_host(raw_host or "")
        if host in _METADATA_HOSTS:
            raise EgressPolicyError(f"blocked metadata host: {host!r}")
        for pattern in self.deny:
            if fnmatch.fnmatch(host, pattern.lower()) or fnmatch.fnmatch(value, pattern):
                raise EgressPolicyError(f"blocked by browser deny policy: {safe_url(value)}")
        if self.allow and not any(fnmatch.fnmatch(host, p.lower()) for p in self.allow):
            raise EgressPolicyError(f"host is not in the browser allowlist: {host!r}")
        return host, port or (443 if scheme in ("https", "wss") else 80), scheme

    def _authorize_address(self, host: str, value: str):
        address = _address(value)
        if _hard_blocked(address):
            raise EgressPolicyError(f"blocked destination address: {address.compressed}")
        if address.is_loopback or address.is_private:
            if host not in self.private_hosts or not any(
                address.version == network.version and address in network
                for network in self.private_ranges
            ):
                raise EgressPolicyError(
                    f"blocked private destination address: {address.compressed}"
                )
            return address
        if not address.is_global:
            raise EgressPolicyError(f"blocked non-public destination: {address.compressed}")
        return address

    def resolve(self, host: str, port: int) -> List[Tuple[int, int, int, tuple]]:
        host = _canonical_host(host)
        try:
            literal = _address(host)
        except EgressPolicyError:
            literal = None
        if literal is not None:
            self._authorize_address(host, str(literal))
            family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
            sockaddr = (str(literal), port, 0, 0) if family == socket.AF_INET6 else (str(literal), port)
            return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, sockaddr)]
        try:
            answers = socket.getaddrinfo(
                host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise EgressPolicyError(f"DNS resolution failed for {host!r}: {exc}") from None
        if not answers:
            raise EgressPolicyError(f"DNS resolution returned no addresses for {host!r}")
        validated = []
        seen = set()
        for family, socktype, proto, _canonname, sockaddr in answers:
            if family not in (socket.AF_INET, socket.AF_INET6):
                raise EgressPolicyError("resolver returned an unsupported address family")
            self._authorize_address(host, sockaddr[0])
            key = (family, socktype, proto, sockaddr)
            if key not in seen:
                seen.add(key)
                validated.append(key)
        return validated

    def connect(self, host: str, port: int, timeout, source_address=None) -> socket.socket:
        last_error: Optional[OSError] = None
        for family, socktype, proto, sockaddr in self.resolve(host, port):
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(timeout)
                if source_address:
                    sock.bind(source_address)
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        if last_error is not None:
            raise last_error
        raise EgressPolicyError(f"no usable address for {host!r}")


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
    address = _address(value)
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


def connect_with_policy(
    policy: DestinationPolicy, host: str, port: int, timeout, source_address=None
) -> socket.socket:
    return policy.connect(host, port, timeout, source_address)


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


class PolicyHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, policy: DestinationPolicy, **kwargs):
        self._policy = policy
        super().__init__(*args, **kwargs)

    def connect(self) -> None:
        if self._tunnel_host:
            raise EgressPolicyError("proxy tunnels are disabled")
        self.sock = self._policy.connect(
            self.host, self.port, self.timeout, getattr(self, "source_address", None)
        )


class PolicyHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, policy: DestinationPolicy, **kwargs):
        self._policy = policy
        super().__init__(*args, **kwargs)

    def connect(self) -> None:
        if self._tunnel_host:
            raise EgressPolicyError("proxy tunnels are disabled")
        raw = self._policy.connect(
            self.host, self.port, self.timeout, getattr(self, "source_address", None)
        )
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


class PolicyHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, policy: DestinationPolicy):
        self.policy = policy
        super().__init__()

    def http_open(self, req):
        def connection(host, **kwargs):
            return PolicyHTTPConnection(host, policy=self.policy, **kwargs)
        return self.do_open(connection, req)


class PolicyHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, policy: DestinationPolicy, *, context=None):
        self.policy = policy
        super().__init__(context=context)

    def https_open(self, req):
        def connection(host, **kwargs):
            return PolicyHTTPSConnection(host, policy=self.policy, **kwargs)
        return self.do_open(connection, req, context=self._context)


class PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS
    max_repeats = 2

    def __init__(self, policy: DestinationPolicy):
        self.policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            self.policy.endpoint(newurl)
        except EgressPolicyError as exc:
            raise urllib.error.HTTPError(
                safe_url(newurl), code, f"blocked redirect: {exc}", headers, fp
            ) from None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_policy_opener(policy: DestinationPolicy) -> urllib.request.OpenerDirector:
    """Build an HTTP(S)-only opener with no ambient proxy or alternate schemes."""
    context = ssl.create_default_context()
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.UnknownHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        PolicyRedirectHandler(policy),
        PolicyHTTPHandler(policy),
        PolicyHTTPSHandler(policy, context=context),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


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
