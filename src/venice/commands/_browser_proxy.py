"""Short-lived loopback proxy enforcing browser destination policy per connection."""
from __future__ import annotations

import contextlib
import select
import socket
import socketserver
import threading
import urllib.parse

from venice import _egress

MAX_HEADER_BYTES = 64 * 1024
MAX_REQUESTS = 128
MAX_TRANSFER_BYTES = 32 * 1024 * 1024
MAX_CONCURRENCY = 16
IO_TIMEOUT = 30


class _Budget:
    def __init__(self, requests=MAX_REQUESTS, transfer=MAX_TRANSFER_BYTES):
        self.requests = requests
        self.transfer = transfer
        self.lock = threading.Lock()

    def request(self) -> bool:
        with self.lock:
            if self.requests <= 0:
                return False
            self.requests -= 1
            return True

    def consume(self, count: int) -> bool:
        with self.lock:
            if count > self.transfer:
                self.transfer = 0
                return False
            self.transfer -= count
            return True


def _authority(value: str):
    value = value.strip()
    if not value or "@" in value or any(ord(ch) < 33 for ch in value):
        raise _egress.EgressPolicyError("invalid proxy authority")
    try:
        parts = urllib.parse.urlsplit("//" + value)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise _egress.EgressPolicyError("invalid proxy authority") from None
    if not host or port is None or parts.path or parts.query or parts.fragment:
        raise _egress.EgressPolicyError("proxy authority requires host and port")
    return host, port


def _read_headers(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(min(8192, MAX_HEADER_BYTES + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_HEADER_BYTES:
            raise _egress.EgressPolicyError("proxy request headers are too large")
    if b"\r\n\r\n" not in data:
        raise _egress.EgressPolicyError("incomplete proxy request headers")
    return bytes(data)


def _tunnel(left: socket.socket, right: socket.socket, budget: _Budget) -> None:
    peers = {left: right, right: left}
    while peers:
        ready, _, _ = select.select(list(peers), [], [], IO_TIMEOUT)
        if not ready:
            return
        for source in ready:
            try:
                chunk = source.recv(64 * 1024)
            except OSError:
                return
            if not chunk:
                return
            if not budget.consume(len(chunk)):
                return
            try:
                peers[source].sendall(chunk)
            except OSError:
                return


class _ProxyHandler(socketserver.BaseRequestHandler):
    def _reply(self, code: int, reason: str) -> None:
        try:
            self.request.sendall(
                f"HTTP/1.1 {code} {reason}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".encode("ascii")
            )
        except OSError:
            pass

    def handle(self) -> None:
        server = self.server
        if not server.semaphore.acquire(blocking=False):
            self._reply(503, "Busy")
            return
        try:
            if not server.budget.request():
                self._reply(429, "Request Limit")
                return
            self.request.settimeout(IO_TIMEOUT)
            raw = _read_headers(self.request)
            head, rest = raw.split(b"\r\n\r\n", 1)
            lines = head.split(b"\r\n")
            try:
                method_b, target_b, version_b = lines[0].split(b" ", 2)
                method = method_b.decode("ascii").upper()
                target = target_b.decode("ascii")
                version = version_b.decode("ascii")
            except (ValueError, UnicodeDecodeError):
                raise _egress.EgressPolicyError("invalid proxy request line") from None
            if version not in ("HTTP/1.0", "HTTP/1.1"):
                raise _egress.EgressPolicyError("unsupported proxy protocol")

            if method == "CONNECT":
                host, port = _authority(target)
                scheme = "https" if port == 443 else "http"
                server.policy.endpoint(f"{scheme}://{target}/")
                upstream = server.policy.connect(host, port, IO_TIMEOUT)
                with upstream:
                    self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    if rest:
                        if not server.budget.consume(len(rest)):
                            return
                        upstream.sendall(rest)
                    _tunnel(self.request, upstream, server.budget)
                return

            parts = urllib.parse.urlsplit(target)
            if parts.scheme not in ("http", "ws"):
                raise _egress.EgressPolicyError("proxy requires absolute HTTP URLs")
            host, port, _scheme = server.policy.endpoint(
                target, schemes=("http", "ws")
            )
            upstream = server.policy.connect(host, port, IO_TIMEOUT)
            with upstream:
                path = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
                filtered = []
                for line in lines[1:]:
                    name = line.split(b":", 1)[0].strip().lower()
                    if name in (b"proxy-authorization", b"proxy-connection"):
                        continue
                    filtered.append(line)
                outgoing = (
                    b" ".join((method_b, path.encode("ascii"), version_b))
                    + b"\r\n"
                    + b"\r\n".join(filtered)
                    + b"\r\n\r\n"
                    + rest
                )
                if not server.budget.consume(len(outgoing)):
                    return
                upstream.sendall(outgoing)
                _tunnel(self.request, upstream, server.budget)
        except _egress.EgressPolicyError:
            self._reply(403, "Forbidden")
        except (OSError, UnicodeError, ValueError):
            self._reply(502, "Bad Gateway")
        finally:
            server.semaphore.release()


class _ProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, policy, *, requests=MAX_REQUESTS, transfer=MAX_TRANSFER_BYTES):
        self.policy = policy
        self.budget = _Budget(requests, transfer)
        self.semaphore = threading.BoundedSemaphore(MAX_CONCURRENCY)
        super().__init__(("127.0.0.1", 0), _ProxyHandler)


@contextlib.contextmanager
def policy_proxy(policy: _egress.DestinationPolicy):
    """Yield the enforcing proxy URL and tear it down after the browser exits."""
    server = _ProxyServer(policy)
    thread = threading.Thread(target=server.serve_forever, name="venice-browser-proxy")
    thread.daemon = True
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
