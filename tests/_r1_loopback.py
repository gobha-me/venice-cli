"""Scriptable loopback Venice API for the r1 audit tests.

The only thing faked is the network peer: a stdlib ``ThreadingHTTPServer`` on
``127.0.0.1:0``. The real ``VeniceClient`` (or a real CLI/MCP subprocess pointed
at ``$VENICE_BASE_URL``) talks to it over a socket, so request encoding, error
decoding, excerpt truncation, and redaction all run unmodified.

Routes are scripted per test. A route not scripted is answered 404 and recorded
in ``unexpected`` -- tests assert that list is empty, so a code path that
silently starts calling a new endpoint fails instead of passing.

Credential hygiene (AGENTS.md): the ``Authorization`` header is recorded as a
boolean only. A responder may *read* the bearer to build an echoing error body
(the attack the redaction code defends against), but it is never stored.
Every key used with this fixture is synthetic.
"""
from __future__ import annotations

import base64
import json
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # annotation-only (postponed); no runtime dependency
    from typing_extensions import Self

PREFIX = "/api/v1"

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZQAAAABJRU5ErkJggg=="
)
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(64)
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + bytes(64)


@dataclass
class Seen:
    """One request as the loopback peer received it (never the key itself)."""

    method: str
    path: str
    query: dict
    body: Any
    authorized: bool


@dataclass
class Reply:
    status: int = 200
    content_type: str = "application/json"
    body: Any = field(default_factory=dict)

    def encoded(self) -> bytes:
        if isinstance(self.body, (bytes, bytearray)):
            return bytes(self.body)
        if isinstance(self.body, str):
            return self.body.encode("utf-8")
        return json.dumps(self.body).encode("utf-8")


def json_reply(body, status=200) -> Reply:
    return Reply(status, "application/json", body)


def bytes_reply(data: bytes, content_type: str, status=200) -> Reply:
    return Reply(status, content_type, data)


# A responder receives (seen, bearer) and returns a Reply. ``bearer`` is the
# token the client sent; it exists so a test can script a hostile upstream that
# echoes it. It is not retained anywhere.
Responder = Callable[[Seen, str], Reply]


def video_model(identifier="r1-video", *, default=True) -> dict:
    return {
        "id": identifier,
        "type": "video",
        "model_spec": {
            "traits": ["default"] if default else [],
            "constraints": {
                "durations": ["5s", "10s"],
                "resolutions": ["720p"],
                "aspect_ratios": ["16:9"],
            },
        },
    }


class LoopbackVenice:
    """Context manager serving scripted responses on an ephemeral loopback port."""

    def __init__(self):
        self.seen: list = []
        self.unexpected: list = []
        self._routes: dict = {}
        self._lock = threading.Lock()

    # -- scripting --------------------------------------------------------- #
    def on(self, method: str, path: str, *replies) -> LoopbackVenice:
        """Script ``method path``. Replies are consumed in order; the last one
        repeats. Each is a ``Reply`` or a ``Responder`` callable."""
        if not replies:
            raise ValueError("at least one reply is required")
        self._routes[(method.upper(), path)] = list(replies)
        return self

    def bodies(self, path: str, method: str = "POST") -> list:
        return [s.body for s in self.seen if s.path == path and s.method == method]

    def paths(self) -> list:
        return [s.path for s in self.seen]

    def _next(self, method: str, path: str):
        with self._lock:
            queue = self._routes.get((method, path))
            if not queue:
                return None
            return queue.pop(0) if len(queue) > 1 else queue[0]

    # -- lifecycle --------------------------------------------------------- #
    def __enter__(self) -> Self:
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # stdlib hook; keep test output clean
                pass

            def _handle(self, method):
                parsed = urllib.parse.urlsplit(self.path)
                path = parsed.path
                path = path.removeprefix(PREFIX)
                raw = b""
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    raw = self.rfile.read(length)
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                auth = self.headers.get("Authorization") or ""
                bearer = auth[7:] if auth.startswith("Bearer ") else ""
                seen = Seen(method, path, urllib.parse.parse_qs(parsed.query),
                            body, bool(bearer))
                with fixture._lock:
                    fixture.seen.append(seen)
                scripted = fixture._next(method, path)
                if scripted is None:
                    with fixture._lock:
                        fixture.unexpected.append(f"{method} {path}")
                    reply = json_reply({"error": f"unscripted {method} {path}"}, 404)
                elif isinstance(scripted, Reply):
                    reply = scripted
                else:
                    reply = scripted(seen, bearer)
                payload = reply.encoded()
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # A short poll keeps shutdown() (which waits one poll) from dominating
        # suites that start a fresh peer per subTest.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        return False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}{PREFIX}"


def echo_bearer(status: int, template: str, content_type="application/json") -> Responder:
    """A hostile upstream: put the caller's bearer into the error body.

    ``template`` contains ``{key}``; it is formatted with the *received* bearer,
    optionally transformed (see ``encoded``)."""

    def respond(_seen: Seen, bearer: str) -> Reply:
        return Reply(status, content_type, template.replace("{key}", bearer))

    return respond


def model_catalog(by_type: dict | None = None) -> Responder:
    """GET /models answering per ``?type=`` from ``by_type`` (empty list default)."""
    table = by_type or {}

    def respond(seen: Seen, _bearer: str) -> Reply:
        kind = (seen.query.get("type") or [""])[0]
        return json_reply({"data": table.get(kind, [])})

    return respond
