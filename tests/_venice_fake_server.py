"""A tiny in-process fake Venice API, used as a fixture by the #80 drive suite.

The drive tests (``tests/test_drive_cli.py``) spawn the *real* CLI as a child
process, so they cannot patch ``urlopen`` the way the rest of the suite does --
the seam has to be a socket. This module is that seam: a stdlib
``ThreadingHTTPServer`` bound to ``127.0.0.1:0``, started in a background thread
inside the test process, with the child pointed at it via ``$VENICE_BASE_URL``.

Hermeticity is unchanged from the rest of the suite (see CONTRIBUTING.md): it
binds loopback on an ephemeral port, never reaches the network, and never sees a
real key -- the drive tests pass ``VENICE_API_KEY=test-fake-key``. The
``Authorization`` header is recorded as a **boolean only**; its value is never
stored, logged, or printed (CLAUDE.md).

Unlike ``_mcp_fake_server.py`` this runs in-process rather than as a subprocess:
that one *has* to be spawned because its module-scope ``import mcp`` fails on
3.9, which is stdlib-only code's problem to not have. Running in-process is what
buys the assertions a mocked ``urlopen`` cannot make from the outside -- which
endpoints the child actually hit (``"/image/generate" not in api.paths`` proves
a declined charge never spent money) and what was in each request body (proving
a REPL ``/model`` switch reached the wire, not just the local state dict).

The leading underscore keeps it out of unittest discovery, per the repo's
fixture convention.

Usage::

    with FakeVenice() as api:
        api.reply("HELLO")                 # queue one chat completion
        ...                                # drive the CLI at api.base_url
        assert api.paths == ["/models", "/chat/completions"]
"""
from __future__ import annotations

import base64
import collections
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The real base URL carries this prefix, so the fake mirrors it -- that way the
# child's $VENICE_BASE_URL looks exactly like production and any prefix-handling
# bug in the client shows up here instead of hiding.
PREFIX = "/api/v1"

# Model catalogs. The text shape is copied from the proven fixture in
# tests/test_chat.py::_text_payload -- `model_spec.traits: ["default"]` is what
# _models.default_model() scans for, and without it resolve_model() exits 6.
TEXT_MODELS = [
    {
        "id": "llama-3.3-70b",
        "type": "text",
        "model_spec": {
            "traits": ["default"],
            "capabilities": {"supportsFunctionCalling": True},
        },
    },
    {
        "id": "venice-uncensored",
        "type": "text",
        "model_spec": {
            "traits": [],
            "capabilities": {"supportsFunctionCalling": True},
        },
    },
]

# id matches image.DEFAULT_IMAGE_MODEL so `venice image` needs no --model.
# _usd_from_pricing() digs the first nested {"usd": N} out of model_spec.pricing,
# so 1 image costs $0.0100 -- billing.format_usd uses 4 decimals below $1.
IMAGE_MODELS = [
    {
        "id": "venice-sd35",
        "type": "image",
        "model_spec": {"pricing": {"image": {"usd": 0.01}}},
    },
]

# billing.fetch_balance sums USD + DIEM into `total`, so this renders as
# "$12.50 USD (0.00 DIEM allowance + 12.50 USD cash)".
RATE_LIMITS = {
    "data": {
        "balances": {"USD": 12.5, "DIEM": 0.0},
        "apiTier": {"id": "explorer", "isCharged": True},
        "nextEpochBegins": "2026-01-01T00:00:00Z",
        "keyExpiration": None,
    }
}

# image._decode_images() only base64-decodes and writes, so any bytes will do.
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"

_USAGE = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}


class FakeVenice:
    """A scriptable fake Venice API. Start it with ``with FakeVenice() as api:``."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = []  # [{"method", "path", "query", "body", "has_auth"}]
        self._chat = collections.deque()
        self._httpd = None
        self._thread = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FakeVenice":
        handler = _make_handler(self)
        # Port 0 -> the kernel picks a free one, so parallel runs never collide
        # and there is no readiness sleep to tune: bind() completes before
        # serve_forever() starts, so the port is connectable the moment we
        # return it.
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "FakeVenice":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        """What to put in the child's $VENICE_BASE_URL."""
        return "http://127.0.0.1:%d%s" % (self.port, PREFIX)

    # -- scripting ---------------------------------------------------------

    def reply(self, text: str, *, usage=None) -> None:
        """Queue one chat completion whose whole content is `text`."""
        self.reply_chunks(text, usage=usage)

    def reply_chunks(self, *deltas: str, usage=None) -> None:
        """Queue one chat completion streamed as the given deltas."""
        with self._lock:
            self._chat.append({"deltas": list(deltas), "usage": usage or _USAGE})

    def _next_chat(self) -> dict:
        with self._lock:
            if self._chat:
                return self._chat.popleft()
        # An unscripted turn is a test bug; make it obvious in the transcript
        # rather than hanging or 500-ing.
        return {"deltas": ["UNSCRIPTED-REPLY"], "usage": _USAGE}

    # -- inspection --------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self.requests = []
            self._chat.clear()

    @property
    def paths(self):
        """Paths hit so far, prefix-stripped, in order."""
        with self._lock:
            return [r["path"] for r in self.requests]

    def bodies(self, path: str):
        """Parsed JSON bodies of every request to `path`, in order."""
        with self._lock:
            return [r["body"] for r in self.requests if r["path"] == path]

    def _record(self, entry: dict) -> None:
        with self._lock:
            self.requests.append(entry)


def _make_handler(api: FakeVenice):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 means the response body is connection-close delimited, so the
        # SSE path needs no Content-Length and no chunked framing.
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
            pass  # keep the unittest output clean

        # -- plumbing ------------------------------------------------------

        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None

        def _dispatch(self, method: str):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path.startswith(PREFIX):
                path = path[len(PREFIX):]
            body = self._read_body()
            api._record({
                "method": method,
                "path": path,
                "query": urllib.parse.parse_qs(parsed.query),
                "body": body,
                # Presence only. The value is never recorded (CLAUDE.md).
                "has_auth": bool(self.headers.get("Authorization")),
            })
            return path, urllib.parse.parse_qs(parsed.query), body

        def _send_json(self, payload, status=200):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def _send_404(self, path):
            # Deliberately 404 and never 5xx/429: the OpenAI SDK retries those
            # twice with backoff, which would turn a typo'd route into a slow
            # mystery instead of an immediate, readable failure.
            self._send_json(
                {"error": {"code": "NOT_FOUND", "message": path}}, status=404
            )

        # -- routes --------------------------------------------------------

        def do_GET(self):  # noqa: N802 - stdlib hook name
            path, query, _ = self._dispatch("GET")
            if path == "/models":
                kind = (query.get("type") or ["text"])[0]
                data = {"text": TEXT_MODELS, "image": IMAGE_MODELS}.get(kind, [])
                self._send_json({"object": "list", "data": data})
            elif path == "/api_keys/rate_limits":
                self._send_json(RATE_LIMITS)
            else:
                self._send_404(path)

        def do_POST(self):  # noqa: N802 - stdlib hook name
            path, _, body = self._dispatch("POST")
            if path == "/image/generate":
                self._send_json({
                    "images": [base64.b64encode(PNG_BYTES).decode("ascii")],
                    "request": {"data": body or {}},
                })
            elif path == "/chat/completions":
                scripted = api._next_chat()
                model = (body or {}).get("model", "llama-3.3-70b")
                if (body or {}).get("stream"):
                    self._send_sse(scripted, model)
                else:
                    self._send_completion(scripted, model)
            else:
                self._send_404(path)

        # -- chat completions ----------------------------------------------

        def _send_completion(self, scripted, model):
            self._send_json({
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(scripted["deltas"]),
                    },
                    "finish_reason": "stop",
                }],
                "usage": scripted["usage"],
            })

        def _send_sse(self, scripted, model):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def frame(choices, usage=None):
                payload = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": choices,
                }
                if usage is not None:
                    payload["usage"] = usage
                self.wfile.write(
                    ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")
                )
                self.wfile.flush()

            # No sleep between frames: sequential expect() calls already prove
            # ordering, so artificial delay would only buy flake.
            frame([{"index": 0, "delta": {"role": "assistant"},
                    "finish_reason": None}])
            for piece in scripted["deltas"]:
                frame([{"index": 0, "delta": {"content": piece},
                        "finish_reason": None}])
            frame([{"index": 0, "delta": {}, "finish_reason": "stop"}])
            # `venice chat` always sends stream_options={"include_usage": true},
            # so the final usage frame (empty `choices`) is what produces the
            # "usage: prompt=..." line -- the cleanest end-of-stream anchor.
            frame([], usage=scripted["usage"])
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler
