"""External MCP client for `venice chat --mcp` (issue #21, Direction B of #16).

The built-in agent loop (#15) can already call venice's own endpoints as tools.
This module adds a **second tool provider**: it connects, as an MCP *client*, to
third-party MCP servers registered via `venice config add` (filesystem, exec, git,
...), lists their tools, and exposes each as an `_agent.Tool` so the SAME
`_agent.run_loop` can call them alongside the built-ins. Nothing in the loop
changes -- it consumes a `list[Tool]`; we just append more entries.

Two hard problems shape the design:

1. **Async vs sync.** The `mcp` SDK client (`ClientSession`, `stdio_client`,
   `streamablehttp_client`, `sse_client`) is entirely async/anyio-based, while
   `run_loop` and the `openai` SDK it drives are synchronous. We bridge with a
   dedicated **background thread running its own asyncio loop**. A single
   "supervisor" coroutine owns every session for the whole `venice chat` (or REPL)
   session; each synchronous `Tool.invoke` hands a request to that coroutine over a
   queue and blocks on a `concurrent.futures.Future`.

   LOAD-BEARING INVARIANT: the anyio cancel scopes / task groups opened by
   `stdio_client(...)` and `ClientSession(...)` MUST be entered and exited in the
   *same task*, or anyio raises "Attempted to exit cancel scope in a different task
   than it was entered in". So ONE supervisor task opens the `AsyncExitStack`,
   serves every `call_tool`, and closes the stack -- all session I/O stays in that
   one task. Do NOT `run_coroutine_threadsafe(session.call_tool(...))` from
   elsewhere; route calls through the queue.

2. **Import discipline.** Like `_mcp`, this module must NOT import `mcp` at module
   scope -- it is pulled in eagerly by `register_all` (via `chat`), and the base,
   stdlib-only install (and Python 3.9, where the `mcp` SDK can't install) must keep
   working. Every `mcp` import is function-local, reached only after the caller has
   probed the SDK with `_mcp.import_mcp`.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import re
import sys
import threading
from typing import Dict, List, Optional, Tuple

from .. import auth
from .. import userconfig
from . import _agent


# --------------------------------------------------------------------------- #
# Pure helpers (no `mcp` import -- unit-testable with duck-typed objects)
# --------------------------------------------------------------------------- #
def resolve_specs(names, doc) -> List[Tuple[str, dict]]:
    """Map requested server names to ``(name, entry)`` from the #13 registry.

    Raises ``ValueError`` (naming the unknown server + what is registered) so the
    caller can print it and exit 2. Entry shape is validated loosely here; the
    stdio-vs-http branch happens later in ``_open_transport``.
    """
    specs: List[Tuple[str, dict]] = []
    for name in names:
        entry = userconfig.mcp_get(doc, name)
        if not isinstance(entry, dict) or not ("command" in entry or "url" in entry):
            available = ", ".join(sorted(userconfig.mcp_map(doc))) or "(none)"
            raise ValueError(
                f"unknown MCP server {name!r}; registered: {available}. "
                "Register one with `venice config add`."
            )
        specs.append((name, entry))
    return specs


# Secret-name charset mirrors the #43 store validator (^[A-Za-z0-9_.-]+$).
_SECRET_REF_RE = re.compile(r"@secret:([A-Za-z0-9_.-]+)")


def resolve_secret_refs(mapping, *, where: str):
    """Return a copy of ``mapping`` with each ``@secret:<name>`` token in a value
    replaced by the live secret from the store (``auth.load_secret``).

    ``@secret:<name>`` may appear anywhere inside a value (e.g. a bearer header
    ``"Bearer @secret:cluster"``); every occurrence is substituted independently.
    Values with no ref pass through unchanged, so existing plaintext ``env`` /
    ``headers`` entries are untouched (#70 is fully back-compatible). Raises
    ``ValueError`` (naming the secret + ``where``) if a referenced secret is unset,
    so a missing token surfaces as a clear attach-time error instead of a
    downstream 401.
    """
    if not mapping:
        return mapping

    def _sub(m):
        name = m.group(1)
        val = auth.load_secret(name)
        if val is None:
            raise ValueError(
                f"MCP server secret {name!r} referenced in {where} is not set -- "
                f"add it with 'venice secret set {name}' or set its env var"
            )
        return val

    resolved = {}
    for key, value in mapping.items():
        resolved[key] = _SECRET_REF_RE.sub(_sub, value) if isinstance(value, str) else value
    return resolved


_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_MAX_NAME = 64  # OpenAI/Venice function names must match ^[A-Za-z0-9_-]{1,64}$


def _sanitize(s: str) -> str:
    return _NAME_RE.sub("_", s) or "_"


def _advertised_name(server: str, tool: str, taken: set) -> str:
    """Namespace a remote tool as ``server__tool``, sanitized, <=64 chars, unique.

    Truncation/de-collision only affect the *advertised* name; the invoke closure
    keeps the original ``(server, tool)`` for dispatch, so the wire is unchanged.
    """
    base = f"{_sanitize(server)}__{_sanitize(tool)}"
    name = base[:_MAX_NAME]
    if name not in taken:
        taken.add(name)
        return name
    i = 2
    while True:
        suffix = f"_{i}"
        name = base[: _MAX_NAME - len(suffix)] + suffix
        if name not in taken:
            taken.add(name)
            return name
        i += 1


def _is_side_effecting(annotations) -> bool:
    """A tool is side-effecting unless it is explicitly annotated read-only.

    Conservative default (unknown -> side-effecting) so an unannotated write/exec
    tool is gated, not silently auto-run. Read-only tools skip the confirm gate.
    """
    if annotations is None:
        return True
    return getattr(annotations, "readOnlyHint", None) is not True


def _translate_result(result) -> dict:
    """Turn an MCP ``CallToolResult`` into the loop's JSON-serializable result dict.

    Joins text blocks; marks (never inlines) non-text content; maps ``isError`` to
    a ``{"status": "error"}`` dict the model can recover from; carries
    ``structuredContent`` through when present.
    """
    parts: List[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            btype = getattr(block, "type", None) or type(block).__name__
            parts.append(f"[non-text content: {btype}]")
    joined = "\n".join(parts)
    is_error = bool(getattr(result, "isError", False))
    out: dict = {"status": "error" if is_error else "ok"}
    if is_error:
        out["message"] = joined or "tool call failed"
    else:
        out["content"] = joined
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        out["structured"] = structured
    return out


_CONTROLLED = ("confirm", "max_spend", "output_dir")


def _clean_args(arguments) -> dict:
    """Strip loop-controlled keys before forwarding to the server (mirrors
    ``_agent._clean``): defense-in-depth so a model can't smuggle them through."""
    if not isinstance(arguments, dict):
        return {}
    return {k: v for k, v in arguments.items() if k not in _CONTROLLED}


# --------------------------------------------------------------------------- #
# Timeouts
# --------------------------------------------------------------------------- #
DEFAULT_CONNECT_TIMEOUT = 30.0  # seconds to open + initialize + list_tools a server
DEFAULT_CALL_TIMEOUT = 60.0     # seconds a single tool call may run


def _safe_errlog():
    """A stderr stream with a real file descriptor for the child server's stderr.

    Prefer the live ``sys.stderr``, falling back to the interpreter's original
    ``sys.__stderr__`` when the current one has no ``fileno()`` (a captured
    ``StringIO`` under tests, or a user redirect). anyio's ``open_process`` calls
    ``.fileno()`` on this stream, so it must be a real fd. Passed explicitly rather
    than relying on ``stdio_client``'s default, whose value is bound once at import
    and can be a captured stream if ``mcp`` was first imported mid-redirect.
    """
    for stream in (sys.stderr, sys.__stderr__):
        try:
            if stream is not None:
                stream.fileno()
                return stream
        except (OSError, ValueError, AttributeError):
            continue
    return None


def _resolve_timeout(explicit, env_name: str, default: float) -> float:
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass
    env = os.environ.get(env_name)
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return default


# --------------------------------------------------------------------------- #
# The async<->sync bridge
# --------------------------------------------------------------------------- #
_SHUTDOWN = object()  # sentinel enqueued to break the supervisor's serve loop


class _Bridge:
    """Owns a background asyncio loop + the supervisor coroutine that holds every
    MCP session open for the life of one ``attach()`` context."""

    def __init__(self, specs, *, connect_timeout: float, call_timeout: float):
        self._specs = specs
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._request_q: Optional[asyncio.Queue] = None
        self._ready: "concurrent.futures.Future" = concurrent.futures.Future()
        self._supervisor_future: Optional["concurrent.futures.Future"] = None
        self._closed = False

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> List[_agent.Tool]:
        """Spin up the loop thread, open all servers, return the translated tools.

        Self-cleans and re-raises on any setup failure (a missing binary, a hung
        ``initialize``, a server error) -- ``__exit__`` is not called when
        ``__enter__`` raises, so teardown must happen here.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="venice-mcp", daemon=True
        )
        self._thread.start()
        self._supervisor_future = asyncio.run_coroutine_threadsafe(
            self._supervise(), self._loop
        )
        try:
            descriptors = self._ready.result(timeout=self.connect_timeout + 5)
        except BaseException:
            self.close()
            raise
        return self._build_tools(descriptors)

    def close(self) -> None:
        """Signal shutdown, wait (bounded) for the supervisor to unwind the stack
        in-task, then stop the loop and join the thread. Ordering is fixed:
        aclose() reaps stdio subprocesses, so it must finish before loop.stop()."""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None and self._request_q is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._request_q.put_nowait, _SHUTDOWN)
        if self._supervisor_future is not None:
            with contextlib.suppress(Exception):
                self._supervisor_future.result(timeout=self.connect_timeout + 5)
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    # -- the supervisor (all session I/O lives here, in one task) ---------- #
    async def _supervise(self) -> None:
        self._request_q = asyncio.Queue()
        try:
            async with contextlib.AsyncExitStack() as stack:
                sessions, descriptors = await self._open_all(stack)
                self._ready.set_result(descriptors)
                await self._serve(sessions)
            # AsyncExitStack unwinds HERE, in this task -> anyio-safe.
        except BaseException as e:  # noqa: BLE001 - publish to the sync side
            if not self._ready.done():
                self._ready.set_exception(e)

    async def _open_all(self, stack):
        from mcp import ClientSession  # lazy: only after the caller probed the SDK

        sessions: Dict[str, object] = {}
        descriptors: List[dict] = []
        taken: set = set()
        for name, entry in self._specs:
            read, write = await self._open_transport(stack, entry)
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), self.connect_timeout)
            listed = await asyncio.wait_for(session.list_tools(), self.connect_timeout)
            sessions[name] = session
            for t in listed.tools:
                descriptors.append(
                    {
                        "advertised": _advertised_name(name, t.name, taken),
                        "server": name,
                        "real": t.name,
                        "description": t.description
                        or f"{t.name} (via MCP server {name!r})",
                        "parameters": t.inputSchema
                        or {"type": "object", "properties": {}},
                        "side_effecting": _is_side_effecting(
                            getattr(t, "annotations", None)
                        ),
                    }
                )
        return sessions, descriptors

    async def _open_transport(self, stack, entry: dict):
        """Enter the right transport context and return its ``(read, write)`` pair."""
        if "command" in entry:
            from mcp import StdioServerParameters
            from mcp.client.stdio import get_default_environment, stdio_client

            # StdioServerParameters.env REPLACES the child environment (no merge),
            # so a bare {"TOKEN": ...} would strip PATH/HOME and the server would
            # fail to spawn. Layer the registry env over the SDK's safe default.
            env = {**get_default_environment(),
                   **(resolve_secret_refs(entry.get("env"), where="env") or {})}
            params = StdioServerParameters(
                command=entry["command"],
                args=list(entry.get("args") or []),
                env=env,
            )
            # Pass errlog explicitly (live stderr) -- stdio_client's default is
            # bound at import and may be a captured stream; see _safe_errlog.
            read, write = await stack.enter_async_context(
                stdio_client(params, errlog=_safe_errlog())
            )
            return read, write

        url = entry.get("url")
        headers = resolve_secret_refs(entry.get("headers"), where="headers") or None
        if entry.get("type") == "sse":
            from mcp.client.sse import sse_client

            read, write = await stack.enter_async_context(
                sse_client(url, headers=headers)
            )
            return read, write

        from mcp.client.streamable_http import streamablehttp_client

        # streamablehttp yields (read, write, get_session_id); drop the third.
        transport = await stack.enter_async_context(
            streamablehttp_client(url, headers=headers)
        )
        return transport[0], transport[1]

    async def _serve(self, sessions) -> None:
        assert self._request_q is not None
        while True:
            req = await self._request_q.get()
            if req is _SHUTDOWN:
                return
            server, real, args, fut = req
            try:
                result = await sessions[server].call_tool(real, args)
                _safe_set(fut, _translate_result(result))
            except BaseException as e:  # noqa: BLE001 - report, keep serving
                _safe_set(fut, {"status": "error", "message": f"{server}/{real} failed: {e}"})

    # -- tool construction ------------------------------------------------- #
    def _build_tools(self, descriptors) -> List[_agent.Tool]:
        return [
            _agent.Tool(
                name=d["advertised"],
                description=d["description"],
                parameters=d["parameters"],
                invoke=self._make_invoke(
                    d["server"], d["real"], d["advertised"], d["side_effecting"]
                ),
                paid=d["side_effecting"],
            )
            for d in descriptors
        ]

    def _make_invoke(self, server, real, advertised, side_effecting):
        def invoke(arguments, *, confirm: bool = False):
            # Gate side-effecting tools WITHOUT a server round-trip; the loop's
            # `_resolve_spend` turns this into a TTY prompt / model-visible block,
            # and `--yes` (confirm=True) bypasses it (mirrors the paid built-ins).
            if side_effecting and not confirm:
                return {
                    "status": "confirmation_required",
                    "message": (
                        f"{advertised} is a side-effecting external tool "
                        f"(server {server!r}, not marked read-only); "
                        "re-run with --yes or confirm to proceed."
                    ),
                }
            args = _clean_args(arguments)
            fut: "concurrent.futures.Future" = concurrent.futures.Future()
            try:
                self._loop.call_soon_threadsafe(
                    self._request_q.put_nowait, (server, real, args, fut)
                )
            except RuntimeError as e:  # loop already stopped
                return {"status": "error", "message": f"{advertised}: cannot dispatch ({e})"}
            try:
                return fut.result(timeout=self.call_timeout)
            except concurrent.futures.TimeoutError:
                return {
                    "status": "error",
                    "message": f"{advertised}: timed out after {self.call_timeout}s",
                }
            except Exception as e:  # pragma: no cover - defensive
                return {"status": "error", "message": f"{advertised} failed: {e}"}

        return invoke


def _safe_set(fut: "concurrent.futures.Future", value) -> None:
    """Set a result unless the sync side already abandoned the future (post-timeout)."""
    if not fut.done():
        with contextlib.suppress(concurrent.futures.InvalidStateError):
            fut.set_result(value)


@contextlib.contextmanager
def attach(specs, *, connect_timeout=None, call_timeout=None):
    """Open the given MCP servers for the duration of the ``with`` block.

    ``specs`` is a list of ``(name, entry)`` from :func:`resolve_specs`. Yields a
    ``list[_agent.Tool]`` (possibly empty) ready to concatenate with the built-in
    tools. Servers are torn down -- subprocesses reaped, sockets closed, thread
    joined -- on exit, whether the body succeeds, raises, or setup itself fails.
    """
    bridge = _Bridge(
        specs,
        connect_timeout=_resolve_timeout(
            connect_timeout, "VENICE_MCP_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT
        ),
        call_timeout=_resolve_timeout(
            call_timeout, "VENICE_MCP_CALL_TIMEOUT", DEFAULT_CALL_TIMEOUT
        ),
    )
    tools = bridge.start()  # raises (after self-clean) on setup failure
    try:
        yield tools
    finally:
        bridge.close()
