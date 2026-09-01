"""Pinned web fetch and sandboxed Chromium rails for untrusted pages.

Both paths use :class:`venice._egress.DestinationPolicy`. Browser traffic is forced
through a disposable loopback proxy with no direct fallback, so redirects, frames,
scripts, CSS, images, fetch/XHR, workers, WebSockets, prefetches, and downloads all
cross the same resolve-and-pin boundary.
"""
from __future__ import annotations

import html as _htmlmod
import os
import re
import signal
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

from venice import _egress
from . import _browser_proxy
from ._exec import MAX_OUTPUT_CHARS

DEFAULT_WINDOW = (1280, 900)
DEFAULT_WAIT_MS = 4000
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 60
MAX_WAIT_MS = 30_000
MAX_FETCH_BYTES = 2_000_000
_UA = "venice-cli/web_fetch (+https://github.com/gobha-me/venice-cli)"
_CAPTURE_MODES = ("dom", "text", "screenshot", "both")
_ENV_ALLOW = (
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME",
)
_BROWSERS: List[Tuple[str, str]] = [
    ("chromium", "chromium"),
    ("chromium-browser", "chromium"),
    ("google-chrome", "chromium"),
    ("google-chrome-stable", "chromium"),
    ("chrome", "chromium"),
    ("brave-browser", "chromium"),
    ("brave-browser-stable", "chromium"),
    ("brave", "chromium"),
]
_BROWSER_BINARIES = (
    "/opt/google/chrome/chrome",
    "/opt/brave.com/brave/brave",
    "/usr/lib/chromium/chromium",
    "/usr/lib/chromium-browser/chromium-browser",
)


def find_browser() -> Optional[Tuple[str, str]]:
    for path in _BROWSER_BINARIES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path, "chromium"
    for name, family in _BROWSERS:
        path = shutil.which(name)
        if path:
            return path, family
    return None


def browser_names() -> str:
    return ", ".join(name for name, _family in _BROWSERS)


def capture_filename(url: str) -> str:
    try:
        host = urllib.parse.urlsplit(str(url)).hostname or "page"
    except ValueError:
        host = "page"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", host).strip("-") or "page"
    return f"capture-{slug}.png"


def _browser_env(profile=None) -> dict:
    env = {
        key: value for key, value in os.environ.items()
        if key in _ENV_ALLOW or key.startswith("LC_")
    }
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    if profile is not None:
        env["HOME"] = str(profile)
        env["XDG_CACHE_HOME"] = os.path.join(str(profile), "cache")
        env["XDG_CONFIG_HOME"] = os.path.join(str(profile), "config")
    return env


def _policy(*, allow=(), deny=(), private_hosts=(), private_ranges=()):
    return _egress.DestinationPolicy.create(
        allow=allow, deny=deny, private_hosts=private_hosts,
        private_ranges=private_ranges,
    )


def check_url_policy(
    url, *, allow=(), deny=(), private_hosts=(), private_ranges=()
) -> Optional[str]:
    try:
        policy = _policy(
            allow=allow, deny=deny, private_hosts=private_hosts,
            private_ranges=private_ranges,
        )
        host, port, _scheme = policy.endpoint(str(url))
        policy.resolve(host, port)
    except (_egress.EgressPolicyError, OSError) as exc:
        return str(exc)
    return None


def _bounded_int(value, default, *, minimum=1, maximum):
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _decode(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if match:
        charset = match.group(1)
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, TypeError):
        return raw.decode("utf-8", errors="replace")


_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_INLINE_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n\s*")


def html_to_text(text: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _TAGS.sub(" ", text)
    text = _htmlmod.unescape(text)
    text = _INLINE_WS.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def web_fetch(
    url, *, mode="text", max_bytes=None, timeout=None, allow=(), deny=(),
    private_hosts=(), private_ranges=(),
) -> dict:
    if mode not in ("text", "html"):
        return {"ok": False, "error": "unknown mode (use text or html)"}
    try:
        policy = _policy(
            allow=allow, deny=deny, private_hosts=private_hosts,
            private_ranges=private_ranges,
        )
        policy.endpoint(str(url))
    except _egress.EgressPolicyError as exc:
        return {"ok": False, "error": str(exc)}
    cap = _bounded_int(max_bytes, MAX_FETCH_BYTES, maximum=MAX_FETCH_BYTES)
    seconds = _bounded_int(timeout, DEFAULT_TIMEOUT, maximum=MAX_TIMEOUT)
    opener = _egress.build_policy_opener(policy)
    request = urllib.request.Request(str(url), headers={"User-Agent": _UA})
    try:
        with opener.open(request, timeout=seconds) as response:
            raw = response.read(cap + 1)
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "") or ""
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "error": f"fetch failed: {getattr(exc, 'reason', exc)}"}
    decoded = _decode(raw[:cap], content_type)
    body = decoded if mode == "html" else html_to_text(decoded)
    key = "html" if mode == "html" else "text"
    return {
        "ok": True,
        "final_url": _egress.safe_url(final_url),
        "content_type": content_type,
        key: body[:MAX_OUTPUT_CHARS],
        "truncated": len(raw) > cap or len(body) > MAX_OUTPUT_CHARS,
    }


def _chromium_flags(proxy_url: str, profile: str) -> list:
    return [
        "--headless=new", "--disable-gpu", "--disable-dev-shm-usage",
        "--disable-background-networking", "--disable-component-update",
        "--disable-default-apps", "--disable-extensions", "--disable-sync",
        "--disable-quic", "--metrics-recording-only", "--no-first-run",
        "--no-default-browser-check", "--password-store=basic", "--use-mock-keychain",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        f"--proxy-server={proxy_url}", "--proxy-bypass-list=<-loopback>",
        f"--user-data-dir={profile}",
    ]


def _run(argv, *, timeout: int, profile=None):
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=_browser_env(profile), stdin=subprocess.DEVNULL,
            cwd=str(profile) if profile is not None else None,
            start_new_session=(os.name == "posix"),
        )
        out, err = proc.communicate(timeout=timeout)
    except FileNotFoundError:
        return None, "", "", "no Chromium-family browser available"
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - exercised on Windows CI
            proc.kill()
        proc.communicate()
        return None, "", "", f"browser timed out after {timeout}s"
    except OSError as exc:
        return None, "", "", f"browser failed: {exc}"
    return proc.returncode, out or "", err or "", None


def capture(
    url, *, out_path=None, mode="dom", wait_ms=None, window=None, timeout=None,
    assert_contains=None, allow=(), deny=(), private_hosts=(), private_ranges=(),
) -> dict:
    mode = str(mode or "dom")
    if mode not in _CAPTURE_MODES:
        return {"ok": False, "error": f"unknown mode {mode!r} (use {'/'.join(_CAPTURE_MODES)})"}
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return {"ok": False, "error": "browser_capture refuses to run Chromium as root"}
    try:
        policy = _policy(
            allow=allow, deny=deny, private_hosts=private_hosts,
            private_ranges=private_ranges,
        )
        host, port, _scheme = policy.endpoint(str(url))
        policy.resolve(host, port)
    except (_egress.EgressPolicyError, OSError) as exc:
        return {"ok": False, "error": str(exc)}
    found = find_browser()
    if not found:
        return {"ok": False, "error": f"no Chromium-family browser available (looked for {browser_names()})"}
    path, family = found
    wants_dom = mode in ("dom", "text", "both")
    wants_shot = mode in ("screenshot", "both")
    if wants_shot and not out_path:
        return {"ok": False, "error": "out_path is required for a screenshot"}
    wait = _bounded_int(wait_ms, DEFAULT_WAIT_MS, minimum=0, maximum=MAX_WAIT_MS)
    seconds = _bounded_int(timeout, DEFAULT_TIMEOUT, maximum=MAX_TIMEOUT)
    win = window or DEFAULT_WINDOW
    try:
        width, height = int(win[0]), int(win[1])
        if not (1 <= width <= 7680 and 1 <= height <= 4320):
            raise ValueError
    except (TypeError, ValueError, IndexError):
        return {"ok": False, "error": "window must be a width/height pair within 7680x4320"}

    result = {"ok": True, "browser": os.path.basename(path), "family": family}
    with tempfile.TemporaryDirectory(prefix="venice-browser-") as profile:
        with _browser_proxy.policy_proxy(policy) as proxy_url:
            base = [path, *_chromium_flags(proxy_url, profile),
                    f"--window-size={width},{height}", f"--virtual-time-budget={wait}"]
            argv = list(base)
            if wants_shot:
                argv.append(f"--screenshot={out_path}")
            if wants_dom:
                argv.append("--dump-dom")
            argv.append(str(url))
            rc, out, err, error = _run(argv, timeout=seconds, profile=profile)
            if error:
                return {"ok": False, "error": error}
            if wants_shot:
                if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                    return {"ok": False, "error": f"browser produced no screenshot (exit {rc}); stderr: {err[:500].strip()}"}
                result["screenshot_path"] = str(out_path)
            if wants_dom:
                if rc != 0 and not out:
                    return {"ok": False, "error": f"browser dump-dom failed (exit {rc}); stderr: {err[:500].strip()}"}
                dom = html_to_text(out) if mode == "text" else out
                if assert_contains is not None:
                    needle = str(assert_contains)
                    result["assert_contains"] = needle
                    result["contains"] = needle in dom
                result["truncated"] = len(dom) > MAX_OUTPUT_CHARS
                result["dom"] = dom[:MAX_OUTPUT_CHARS]
    return result
