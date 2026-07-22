"""Web-fetch + headless-browser rails for the agent's `web_fetch` / `browser_capture`
tools (issue #71).

**stdlib only** -- like `_exec`, this module imports nothing from the package except the
shared output cap, so it stays a dependency leaf (no import cycle with `_agent`, which
builds the Tools from these primitives). It provides:

- a browser-binary probe (`find_browser`) mirroring `audio_player.find_player`:
  Chromium-family first (it adds `--dump-dom` = post-JS HTML + `--virtual-time-budget`),
  Firefox last (its headless CLI is screenshot-only);
- a URL safety policy (`check_url_policy`) mirroring the *shape* of `_exec.check_policy`
  (deny-wins, returns a refusal string or None) with a hardcoded default-deny that config
  cannot re-open: http/https only, cloud-metadata endpoint always blocked;
- `web_fetch` -- a urllib GET with a per-redirect-hop policy guard and size caps;
- `capture` -- a headless screenshot and/or post-JS DOM dump via a subprocess.

A headless browser executes untrusted remote JS and inherits the parent env, so the child
env here is an *allowlist* (`_browser_env`), stricter than `_exec._scrubbed_env` which only
drops two named keys -- that keeps `VENICE_API_KEY` and every other ambient token out of a
hostile page's reach. The browser also runs under a throwaway `--user-data-dir`.
"""
from __future__ import annotations

import fnmatch
import html as _htmlmod
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

# Single source of truth for the output cap (shared with the shell rail).
from ._exec import MAX_OUTPUT_CHARS

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_WINDOW = (1280, 900)
DEFAULT_WAIT_MS = 4000            # chromium --virtual-time-budget: let JS settle
DEFAULT_TIMEOUT = 30             # seconds per fetch / browser subprocess
MAX_FETCH_BYTES = 2_000_000     # cap on a web_fetch download (~2 MB)

_UA = "venice-cli/web_fetch (+https://github.com/gobha-me/venice-cli)"
_SCHEMES = ("http", "https")

# Hosts refused regardless of operator policy (SSRF hard stops: cloud metadata services).
_BLOCKED_HOSTS = frozenset({"169.254.169.254", "metadata.google.internal"})
# The IPv4 link-local range covers the metadata IP in ANY of inet_aton's accepted encodings
# (dotted/decimal/hex/octal), closing the trivial `http://2852039166/` SSRF bypass. Plus the
# AWS IPv6 metadata literal. DNS names are matched only literally (we don't resolve -- that
# would add a lookup + a TOCTOU window).
_LINK_LOCAL_V4 = ipaddress.ip_network("169.254.0.0/16")
_BLOCKED_V6 = frozenset({ipaddress.ip_address("fd00:ec2::254")})

# Env keys allowed through to the browser child. Everything else -- VENICE_API_KEY and any
# other ambient token -- is dropped, since the child runs untrusted remote JS. `LC_*` is
# passed by prefix. `DISPLAY` is intentionally absent to keep it headless.
_ENV_ALLOW = (
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR",
    "XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME",
)


# --------------------------------------------------------------------------- #
# Browser-binary probe (mirrors audio_player.find_player)
# --------------------------------------------------------------------------- #
# (binary name, family). Chromium-family first: it supports --dump-dom (post-JS HTML) and
# --virtual-time-budget. Firefox is a screenshot-only fallback (no DOM dump in its CLI).
_BROWSERS: List[Tuple[str, str]] = [
    ("chromium", "chromium"),
    ("chromium-browser", "chromium"),
    ("google-chrome", "chromium"),
    ("google-chrome-stable", "chromium"),
    ("chrome", "chromium"),
    ("brave-browser", "chromium"),
    ("brave-browser-stable", "chromium"),
    ("brave", "chromium"),
    ("firefox", "firefox"),
]


def find_browser() -> Optional[Tuple[str, str]]:
    """First available ``(path, family)`` from the probe order, or ``None``."""
    for name, family in _BROWSERS:
        path = shutil.which(name)
        if path:
            return path, family
    return None


def browser_names() -> str:
    return ", ".join(name for name, _ in _BROWSERS)


def _parse_ip(host: str):
    """`host` as an ipaddress, accepting inet_aton's int/hex/octal IPv4 forms; else None."""
    try:
        return ipaddress.ip_address(socket.inet_aton(host))   # 2852039166, 0xA9FEA9FE, ...
    except OSError:
        pass
    try:
        return ipaddress.ip_address(host)                     # canonical v4/v6 literal
    except ValueError:
        return None


def _is_blocked_host(host: str) -> bool:
    """True for a cloud-metadata endpoint (name or IP, in any IPv4 encoding / the v6 literal)."""
    if host in _BLOCKED_HOSTS:
        return True
    ip = _parse_ip(host)
    if ip is None:
        return False
    if ip.version == 4:
        return ip in _LINK_LOCAL_V4
    return ip in _BLOCKED_V6


def capture_filename(url: str) -> str:
    """A stable, filesystem-safe screenshot filename derived from a URL's host."""
    try:
        host = urllib.parse.urlsplit(str(url)).hostname or "page"
    except ValueError:
        host = "page"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", host).strip("-") or "page"
    return f"capture-{slug}.png"


def _browser_env() -> dict:
    """A minimal allowlisted env for the browser child (drops VENICE_API_KEY + tokens)."""
    env = {k: v for k, v in os.environ.items()
           if k in _ENV_ALLOW or k.startswith("LC_")}
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    return env


# --------------------------------------------------------------------------- #
# URL policy (mirrors the shape of _exec.check_policy)
# --------------------------------------------------------------------------- #
def check_url_policy(url, *, allow=(), deny=()) -> Optional[str]:
    """Return a refusal message if `url` is blocked, else ``None``.

    Hardcoded default-deny that config cannot re-open: the scheme must be http/https and
    the cloud-metadata host is always blocked. On top of that, operator `deny` globs
    (matched against the host and the full URL) always refuse and win over allow; a
    non-empty `allow` restricts to hosts matching one of its globs. Empty allow = any
    host permitted (still subject to the hardcoded stops + deny).
    """
    u = str(url or "").strip()
    if not u:
        return "url is required"
    try:
        parts = urllib.parse.urlsplit(u)
    except ValueError:
        return f"could not parse url: {u}"
    scheme = (parts.scheme or "").lower()
    if scheme not in _SCHEMES:
        return (f"blocked url scheme {scheme or '(none)'!r}: only http/https are allowed "
                f"(no file://, data:, ftp, ...): {u}")
    try:
        host = (parts.hostname or "").lower()
    except ValueError:
        return f"could not parse url host: {u}"
    if not host:
        return f"url has no host: {u}"
    if _is_blocked_host(host):
        return f"blocked host {host!r} (cloud metadata / link-local endpoint): {u}"

    deny = [str(d) for d in (deny or [])]
    for pat in deny:
        if fnmatch.fnmatch(host, pat) or fnmatch.fnmatch(u, pat):
            return f"blocked by browser deny policy ({pat!r}): {u}"

    allow = [str(a) for a in (allow or [])]
    if allow and not any(fnmatch.fnmatch(host, a) for a in allow):
        return f"{host!r} is not in the browser allowlist ({', '.join(allow)}): {u}"
    return None


# --------------------------------------------------------------------------- #
# web_fetch (stdlib urllib, redirect-hop guarded)
# --------------------------------------------------------------------------- #
class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Re-apply the URL policy to every redirect hop. urllib auto-follows redirects, so a
    302 -> the metadata endpoint (or file://) must be re-checked, not just the initial URL."""

    def __init__(self, allow, deny):
        self._allow = allow
        self._deny = deny

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        blocked = check_url_policy(newurl, allow=self._allow, deny=self._deny)
        if blocked:
            raise urllib.error.HTTPError(newurl, code, f"blocked redirect: {blocked}",
                                         headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if m:
        charset = m.group(1)
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, TypeError):
        return raw.decode("utf-8", errors="replace")


_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_INLINE_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n\s*")


def html_to_text(text: str) -> str:
    """Very light HTML -> text: drop script/style, strip tags, unescape, collapse space."""
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _TAGS.sub(" ", text)
    text = _htmlmod.unescape(text)
    text = _INLINE_WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def web_fetch(url, *, mode="text", max_bytes=None, timeout=None, allow=(), deny=()) -> dict:
    """GET `url` and return its body as text or raw html. Never raises.

    Returns ``{"ok": True, "final_url", "content_type", "text"|"html", "truncated"}`` or
    ``{"ok": False, "error": msg}``.
    """
    blocked = check_url_policy(url, allow=allow, deny=deny)
    if blocked:
        return {"ok": False, "error": blocked}
    try:
        cap = int(max_bytes) if max_bytes else MAX_FETCH_BYTES
    except (TypeError, ValueError):
        cap = MAX_FETCH_BYTES
    try:
        t = int(timeout) if timeout else DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        t = DEFAULT_TIMEOUT

    opener = urllib.request.build_opener(_GuardedRedirect(allow, deny))
    req = urllib.request.Request(str(url), headers={"User-Agent": _UA})
    try:
        with opener.open(req, timeout=t) as resp:
            raw = resp.read(cap + 1)
            final_url = resp.geturl()
            ctype = resp.headers.get("Content-Type", "") or ""
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http {e.code}: {e.reason} ({url})"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", e)
        return {"ok": False, "error": f"fetch failed: {reason} ({url})"}

    over_bytes = len(raw) > cap
    text = _decode(raw[:cap], ctype)
    if mode == "html":
        body, key = text, "html"
    else:
        body, key = html_to_text(text), "text"
    return {
        "ok": True,
        "final_url": final_url,
        "content_type": ctype,
        key: body[:MAX_OUTPUT_CHARS],
        "truncated": over_bytes or len(body) > MAX_OUTPUT_CHARS,
    }


# --------------------------------------------------------------------------- #
# capture (headless screenshot / post-JS DOM dump via subprocess)
# --------------------------------------------------------------------------- #
_CAPTURE_MODES = ("dom", "text", "screenshot", "both")


def _run(argv, *, timeout: int) -> Tuple[Optional[int], str, str, Optional[str]]:
    """Run `argv` (no shell) with the scrubbed browser env. (rc, stdout, stderr, error)."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=int(timeout),
            env=_browser_env(), stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None, "", "", "no headless browser available"
    except subprocess.TimeoutExpired:
        return None, "", "", f"browser timed out after {int(timeout)}s"
    except OSError as e:
        return None, "", "", f"browser failed: {e}"
    return proc.returncode, proc.stdout or "", proc.stderr or "", None


def _shot_argv(path, family, url, *, out_path, wait_ms, window, profile) -> list:
    w, h = window
    if family == "firefox":
        # Verified: firefox --headless --window-size=W,H --screenshot out.png <url>
        return [path, "--headless", f"--window-size={w},{h}",
                "--screenshot", str(out_path), str(url)]
    return [path, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--disable-dev-shm-usage", f"--user-data-dir={profile}",
            f"--window-size={w},{h}", f"--virtual-time-budget={wait_ms}",
            f"--screenshot={out_path}", str(url)]


def _dom_argv(path, url, *, wait_ms, profile) -> list:
    return [path, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--disable-dev-shm-usage", f"--user-data-dir={profile}",
            f"--virtual-time-budget={wait_ms}", "--dump-dom", str(url)]


def capture(url, *, out_path=None, mode="dom", wait_ms=None, window=None, timeout=None,
            assert_contains=None, allow=(), deny=()) -> dict:
    """Headless screenshot and/or post-JS DOM dump of `url`. Never raises.

    Returns ``{"ok": True, "browser", "family", "screenshot_path"?, "dom"?, "truncated"?,
    "assert_contains"?, "contains"?}`` or ``{"ok": False, "error": msg}``. `out_path` is
    required for modes that produce a screenshot. DOM modes need a Chromium-family browser.
    """
    m = str(mode or "dom")
    if m not in _CAPTURE_MODES:
        return {"ok": False, "error": f"unknown mode {m!r} (use {'/'.join(_CAPTURE_MODES)})"}
    blocked = check_url_policy(url, allow=allow, deny=deny)
    if blocked:
        return {"ok": False, "error": blocked}
    found = find_browser()
    if not found:
        return {"ok": False, "error": ("no headless browser available (looked for "
                                        f"{browser_names()}); install one to use "
                                        "browser_capture")}
    path, family = found
    try:
        wait = int(wait_ms) if wait_ms else DEFAULT_WAIT_MS
    except (TypeError, ValueError):
        wait = DEFAULT_WAIT_MS
    try:
        t = int(timeout) if timeout else DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        t = DEFAULT_TIMEOUT
    win = window or DEFAULT_WINDOW

    wants_dom = m in ("dom", "text", "both")
    wants_shot = m in ("screenshot", "both")
    if wants_dom and family == "firefox":
        return {"ok": False, "error": (
            "the available browser is Firefox, whose headless CLI is screenshot-only; "
            "use mode='screenshot', or install a Chromium-family browser "
            "(chromium/chrome/brave) for a DOM dump")}
    if wants_shot and not out_path:
        return {"ok": False, "error": "out_path is required for a screenshot"}

    result = {"ok": True, "browser": os.path.basename(path), "family": family}
    with tempfile.TemporaryDirectory(prefix="venice-browser-") as profile:
        if wants_shot:
            argv = _shot_argv(path, family, url, out_path=out_path, wait_ms=wait,
                              window=win, profile=profile)
            rc, _out, err, error = _run(argv, timeout=t)
            if error:
                return {"ok": False, "error": error}
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                return {"ok": False, "error": (
                    f"browser produced no screenshot (exit {rc}); "
                    f"stderr: {err[:500].strip()}")}
            result["screenshot_path"] = str(out_path)
        if wants_dom:
            argv = _dom_argv(path, url, wait_ms=wait, profile=profile)
            rc, out, err, error = _run(argv, timeout=t)
            if error:
                return {"ok": False, "error": error}
            if rc != 0 and not out:
                return {"ok": False, "error": (
                    f"browser dump-dom failed (exit {rc}); stderr: {err[:500].strip()}")}
            dom = html_to_text(out) if m == "text" else out
            if assert_contains is not None:
                needle = str(assert_contains)
                result["assert_contains"] = needle
                result["contains"] = needle in dom  # full DOM, before truncation
            result["truncated"] = len(dom) > MAX_OUTPUT_CHARS
            result["dom"] = dom[:MAX_OUTPUT_CHARS]
    return result
