"""Fail-closed browser rail containment.

The former implementation launched a headless browser after checking only the
initial URL. Redirects, DNS resolution, subresources, and page-script requests
then escaped that policy boundary. Until the rail has an enforced egress
boundary, keep every entry point unavailable and dependency-free.
"""
from __future__ import annotations


UNAVAILABLE_MESSAGE = (
    "web and browser tools are temporarily disabled for security; "
    "see GHSA-mqjr-2vh8-6fvg"
)


def web_fetch(
    url, *, mode="text", max_bytes=None, timeout=None, allow=(), deny=()
) -> dict:
    """Fail closed without parsing, resolving, or opening ``url``."""
    return {"ok": False, "error": UNAVAILABLE_MESSAGE}


def capture(
    url, *, out_path=None, mode="dom", wait_ms=None, window=None,
    timeout=None, assert_contains=None, allow=(), deny=(),
) -> dict:
    """Fail closed without probing or launching a browser."""
    return {"ok": False, "error": UNAVAILABLE_MESSAGE}
