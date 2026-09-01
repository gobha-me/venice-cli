"""Shared `openai` SDK plumbing for Venice's OpenAI-compatible endpoints.

Extracted from `chat` so it and `embed` share one copy of the lazy-import probe,
the SDK client construction, and the exception -> exit-code mapping rather than
each carrying its own. These helpers take primitive args (a label for messages,
the imported module, an exception) so they stay independent of any one command's
argument shape.

The SDK is imported lazily -- the rest of the CLI is stdlib-only, so a missing
`openai` must degrade to a hint and exit 2 rather than break `venice --help`.
Callers probe the import *first*, before building a client or fetching a
catalog, so the missing-SDK path never touches the network.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


_PROMPT_CACHE_KEY = "prompt_cache_key"


def prompt_cache_key(kwargs: Optional[dict]) -> Optional[str]:
    """The OpenAI-compatible cache-affinity key carried by ``extra_body``.

    The project supports ``openai>=1.40``, predating the SDK's typed
    ``prompt_cache_key`` argument. Keeping the wire field in ``extra_body`` makes it
    available on every supported SDK version; the SDK merges it into the request's
    top-level JSON object.
    """
    extra = (kwargs or {}).get("extra_body")
    value = extra.get(_PROMPT_CACHE_KEY) if isinstance(extra, dict) else None
    return value if isinstance(value, str) and value else None


def with_prompt_cache_key(kwargs: Optional[dict], key: Optional[str] = None) -> dict:
    """Copy generation kwargs and set one opaque prompt-cache routing key.

    A missing ``key`` mints a new conversation identity. Nested ``extra_body`` is
    copied before mutation so a disposable subagent can replace its parent's key
    without changing the parent session or its Venice extension parameters.
    """
    out = dict(kwargs or {})
    extra = out.get("extra_body")
    extra = dict(extra) if isinstance(extra, dict) else {}
    extra[_PROMPT_CACHE_KEY] = key or f"venice-{os.urandom(16).hex()}"
    out["extra_body"] = extra
    return out


def without_prompt_cache_key(kwargs: Optional[dict]) -> dict:
    """Copy generation kwargs without a parent conversation's affinity key."""
    out = dict(kwargs or {})
    extra = out.get("extra_body")
    if not isinstance(extra, dict) or _PROMPT_CACHE_KEY not in extra:
        return out
    extra = dict(extra)
    extra.pop(_PROMPT_CACHE_KEY, None)
    if extra:
        out["extra_body"] = extra
    else:
        out.pop("extra_body", None)
    return out


def import_openai(label: str):
    """Import the openai SDK lazily. None (after printing a hint) if absent.

    `label` names the command in the hint (e.g. "chat").
    """
    try:
        import openai
    except ImportError:
        print(
            f"venice {label} needs the openai package: "
            'pip install "venice-cli[openai]" (or: pip install openai)',
            file=sys.stderr,
        )
        return None
    return openai


def build_openai(module, client=None, *, base_url=None, api_key=None, verify=None):
    """Build an SDK client pointed at Venice, borrowing the lean client's auth.

    When `base_url` is given (an alternate OpenAI-compatible backend, e.g. a
    local embeddings server), use it and `api_key` directly instead of the
    Venice client -- which may then be None. Local servers usually need no key,
    so `api_key` falls back to a placeholder the SDK accepts.

    `verify` overrides TLS verification for that alternate backend (a CA-bundle
    path to trust a private CA, or False to disable checks for a self-signed
    cert). It is opt-in and only reaches non-Venice endpoints. When set we use
    the SDK's own HTTP-client factory, which follows the transport bundled by
    that SDK version (`httpx` on older releases, `httpx2` on newer ones). The
    client is not explicitly closed -- fine for a one-shot CLI process that exits
    right after; don't copy this into a long-lived caller without closing it.
    """
    extra = {}
    if verify is not None:
        extra["http_client"] = module.DefaultHttpxClient(verify=verify)
    if base_url is not None:
        return module.OpenAI(
            api_key=api_key or "not-needed", base_url=base_url, **extra
        )
    return module.OpenAI(
        api_key=client.api_key, base_url=client.base_url, **extra
    )


def status_to_exit(module, e, label: str) -> int:
    """Map an openai SDK exception to a venice exit code.

    `module` is the imported openai module (for its exception types) and `label`
    prefixes the message (e.g. "chat").
    """
    if isinstance(e, module.APIConnectionError):
        print(f"{label}: connection error: {e}", file=sys.stderr)
        return 8
    status = getattr(e, "status_code", None)
    print(f"{label}: API error: {e}", file=sys.stderr)
    if status == 401:
        return 2
    if status == 404:
        return 6
    if status == 429:
        return 4
    if isinstance(status, int) and 500 <= status < 600:
        return 5
    if isinstance(status, int) and 400 <= status < 500:
        return 2
    return 5
