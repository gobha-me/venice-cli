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

import sys


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
    cert). It is opt-in and only reaches non-Venice endpoints. When set we hand
    the SDK an httpx client (httpx ships transitively with the openai SDK). The
    client is not explicitly closed -- fine for a one-shot CLI process that exits
    right after; don't copy this into a long-lived caller without closing it.
    """
    extra = {}
    if verify is not None:
        import httpx
        extra["http_client"] = httpx.Client(verify=verify)
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
