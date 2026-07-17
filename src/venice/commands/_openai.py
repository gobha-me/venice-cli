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


def build_openai(module, client):
    """Build an SDK client pointed at Venice, borrowing the lean client's auth."""
    return module.OpenAI(api_key=client.api_key, base_url=client.base_url)


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
