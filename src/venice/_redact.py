"""Remove known API credentials from diagnostic text without reading files."""
import json
import os
import urllib.parse


def redact_text(value, *secrets: str) -> str:
    """Redact complete known secrets before truncation or diagnostic rendering.

    Callers with file-loaded credentials pass them explicitly. The environment
    fallback never searches the filesystem. Escaped JSON and URL forms are
    included because upstream error messages may echo encoded credentials.
    """
    text = str(value)
    known = set(secrets)
    known.add(os.environ.get("VENICE_API_KEY", ""))
    variants = set()
    for secret in known:
        if isinstance(secret, str) and secret:
            variants.update((
                secret, json.dumps(secret)[1:-1],
                urllib.parse.quote(secret, safe=""),
                urllib.parse.quote_plus(secret, safe=""),
            ))
    for secret in sorted(variants, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text
