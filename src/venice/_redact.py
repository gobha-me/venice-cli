"""Remove known credentials from diagnostics without resolving credentials."""
import json
import os
import re
import urllib.parse


def redact_text(value, *secrets: str) -> str:
    """Redact raw, JSON-escaped and URL-encoded keys before truncation.

    Callers pass already-loaded client keys explicitly. Diagnostic rendering
    must never open a credential file or trigger authentication.
    """
    known = {secret for secret in secrets if isinstance(secret, str) and secret}
    known.add(os.environ.get("VENICE_API_KEY", ""))
    variants = {}
    for secret in known:
        if not secret:
            continue
        escaped = json.dumps(secret)[1:-1]
        for literal in (secret, escaped, escaped.replace("/", r"\/")):
            variants[re.escape(literal)] = len(literal)
        for quote in (urllib.parse.quote, urllib.parse.quote_plus):
            encoded = quote(secret, safe="")
            # Only hexadecimal escapes ignore case; literal key characters
            # retain their case. Mixed %2f/%2B escapes are legal too.
            pattern = re.sub(r"%[0-9A-Fa-f]{2}", lambda m: f"(?i:{m[0]})", re.escape(encoded))
            variants[pattern] = len(encoded)
    if not variants:
        return str(value)
    # One pass, longest first, so overlapping keys and the replacement marker
    # cannot cause a second replacement to corrupt an earlier redaction.
    pattern = "|".join(sorted(variants, key=lambda item: (-variants[item], item)))
    return re.sub(pattern, "[REDACTED]", str(value))
