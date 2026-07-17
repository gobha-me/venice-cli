"""Shared model-catalog resolution for commands that pick a model by id.

Extracted from `chat` so it, `embed`, and `video` share one copy of the free
`/models?type=...` GET plus the default-trait/validation logic rather than each
carrying its own. These helpers take primitive args (a model type, the requested
id, a label and noun for messages) so they stay independent of any one command's
argument shape.

The catalog GET is free, so commands call it before the paid request to validate
`--model` and resolve a default without spending.
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from ..client import VeniceAPIError


def catalog(client, model_type: str) -> Optional[List[dict]]:
    """Fetch the model catalog for `model_type` ("text", "embedding", "video").

    None if the (free) GET is unavailable, which leaves the caller unable to
    validate or pick a default.
    """
    try:
        doc = client.get_json("/models", params={"type": model_type})
    except VeniceAPIError:
        return None
    data = doc.get("data") if isinstance(doc, dict) else None
    return list(data) if isinstance(data, list) else None


def default_model(models: List[dict]) -> Optional[str]:
    """The id of the first model advertising the 'default' trait, if any."""
    for m in models:
        spec = m.get("model_spec") if isinstance(m, dict) else None
        traits = spec.get("traits") if isinstance(spec, dict) else None
        if isinstance(traits, list) and "default" in traits:
            return m.get("id")
    return None


def resolve_model(
    requested: Optional[str],
    models: Optional[List[dict]],
    *,
    label: str,
    noun: str,
) -> Tuple[Optional[str], Optional[int]]:
    """Validate `requested` against the catalog, or pick the default.

    Returns (model_id, exit_code). exit_code is None on success. `label` prefixes
    error messages (e.g. "chat") and `noun` names the model kind in them (e.g.
    "text model").
    """
    if models is None:
        # Catalog unavailable: can't validate or pick a default.
        if requested:
            return requested, None
        print(
            f"{label}: could not fetch the model catalog; pass --model explicitly",
            file=sys.stderr,
        )
        return None, 2

    ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
    if requested:
        if requested in ids:
            return requested, None
        print(f"{label}: unknown {noun} {requested!r}", file=sys.stderr)
        print("available: " + ", ".join(ids), file=sys.stderr)
        return None, 6

    default = default_model(models)
    if default:
        return default, None
    print(
        f"{label}: no default {noun} advertised; pass --model. "
        "available: " + ", ".join(ids),
        file=sys.stderr,
    )
    return None, 6
