"""Shared model-catalog resolution for commands that pick a model by id.

Extracted from `chat` so it, `embed`, and `video` share one copy of the free
`/models?type=...` GET plus the default-trait/validation logic rather than each
carrying its own. These helpers take primitive args (a model type, the requested
id, a label and noun for messages, and an optional config key naming the
`defaults.<cmd>.model` that would make the choice permanent) so they stay
independent of any one command's argument shape.

The catalog GET is free, so commands call it before the paid request to validate
`--model` and resolve a default without spending.
"""
from __future__ import annotations

import re
import sys
from typing import List, Optional, Tuple

from ..client import VeniceAPIError


def model_family(model_id: str) -> str:
    """The leading vendor/family token of a model id.

    ``qwen3-4b`` and ``qwen-2.5-coder`` share family ``qwen``.  Keeping the
    heuristic beside the catalog helpers lets review decorrelation and the
    cache-probe control selection agree without importing either command.
    """
    return re.split(
        r"[-._0-9]", (model_id or "").strip().lower(), maxsplit=1
    )[0]


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


def supports_capability(models, model_id, key) -> Optional[bool]:
    """Whether `model_id` advertises the boolean capability `key` in the catalog.

    `key` is matched case/underscore-insensitively against
    `model_spec.capabilities` (e.g. "supportsVision"). True/False when the model
    is found and carries the field; None when it can't be determined (no
    catalog, model absent, or the field missing) -- callers treat None as
    "unknown, attempt anyway".
    """
    if not models:
        return None
    want = str(key).lower().replace("_", "")
    for m in models:
        if not isinstance(m, dict) or m.get("id") != model_id:
            continue
        spec = m.get("model_spec") or {}
        caps = spec.get("capabilities")
        if not isinstance(caps, dict):
            return None
        norm = {str(k).lower().replace("_", ""): v for k, v in caps.items()}
        val = norm.get(want)
        return bool(val) if val is not None else None
    return None


def _print_config_hint(label: str, config_key: Optional[str]) -> None:
    """Name the durable fix on the two "pass --model" failures, when the caller
    has one.

    Opt-in per call site and deliberately NEVER derived from `label`: `venice
    video-status` restores `args.model` over `apply_defaults` on purpose (there
    the model is job identity, not a preference), and the MCP `vision` tool has
    no config section at all -- an auto-derived `defaults.<label>.model` would
    advertise a key that does nothing in both cases.
    """
    if config_key:
        print(
            f"{label}: set one permanently with: "
            f"venice config set {config_key} <id>",
            file=sys.stderr,
        )


def resolve_model(
    requested: Optional[str],
    models: Optional[List[dict]],
    *,
    label: str,
    noun: str,
    config_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Validate `requested` against the catalog, or pick the default.

    Returns (model_id, exit_code). exit_code is None on success. `label` prefixes
    error messages (e.g. "chat") and `noun` names the model kind in them (e.g.
    "text model").

    `config_key` is the dotted key (e.g. "defaults.embed.model") that would make
    a choice permanent. When given it adds a trailing hint to the two failures a
    config value would actually fix -- an unfetchable catalog and a catalog with
    no default advertised. The unknown-model failure deliberately gets no hint:
    it already prints every legal id, and pointing at the config key would be
    right for a stale `defaults.*.model` but wrong for a mistyped `--model`,
    which this function has no way to tell apart.
    """
    if models is None:
        # Catalog unavailable: can't validate or pick a default.
        if requested:
            return requested, None
        print(
            f"{label}: could not fetch the model catalog; pass --model explicitly",
            file=sys.stderr,
        )
        _print_config_hint(label, config_key)
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
    _print_config_hint(label, config_key)
    return None, 6
