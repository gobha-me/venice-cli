"""`venice image` -- generate images via Venice's /image/generate endpoint.

Sync flow (no queue), mirroring `tts`. Pricing is per image; the command
fetches the live per-model rate from /models?type=image and shows the
estimate upfront alongside the current balance. Supports 1-4 variants,
meaningful --name output, full parameter passthrough, sizing presets, and
batch generation of a whole card set from --from-file.

Every call uses JSON mode (return_binary omitted), so the response is
{"images": ["<base64>", ...]} which we decode -- one code path for 1-4
variants, no client changes needed.

Reproducibility round-trip: --save-json writes a .json sidecar of the
resolved params (including the actual seed Venice used) next to the generated
image; --from-json replays such a sidecar to regenerate, with explicitly-passed
CLI flags (and a positional prompt) overriding the saved values. A call has one
seed, and replaying it reproduces the first variant, so with --variants>1 only
that first (reproducible) variant gets a sidecar.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .. import _numeric, auth, billing, config, userconfig
from ..client import VeniceAPIError, build_client_from_auth
from . import _models, _shared
from ._shared import (
    add_balance_flag as _add_balance_flag,
    confirm_or_exit as _confirm_or_exit,
    over_budget as _over_budget,
    print_balance_and_remaining as _print_balance_and_remaining,
    print_estimate as _print_estimate,
    status_to_exit as _status_to_exit,
    write_bytes_outputs as _write_images,
)

DEFAULT_IMAGE_MODEL = "venice-sd35"  # pixel-based; honors --width/--height

FORMATS = ("png", "webp", "jpeg")
DEFAULT_FORMAT = "png"  # lossless; best for card art / upscaling

EXT_BY_FORMAT = {
    "png": ".png",
    "webp": ".webp",
    "jpeg": ".jpg",
}

MIN_VARIANTS = 1
MAX_VARIANTS = 4
DEFAULT_VARIANTS = 1
QUALITY_CHOICES = ("low", "medium", "high")
STYLE_REFERENCE_MAX_BYTES = 8 * 1024 * 1024


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "image",
        help="Generate images via /image/generate (sync).",
        description=(
            "Generates one or more images from a text prompt. Single card: "
            "pass a positional prompt. Whole set: pass --from-file with one "
            "prompt per line (optionally 'name<TAB>prompt'). Pricing is per "
            "image; cost is estimated from the live model rate. Use "
            "`venice models --type image --detail` to see models and pricing."
        ),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "prompt",
        nargs="?",
        help="Image description. Omit and pass --from-file for batch mode.",
    )
    src.add_argument(
        "--from-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Batch mode: read prompts from PATH (one per line; blank lines "
        "and '#' comments skipped; optional 'name<TAB>prompt').",
    )
    # Replay lives outside the mutually-exclusive group so a positional prompt
    # can coexist as an override of the saved one.
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Replay a sidecar: load resolved params from PATH and regenerate. "
        "Explicitly-passed flags (and a positional prompt) override saved values.",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"Image model id (default {DEFAULT_IMAGE_MODEL}). "
        "See `venice models --type image`. Config-backable via "
        "defaults.image.model; an explicit --model still wins.",
    )
    p.add_argument(
        "--format",
        choices=FORMATS,
        default=None,
        help=f"Output image format (default {DEFAULT_FORMAT}). Config-backable "
        "via defaults.image.format; an explicit --format still wins.",
    )
    # Sizing presets.
    p.add_argument("--width", type=int, default=None,
                   help="Width in px (pixel-based models, max 1280).")
    p.add_argument("--height", type=int, default=None,
                   help="Height in px (pixel-based models, max 1280).")
    p.add_argument("--aspect-ratio", default=None, metavar="W:H",
                   help="Aspect ratio (aspect-ratio/resolution-tier models).")
    p.add_argument("--resolution", default=None, metavar="TIER",
                   help="Resolution tier, e.g. 1K/2K/4K (tier models).")
    # Shared style templating (local; applies to single + batch).
    p.add_argument(
        "--style-prefix",
        default=None,
        metavar="TEXT",
        help="Text prepended to every prompt (single + batch) for a shared "
        "look. Does not affect output filenames.",
    )
    p.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help="Named {style_prefix, negative_prompt} bundle from the presets "
        "file. Explicit --style-prefix/--negative-prompt override it.",
    )
    p.add_argument(
        "--preset-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Presets JSON location (default {config.PRESETS_FILE}).",
    )
    # Parameter passthrough.
    p.add_argument("--negative-prompt", default=None,
                   help="What to exclude from the image.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for reproducibility.")
    p.add_argument("--cfg-scale", type=_numeric.finite_float, default=None, metavar="N",
                   help="Prompt adherence 0-20 (higher = stricter).")
    p.add_argument("--steps", type=int, default=None,
                   help="Inference steps (model-dependent).")
    p.add_argument("--style-preset", default=None,
                   help="Predefined style preset (model-dependent).")
    p.add_argument(
        "--style-reference",
        action="append",
        dest="style_references",
        default=None,
        metavar="JSON",
        help="Style reference as JSON with image=PATH|URL|data: and optional "
        "strength=0.1-1. Repeatable; count is validated from the live model catalog.",
    )
    p.add_argument(
        "--embed-exif-metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Embed generation metadata in the output image when supported.",
    )
    p.add_argument("--lora-strength", type=int, default=None, metavar="N",
                   help="LoRA strength 0-100 for models using additional LoRAs.")
    p.add_argument("--quality", choices=QUALITY_CHOICES, default=None,
                   help="Model-supported output quality; may change the price.")
    p.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        dest="enable_web_search",
        default=None,
        help="Allow supported image models to use web search (additional cost).",
    )
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument(
        "--disable-prompt-optimization-thinking",
        action="store_true",
        dest="disable_prompt_optimization_thinking",
        default=None,
        help="Skip supported models' prompt-optimization thinking step.",
    )
    thinking.add_argument(
        "--enable-prompt-optimization-thinking",
        action="store_false",
        dest="disable_prompt_optimization_thinking",
        default=None,
        help="Force supported models' prompt-optimization thinking step on.",
    )
    p.add_argument(
        "--enhance-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rewrite the prompt before generation (additional cost when applied).",
    )
    p.add_argument(
        "--variants",
        type=int,
        default=None,
        metavar="N",
        help=f"Images to generate per prompt ({MIN_VARIANTS}-{MAX_VARIANTS}; "
        f"default {DEFAULT_VARIANTS}). Config-backable via "
        "defaults.image.variants; an explicit --variants still wins.",
    )
    p.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Safe mode blurs flagged art (on by default). --no-safe-mode "
        "disables it. Config-backable via defaults.image.safe_mode; an explicit "
        "--safe-mode/--no-safe-mode still wins.",
    )
    p.add_argument(
        "--hide-watermark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask Venice to omit its watermark (may be ignored for some content). "
        "Config-backable via defaults.image.hide_watermark; --no-hide-watermark "
        "forces it back on.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Base output filename (single mode), e.g. 'fire-dragon'.",
    )
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file (single) or directory. Default: cwd.")
    p.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Write a .json sidecar of resolved params (incl. seed) for "
        "reproducible replay via --from-json (with --variants>1, only the "
        "first, reproducible variant gets one).",
    )
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost and exit; don't call /image/generate.")
    p.add_argument(
        "--max-spend",
        type=_numeric.non_negative_float,
        default=None,
        metavar="USD",
        help="Refuse to generate if the estimated cost exceeds this USD cap.",
    )
    _add_balance_flag(p)
    p.set_defaults(handler=_run)


# ---- request body ------------------------------------------------------------

def _build_body(prompt: str, args) -> dict:
    prefix = getattr(args, "style_prefix", None)
    if prefix and prefix.strip():
        prompt = f"{prefix.strip()} {prompt}"
    body: dict = {
        "model": args.model,
        "prompt": prompt,
        "format": args.format,
        # None (neither flag nor config set) -> True, i.e. stay safe by default.
        "safe_mode": args.safe_mode if args.safe_mode is not None else True,
        # None (neither flag nor config set) -> False, i.e. keep the watermark.
        "hide_watermark": bool(args.hide_watermark),
    }
    if args.variants > 1:
        body["variants"] = args.variants
    optional = {
        "width": args.width,
        "height": args.height,
        "aspect_ratio": args.aspect_ratio,
        "resolution": args.resolution,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
        "style_preset": args.style_preset,
        "style_references": getattr(args, "style_references", None),
        "embed_exif_metadata": getattr(args, "embed_exif_metadata", None),
        "lora_strength": getattr(args, "lora_strength", None),
        "quality": getattr(args, "quality", None),
        "enable_web_search": getattr(args, "enable_web_search", None),
        "disable_prompt_optimization_thinking": getattr(
            args, "disable_prompt_optimization_thinking", None
        ),
        "enhance_prompt": getattr(args, "enhance_prompt", None),
    }
    for k, v in optional.items():
        if v is not None:
            body[k] = v
    return body


# ---- response decode ---------------------------------------------------------

def _decode_images(doc: dict) -> List[bytes]:
    """Base64-decode the `images` array. Strips a data-URI prefix if present."""
    out: List[bytes] = []
    imgs = doc.get("images") if isinstance(doc, dict) else None
    if not isinstance(imgs, list):
        return out
    for entry in imgs:
        if not isinstance(entry, str):
            continue
        b64 = entry
        if b64.startswith("data:") and "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            out.append(base64.b64decode(b64))
        except (binascii.Error, ValueError):
            continue
    return out


def _sidecar_params(doc: dict, body: dict) -> dict:
    """Flat, replayable spec of resolved params for the sidecar.

    Prefers the resolved `request.data` block Venice echoes -- it carries the
    actual seed even when the user passed none -- and falls back to the sent
    body. `variants` is dropped so the sidecar reproduces a single image on
    replay.
    """
    resolved = None
    req = doc.get("request") if isinstance(doc, dict) else None
    if isinstance(req, dict):
        data = req.get("data")
        if isinstance(data, dict) and data:
            resolved = data
    # Keep every provider-shaped request field even when the response echoes only
    # a subset, then let resolved values (especially the server-selected seed)
    # override what was sent.
    params = dict(body)
    if resolved is not None:
        params.update(resolved)
    params.pop("variants", None)
    params.setdefault("model", body.get("model"))
    params.setdefault("prompt", body.get("prompt"))
    params.setdefault("format", body.get("format"))
    return params


# ---- native controls, catalog validation, and pricing -----------------------

def validate_lora_strength(value) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 100
    ):
        raise ValueError("lora_strength must be an integer between 0 and 100")


def _decode_data_image(value: str) -> None:
    """Validate a style-reference data URL without rewriting it."""
    if "," not in value:
        raise ValueError("style reference data URL is missing its payload")
    header, payload = value.split(",", 1)
    if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
        raise ValueError("style reference data URL must be a base64 image")
    max_encoded = 4 * ((STYLE_REFERENCE_MAX_BYTES + 2) // 3)
    if len(payload) > max_encoded:
        raise ValueError(
            "style reference data URL exceeds the encoded 8 MiB image limit"
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("style reference data URL has invalid base64") from None
    if not data:
        raise ValueError("style reference data URL is empty")
    if _shared._sniff_media_mime(data[:64], "image") is None:
        raise ValueError("style reference data URL is not a recognized image container")
    if len(data) >= STYLE_REFERENCE_MAX_BYTES:
        raise ValueError(
            f"style reference data is {len(data)} bytes; limit is less than "
            f"{STYLE_REFERENCE_MAX_BYTES} bytes"
        )


def _is_raw_base64_image(value: str) -> bool:
    """Recognize a bounded raw-base64 image without mistaking paths for data."""
    max_encoded = 4 * ((STYLE_REFERENCE_MAX_BYTES + 2) // 3)
    if len(value) > max_encoded:
        return False
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return bool(data) and len(data) < STYLE_REFERENCE_MAX_BYTES and (
        _shared._sniff_media_mime(data[:64], "image") is not None
    )


def _resolve_style_image(value, *, path_authority=None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("style reference image must be a non-empty string")
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("data:"):
        _decode_data_image(value)
        return value
    if len(value) > 4 * ((STYLE_REFERENCE_MAX_BYTES + 2) // 3):
        raise ValueError("style reference exceeds the encoded 8 MiB image limit")
    if _is_raw_base64_image(value):
        return value

    if path_authority is not None:
        try:
            path, mime = path_authority.resolve(
                value, kind="image", max_bytes=STYLE_REFERENCE_MAX_BYTES - 1
            )
        except _shared.MediaPathError as e:
            raise ValueError(str(e)) from None
    else:
        path = Path(value).expanduser()
        try:
            info = path.stat()
        except OSError:
            raise ValueError(f"style reference file does not exist: {value}") from None
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("style reference path must name a regular file")
        if info.st_size == 0:
            raise ValueError("style reference file is empty")
        if info.st_size >= STYLE_REFERENCE_MAX_BYTES:
            raise ValueError(
                f"style reference file is {info.st_size} bytes; limit is less than "
                f"{STYLE_REFERENCE_MAX_BYTES} bytes"
            )
        try:
            with path.open("rb") as fh:
                prefix = fh.read(64)
        except OSError:
            raise ValueError("style reference file could not be read") from None
        mime = _shared._sniff_media_mime(prefix, "image")
        if mime is None:
            raise ValueError("style reference is not a recognized image container")
    return _shared.encode_data_url(path, detected_mime=mime)


def resolve_style_references(raw_values, *, path_authority=None) -> Optional[List[dict]]:
    """Parse CLI/config/tool style references and normalize their image values."""
    if raw_values is None:
        return None
    if not isinstance(raw_values, list):
        raise ValueError("style_references must be a list")
    out: List[dict] = []
    for raw in raw_values:
        if isinstance(raw, str):
            try:
                item = json.loads(raw)
            except ValueError as e:
                raise ValueError(f"--style-reference is not valid JSON: {e}") from None
        else:
            item = raw
        if not isinstance(item, dict):
            raise ValueError("each style reference must be a JSON object")
        unknown = set(item) - {"image", "strength"}
        if unknown:
            raise ValueError(
                "style reference has unknown field(s): " + ", ".join(sorted(unknown))
            )
        resolved = {"image": _resolve_style_image(
            item.get("image"), path_authority=path_authority
        )}
        if item.get("strength") is not None:
            try:
                strength = _numeric.finite_float(item["strength"])
            except ValueError as e:
                raise ValueError(f"invalid style reference strength: {e}") from None
            if not 0.1 <= strength <= 1.0:
                raise ValueError("style reference strength must be between 0.1 and 1")
            resolved["strength"] = strength
        out.append(resolved)
    return out


def _model_entry(models, model: str) -> Optional[dict]:
    if not isinstance(models, list):
        return None
    return next(
        (m for m in models if isinstance(m, dict) and m.get("id") == model), None
    )


def _validated_usd(node) -> Optional[float]:
    if not isinstance(node, dict) or "usd" not in node:
        return None
    try:
        return _numeric.non_negative_float(node["usd"])
    except ValueError:
        raise ValueError("model catalog contains an invalid image price") from None


def _price_from_entry(entry, *, resolution=None, quality=None) -> Optional[float]:
    spec = entry.get("model_spec") if isinstance(entry, dict) else None
    pricing = spec.get("pricing") if isinstance(spec, dict) else None
    constraints = spec.get("constraints") if isinstance(spec, dict) else None
    if not isinstance(pricing, dict):
        return None
    constraints = constraints if isinstance(constraints, dict) else {}

    quality_matrix = pricing.get("quality")
    if quality is not None or isinstance(quality_matrix, dict):
        selected_quality = quality or constraints.get("defaultQuality")
        selected_resolution = resolution or constraints.get("defaultResolution")
        if selected_quality and selected_resolution and isinstance(quality_matrix, dict):
            by_resolution = quality_matrix.get(selected_resolution)
            if isinstance(by_resolution, dict):
                return _validated_usd(by_resolution.get(selected_quality))
        if quality is not None:
            return None

    resolution_prices = pricing.get("resolutions")
    selected_resolution = resolution or constraints.get("defaultResolution")
    if selected_resolution and isinstance(resolution_prices, dict):
        return _validated_usd(resolution_prices.get(selected_resolution))

    for key in ("generation", "image", "output"):
        price = _validated_usd(pricing.get(key))
        if price is not None:
            return price
    return _validated_usd(pricing)


def _validate_model_controls(entry, args) -> None:
    spec = entry.get("model_spec") if isinstance(entry, dict) else None
    constraints = spec.get("constraints") if isinstance(spec, dict) else None
    constraints = constraints if isinstance(constraints, dict) else {}
    refs = getattr(args, "style_references", None) or []
    if refs:
        if not isinstance(spec, dict) or spec.get("supportsStyleReferences") is not True:
            raise ValueError(f"model {args.model!r} does not support style references")
        cap = constraints.get("maxStyleReferences")
        if (
            isinstance(cap, bool)
            or not isinstance(cap, (int, float))
            or int(cap) != cap
            or cap < 1
        ):
            raise ValueError(
                f"live image catalog entry for {args.model!r} has no valid "
                "maxStyleReferences"
            )
        if len(refs) > int(cap):
            raise ValueError(
                f"model {args.model!r} accepts at most {int(cap)} style references"
            )
        if any("strength" in ref for ref in refs):
            if constraints.get("supportsStyleReferenceStrength") is not True:
                raise ValueError(
                    f"model {args.model!r} does not support style reference strength"
                )

    quality = getattr(args, "quality", None)
    if quality is not None:
        choices = constraints.get("qualities")
        if not isinstance(choices, list) or any(not isinstance(v, str) for v in choices):
            raise ValueError(
                f"live image catalog entry for {args.model!r} has no valid qualities"
            )
        if quality not in choices:
            raise ValueError(
                f"quality {quality!r} is not supported by {args.model!r}; "
                "choose from " + ", ".join(choices)
            )


def prepare_request(client, args, *, path_authority=None) -> Optional[float]:
    """Normalize native inputs, validate the live model contract, and return price."""
    args.style_references = resolve_style_references(
        getattr(args, "style_references", None), path_authority=path_authority
    )
    models = _models.catalog(client, "image")
    entry = _model_entry(models, args.model)
    quality = getattr(args, "quality", None)
    if (args.style_references or quality is not None) and entry is None:
        if models is None:
            raise ValueError("could not fetch the live image catalog")
        raise ValueError(f"model {args.model!r} is not in the live image catalog")
    if entry is not None:
        _validate_model_controls(entry, args)
    price = _price_from_entry(
        entry, resolution=getattr(args, "resolution", None), quality=quality
    )
    # These switches explicitly carry additional, unlisted charges. Never let the
    # base image price masquerade as a complete max-spend estimate.
    if (
        getattr(args, "enable_web_search", None) is True
        or getattr(args, "enhance_prompt", None) is True
    ):
        return None
    return price


def _price_description(args, prefix: str) -> str:
    parts = [prefix, f"model={args.model}"]
    if getattr(args, "resolution", None):
        parts.append(f"resolution={args.resolution}")
    if getattr(args, "quality", None):
        parts.append(f"quality={args.quality}")
    if getattr(args, "style_references", None):
        parts.append(f"style_refs={len(args.style_references)}")
    if getattr(args, "enable_web_search", None):
        parts.append("web_search=on")
    if getattr(args, "enhance_prompt", None):
        parts.append("enhance_prompt=on")
    return ", ".join(parts)


# ---- pricing + cost estimation ----------------------------------------------

def _fetch_image_price(client, model: str) -> Optional[float]:
    """Compatibility wrapper for callers/tests that need only the base price."""
    return _price_from_entry(_model_entry(_models.catalog(client, "image"), model))


def _usd_from_pricing(pricing) -> Optional[float]:
    """Compatibility parser for a base-price block, without matrix guessing."""
    if not isinstance(pricing, dict):
        return None
    for key in ("generation", "image", "output"):
        price = _validated_usd(pricing.get(key))
        if price is not None:
            return price
    return _validated_usd(pricing)


def _estimate_cost(
    price: Optional[float], variants: int, n_prompts: int = 1
) -> Optional[float]:
    if price is None:
        return None
    return _numeric.non_negative_float(price * variants * n_prompts)


# ---- output paths ------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, fallback: str = "image") -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug or fallback


def _short_id(prompt: str, model: str, seed: Optional[int]) -> str:
    h = hashlib.sha1()
    h.update(prompt.encode("utf-8"))
    h.update(model.encode("utf-8"))
    if seed is not None:
        h.update(str(seed).encode("utf-8"))
    return h.hexdigest()[:8]


def _variant_path(directory: Path, base: str, idx: int, total: int, ext: str) -> Path:
    name = base if total == 1 else f"{base}-{idx}"
    return directory / f"{name}{ext}"


def _resolve_single_paths(args, prompt: str, total: int) -> List[Path]:
    """Output paths for single mode (honors --name and --output file/dir)."""
    ext = EXT_BY_FORMAT.get(args.format, ".bin")
    if args.name:
        base = _slugify(args.name)
    else:
        base = f"venice-image-{_short_id(prompt, args.model, args.seed)}"

    out = args.output
    # A single explicit file path is only meaningful for one image.
    if out is not None and total == 1 and not out.is_dir():
        return [out]
    directory = out if (out is not None and out.is_dir()) else Path.cwd()
    return [_variant_path(directory, base, i + 1, total, ext) for i in range(total)]


# ---- batch parsing -----------------------------------------------------------

def _read_batch(path: Path) -> Tuple[Optional[List[Tuple[str, str]]], int]:
    """Parse a prompts file into [(name, prompt), ...]. Returns (items, rc)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"image: cannot read {path}: {e}", file=sys.stderr)
        return None, 2
    items: List[Tuple[str, str]] = []
    used: dict = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in line:
            name_part, prompt_part = line.split("\t", 1)
            name = _slugify(name_part) if name_part.strip() else ""
            prompt = prompt_part.strip()
        else:
            name = ""
            prompt = stripped
        if not prompt:
            continue
        if not name:
            name = _slugify(" ".join(prompt.split()[:4]), fallback="card")
        # Deduplicate names so files don't clobber each other.
        n = used.get(name, 0)
        used[name] = n + 1
        if n:
            name = f"{name}-{n + 1}"
        items.append((name, prompt))
    if not items:
        print(f"image: no prompts found in {path}", file=sys.stderr)
        return None, 2
    return items, 0


# ---- sidecar -----------------------------------------------------------------

def _write_sidecar(json_path: Path, params: dict) -> None:
    """Best-effort: write a params sidecar. Warns but never fails the run --
    the image is already on disk; the sidecar is reproducibility metadata."""
    try:
        fd = os.open(json_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o600)
            json.dump(params, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.chmod(json_path, 0o600)
    except OSError as e:
        print(f"warning: could not write sidecar {json_path}: {e}", file=sys.stderr)
        return
    print(f"wrote sidecar {json_path.resolve()}", file=sys.stderr)


def _generate_one(client, prompt: str, args, paths: List[Path]) -> int:
    body = _build_body(prompt, args)
    try:
        doc = client.post_json("/image/generate", body)
    except VeniceAPIError as e:
        print(f"image failed: {e}", file=sys.stderr)
        return _status_to_exit(e)
    images = _decode_images(doc)
    if not images:
        print("image: server returned no images", file=sys.stderr)
        return 5
    written = paths[: len(images)]
    rc = _write_images(images, written)
    if rc is not None:
        return rc
    if getattr(args, "save_json", False):
        # One call-level seed backs all variants, and replaying it reproduces
        # only the first variant byte-for-byte -- so the sidecar goes next to
        # variant 1 alone; a sidecar beside variants 2..N would falsely imply
        # they're individually reproducible.
        params = _sidecar_params(doc, body)
        _write_sidecar(written[0].with_suffix(".json"), params)
    return 0


# ---- replay ------------------------------------------------------------------

# Generation fields a sidecar can supply and CLI flags can override
# (== the inputs of _build_body).
_REPLAY_FIELDS = (
    "model", "prompt", "format", "width", "height", "aspect_ratio",
    "resolution", "negative_prompt", "seed", "cfg_scale", "steps",
    "style_preset", "style_references", "variants", "safe_mode",
    "hide_watermark", "embed_exif_metadata", "lora_strength", "quality",
    "enable_web_search", "disable_prompt_optimization_thinking", "enhance_prompt",
)


def _load_sidecar(path: Path) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"image: cannot read {path}: {e}", file=sys.stderr)
        return None
    try:
        spec = json.loads(raw)
    except ValueError as e:
        print(f"image: {path} is not valid JSON: {e}", file=sys.stderr)
        return None
    if not isinstance(spec, dict):
        print(f"image: {path} must contain a JSON object", file=sys.stderr)
        return None
    return spec


def _apply_replay(args):
    """Merge a sidecar with CLI overrides into a synthetic namespace.

    Returns the merged namespace, or an int exit code on error. A generation
    field is taken from the CLI when it was set explicitly (differs from the
    parser default); otherwise it falls back to the saved spec.
    """
    if args.from_file is not None:
        print("image: --from-json and --from-file are mutually exclusive",
              file=sys.stderr)
        return 2
    spec = _load_sidecar(args.from_json)
    if spec is None:
        return 2

    from ..cli import build_parser  # lazy: avoids a circular import at load
    # A VIRGIN namespace: no config, no built-in literals. That is the whole
    # point -- `cur != defaults.<f>` below is how "the user (or their config)
    # supplied this" is detected, so anything pre-filled here would read as
    # explicit and the sidecar would never win. `_run` deliberately applies the
    # Class C literals AFTER this merge for the same reason. (#57 Class C1)
    defaults = build_parser().parse_args(["image"])

    merged = argparse.Namespace(**vars(args))
    for f in _REPLAY_FIELDS:
        cur = getattr(args, f, None)
        if f == "prompt":
            explicit = bool(cur and cur.strip())
        else:
            explicit = cur != getattr(defaults, f, None)
        if explicit:
            continue
        if spec.get(f) is not None:
            setattr(merged, f, spec[f])

    merged.from_file = None
    merged.from_json = None
    return merged


# ---- style presets -----------------------------------------------------------

def _resolve_preset(args) -> Optional[int]:
    """Fill style_prefix/negative_prompt from a named preset when --preset is set.

    Explicit CLI flags win: a preset value is applied only where the arg is
    still None. Returns an int exit code on error, else None.
    """
    name = getattr(args, "preset", None)
    if not name:
        return None
    path = args.preset_file or config.PRESETS_FILE
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        print(f"image: cannot read presets file {path}: {e}", file=sys.stderr)
        return 2
    try:
        presets = json.loads(raw)
    except ValueError as e:
        print(f"image: {path} is not valid JSON: {e}", file=sys.stderr)
        return 2
    if not isinstance(presets, dict):
        print(f"image: {path} must contain a JSON object of presets",
              file=sys.stderr)
        return 2
    entry = presets.get(name)
    if not isinstance(entry, dict):
        available = ", ".join(sorted(presets)) or "(none)"
        print(f"image: preset {name!r} not found in {path}; available: {available}",
              file=sys.stderr)
        return 2
    for field in ("style_prefix", "negative_prompt"):
        if getattr(args, field, None) is None and entry.get(field) is not None:
            setattr(args, field, entry[field])
    return None


# ---- main flow ---------------------------------------------------------------

def _run(args) -> int:
    # Capture explicitness BEFORE config can fill the dest -- afterwards a
    # config-sourced value is indistinguishable from a typed --variants.
    variants_explicit = args.variants is not None
    userconfig.apply_defaults(args, "image")
    if args.from_json is not None:
        merged = _apply_replay(args)
        if isinstance(merged, int):
            return merged
        args = merged
        if not variants_explicit:
            # A sidecar deliberately records no `variants` (see _sidecar_params)
            # so a replay reproduces the ONE saved image. A standing
            # defaults.image.variants must not silently turn that into an N-image
            # charge; an explicit --variants on the command line still can.
            merged.variants = DEFAULT_VARIANTS
    # AFTER the replay merge, never before: a literal applied up front would
    # make every _REPLAY_FIELDS comparison read as "explicit" and the sidecar
    # would be ignored. Also before the --variants range check below, which
    # cannot compare None. (#57 Class C1)
    userconfig.apply_literals(
        args,
        model=DEFAULT_IMAGE_MODEL,
        format=DEFAULT_FORMAT,
        variants=DEFAULT_VARIANTS,
    )

    rc = _resolve_preset(args)
    if rc is not None:
        return rc

    if not (MIN_VARIANTS <= args.variants <= MAX_VARIANTS):
        print(
            f"image: --variants {args.variants} out of range "
            f"({MIN_VARIANTS}-{MAX_VARIANTS})",
            file=sys.stderr,
        )
        return 2

    try:
        validate_lora_strength(getattr(args, "lora_strength", None))
    except ValueError:
        print(
            "image: --lora-strength must be an integer between 0 and 100",
            file=sys.stderr,
        )
        return 2

    if args.from_file is None and not (args.prompt and args.prompt.strip()):
        print(
            "image: prompt required (positional prompt or --from-file PATH)",
            file=sys.stderr,
        )
        return 2

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.from_file is not None:
        return _run_batch(client, args)
    return _run_single(client, args)


def _run_single(client, args) -> int:
    prompt = args.prompt.strip()
    try:
        price = prepare_request(client, args)
    except ValueError as e:
        print(f"image: {e}", file=sys.stderr)
        return 2
    cost = _estimate_cost(price, args.variants)
    desc = f"{args.variants} image" + ("s" if args.variants != 1 else "")
    _print_estimate(cost, _price_description(args, desc))
    _print_balance_and_remaining(client, cost, show=not args.no_balance)

    if _over_budget(cost, args.max_spend):
        shown = billing.format_usd(cost) if cost is not None else "unknown"
        print(
            f"image: estimate {shown} cannot be kept within "
            f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
            file=sys.stderr,
        )
        return 1

    paths = _resolve_single_paths(args, prompt, args.variants)
    if args.dry_run:
        for p in paths:
            print(f"would write: {p.resolve()}", file=sys.stderr)
        return 0

    rc = _confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    return _generate_one(client, prompt, args, paths)


def _run_batch(client, args) -> int:
    items, rc = _read_batch(args.from_file)
    if items is None:
        return rc

    n = len(items)
    try:
        price = prepare_request(client, args)
    except ValueError as e:
        print(f"image: {e}", file=sys.stderr)
        return 2
    cost = _estimate_cost(price, args.variants, n)
    desc = f"{n} prompt" + ("s" if n != 1 else "") + f" x {args.variants}"
    _print_estimate(cost, _price_description(args, desc))
    _print_balance_and_remaining(client, cost, show=not args.no_balance)

    if _over_budget(cost, args.max_spend):
        shown = billing.format_usd(cost) if cost is not None else "unknown"
        print(
            f"image: estimate {shown} cannot be kept within "
            f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
            file=sys.stderr,
        )
        return 1

    ext = EXT_BY_FORMAT.get(args.format, ".bin")
    out_dir = args.output if (args.output and args.output.is_dir()) else Path.cwd()
    plan: List[Tuple[str, List[Path]]] = []
    for name, prompt in items:
        paths = [
            _variant_path(out_dir, name, i + 1, args.variants, ext)
            for i in range(args.variants)
        ]
        plan.append((prompt, paths))

    if args.dry_run:
        for _, paths in plan:
            for p in paths:
                print(f"would write: {p.resolve()}", file=sys.stderr)
        return 0

    rc = _confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    first_failure = 0
    for prompt, paths in plan:
        item_rc = _generate_one(client, prompt, args, paths)
        if item_rc != 0 and first_failure == 0:
            first_failure = item_rc
    return first_failure
