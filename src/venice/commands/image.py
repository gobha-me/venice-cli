"""`venice image` -- generate images via Venice's /image/generate endpoint.

Sync flow (no queue), mirroring `tts`. Pricing is per image; the command
fetches the live per-model rate from /models?type=image and shows the
estimate upfront alongside the current balance. Supports 1-4 variants,
meaningful --name output, full parameter passthrough, sizing presets, and
batch generation of a whole card set from --from-file.

Every call uses JSON mode (return_binary omitted), so the response is
{"images": ["<base64>", ...]} which we decode -- one code path for 1-4
variants, no client changes needed.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .. import auth, billing, config
from ..client import VeniceAPIError, build_client_from_auth

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
    p.add_argument(
        "--model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"Image model id (default {DEFAULT_IMAGE_MODEL}). "
        "See `venice models --type image`.",
    )
    p.add_argument(
        "--format",
        choices=FORMATS,
        default=DEFAULT_FORMAT,
        help=f"Output image format (default {DEFAULT_FORMAT}).",
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
    # Parameter passthrough.
    p.add_argument("--negative-prompt", default=None,
                   help="What to exclude from the image.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for reproducibility.")
    p.add_argument("--cfg-scale", type=float, default=None, metavar="N",
                   help="Prompt adherence 0-20 (higher = stricter).")
    p.add_argument("--steps", type=int, default=None,
                   help="Inference steps (model-dependent).")
    p.add_argument("--style-preset", default=None,
                   help="Predefined style preset (model-dependent).")
    p.add_argument(
        "--variants",
        type=int,
        default=1,
        metavar="N",
        help=f"Images to generate per prompt ({MIN_VARIANTS}-{MAX_VARIANTS}).",
    )
    p.add_argument(
        "--no-safe-mode",
        dest="safe_mode",
        action="store_false",
        default=True,
        help="Disable safe_mode so flagged art isn't silently blurred.",
    )
    p.add_argument(
        "--hide-watermark",
        action="store_true",
        default=False,
        help="Ask Venice to omit its watermark (may be ignored for some content).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Base output filename (single mode), e.g. 'fire-dragon'.",
    )
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file (single) or directory. Default: cwd.")
    p.add_argument("--yes", "-y", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Estimate cost and exit; don't call /image/generate.")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse to generate if the estimated cost exceeds this USD cap.",
    )
    p.add_argument("--no-balance", action="store_true",
                   help="Skip the upfront balance display.")
    p.set_defaults(handler=_run)


# ---- request body ------------------------------------------------------------

def _build_body(prompt: str, args) -> dict:
    body: dict = {
        "model": args.model,
        "prompt": prompt,
        "format": args.format,
        "safe_mode": args.safe_mode,
        "hide_watermark": args.hide_watermark,
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


# ---- pricing + cost estimation ----------------------------------------------

def _fetch_image_price(client, model: str) -> Optional[float]:
    """Per-image USD price for the model from /models?type=image. Best-effort.

    Schema-tolerant: returns the first USD value found in model_spec.pricing
    (values may be nested one level, as with tts's {"input": {"usd": ...}}).
    """
    try:
        doc = client.get_json("/models", params={"type": "image"})
    except VeniceAPIError:
        return None
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, list):
        return None
    for m in data:
        if isinstance(m, dict) and m.get("id") == model:
            spec = m.get("model_spec")
            pricing = spec.get("pricing") if isinstance(spec, dict) else None
            return _usd_from_pricing(pricing)
    return None


def _usd_from_pricing(pricing) -> Optional[float]:
    if not isinstance(pricing, dict):
        return None
    for v in pricing.values():
        if isinstance(v, (int, float)):
            # e.g. {"usd": 0.01} handled below; a bare number is unlabeled.
            continue
        if isinstance(v, dict) and "usd" in v:
            try:
                return float(v["usd"])
            except (TypeError, ValueError):
                return None
    # Fall back to a top-level "usd" key.
    if "usd" in pricing:
        try:
            return float(pricing["usd"])
        except (TypeError, ValueError):
            return None
    return None


def _estimate_cost(
    price: Optional[float], variants: int, n_prompts: int = 1
) -> Optional[float]:
    if price is None:
        return None
    return price * variants * n_prompts


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


# ---- shared UX ---------------------------------------------------------------

def _print_estimate(cost: Optional[float], count_desc: str, model: str) -> None:
    if cost is None:
        print(
            f"Estimated cost: (unknown -- could not fetch {model} pricing) "
            f"[{count_desc}]",
            file=sys.stderr,
        )
    else:
        print(
            f"Estimated cost: {billing.format_usd(cost)} "
            f"({count_desc}, model={model})",
            file=sys.stderr,
        )


def _print_balance_and_remaining(client, cost: Optional[float], *, show: bool) -> None:
    if not show:
        return
    try:
        info = billing.fetch_balance(client)
    except VeniceAPIError:
        info = None
    if not info or info.get("total") is None:
        return
    print(f"Balance:        {billing.format_balance_breakdown(info)}", file=sys.stderr)
    if cost is not None:
        try:
            remaining = float(info["total"]) - float(cost)
            print(f"After charge:   {billing.format_usd(remaining)}", file=sys.stderr)
        except (TypeError, ValueError):
            pass


def _over_budget(cost: Optional[float], max_spend: Optional[float]) -> bool:
    if max_spend is None or cost is None:
        return False
    try:
        return float(cost) > float(max_spend)
    except (TypeError, ValueError):
        return False


def _confirm_or_exit(yes: bool) -> Optional[int]:
    if yes:
        return None
    if not sys.stdin.isatty():
        print("non-interactive; pass --yes to confirm the charge.", file=sys.stderr)
        return 1
    try:
        ans = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("aborted by user", file=sys.stderr)
        return 1
    return None


def _status_to_exit(err: VeniceAPIError) -> int:
    s = err.status
    if s == 400:
        return 2
    if s == 402:
        return 1  # insufficient balance ~ declined
    if s == 422:
        return 3
    if s == 429:
        return 4
    if s == 503:
        return 5
    if 500 <= s < 600:
        return 5
    if s == 404:
        return 6
    if s == 0:
        return 8
    return 2


def _write_images(images: List[bytes], paths: List[Path]) -> Optional[int]:
    """Write decoded images to paths. Returns an exit code on failure, else None."""
    for data, path in zip(images, paths):
        try:
            path.write_bytes(data)
        except OSError as e:
            print(f"could not write {path}: {e}", file=sys.stderr)
            return 9
        abs_path = path.resolve()
        print(str(abs_path))
        print(f"wrote {len(data)} bytes to {abs_path}", file=sys.stderr)
    return None


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
    rc = _write_images(images, paths[: len(images)])
    return rc if rc is not None else 0


# ---- main flow ---------------------------------------------------------------

def _run(args) -> int:
    if not (MIN_VARIANTS <= args.variants <= MAX_VARIANTS):
        print(
            f"image: --variants {args.variants} out of range "
            f"({MIN_VARIANTS}-{MAX_VARIANTS})",
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
    price = _fetch_image_price(client, args.model)
    cost = _estimate_cost(price, args.variants)
    desc = f"{args.variants} image" + ("s" if args.variants != 1 else "")
    _print_estimate(cost, desc, args.model)
    _print_balance_and_remaining(client, cost, show=not args.no_balance)

    if _over_budget(cost, args.max_spend):
        print(
            f"image: estimate {billing.format_usd(cost)} exceeds "
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
    price = _fetch_image_price(client, args.model)
    cost = _estimate_cost(price, args.variants, n)
    desc = f"{n} prompt" + ("s" if n != 1 else "") + f" x {args.variants}"
    _print_estimate(cost, desc, args.model)
    _print_balance_and_remaining(client, cost, show=not args.no_balance)

    if _over_budget(cost, args.max_spend):
        print(
            f"image: estimate {billing.format_usd(cost)} exceeds "
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
