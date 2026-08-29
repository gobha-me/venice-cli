"""`venice image-edit` -- edit/inpaint an image via /image/edit (sync).

Iterate on already-generated art without regenerating: tweak a color, change
the sky, or composite a mask onto a base image. With no `--layer`, the base
image + prompt go to `/image/edit`. One or more `--layer` images route the base
plus those layers/masks to `/image/multi-edit`; the live model catalog supplies
the input-image limit. The base image is a local file (sent as a base64 string
in a JSON body) or an image URL; layers are local files. The endpoint returns
raw image bytes (png/jpeg/webp), so we use the client's bytes-or-json path.
Pricing is dynamic ($0.001-$10.00/call); the balance is shown and you confirm
before the charge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .. import _numeric, auth, userconfig
from ..client import build_client_from_auth
from . import _models
from .image import QUALITY_CHOICES
from ._shared import (
    add_balance_flag,
    check_image_file,
    confirm_or_exit,
    encode_base64,
    over_budget,
    post_binary_op,
    print_balance_and_remaining,
    print_estimate,
    resolve_output,
)

EDIT_ENDPOINT = "/image/edit"
MULTI_EDIT_ENDPOINT = "/image/multi-edit"
DEFAULT_EDIT_MODEL = "firered-image-edit"
CONTRACT_DEFAULT_MAX_INPUT_IMAGES = 3
MAX_PROMPT = 32768
URL_DEFAULT_STEM = "venice-edit"

ASPECT_RATIOS = ["auto", "1:1", "3:2", "16:9", "21:9", "9:16", "2:3", "3:4", "4:5"]
OUTPUT_FORMATS = ["png", "jpeg", "webp"]

# Map --output-format to a file extension for the default output name. Mirrors
# image.py's EXT_BY_FORMAT; the response content-type is not inspected by
# post_binary_op, so we name the file from the requested format (default png).
EXT_BY_FORMAT = {
    "png": ".png",
    "webp": ".webp",
    "jpeg": ".jpg",
}


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "image-edit",
        help="Edit/inpaint an image via /image/edit (sync).",
        description=(
            "Edit an already-generated image from a text prompt without "
            "regenerating it, e.g. `venice image-edit card.png -p 'change the "
            "sky to a sunrise'`. Pass repeatable `--layer` images (masks/"
            "overlays) to composite via /image/multi-edit; the selected model's "
            "live maxInputImages limit is enforced. The base is a local "
            "file (positional) or --image-url. Pricing is dynamic; the balance "
            "is shown and you confirm before the charge."
        ),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("input", type=Path, nargs="?", default=None,
                     help="Base image file to edit.")
    src.add_argument("--image-url", default=None, metavar="URL",
                     help="Edit the image at this URL instead of a local file.")
    p.add_argument("--prompt", "-p", default=None,
                   help="Text directions for the edit, e.g. 'remove the tree'.")
    p.add_argument("--layer", type=Path, action="append", default=None,
                   metavar="PATH",
                   help="Extra image (mask/overlay) layered onto the base; "
                   "routes to /image/multi-edit. Repeatable; the limit is "
                   "model-specific.")
    p.add_argument("--model", default=None, metavar="ID",
                   help="Edit model id (default: server picks firered-image-edit).")
    p.add_argument("--aspect-ratio", default=None, choices=ASPECT_RATIOS,
                   help="Output aspect ratio ('auto' infers from the input).")
    p.add_argument("--resolution", default=None, metavar="TIER",
                   help="Output resolution tier, e.g. 1K/2K/4K (default 1K).")
    p.add_argument("--output-format", default=None,
                   choices=OUTPUT_FORMATS,
                   help="Output image format (default inferred; PNG for 1K).")
    p.add_argument(
        "--quality",
        choices=QUALITY_CHOICES,
        default=None,
        help="Output quality for supported multi-edit models; requires --layer.",
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
        help="Rewrite the edit prompt before generation (additional cost when applied).",
    )
    p.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Safe mode blurs flagged art (on by default). --no-safe-mode "
        "disables it. Config-backable via defaults.image_edit.safe_mode; an "
        "explicit --safe-mode/--no-safe-mode still wins.",
    )
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file or directory. Default: cwd/<input>-edit.<ext> "
                   f"(or {URL_DEFAULT_STEM}.<ext> for --image-url).")
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Show the planned output and exit; don't call the API.")
    p.add_argument(
        "--max-spend",
        type=_numeric.non_negative_float,
        default=None,
        metavar="USD",
        help="Refuse if the estimated cost exceeds this cap. Note: image-edit "
        "pricing is dynamic, so no pre-charge estimate is available.",
    )
    add_balance_flag(p)
    p.set_defaults(handler=_run)


def _validate(args) -> Optional[int]:
    if (args.input is None) == (args.image_url is None):
        print("image-edit: provide exactly one of INPUT file or --image-url",
              file=sys.stderr)
        return 2
    if not args.prompt:
        print("image-edit: --prompt is required", file=sys.stderr)
        return 2
    if len(args.prompt) > MAX_PROMPT:
        print(f"image-edit: --prompt exceeds {MAX_PROMPT} chars", file=sys.stderr)
        return 2
    if args.input is not None:
        rc = check_image_file(args.input, label="image-edit")
        if rc is not None:
            return rc
    layers = args.layer or []
    if getattr(args, "quality", None) is not None and not layers:
        print("image-edit: --quality is supported only with --layer", file=sys.stderr)
        return 2
    for layer in layers:
        rc = check_image_file(layer, label="image-edit")
        if rc is not None:
            return rc
    return None


def _add_common(body: dict, args) -> None:
    """Optional params shared by both endpoints; added only when set."""
    if args.aspect_ratio is not None:
        body["aspect_ratio"] = args.aspect_ratio
    if args.resolution is not None:
        body["resolution"] = args.resolution
    if args.output_format is not None:
        body["output_format"] = args.output_format
    # None (neither flag nor config set) -> True, i.e. stay safe by default.
    # Unconditional, matching `image._build_body`: when unset the body now
    # carries safe_mode=true rather than omitting the key, which the API
    # schema declares as the default anyway. (#57 Class B)
    body["safe_mode"] = args.safe_mode if args.safe_mode is not None else True
    if getattr(args, "disable_prompt_optimization_thinking", None) is not None:
        body["disable_prompt_optimization_thinking"] = (
            args.disable_prompt_optimization_thinking
        )
    if getattr(args, "enhance_prompt", None) is not None:
        body["enhance_prompt"] = args.enhance_prompt


def _build_body(args, base_image: str, layers_b64: List[str]) -> tuple:
    """Return (endpoint, body). base_image is a base64 string or an image URL;
    /image/edit and /image/multi-edit both accept either in the image field."""
    if layers_b64:
        body: dict = {"images": [base_image, *layers_b64], "prompt": args.prompt}
        if args.model is not None:
            body["modelId"] = args.model  # multi-edit uses modelId, not model
        if getattr(args, "quality", None) is not None:
            body["quality"] = args.quality
        _add_common(body, args)
        return MULTI_EDIT_ENDPOINT, body
    body = {"image": base_image, "prompt": args.prompt}
    if args.model is not None:
        body["model"] = args.model
    _add_common(body, args)
    return EDIT_ENDPOINT, body


def resolve_multi_edit_model(
    client, requested: Optional[str], image_count: int, quality: Optional[str] = None
) -> str:
    """Resolve and validate an inpaint model before a paid multi-edit request."""
    models = _models.catalog(client, "inpaint")
    if models is None:
        raise ValueError("could not fetch the live inpaint model catalog")
    model = requested or _models.default_model(models)
    ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
    if model is None and DEFAULT_EDIT_MODEL in ids:
        model = DEFAULT_EDIT_MODEL
    if model is None:
        raise ValueError(
            "no default inpaint model is advertised; pass --model. available: "
            + ", ".join(ids)
        )
    entry = next(
        (m for m in models if isinstance(m, dict) and m.get("id") == model), None
    )
    if entry is None:
        raise ValueError(
            f"unknown inpaint model {model!r}; available: " + ", ".join(ids)
        )
    spec = entry.get("model_spec")
    constraints = spec.get("constraints") if isinstance(spec, dict) else None
    if not isinstance(constraints, dict):
        raise ValueError(f"live inpaint catalog entry for {model!r} has no constraints")
    combine = constraints.get("combineImages")
    if combine is not True:
        raise ValueError(f"model {model!r} does not support multi-image editing")
    cap = constraints.get(
        "maxInputImages", CONTRACT_DEFAULT_MAX_INPUT_IMAGES
    )
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(
            f"live inpaint catalog entry for {model!r} has invalid maxInputImages"
        )
    if image_count > cap:
        raise ValueError(f"model {model!r} accepts at most {cap} input images")
    if quality is not None:
        qualities = constraints.get("qualities")
        if not isinstance(qualities, list) or any(
            not isinstance(value, str) for value in qualities
        ):
            raise ValueError(
                f"live inpaint catalog entry for {model!r} has no valid qualities"
            )
        if quality not in qualities:
            raise ValueError(
                f"quality {quality!r} is not supported by {model!r}; choose from "
                + ", ".join(qualities)
            )
    return model


def _run(args) -> int:
    userconfig.apply_defaults(args, "image_edit")
    rc = _validate(args)
    if rc is not None:
        return rc

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.layer:
        try:
            args.model = resolve_multi_edit_model(
                client, args.model, 1 + len(args.layer), getattr(args, "quality", None)
            )
        except ValueError as e:
            print(f"image-edit: {e}", file=sys.stderr)
            return 2

    cost = None  # dynamic pricing -- no reliable upfront quote
    print_estimate(cost, "image edit; dynamic $0.001-$10.00/call")
    print_balance_and_remaining(client, cost, show=not args.no_balance)
    if over_budget(cost, args.max_spend):
        print(
            "image-edit: dynamic price cannot be bounded by --max-spend; aborting",
            file=sys.stderr,
        )
        return 1

    ext = EXT_BY_FORMAT.get(args.output_format or "png", ".png")
    default_name = (
        f"{args.input.stem}-edit{ext}" if args.input is not None
        else f"{URL_DEFAULT_STEM}{ext}"
    )
    out_path = resolve_output(args.output, default_name)
    if args.dry_run:
        print(f"would write: {out_path.resolve()}", file=sys.stderr)
        return 0

    rc = confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    base_image = encode_base64(args.input) if args.input is not None else args.image_url
    layers_b64 = [encode_base64(p) for p in (args.layer or [])]
    endpoint, body = _build_body(args, base_image, layers_b64)
    return post_binary_op(client, endpoint, body, out_path, "image-edit")
