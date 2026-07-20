"""`venice image-edit` -- edit/inpaint an image via /image/edit (sync).

Iterate on already-generated art without regenerating: tweak a color, change
the sky, or composite a mask onto a base image. With no `--layer`, the base
image + prompt go to `/image/edit`. With one or two `--layer` images, the base
plus those layers/masks go to `/image/multi-edit` (max 3 images total, base
first). The base image is a local file (sent as a base64 string in a JSON body)
or an image URL; layers are local files. The endpoint returns raw image bytes
(png/jpeg/webp), so we use the client's bytes-or-json path. Pricing is dynamic
($0.001-$10.00/call); the balance is shown and you confirm before the charge.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import List, Optional

from .. import auth, userconfig
from ..client import build_client_from_auth
from ._shared import (
    confirm_or_exit,
    over_budget,
    post_binary_op,
    print_balance_and_remaining,
    print_estimate,
    resolve_output,
)

EDIT_ENDPOINT = "/image/edit"
MULTI_EDIT_ENDPOINT = "/image/multi-edit"
MAX_INPUT_BYTES = 25 * 1024 * 1024  # API limit: each input file < 25 MB
MAX_LAYERS = 2  # /image/multi-edit takes up to 3 images (base + 2 layers)
MAX_PROMPT = 32768
URL_DEFAULT_STEM = "venice-edit"

ASPECT_RATIOS = ["auto", "1:1", "3:2", "16:9", "21:9", "9:16", "2:3", "3:4", "4:5"]

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
            "sky to a sunrise'`. Pass one or two `--layer` images (masks/"
            "overlays) to composite via /image/multi-edit. The base is a local "
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
                   "routes to /image/multi-edit. Repeatable, up to 2.")
    p.add_argument("--model", default=None, metavar="ID",
                   help="Edit model id (default: server picks firered-image-edit).")
    p.add_argument("--aspect-ratio", default=None, choices=ASPECT_RATIOS,
                   help="Output aspect ratio ('auto' infers from the input).")
    p.add_argument("--resolution", default=None, metavar="TIER",
                   help="Output resolution tier, e.g. 1K/2K/4K (default 1K).")
    p.add_argument("--output-format", default=None,
                   choices=["png", "jpeg", "webp"],
                   help="Output image format (default inferred; PNG for 1K).")
    p.add_argument("--no-safe-mode", action="store_true",
                   help="Disable safe mode (safe_mode defaults to true).")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file or directory. Default: cwd/<input>-edit.<ext> "
                   f"(or {URL_DEFAULT_STEM}.<ext> for --image-url).")
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Show the planned output and exit; don't call the API.")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse if the estimated cost exceeds this cap. Note: image-edit "
        "pricing is dynamic, so no pre-charge estimate is available.",
    )
    p.add_argument("--no-balance", action="store_true",
                   help="Skip the upfront balance display.")
    p.set_defaults(handler=_run)


def _check_image_file(inp: Path) -> Optional[int]:
    if not inp.is_file():
        print(f"image-edit: input file not found: {inp}", file=sys.stderr)
        return 2
    size = inp.stat().st_size
    if size == 0:
        print(f"image-edit: input {inp} is empty", file=sys.stderr)
        return 2
    if size > MAX_INPUT_BYTES:
        print(f"image-edit: input {inp} is {size} bytes; must be < 25 MB",
              file=sys.stderr)
        return 2
    return None


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
        rc = _check_image_file(args.input)
        if rc is not None:
            return rc
    layers = args.layer or []
    if len(layers) > MAX_LAYERS:
        print(f"image-edit: at most {MAX_LAYERS} --layer images "
              "(base + 2 = 3 total)", file=sys.stderr)
        return 2
    for layer in layers:
        rc = _check_image_file(layer)
        if rc is not None:
            return rc
    return None


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _add_common(body: dict, args) -> None:
    """Optional params shared by both endpoints; added only when set."""
    if args.aspect_ratio is not None:
        body["aspect_ratio"] = args.aspect_ratio
    if args.resolution is not None:
        body["resolution"] = args.resolution
    if args.output_format is not None:
        body["output_format"] = args.output_format
    if args.no_safe_mode:
        body["safe_mode"] = False


def _build_body(args, base_image: str, layers_b64: List[str]) -> tuple:
    """Return (endpoint, body). base_image is a base64 string or an image URL;
    /image/edit and /image/multi-edit both accept either in the image field."""
    if layers_b64:
        body: dict = {"images": [base_image, *layers_b64], "prompt": args.prompt}
        if args.model is not None:
            body["modelId"] = args.model  # multi-edit uses modelId, not model
        _add_common(body, args)
        return MULTI_EDIT_ENDPOINT, body
    body = {"image": base_image, "prompt": args.prompt}
    if args.model is not None:
        body["model"] = args.model
    _add_common(body, args)
    return EDIT_ENDPOINT, body


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

    cost = None  # dynamic pricing -- no reliable upfront quote
    print_estimate(cost, "image edit; dynamic $0.001-$10.00/call")
    print_balance_and_remaining(client, cost, show=not args.no_balance)
    if over_budget(cost, args.max_spend):  # no-op while cost is unknown
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

    base_image = _encode(args.input) if args.input is not None else args.image_url
    layers_b64 = [_encode(p) for p in (args.layer or [])]
    endpoint, body = _build_body(args, base_image, layers_b64)
    return post_binary_op(client, endpoint, body, out_path, "image-edit")
