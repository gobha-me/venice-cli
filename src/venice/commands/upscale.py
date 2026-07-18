"""`venice upscale` -- upscale/enhance an image via /image/upscale (sync).

`/image/generate` caps width/height at 1280, so large environment art is made
≤1280 then upscaled here (e.g. ×2: 960×540 -> 1920×1080). The input image is
sent as a base64 string in a JSON body; the endpoint returns raw image/png
bytes (not the base64 array `/image/generate` returns), so we use the client's
bytes-or-json path. Pricing is dynamic (Venice bills $0.001-$10.00 per call),
so there is no reliable upfront quote -- we show the balance and confirm.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

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

ENDPOINT = "/image/upscale"
MAX_INPUT_BYTES = 25 * 1024 * 1024  # API limit: input file < 25 MB
MAX_ENHANCE_PROMPT = 1500


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "upscale",
        help="Upscale/enhance an image via /image/upscale (sync).",
        description=(
            "Upscales an image by a factor of 1-4 (default 2), optionally running "
            "Venice's enhancer. Use this to take generated art above the 1280px "
            "generate cap, e.g. `venice upscale env.png --scale 2` -> 2x. Pricing "
            "is dynamic; the balance is shown and you confirm before the charge."
        ),
    )
    p.add_argument("input", type=Path, help="Image file to upscale.")
    p.add_argument(
        "--scale",
        type=float,
        default=2.0,
        metavar="N",
        help="Upscale factor 1-4 (default 2). Scale 1 only runs the enhancer "
        "and requires --enhance.",
    )
    p.add_argument(
        "--enhance",
        action="store_true",
        default=False,
        help="Run Venice's enhancer during upscaling (required when --scale 1).",
    )
    p.add_argument(
        "--enhance-creativity",
        type=float,
        default=None,
        metavar="F",
        help="0-1 (default 0.5); higher lets the enhancer change the image more.",
    )
    p.add_argument(
        "--enhance-prompt",
        default=None,
        metavar="TEXT",
        help="Short style to steer enhancement, e.g. 'gold' (<=1500 chars).",
    )
    p.add_argument(
        "--replication",
        type=float,
        default=None,
        metavar="R",
        help="0-1 (default 0.35); how strongly base-image lines/noise are kept.",
    )
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file or directory. Default: cwd/<input>-upscaled.png.")
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Show the planned output and exit; don't call the API.")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse if the estimated cost exceeds this cap. Note: upscale "
        "pricing is dynamic, so no pre-charge estimate is available.",
    )
    p.add_argument("--no-balance", action="store_true",
                   help="Skip the upfront balance display.")
    p.set_defaults(handler=_run)


def _validate(args) -> Optional[int]:
    inp = args.input
    if not inp.is_file():
        print(f"upscale: input file not found: {inp}", file=sys.stderr)
        return 2
    size = inp.stat().st_size
    if size == 0:
        print(f"upscale: input {inp} is empty", file=sys.stderr)
        return 2
    if size > MAX_INPUT_BYTES:
        print(f"upscale: input {inp} is {size} bytes; must be < 25 MB",
              file=sys.stderr)
        return 2
    if not (1 <= args.scale <= 4):
        print(f"upscale: --scale {args.scale} out of range (1-4)", file=sys.stderr)
        return 2
    if args.scale == 1 and not args.enhance:
        print("upscale: --scale 1 only runs the enhancer; pass --enhance "
              "(or use --scale >1)", file=sys.stderr)
        return 2
    if args.enhance_creativity is not None and not (0 <= args.enhance_creativity <= 1):
        print("upscale: --enhance-creativity must be between 0 and 1", file=sys.stderr)
        return 2
    if args.replication is not None and not (0 <= args.replication <= 1):
        print("upscale: --replication must be between 0 and 1", file=sys.stderr)
        return 2
    if args.enhance_prompt is not None and len(args.enhance_prompt) > MAX_ENHANCE_PROMPT:
        print(f"upscale: --enhance-prompt exceeds {MAX_ENHANCE_PROMPT} chars",
              file=sys.stderr)
        return 2
    return None


def _build_body(args, image_b64: str) -> dict:
    body: dict = {
        "image": image_b64,
        "scale": args.scale,
        "enhance": bool(args.enhance),
    }
    if args.enhance_creativity is not None:
        body["enhanceCreativity"] = args.enhance_creativity
    if args.enhance_prompt is not None:
        body["enhancePrompt"] = args.enhance_prompt
    if args.replication is not None:
        body["replication"] = args.replication
    return body


def _fmt_scale(scale: float) -> str:
    return str(int(scale)) if float(scale).is_integer() else str(scale)


def _run(args) -> int:
    userconfig.apply_defaults(args, "upscale")
    rc = _validate(args)
    if rc is not None:
        return rc

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    cost = None  # dynamic pricing -- no reliable upfront quote
    print_estimate(cost, f"×{_fmt_scale(args.scale)} upscale; dynamic $0.001-$10.00/call")
    print_balance_and_remaining(client, cost, show=not args.no_balance)
    if over_budget(cost, args.max_spend):  # no-op while cost is unknown
        return 1

    out_path = resolve_output(args.output, f"{args.input.stem}-upscaled.png")
    if args.dry_run:
        print(f"would write: {out_path.resolve()}", file=sys.stderr)
        return 0

    rc = confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    image_b64 = base64.b64encode(args.input.read_bytes()).decode("ascii")
    body = _build_body(args, image_b64)
    return post_binary_op(client, ENDPOINT, body, out_path, "upscale")
