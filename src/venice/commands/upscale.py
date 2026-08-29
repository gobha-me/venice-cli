"""`venice upscale` -- upscale an image via /image/upscale (sync).

`/image/generate` caps width/height at 1280, so large environment art is made
≤1280 then upscaled here (e.g. ×2: 960×540 -> 1920×1080). The input image is
sent as a base64 string in a JSON body; the endpoint returns raw image/png
bytes (not the base64 array `/image/generate` returns), so we use the client's
bytes-or-json path. Pricing is dynamic (Venice bills $0.001-$10.00 per call),
so there is no reliable upfront quote -- we show the balance and confirm.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .. import _numeric, auth, userconfig
from ..client import build_client_from_auth
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

ENDPOINT = "/image/upscale"
DEFAULT_SCALE = 2.0
SCALES = (2.0, 4.0)
MIN_CREATIVITY = 0.0
MAX_CREATIVITY = 0.02
SERVER_DEFAULT_CREATIVITY = 0.01
RETIRED_CONFIG_KEYS = (
    "enhance",
    "enhance_creativity",
    "enhance_prompt",
    "replication",
)


def retired_config_keys(doc) -> tuple[str, ...]:
    """Retired ``defaults.upscale`` keys present in a config document.

    Unknown config keys deliberately survive normal config round-trips.  These
    four cannot merely become unknown: silently ignoring an old paid-operation
    preference would make a request whose output no longer matches the user's
    stated intent.  Callers therefore fail the upscale operation closed while
    leaving unrelated commands/tools usable.
    """
    if not isinstance(doc, dict):
        return ()
    defaults = doc.get("defaults")
    if not isinstance(defaults, dict):
        return ()
    section = defaults.get("upscale")
    if not isinstance(section, dict):
        return ()
    return tuple(key for key in RETIRED_CONFIG_KEYS if key in section)


def retired_config_message(keys) -> str:
    joined = ", ".join(f"defaults.upscale.{key}" for key in keys)
    commands = "; ".join(
        f"venice config unset defaults.upscale.{key}" for key in keys
    )
    return (
        f"upscale: retired config setting(s): {joined}; Venice no longer accepts "
        f"the enhancer controls. Remove them with: {commands}"
    )


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "upscale",
        help="Upscale an image via /image/upscale (sync).",
        description=(
            "Upscales an image by a factor of 2 or 4 (default 2). Use this to "
            "take generated art above the 1280px "
            "generate cap, e.g. `venice upscale env.png --scale 2` -> 2x. Pricing "
            "is dynamic; the balance is shown and you confirm before the charge."
        ),
    )
    p.add_argument("input", type=Path, help="Image file to upscale.")
    p.add_argument(
        "--scale",
        type=_numeric.finite_float,
        default=None,
        metavar="N",
        help=f"Upscale factor: 2 or 4 (default {DEFAULT_SCALE:g}). "
        "Config-backable via defaults.upscale.scale; an explicit --scale still wins.",
    )
    p.add_argument(
        "--creativity",
        type=_numeric.finite_float,
        default=None,
        metavar="F",
        help=f"Detail/texture creativity from {MIN_CREATIVITY:g} to "
        f"{MAX_CREATIVITY:g} (server default {SERVER_DEFAULT_CREATIVITY:g}). "
        "Config-backable via defaults.upscale.creativity; omitted values are "
        "left to the server.",
    )
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file or directory. Default: cwd/<input>-upscaled.png.")
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Show the planned output and exit; don't call the API.")
    p.add_argument(
        "--max-spend",
        type=_numeric.non_negative_float,
        default=None,
        metavar="USD",
        help="Refuse if the estimated cost exceeds this cap. Note: upscale "
        "pricing is dynamic, so no pre-charge estimate is available.",
    )
    add_balance_flag(p)
    p.set_defaults(handler=_run)


def _validate(args) -> Optional[int]:
    rc = check_image_file(args.input, label="upscale")
    if rc is not None:
        return rc
    if args.scale not in SCALES:
        print(f"upscale: --scale must be 2 or 4 (got {args.scale})", file=sys.stderr)
        return 2
    if args.creativity is not None and not (
        MIN_CREATIVITY <= args.creativity <= MAX_CREATIVITY
    ):
        print(
            f"upscale: --creativity must be between {MIN_CREATIVITY:g} and "
            f"{MAX_CREATIVITY:g}",
            file=sys.stderr,
        )
        return 2
    return None


def _build_body(args, image_b64: str) -> dict:
    body: dict = {
        "image": image_b64,
        "scale": args.scale,
    }
    if args.creativity is not None:
        body["creativity"] = args.creativity
    return body


def _fmt_scale(scale: float) -> str:
    return str(int(scale)) if float(scale).is_integer() else str(scale)


def _run(args) -> int:
    doc = userconfig.load_config()
    retired = retired_config_keys(doc)
    if retired:
        print(retired_config_message(retired), file=sys.stderr)
        return 2
    userconfig.apply_defaults(args, "upscale", doc)
    # #57 Class C1: built-in literal last, before `_validate`'s range check,
    # which cannot compare None. A config-set `scale: 0` deliberately survives
    # this to reach that check's own out-of-range message.
    userconfig.apply_literals(args, scale=DEFAULT_SCALE)
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
    if over_budget(cost, args.max_spend):
        print(
            "upscale: dynamic price cannot be bounded by --max-spend; aborting",
            file=sys.stderr,
        )
        return 1

    out_path = resolve_output(args.output, f"{args.input.stem}-upscaled.png")
    if args.dry_run:
        print(f"would write: {out_path.resolve()}", file=sys.stderr)
        return 0

    rc = confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    image_b64 = encode_base64(args.input)
    body = _build_body(args, image_b64)
    return post_binary_op(client, ENDPOINT, body, out_path, "upscale")
