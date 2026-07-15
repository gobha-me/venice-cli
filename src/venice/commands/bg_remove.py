"""`venice bg-remove` -- strip an image's background via /image/background-remove.

Venice's generate call treats `background: transparent` as a no-op, so
transparent assets (e.g. rank insignia) are made opaque then run through this
endpoint, which returns a PNG with a transparent background. The source is
either a local file (sent as base64 in a JSON body) or an image URL. The
response is raw image/png bytes, so we use the client's bytes-or-json path.
Pricing is dynamic ($0.001-$10.00/call); the balance is shown and you confirm.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

from .. import auth
from ..client import build_client_from_auth
from ._shared import (
    confirm_or_exit,
    over_budget,
    post_binary_op,
    print_balance_and_remaining,
    print_estimate,
    resolve_output,
)

ENDPOINT = "/image/background-remove"
MAX_INPUT_BYTES = 25 * 1024 * 1024  # API limit: input file < 25 MB
URL_DEFAULT_NAME = "venice-nobg.png"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "bg-remove",
        help="Remove an image's background via /image/background-remove.",
        description=(
            "Returns a PNG with a transparent background. Source is a local file "
            "(positional) or --image-url. Use for assets that need alpha, e.g. "
            "`venice bg-remove insignia.png`. Pricing is dynamic; the balance is "
            "shown and you confirm before the charge."
        ),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("input", type=Path, nargs="?", default=None,
                     help="Image file whose background to remove.")
    src.add_argument("--image-url", default=None, metavar="URL",
                     help="Remove the background from an image at this URL instead.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file or directory. Default: cwd/<input>-nobg.png "
                   f"(or {URL_DEFAULT_NAME} for --image-url).")
    p.add_argument("--yes", "-y", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the planned output and exit; don't call the API.")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse if the estimated cost exceeds this cap. Note: bg-remove "
        "pricing is dynamic, so no pre-charge estimate is available.",
    )
    p.add_argument("--no-balance", action="store_true",
                   help="Skip the upfront balance display.")
    p.set_defaults(handler=_run)


def _validate(args) -> Optional[int]:
    if (args.input is None) == (args.image_url is None):
        print("bg-remove: provide exactly one of INPUT file or --image-url",
              file=sys.stderr)
        return 2
    if args.input is not None:
        inp = args.input
        if not inp.is_file():
            print(f"bg-remove: input file not found: {inp}", file=sys.stderr)
            return 2
        size = inp.stat().st_size
        if size == 0:
            print(f"bg-remove: input {inp} is empty", file=sys.stderr)
            return 2
        if size > MAX_INPUT_BYTES:
            print(f"bg-remove: input {inp} is {size} bytes; must be < 25 MB",
                  file=sys.stderr)
            return 2
    return None


def _build_body(args) -> dict:
    if args.input is not None:
        b64 = base64.b64encode(args.input.read_bytes()).decode("ascii")
        return {"image": b64}
    return {"image_url": args.image_url}


def _run(args) -> int:
    rc = _validate(args)
    if rc is not None:
        return rc

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    cost = None  # dynamic pricing -- no reliable upfront quote
    print_estimate(cost, "background removal; dynamic $0.001-$10.00/call")
    print_balance_and_remaining(client, cost, show=not args.no_balance)
    if over_budget(cost, args.max_spend):  # no-op while cost is unknown
        return 1

    default_name = (
        f"{args.input.stem}-nobg.png" if args.input is not None else URL_DEFAULT_NAME
    )
    out_path = resolve_output(args.output, default_name)
    if args.dry_run:
        print(f"would write: {out_path.resolve()}", file=sys.stderr)
        return 0

    rc = confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    body = _build_body(args)
    return post_binary_op(client, ENDPOINT, body, out_path, "bg-remove")
