"""`venice balance` -- show current account balance and tier."""
from __future__ import annotations

import json
import sys

from .. import auth
from ..billing import fetch_balance, format_usd
from ..client import VeniceAPIError


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "balance",
        help="Show current account balance (USD + DIEM).",
        description=(
            "Queries /api_keys/rate_limits. Default: print '$X.XX USD' to "
            "stdout. --json for raw, --verbose for tier + next epoch. "
            "--min sets a floor: exits 1 if balance < floor (script-friendly)."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Dump raw {USD, DIEM, tier, next_epoch, key_expires} to stdout.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print a human-readable multi-line summary.",
    )
    p.add_argument(
        "--min",
        type=float,
        default=None,
        metavar="USD",
        help="Exit 1 if balance is below this USD threshold.",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    try:
        from ..client import build_client_from_auth
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        info = fetch_balance(client)
    except VeniceAPIError as e:
        print(f"balance: API error: {e}", file=sys.stderr)
        if e.status == 401:
            return 2
        if e.status == 429:
            return 4
        return 5

    if info is None:
        print("balance: API returned no data block", file=sys.stderr)
        return 5

    usd = info.get("usd")
    diem = info.get("diem")
    tier = info.get("tier")
    next_epoch = info.get("next_epoch")
    key_exp = info.get("key_expires")

    if args.json:
        json.dump(
            {
                "USD": usd,
                "DIEM": diem,
                "tier": tier,
                "next_epoch": next_epoch,
                "key_expires": key_exp,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    elif args.verbose:
        print(f"Tier:        {tier or 'unknown'}")
        print(f"Balance:     {format_usd(usd)}")
        if diem is not None:
            print(f"             {float(diem):.4f} DIEM")
        if next_epoch:
            print(f"Next epoch:  {next_epoch}")
        print(f"Key expires: {key_exp or 'never'}")
    else:
        print(format_usd(usd))

    if args.min is not None and usd is not None:
        try:
            if float(usd) < float(args.min):
                print(
                    f"balance: ${float(usd):.4f} is below floor ${float(args.min):.4f}",
                    file=sys.stderr,
                )
                return 1
        except (TypeError, ValueError):
            pass

    return 0
