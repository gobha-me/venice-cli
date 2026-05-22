"""`venice models` -- browse the Venice model catalog."""
from __future__ import annotations

import json
import sys
from typing import Iterable, List, Optional

from .. import auth
from ..client import VeniceAPIError, build_client_from_auth

MODEL_TYPES = (
    "text",
    "code",
    "image",
    "video",
    "music",
    "tts",
    "embedding",
    "upscale",
)


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "models",
        help="List Venice models (filter by type, or show one in detail).",
        description=(
            "Without args: count by type. With --type, list ids of that "
            "type. With a slug arg, show full detail for that one model."
        ),
    )
    p.add_argument(
        "slug",
        nargs="?",
        help="Optional model id; if given, print full details for it.",
    )
    p.add_argument(
        "--type",
        "-t",
        choices=("all", *MODEL_TYPES),
        help="Filter to one type (or 'all' to list every id).",
    )
    p.add_argument(
        "--detail",
        "-d",
        action="store_true",
        help="With --type: include name, capabilities, pricing per row.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Dump raw JSON (the /models data array) to stdout.",
    )
    p.set_defaults(handler=_run)


def _fetch_type(client, mtype: str) -> List[dict]:
    doc = client.get_json("/models", params={"type": mtype})
    data = doc.get("data") if isinstance(doc, dict) else None
    return list(data) if isinstance(data, list) else []


def _fetch_all(client) -> dict:
    return {t: _fetch_type(client, t) for t in MODEL_TYPES}


def _print_counts(by_type: dict) -> None:
    rows = [(t, len(by_type.get(t, []))) for t in MODEL_TYPES]
    total = sum(c for _, c in rows)
    width = max(len(t) for t, _ in rows)
    print(f"{'TYPE':<{width}}  COUNT")
    for t, c in rows:
        print(f"{t:<{width}}  {c:5d}")
    print(f"{'TOTAL':<{width}}  {total:5d}")


def _format_pricing(pricing) -> str:
    if not isinstance(pricing, dict) or not pricing:
        return ""
    parts = []
    for k, v in pricing.items():
        if isinstance(v, dict) and "usd" in v:
            parts.append(f"{k}=${v['usd']}")
    return " ".join(parts)


def _format_caps(caps) -> str:
    if not isinstance(caps, dict):
        return ""
    on = [k.replace("supports", "").lstrip("_") for k, v in caps.items()
          if isinstance(v, bool) and v and k.startswith("supports")]
    return ",".join(sorted(on))


def _print_listing(models: Iterable[dict], detail: bool) -> None:
    for m in models:
        mid = m.get("id", "")
        if not detail:
            print(mid)
            continue
        spec = m.get("model_spec") or {}
        name = spec.get("name", "")
        caps = _format_caps(spec.get("capabilities"))
        price = _format_pricing(spec.get("pricing"))
        line = mid
        if name:
            line += f"  -- {name}"
        print(line)
        if price:
            print(f"    pricing: {price}")
        if caps:
            print(f"    capabilities: {caps}")


def _find_model(client, slug: str) -> Optional[dict]:
    for t in MODEL_TYPES:
        for m in _fetch_type(client, t):
            if m.get("id") == slug:
                return m
    return None


def _run(args) -> int:
    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        if args.slug:
            m = _find_model(client, args.slug)
            if not m:
                print(f"models: no model with id {args.slug!r}", file=sys.stderr)
                return 6
            json.dump(m, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0

        if args.type and args.type != "all":
            data = _fetch_type(client, args.type)
            if args.json:
                json.dump(data, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                _print_listing(data, detail=args.detail)
            return 0

        by_type = _fetch_all(client)
        if args.json:
            json.dump(by_type, sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0

        if args.type == "all":
            for t in MODEL_TYPES:
                print(f"### {t} ({len(by_type[t])})")
                _print_listing(by_type[t], detail=args.detail)
                print()
            return 0

        _print_counts(by_type)
        return 0
    except VeniceAPIError as e:
        print(f"models: API error: {e}", file=sys.stderr)
        if e.status == 401:
            return 2
        if e.status == 429:
            return 4
        if 500 <= e.status < 600:
            return 5
        if e.status == 0:
            return 8
        return 5
