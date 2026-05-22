"""Small wrapper over /api_keys/rate_limits for balance + tier info.

Kept separate from client.py so commands can import a stable, narrow API
without depending on the raw HTTP shape.

DIEM is Venice's credit unit, equivalent to USD for billing purposes
(per-model pricing lists the same number in both `usd` and `diem`
fields, which means 1 DIEM = $1 of purchasing power). So the user's
effective spendable balance is `USD + DIEM` USD-equivalent.
"""
from __future__ import annotations

from typing import Optional


def fetch_balance(client) -> Optional[dict]:
    """Return {usd, diem, total, tier, next_epoch, key_expires} or None.

    `total` is the combined spendable balance in USD-equivalent units
    (USD + DIEM, since 1 DIEM = $1).
    """
    data = client.get_balance()
    if not isinstance(data, dict):
        return None
    balances = data.get("balances") if isinstance(data.get("balances"), dict) else {}
    tier_block = data.get("apiTier") if isinstance(data.get("apiTier"), dict) else {}
    usd = balances.get("USD")
    diem = balances.get("DIEM")
    total = _safe_sum(usd, diem)
    return {
        "usd": usd,
        "diem": diem,
        "total": total,
        "tier": tier_block.get("id"),
        "is_charged": tier_block.get("isCharged"),
        "next_epoch": data.get("nextEpochBegins"),
        "key_expires": data.get("keyExpiration"),
    }


def _safe_sum(*vals) -> Optional[float]:
    """Sum that returns None if any input is non-numeric (and 0 if all None)."""
    total = 0.0
    any_seen = False
    for v in vals:
        if v is None:
            continue
        try:
            total += float(v)
            any_seen = True
        except (TypeError, ValueError):
            return None
    return total if any_seen else None


def format_usd(amt) -> str:
    """Pretty-print a USD amount. 4 decimals for sub-dollar, 2 otherwise."""
    if amt is None:
        return "$?.?? USD"
    try:
        v = float(amt)
    except (TypeError, ValueError):
        return f"{amt} USD"
    if abs(v) < 1:
        return f"${v:.4f} USD"
    return f"${v:.2f} USD"


def format_balance_breakdown(info: dict) -> str:
    """Render a one-line balance summary with the spendable total + breakdown.

    Example: "$27.04 USD (26.14 USD + 0.90 DIEM credit)"
    Falls back gracefully if any component is missing.
    """
    if not isinstance(info, dict):
        return "(no balance info)"
    usd = info.get("usd")
    diem = info.get("diem")
    total = info.get("total")
    head = format_usd(total)
    parts = []
    if usd is not None:
        try:
            parts.append(f"{float(usd):.2f} USD")
        except (TypeError, ValueError):
            pass
    if diem is not None:
        try:
            parts.append(f"{float(diem):.2f} DIEM credit")
        except (TypeError, ValueError):
            pass
    if parts:
        return f"{head} ({' + '.join(parts)})"
    return head
