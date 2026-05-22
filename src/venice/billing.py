"""Small wrapper over /api_keys/rate_limits for balance + tier info.

Kept separate from client.py so commands can import a stable, narrow API
without depending on the raw HTTP shape.
"""
from __future__ import annotations

from typing import Optional


def fetch_balance(client) -> Optional[dict]:
    """Return {usd, diem, tier, next_epoch, key_expires} or None."""
    data = client.get_balance()
    if not isinstance(data, dict):
        return None
    balances = data.get("balances") if isinstance(data.get("balances"), dict) else {}
    tier_block = data.get("apiTier") if isinstance(data.get("apiTier"), dict) else {}
    return {
        "usd": balances.get("USD"),
        "diem": balances.get("DIEM"),
        "tier": tier_block.get("id"),
        "is_charged": tier_block.get("isCharged"),
        "next_epoch": data.get("nextEpochBegins"),
        "key_expires": data.get("keyExpiration"),
    }


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
