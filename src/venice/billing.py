"""Small wrapper over /api_keys/rate_limits for balance + tier info.

Kept separate from client.py so commands can import a stable, narrow API
without depending on the raw HTTP shape.

## Venice's billing model

Venice has up to four spendable buckets, drained in this order:

  1. DIEM allowance     -- daily credit derived from staked DIEM (1 DIEM
                           staked = $1/day). Resets every 24 h at
                           `nextEpochBegins`. Per-epoch use-it-or-lose-it.
  2. Monthly credit     -- BUNDLED_CREDITS, a recently-added bundle
                           granted with paid subscriptions. Drains BEFORE
                           cash. NOT exposed via the inference-key
                           endpoint -- only via /billing/balance (admin).
  3. VCU                -- Venice Compute Units (per-tier inclusions).
                           Also not exposed via the inference key.
  4. USD cash           -- one-and-done prepaid USD balance.

1 unit of any of these == $1 of purchasing power: per-model pricing in
/models lists the same number in both `usd` and `diem` fields.

What we can read with an inference key (`VENICE-INFERENCE-KEY-...`):
  - balances.USD       (cash)
  - balances.DIEM      (epoch allowance, the remaining portion for the
                        current 24 h)
  - nextEpochBegins    (when DIEM resets)
  - apiTier, keyExpiration, rateLimits

What we cannot read with this key:
  - BUNDLED_CREDITS    (monthly bundle balance)
  - VCU                (compute-unit balance)

Practical implication: when the user has monthly credit, the CLI's
"after charge" line will be slightly pessimistic about USD cash (since
the actual debit lands on monthly first). The combined spendable total
shown still reflects what we can see; if your real spendable is higher,
it means monthly credit / VCU is doing some of the lifting silently.
"""
from __future__ import annotations

from typing import Optional

# Hard-coded because the spec doesn't expose it as a field on this endpoint.
SPEND_ORDER = ("DIEM allowance", "monthly credit", "USD cash")


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

    Example: "$32.70 USD (6.56 DIEM allowance + 26.14 USD cash)"
    Order mirrors the spend order (DIEM drains first), so the leftmost
    bucket is what funds the next call. Falls back gracefully if a
    component is missing.
    """
    if not isinstance(info, dict):
        return "(no balance info)"
    usd = info.get("usd")
    diem = info.get("diem")
    total = info.get("total")
    head = format_usd(total)
    parts = []
    if diem is not None:
        try:
            parts.append(f"{float(diem):.2f} DIEM allowance")
        except (TypeError, ValueError):
            pass
    if usd is not None:
        try:
            parts.append(f"{float(usd):.2f} USD cash")
        except (TypeError, ValueError):
            pass
    if parts:
        return f"{head} ({' + '.join(parts)})"
    return head
