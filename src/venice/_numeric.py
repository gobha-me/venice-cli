"""Strict numeric coercion shared by CLI, config, MCP, and client rails."""
from __future__ import annotations

import math


def finite_float(value) -> float:
    """Return *value* as a finite float, rejecting booleans and JSON extensions."""
    if isinstance(value, bool):
        raise ValueError("must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError("must be finite")
    return number


def non_negative_float(value) -> float:
    """Return a finite float greater than or equal to zero."""
    number = finite_float(value)
    if number < 0:
        raise ValueError("must be >= 0")
    return number


def reject_json_constant(value: str):
    """Reject Python's non-standard JSON NaN/Infinity extensions on input."""
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")
