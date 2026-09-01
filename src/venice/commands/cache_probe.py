"""``venice cache-probe`` -- controlled live prefix-cache diagnostics (#97).

The probe deliberately uses the stdlib Venice client rather than the OpenAI SDK:
the response's raw ``prompt_tokens_details`` object is the deliverable, and a
diagnostic for an OpenAI-compatible API should not require an optional client SDK.

Every paid request is preceded by a conservative all-uncached estimate and the
shared confirmation gate.  Tests replace the transport or use the loopback fake
API; this module never needs a real credential during development.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from typing import Iterable, List, Optional, Tuple

from .. import _numeric, auth, billing, userconfig
from ..client import VeniceAPIError, build_client_from_auth
from . import _models, _shared


DEFAULT_PREFIX_TOKENS = 8192
DEFAULT_REPEAT = 2
MAX_PREFIX_TOKENS = 1_000_000
MAX_REPEAT = 10
_MESSAGE_OVERHEAD_TOKENS = 256
_PROBE_QUESTION = "Reply with a single period."


def _prefix_tokens(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not 1 <= parsed <= MAX_PREFIX_TOKENS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_PREFIX_TOKENS}"
        )
    return parsed


def _repeat_count(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not 2 <= parsed <= MAX_REPEAT:
        raise argparse.ArgumentTypeError(f"must be between 2 and {MAX_REPEAT}")
    return parsed


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "cache-probe",
        help="Test whether prefix caching is live for one or more text models.",
        description=(
            "Send a deterministic synthetic prefix repeatedly and report the raw "
            "prompt_tokens_details returned by the API. This makes paid live calls; "
            "the worst-case estimate is shown before confirmation."
        ),
    )
    p.add_argument(
        "--model", "-m", action="append", default=None, metavar="ID",
        help=(
            "Text model to probe (repeatable). Explicit values replace the default "
            "chat model plus automatic different-family control."
        ),
    )
    p.add_argument(
        "--prefix-tokens", action="append", type=_prefix_tokens, default=None,
        metavar="N", help=(
            f"Approximate synthetic prefix size (repeatable; default "
            f"{DEFAULT_PREFIX_TOKENS}). Six-digit threshold probes are opt-in."
        ),
    )
    p.add_argument(
        "--repeat", type=_repeat_count, default=DEFAULT_REPEAT, metavar="N",
        help=f"Calls per model and prefix size (default {DEFAULT_REPEAT}, max {MAX_REPEAT}).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print one JSON result containing raw per-call usage details.",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Confirm the complete estimated probe matrix without prompting.",
    )
    p.add_argument(
        "--max-spend", type=_numeric.non_negative_float, default=None, metavar="USD",
        help="Refuse the complete matrix if its worst-case USD estimate exceeds this cap.",
    )
    p.set_defaults(handler=_run)


def _ordered_unique(values: Iterable) -> list:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _advertises_cache(model: dict) -> bool:
    spec = model.get("model_spec") if isinstance(model, dict) else None
    pricing = spec.get("pricing") if isinstance(spec, dict) else None
    return isinstance(pricing, dict) and "cache_input" in pricing


def _control_model(models, primary: str) -> Optional[str]:
    """First cache-advertising model from a different family, in catalog order."""
    family = _models.model_family(primary)
    for model in models or []:
        mid = model.get("id") if isinstance(model, dict) else None
        if (
            isinstance(mid, str)
            and mid != primary
            and _models.model_family(mid) != family
            and _advertises_cache(model)
        ):
            return mid
    return None


def _resolve_models(args, models) -> Tuple[Optional[List[str]], Optional[int]]:
    explicit = _ordered_unique(args.model or [])
    if explicit:
        resolved = []
        for requested in explicit:
            model, rc = _models.resolve_model(
                requested, models, label="cache-probe", noun="text model"
            )
            if rc is not None:
                return None, rc
            resolved.append(model)
        return resolved, None

    configured = userconfig.resolve_default("chat", "model")
    primary, rc = _models.resolve_model(
        configured,
        models,
        label="cache-probe",
        noun="text model",
        config_key="defaults.chat.model",
    )
    if rc is not None:
        return None, rc
    control = _control_model(models, primary)
    if control is None:
        print(
            "cache-probe: no cache-advertising different-family control model "
            "is available; probing only the primary model",
            file=sys.stderr,
        )
        return [primary], None
    return [primary, control], None


def _build_prefix(size: int) -> str:
    # The size marker is at the start, so different sweep sizes cannot warm one
    # another.  The remainder is intentionally boring ASCII and carries no user or
    # project content.  Repeating "x " is approximately one token per requested
    # unit on common tokenizers; the API's actual prompt_tokens is always reported.
    return f"CACHE-PROBE-V1 SIZE={size}\n" + ("x " * size)


def _prefix_byte_bound(size: int) -> int:
    """Conservative input-token bound without allocating the synthetic prefix."""
    marker = f"CACHE-PROBE-V1 SIZE={size}\n"
    return len(marker.encode("ascii")) + (2 * size) + _MESSAGE_OVERHEAD_TOKENS


def _body(model: str, prefix: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": _PROBE_QUESTION},
        ],
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
    }


def _price(models, model_id: str, key: str) -> Optional[float]:
    for model in models or []:
        if not isinstance(model, dict) or model.get("id") != model_id:
            continue
        spec = model.get("model_spec")
        pricing = spec.get("pricing") if isinstance(spec, dict) else None
        node = pricing.get(key) if isinstance(pricing, dict) else None
        raw = node.get("usd") if isinstance(node, dict) else None
        try:
            return _numeric.non_negative_float(raw) / 1_000_000.0
        except (TypeError, ValueError):
            return None
    return None


def _estimate(models, model_ids, sizes: List[int], repeat: int) -> dict:
    calls = len(model_ids) * len(sizes) * repeat
    input_upper = sum(
        _prefix_byte_bound(size) for size in sizes
    ) * len(model_ids) * repeat
    output_upper = calls
    total = 0.0
    missing = []
    for model_id in model_ids:
        input_rate = _price(models, model_id, "input")
        output_rate = _price(models, model_id, "output")
        if input_rate is None or output_rate is None:
            missing.append(model_id)
            continue
        for size in sizes:
            max_input = _prefix_byte_bound(size)
            model_cost = repeat * (max_input * input_rate + output_rate)
            total += model_cost
            if not math.isfinite(model_cost) or not math.isfinite(total):
                missing.append(model_id)
                break
    return {
        # Keep the unrounded value for --max-spend. Display formatting can round,
        # but a budget rail must never admit a value just above the cap.
        "usd_upper_bound": None if missing else total,
        "calls": calls,
        "input_tokens_upper_bound": input_upper,
        "completion_tokens_upper_bound": output_upper,
        "pricing_complete": not missing,
        "unpriced_models": missing,
    }


def _print_estimate(estimate: dict) -> None:
    total = estimate["usd_upper_bound"]
    detail = (
        f"{estimate['calls']} calls, up to "
        f"{estimate['input_tokens_upper_bound']} uncached input tokens + "
        f"{estimate['completion_tokens_upper_bound']} output tokens"
    )
    if total is None:
        missing = ", ".join(estimate["unpriced_models"])
        print(
            f"Estimated maximum cost: (unknown -- missing input/output pricing "
            f"for {missing}; {detail})",
            file=sys.stderr,
        )
    else:
        print(
            f"Estimated maximum cost: {billing.format_usd(total)} ({detail})",
            file=sys.stderr,
        )


def _numeric_cache_value(details) -> Optional[float]:
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _verdict(calls: List[dict]) -> str:
    values = [_numeric_cache_value(call.get("prompt_tokens_details")) for call in calls]
    if any(value is not None and value > 0 for value in values):
        return "warms"
    if values and all(value == 0 for value in values):
        return "never warms"
    return "field absent"


def _call_row(response, n: int) -> dict:
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    row = {
        "n": n,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # Preserve the raw object only when it is an object. null/garbage is the
        # three-state "field absent" result, never normalized into an empty object.
        "prompt_tokens_details": details if isinstance(details, dict) else None,
    }
    try:
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("usage block is not valid strict JSON") from None
    return row


def _print_call(model: str, size: int, row: dict, repeat: int) -> None:
    raw = json.dumps(
        row["prompt_tokens_details"], allow_nan=False,
        separators=(",", ":"),
    )
    print(
        f"model={model} prefix_tokens={size} call={row['n']}/{repeat} "
        f"prompt_tokens={row['prompt_tokens']} prompt_tokens_details={raw}"
    )


def _summaries(model_ids: List[str], results: List[dict]) -> List[dict]:
    summaries = []
    for model_id in model_ids:
        rows = [result for result in results if result["model"] == model_id]
        summaries.append({
            "model": model_id,
            "warms": [row["prefix_tokens"] for row in rows if row["verdict"] == "warms"],
            "never_warms": [
                row["prefix_tokens"] for row in rows
                if row["verdict"] == "never warms"
            ],
            "field_absent": [
                row["prefix_tokens"] for row in rows
                if row["verdict"] == "field absent"
            ],
        })
    return summaries


def _print_summaries(summaries: List[dict]) -> None:
    for row in summaries:
        def sizes(key):
            return ",".join(str(value) for value in row[key]) or "none"

        print(
            f"{row['model']}: warms={sizes('warms')} "
            f"never-warms={sizes('never_warms')} "
            f"field-absent={sizes('field_absent')}"
        )


def _run(args) -> int:
    sizes = _ordered_unique(args.prefix_tokens or [DEFAULT_PREFIX_TOKENS])
    try:
        client = build_client_from_auth()
    except auth.AuthError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    models = _models.catalog(client, "text")
    model_ids, rc = _resolve_models(args, models)
    if rc is not None:
        return rc

    estimate = _estimate(models, model_ids, sizes, args.repeat)
    _print_estimate(estimate)
    if _shared.over_budget(estimate["usd_upper_bound"], args.max_spend):
        shown = (
            billing.format_usd(estimate["usd_upper_bound"])
            if estimate["usd_upper_bound"] is not None else "unknown"
        )
        print(
            f"cache-probe: estimate {shown} cannot be kept within --max-spend "
            f"{billing.format_usd(args.max_spend)}; aborting",
            file=sys.stderr,
        )
        return 1
    # ``input`` writes its prompt to stdout. Keep a JSON run's stdout as one valid
    # document by routing only that interactive prompt to stderr.
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            rc = _shared.confirm_or_exit(args.yes)
    else:
        rc = _shared.confirm_or_exit(args.yes)
    if rc is not None:
        return rc

    results = []
    try:
        for model_id in model_ids:
            for size in sizes:
                # Keep at most one synthetic prefix resident. A wide threshold
                # sweep should not multiply its largest allocation by row count.
                prefix = _build_prefix(size)
                calls = []
                for n in range(1, args.repeat + 1):
                    response = client.post_json(
                        "/chat/completions", _body(model_id, prefix)
                    )
                    row = _call_row(response, n)
                    calls.append(row)
                    if not args.json:
                        _print_call(model_id, size, row, args.repeat)
                results.append({
                    "model": model_id,
                    "prefix_tokens": size,
                    "calls": calls,
                    "verdict": _verdict(calls),
                })
    except VeniceAPIError as exc:
        print(f"cache-probe: {exc}", file=sys.stderr)
        return _shared.status_to_exit(exc)
    except ValueError as exc:
        print(f"cache-probe: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ncache-probe: aborted", file=sys.stderr)
        return 130

    summaries = _summaries(model_ids, results)
    if args.json:
        json.dump({
            "estimate": estimate,
            "models": model_ids,
            "prefix_tokens": sizes,
            "repeat": args.repeat,
            "results": results,
            "summary": summaries,
        }, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    else:
        _print_summaries(summaries)
    return 0
