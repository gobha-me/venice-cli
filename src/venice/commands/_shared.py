"""Shared UX/budget rails for spend-incurring commands.

Extracted from `image` so it, `upscale`, and `bg-remove` share one copy of the
estimate/confirm/status/write plumbing rather than each carrying its own. These
helpers take primitive args (cost, max_spend, err, byte blobs) so they stay
independent of any one command's argument shape.
"""
from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path
from typing import List, Optional

from .. import billing
from ..client import VeniceAPIError


def resolve_output(arg_output: Optional[Path], default_name: str) -> Path:
    """Pick an output path: an explicit file, a file inside an explicit dir, or
    `default_name` in the cwd."""
    if arg_output is None:
        return Path.cwd() / default_name
    if arg_output.is_dir():
        return arg_output / default_name
    return arg_output


def encode_data_url(path: Path, *, default_mime: str = "application/octet-stream") -> str:
    """Read a local file and return a `data:<mime>;base64,<b64>` URL.

    The MIME type is sniffed from the filename; callers pass `default_mime` as
    the fallback for extensions `mimetypes` doesn't recognise. Unlike the raw
    base64 that `bg-remove`/`image-edit` send in an `image` field, the Venice
    `/video` media inputs want a full data URL, so this prepends the prefix.
    """
    mime = mimetypes.guess_type(str(path))[0] or default_mime
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


MAX_IMAGE_BYTES = 25 * 1024 * 1024  # API limit: each input image < 25 MB


def encode_base64(path: Path) -> str:
    """Read a local file and return raw base64 (no `data:` prefix).

    This is the form `upscale`/`bg-remove`/`image-edit` send in an
    `image`/`images` field. Contrast `encode_data_url`, which prepends the
    `data:<mime>;base64,` prefix the `/video` media inputs require.
    """
    return base64.b64encode(path.read_bytes()).decode("ascii")


def check_image_file(
    path: Path, *, label: str, max_bytes: int = MAX_IMAGE_BYTES
) -> Optional[int]:
    """Gate a local image input: exists, non-empty, and under `max_bytes`.

    Returns exit code 2 with a `label`-prefixed stderr message on failure, else
    None. Shared by the image-input commands so the exists/empty/size check
    lives in one place (`label` = "upscale"/"bg-remove"/"image-edit").
    """
    if not path.is_file():
        print(f"{label}: input file not found: {path}", file=sys.stderr)
        return 2
    size = path.stat().st_size
    if size == 0:
        print(f"{label}: input {path} is empty", file=sys.stderr)
        return 2
    if size > max_bytes:
        print(
            f"{label}: input {path} is {size} bytes; "
            f"must be < {max_bytes // (1024 * 1024)} MB",
            file=sys.stderr,
        )
        return 2
    return None


def print_estimate(cost: Optional[float], label: str) -> None:
    if cost is None:
        print(f"Estimated cost: (unknown — {label})", file=sys.stderr)
    else:
        print(
            f"Estimated cost: {billing.format_usd(cost)} ({label})",
            file=sys.stderr,
        )


def print_balance_and_remaining(client, cost: Optional[float], *, show: bool) -> None:
    if not show:
        return
    try:
        info = billing.fetch_balance(client)
    except VeniceAPIError:
        info = None
    if not info or info.get("total") is None:
        return
    print(f"Balance:        {billing.format_balance_breakdown(info)}", file=sys.stderr)
    if cost is not None:
        try:
            remaining = float(info["total"]) - float(cost)
            print(f"After charge:   {billing.format_usd(remaining)}", file=sys.stderr)
        except (TypeError, ValueError):
            pass


def over_budget(cost: Optional[float], max_spend: Optional[float]) -> bool:
    if max_spend is None or cost is None:
        return False
    try:
        return float(cost) > float(max_spend)
    except (TypeError, ValueError):
        return False


def confirm_or_exit(yes: bool) -> Optional[int]:
    if yes:
        return None
    if not sys.stdin.isatty():
        print("non-interactive; pass --yes to confirm the charge.", file=sys.stderr)
        return 1
    try:
        ans = input("Proceed? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("aborted by user", file=sys.stderr)
        return 1
    return None


def status_to_exit(err: VeniceAPIError) -> int:
    s = err.status
    if s == 400:
        return 2
    if s == 402:
        return 1  # insufficient balance ~ declined
    if s == 422:
        return 3
    if s == 429:
        return 4
    if s == 503:
        return 5
    if 500 <= s < 600:
        return 5
    if s == 404:
        return 6
    if s == 0:
        return 8
    return 2


def write_bytes_outputs(blobs: List[bytes], paths: List[Path]) -> Optional[int]:
    """Write byte blobs to paths. Returns an exit code on failure, else None."""
    for data, path in zip(blobs, paths):
        try:
            path.write_bytes(data)
        except OSError as e:
            print(f"could not write {path}: {e}", file=sys.stderr)
            return 9
        abs_path = path.resolve()
        print(str(abs_path))
        print(f"wrote {len(data)} bytes to {abs_path}", file=sys.stderr)
    return None


def post_binary_op(client, endpoint: str, body: dict, out_path: Path, label: str) -> int:
    """POST a JSON body to an endpoint that returns raw image bytes and write
    them to `out_path`.

    Returns an exit code (0 on success). API errors map through
    `status_to_exit`; a JSON (non-image) 200 is treated as an unexpected
    server response. `label` prefixes error messages (e.g. "upscale").
    """
    try:
        _ctype, payload = client.post_for_bytes_or_json(endpoint, body)
    except VeniceAPIError as e:
        print(f"{label} failed: {e}", file=sys.stderr)
        return status_to_exit(e)
    if isinstance(payload, (bytes, bytearray)):
        return write_bytes_outputs([bytes(payload)], [out_path]) or 0
    print(f"{label}: unexpected non-image response: {payload!r}", file=sys.stderr)
    return 5
