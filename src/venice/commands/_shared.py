"""Shared UX/budget rails for spend-incurring commands.

Extracted from `image` so it, `upscale`, and `bg-remove` share one copy of the
estimate/confirm/status/write plumbing rather than each carrying its own. These
helpers take primitive args (cost, max_spend, err, byte blobs) so they stay
independent of any one command's argument shape.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import stat
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

from .. import billing
from ..client import VeniceAPIError
from . import _index


class MediaPathError(ValueError):
    """A model-supplied local-media path failed the session's read authority."""


_RootSource = Union[str, Path, Callable[[], Union[str, Path]]]
_RootsSource = Union[Iterable[Union[str, Path]], Callable[[], Iterable[Union[str, Path]]]]
_MEDIA_SNIFF_BYTES = 64


def _root_value(source: _RootSource) -> str:
    value = source() if callable(source) else source
    return os.path.realpath(os.fspath(value))


def _roots_value(source: _RootsSource) -> List[str]:
    values = source() if callable(source) else source
    return [os.path.realpath(os.fspath(value)) for value in values]


def _sniff_media_mime(prefix: bytes, kind: str) -> Optional[str]:
    """Return a conservative MIME type from a bounded media-file prefix."""
    if kind == "image":
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if prefix.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if prefix.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
            return "image/webp"
        return None

    if kind == "video":
        if prefix.startswith(b"\x1aE\xdf\xa3"):
            return "video/webm"
        if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
            return "video/quicktime" if prefix[8:12] == b"qt  " else "video/mp4"
        return None

    if kind == "audio":
        if prefix.startswith(b"fLaC"):
            return "audio/flac"
        if prefix.startswith(b"OggS"):
            return "audio/ogg"
        if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WAVE":
            return "audio/wav"
        if prefix.startswith(b"ID3"):
            return "audio/mpeg"
        if len(prefix) >= 2 and prefix[0] == 0xFF:
            if prefix[1] & 0xF6 == 0xF0:
                return "audio/aac"
            if prefix[1] & 0xE0 == 0xE0 and prefix[1] & 0x06:
                return "audio/mpeg"
        if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
            return "audio/mp4"
        return None

    raise ValueError(f"unknown media kind: {kind}")


class MediaPathAuthority:
    """Resolve model-supplied media paths inside operator-authorized roots.

    ``base`` controls relative-path resolution and ``roots`` controls readable
    containment. Either may be a callable so ``venice code`` can bind this guard to
    its mutable active/attached-root set without rebuilding every tool closure.
    """

    def __init__(self, base: _RootSource, roots: Optional[_RootsSource] = None):
        self._base = base
        self._roots = roots if roots is not None else lambda: [_root_value(self._base)]

    def resolve(
        self, value: Union[str, Path], *, kind: str, max_bytes: int
    ) -> Tuple[Path, str]:
        raw = os.fspath(value)
        if not raw.strip():
            raise MediaPathError("local media path is required")

        base = _root_value(self._base)
        roots = _roots_value(self._roots)
        if base not in roots:
            roots.append(base)
        joined = raw if os.path.isabs(raw) else os.path.join(base, raw)
        real = os.path.realpath(os.path.normpath(joined))
        containers = [root for root in roots if _index.resolves_inside(Path(real), Path(root))]
        if not containers:
            raise MediaPathError("local media path escapes the authorized project roots")
        container = max(containers, key=len)
        rel = Path(os.path.relpath(real, container)).as_posix()
        if _index.is_secret_path(rel) or _index.is_protected_dir_path(rel):
            raise MediaPathError("local media path is in a protected location")

        try:
            info = os.stat(real)
        except OSError:
            raise MediaPathError("local media file does not exist") from None
        if not stat.S_ISREG(info.st_mode):
            raise MediaPathError("local media path must name a regular file")
        if info.st_size == 0:
            raise MediaPathError("local media file is empty")
        if info.st_size > max_bytes:
            raise MediaPathError(
                f"local {kind} file is {info.st_size} bytes; limit is {max_bytes} bytes"
            )

        try:
            with open(real, "rb") as handle:
                prefix = handle.read(_MEDIA_SNIFF_BYTES)
        except OSError:
            raise MediaPathError("local media file could not be read") from None
        mime = _sniff_media_mime(prefix, kind)
        if mime is None:
            raise MediaPathError(f"local file is not a recognized {kind} container")
        return Path(real), mime


def resolve_output(arg_output: Optional[Path], default_name: str) -> Path:
    """Pick an output path: an explicit file, a file inside an explicit dir, or
    `default_name` in the cwd."""
    if arg_output is None:
        return Path.cwd() / default_name
    if arg_output.is_dir():
        return arg_output / default_name
    return arg_output


def add_balance_flag(parser) -> None:
    """Register the tri-stated `--no-balance` / `--balance` pair (#57 Class B).

    `default=None` (not False) is what lets `defaults.no_balance` reach the dest:
    `userconfig.apply_defaults` only fills a dest that is still None. Consumers
    read `show=not args.no_balance`, and `not None == not False`, so no
    consumption site needs a None branch.

    The positive spelling is `--show-balance`: a bare `--balance` would make the
    abbreviation `--ba` ambiguous against `--background` on sfx/music/video.
    """
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--no-balance",
        dest="no_balance",
        action="store_true",
        default=None,
        help="Skip the upfront balance display. Config-backable via "
        "defaults.no_balance; an explicit --show-balance/--no-balance still wins.",
    )
    grp.add_argument(
        "--show-balance",
        dest="no_balance",
        action="store_false",
        default=None,
        help="Force the upfront balance display on (beats defaults.no_balance). "
        "Spelled --show-balance, not --balance, so the abbreviation --ba keeps "
        "resolving to --background on the commands that have it.",
    )


def add_cleanup_flag(parser, *, section: str, endpoint: str) -> None:
    """Register the tri-stated `--no-cleanup` / `--cleanup` pair (#57 Class B).

    Same shape and rationale as :func:`add_balance_flag`; `section` names the
    config table (sfx/music/video) and `endpoint` the completion call skipped
    when set. Registered on the generate AND `-status` parsers of each command.
    """
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--no-cleanup",
        dest="no_cleanup",
        action="store_true",
        default=None,
        help=f"Keep the server-side job after download (skip {endpoint}). "
        f"Config-backable via defaults.{section}.no_cleanup; an explicit "
        "--no-cleanup/--cleanup still wins.",
    )
    grp.add_argument(
        "--cleanup",
        dest="no_cleanup",
        action="store_false",
        default=None,
        help=f"Force the completion call on (beats defaults.{section}.no_cleanup).",
    )


def add_poll_flags(parser, *, section: str, interval: float, max_wait: float) -> None:
    """Register the tri-stated `--poll-interval` / `--max-wait` pair (#57 C2).

    Same shape and rationale as :func:`add_cleanup_flag`; `section` names the
    config table (sfx/music/video). Registered on the generate AND `-status`
    parsers of each command, so a cadence preference applies whether you wait
    inline or reattach later. `default=None` is what lets
    `defaults.<section>.{poll_interval,max_wait}` reach the dests; the literals
    go back on in the handler's :func:`resolve_poll` call.

    The two are asymmetric on the tool path and that is deliberate: `max_wait`
    is a parameter of `_mcp.{sfx,music,video}_tool`, so a config row reaches the
    agent/MCP surface through `config_defaults_for`, while the cadence itself is
    fixed by those impls and stays CLI-only (the `no_cleanup` precedent).
    """
    parser.add_argument(
        "--poll-interval", type=float, default=None, metavar="SEC",
        help=f"Seconds between status polls (default {interval:g}). "
             f"Config-backable via defaults.{section}.poll_interval (CLI only -- "
             "the agent/MCP tools fix their own cadence).",
    )
    parser.add_argument(
        "--max-wait", type=float, default=None, metavar="SEC",
        help=f"Give up waiting after this many seconds (default {max_wait:g}). "
             f"Config-backable via defaults.{section}.max_wait, which the "
             f"venice_{section} agent/MCP tool honors too.",
    )


def resolve_poll(poll_interval, max_wait, *, label: str,
                 interval: float, max_wait_default: float):
    """Resolve the poll cadence to `(poll_interval, max_wait)` (#57 C2).

    Call AFTER `userconfig.apply_defaults`. Every one of the six poll handlers
    needs this: a leftover None is a TypeError inside the poll loop -- immediate
    for `max_wait` (`time.monotonic() + None`), but for `poll_interval` only on
    the SECOND iteration (`time.sleep(None)`), so a job that finishes fast hides
    the bug entirely and a slow one crashes after the spend.

    Takes and returns primitives rather than a namespace, per CONTRIBUTING's
    house style for `commands/_*.py`, which also keeps this layer free of a
    dependency on `userconfig`.

    The negative check here covers a value the user TYPED. A negative from
    CONFIG never reaches this -- `userconfig._non_negative` rejects it at the
    coercer, which is the only layer that also covers the agent/MCP tool path.
    `0` is left alone deliberately on both: `max_wait=0` is a single
    non-blocking probe and `poll_interval=0` a tight loop, both meaningful.
    """
    if poll_interval is None:
        poll_interval = interval
    if max_wait is None:
        max_wait = max_wait_default
    if poll_interval < 0:
        print(f"{label}: --poll-interval must be >= 0; using {interval:g}",
              file=sys.stderr)
        poll_interval = interval
    if max_wait < 0:
        print(f"{label}: --max-wait must be >= 0; using {max_wait_default:g}",
              file=sys.stderr)
        max_wait = max_wait_default
    return poll_interval, max_wait


def encode_data_url(
    path: Path, *, default_mime: str = "application/octet-stream",
    detected_mime: Optional[str] = None,
) -> str:
    """Read a local file and return a `data:<mime>;base64,<b64>` URL.

    The MIME type is sniffed from the filename; callers pass `default_mime` as
    the fallback for extensions `mimetypes` doesn't recognise. Unlike the raw
    base64 that `bg-remove`/`image-edit` send in an `image` field, the Venice
    `/video` media inputs want a full data URL, so this prepends the prefix.
    """
    mime = detected_mime or mimetypes.guess_type(str(path))[0] or default_mime
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
