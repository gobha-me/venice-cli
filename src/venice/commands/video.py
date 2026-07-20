"""`venice video` -- generate a video via Venice's async queue.

Same quote -> queue -> poll -> retrieve -> complete flow as `sfx`/`music`
(shared engine in `_queue`), but against the `/video` endpoints. Two wrinkles
versus audio: `duration` is a string enum ("5s", "Auto", ...), and VPS-backed
models return a `download_url` at queue time -- for those, `/video/retrieve`
reports JSON status only and the mp4 is fetched from that presigned URL once
COMPLETED. Non-VPS models stream `video/mp4` straight from `/video/retrieve`.

A free `/models?type=video` catalog GET (via the lean client) validates
`--model` and resolves a default before the paid quote -- mirrors `venice chat`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from .. import billing, config, userconfig
from ..client import VeniceAPIError
from . import _models, _queue, _shared

VIDEO_EXT_BY_CTYPE = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}

# Media inputs (issue #18): a value is passed through when it is already an
# http(s)/data URL, otherwise it is treated as a local file, size-checked, and
# encoded to a `data:` URL via `_shared.encode_data_url`. Size caps and array
# caps below mirror the `/video/queue` (`QueueVideoRequest`) schema.
MEDIA_LIMITS = {
    "image": 25 * 1024 * 1024,  # image inputs < 25 MB
    "video": 50 * 1024 * 1024,  # reference/input video <= 50 MB per clip
    "audio": 15 * 1024 * 1024,  # audio inputs <= 15 MB
}
DEFAULT_MIME = {"image": "image/png", "video": "video/mp4", "audio": "audio/mpeg"}
REF_IMAGE_MAX = 9
REF_VIDEO_MAX = 3
REF_AUDIO_MAX = 3
SCENE_MAX = 4
ELEMENTS_MAX = 4
ELEMENT_REF_IMAGE_MAX = 3

DEFAULT_VIDEO_DURATION = "5s"
DURATION_CHOICES = (
    "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "10s", "11s", "12s", "13s",
    "14s", "15s", "16s", "18s", "20s", "25s", "30s", "1 gen", "Auto",
)
RESOLUTION_CHOICES = (
    "256p", "360p", "480p", "540p", "580p", "720p", "1080p", "1440p", "2160p",
    "4k", "2x", "4x", "true_1080p",
)
ASPECT_CHOICES = ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9")


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "video",
        help="Generate a video via Venice's async queue.",
        description=(
            "Generates a video from a text prompt. Async flow: quote -> queue "
            "-> poll -> save (mp4). Use --dry-run to see only the cost quote. "
            "To fetch a backgrounded job by its queue_id, use `venice "
            "video-status`. Available durations/resolutions vary by model."
        ),
    )
    p.add_argument("prompt", nargs="?", help="Video description.")
    p.add_argument(
        "--model",
        default=None,
        help="Video model id (default: the catalog's 'default'-trait model).",
    )
    p.add_argument("--duration", choices=DURATION_CHOICES, default=DEFAULT_VIDEO_DURATION)
    p.add_argument("--resolution", choices=RESOLUTION_CHOICES, default=None)
    p.add_argument(
        "--aspect-ratio", choices=ASPECT_CHOICES, default=None, dest="aspect_ratio"
    )
    p.add_argument("--negative-prompt", default=None, dest="negative_prompt")
    p.add_argument(
        "--no-audio",
        action="store_true",
        dest="no_audio",
        help="Disable audio (models that support it generate audio by default).",
    )
    # Media inputs (#18). Each accepts a local file path OR an http(s)/data URL;
    # local files are encoded to a `data:` URL. See _collect_media below.
    p.add_argument(
        "--image", default=None, metavar="PATH|URL",
        help="Start-frame image for image-to-video (local file or URL).",
    )
    p.add_argument(
        "--end-image", default=None, dest="end_image", metavar="PATH|URL",
        help="End-frame image for models that support transitions.",
    )
    p.add_argument(
        "--video", default=None, metavar="PATH|URL",
        help="Input video for video-to-video / upscale models.",
    )
    p.add_argument(
        "--audio", default=None, dest="audio_input", metavar="PATH|URL",
        help="Background-music audio input, e.g. WAV/MP3 (distinct from --no-audio).",
    )
    p.add_argument(
        "--reference-image", action="append", default=None, dest="reference_image",
        metavar="PATH|URL",
        help=f"Reference image for character/style consistency (repeatable, up to {REF_IMAGE_MAX}).",
    )
    p.add_argument(
        "--reference-video", action="append", default=None, dest="reference_video",
        metavar="PATH|URL",
        help=f"Reference video (repeatable, up to {REF_VIDEO_MAX}).",
    )
    p.add_argument(
        "--reference-audio", action="append", default=None, dest="reference_audio",
        metavar="PATH|URL",
        help=f"Reference audio (repeatable, up to {REF_AUDIO_MAX}); must accompany a "
             "reference image or video.",
    )
    p.add_argument(
        "--scene-image", action="append", default=None, dest="scene_image",
        metavar="PATH|URL",
        help=f"Scene reference image (@Image1.., repeatable, up to {SCENE_MAX}).",
    )
    p.add_argument(
        "--reference-video-duration", type=float, default=None,
        dest="reference_video_duration", metavar="SECONDS",
        help="Aggregate reference-video duration (s); sent to the quote for R2V pricing.",
    )
    p.add_argument(
        "--element", action="append", default=None, metavar="JSON",
        help="Advanced @Element as a JSON object (frontal_image_url / "
             f"reference_image_urls / video_url), repeatable up to {ELEMENTS_MAX}. "
             "Local paths inside are encoded like the other media flags.",
    )
    p.add_argument("--output", "-o", type=Path, default=None)
    p.add_argument("--yes", "-y", action="store_true", default=None)
    p.add_argument("--background", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument(
        "--max-spend",
        type=float,
        default=None,
        metavar="USD",
        help="Refuse to queue if the quote exceeds this USD cap.",
    )
    p.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip the upfront balance display.",
    )
    p.add_argument("--poll-interval", type=float, default=config.VIDEO_POLL_INTERVAL_SEC)
    p.add_argument("--max-wait", type=float, default=config.VIDEO_POLL_MAX_WAIT_SEC)
    p.set_defaults(handler=_run_generate)


def register_status(subparsers) -> None:
    sp = subparsers.add_parser(
        "video-status",
        help="Fetch a previously-backgrounded video job by queue_id.",
        description=(
            "Polls /video/retrieve for an already-queued job (typically from "
            "`venice video ... --background`) and downloads the mp4 when ready. "
            "For VPS-backed models, pass the --download-url printed at queue time."
        ),
    )
    sp.add_argument("queue_id")
    sp.add_argument(
        "--model",
        default=None,
        help="Model used for the job (default: catalog 'default'-trait model).",
    )
    sp.add_argument(
        "--download-url",
        default=None,
        dest="download_url",
        help="Presigned URL from the original queue (VPS-backed models only).",
    )
    sp.add_argument("--output", "-o", type=Path, default=None)
    sp.add_argument("--no-cleanup", action="store_true")
    sp.add_argument("--poll-interval", type=float, default=config.VIDEO_POLL_INTERVAL_SEC)
    sp.add_argument("--max-wait", type=float, default=config.VIDEO_POLL_MAX_WAIT_SEC)
    sp.set_defaults(handler=_run_status)


def _shared_params(args) -> dict:
    """Optional params accepted by both /video/quote and /video/queue."""
    extra: dict = {}
    if args.resolution:
        extra["resolution"] = args.resolution
    if args.aspect_ratio:
        extra["aspect_ratio"] = args.aspect_ratio
    if args.no_audio:
        extra["audio"] = False
    return extra


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def _resolve_media(value: str, *, kind: str):
    """Resolve one media input to a URL string. A value that is already an
    http(s)/data URL passes through; anything else is a local file that is
    size-checked and encoded to a `data:` URL. Returns (url_or_None, rc_or_None).
    """
    if _is_url(value):
        return value, None
    path = Path(value)
    if not path.is_file():
        print(f"video: input file not found: {value}", file=sys.stderr)
        return None, 2
    size = path.stat().st_size
    if size == 0:
        print(f"video: input {value} is empty", file=sys.stderr)
        return None, 2
    limit = MEDIA_LIMITS[kind]
    if size > limit:
        print(
            f"video: input {value} is {size} bytes; {kind} inputs must be "
            f"< {limit // (1024 * 1024)} MB",
            file=sys.stderr,
        )
        return None, 2
    return _shared.encode_data_url(path, default_mime=DEFAULT_MIME[kind]), None


def _resolve_media_list(values, *, kind: str, cap: int, flag: str):
    """Resolve a repeatable media flag. Returns (list_or_None, rc_or_None);
    (None, None) when the flag was not given."""
    if values is None:
        return None, None
    if len(values) > cap:
        print(f"video: at most {cap} {flag}", file=sys.stderr)
        return None, 2
    out = []
    for value in values:
        url, rc = _resolve_media(value, kind=kind)
        if rc is not None:
            return None, rc
        out.append(url)
    return out, None


def _resolve_elements(raw_list):
    """Parse `--element` JSON objects, resolving any media paths inside. Returns
    (list_or_None, rc_or_None). Unknown keys pass through verbatim so the JSON
    escape hatch stays forward-compatible with the schema."""
    if raw_list is None:
        return None, None
    if len(raw_list) > ELEMENTS_MAX:
        print(f"video: at most {ELEMENTS_MAX} --element", file=sys.stderr)
        return None, 2
    out = []
    for raw in raw_list:
        try:
            el = json.loads(raw)
        except (ValueError, TypeError) as e:
            print(f"video: --element is not valid JSON: {e}", file=sys.stderr)
            return None, 2
        if not isinstance(el, dict):
            print("video: --element must be a JSON object", file=sys.stderr)
            return None, 2
        resolved = dict(el)  # keep any unknown keys as-is
        for key, kind in (("frontal_image_url", "image"), ("video_url", "video")):
            if el.get(key) is not None:
                url, rc = _resolve_media(el[key], kind=kind)
                if rc is not None:
                    return None, rc
                resolved[key] = url
        refs = el.get("reference_image_urls")
        if refs is not None:
            if not isinstance(refs, list):
                print("video: element reference_image_urls must be a list", file=sys.stderr)
                return None, 2
            urls, rc = _resolve_media_list(
                refs, kind="image", cap=ELEMENT_REF_IMAGE_MAX,
                flag="element reference_image_urls",
            )
            if rc is not None:
                return None, rc
            resolved["reference_image_urls"] = urls
        out.append(resolved)
    return out, None


def _collect_media(args):
    """Resolve every media input up front (before any spend). Returns
    (quote_media, queue_media, rc). `quote_media` holds fields valid on
    /video/quote (video_url + reference_video_total_duration); `queue_media`
    holds the /video/queue-only fields. On any failure returns (None, None, rc).
    """
    quote_media: dict = {}
    queue_media: dict = {}

    # Single-value inputs: (arg attr, kind, queue field, also-in-quote?)
    for attr, kind, field, in_quote in (
        ("image", "image", "image_url", False),
        ("end_image", "image", "end_image_url", False),
        ("audio_input", "audio", "audio_url", False),
        ("video", "video", "video_url", True),  # valid on both endpoints
    ):
        value = getattr(args, attr, None)
        if value is None:
            continue
        url, rc = _resolve_media(value, kind=kind)
        if rc is not None:
            return None, None, rc
        queue_media[field] = url
        if in_quote:
            quote_media[field] = url

    # Repeatable array inputs (queue-only): (arg attr, kind, cap, flag, field)
    for attr, kind, cap, flag, field in (
        ("reference_image", "image", REF_IMAGE_MAX, "--reference-image", "reference_image_urls"),
        ("reference_video", "video", REF_VIDEO_MAX, "--reference-video", "reference_video_urls"),
        ("reference_audio", "audio", REF_AUDIO_MAX, "--reference-audio", "reference_audio_urls"),
        ("scene_image", "image", SCENE_MAX, "--scene-image", "scene_image_urls"),
    ):
        urls, rc = _resolve_media_list(getattr(args, attr, None), kind=kind, cap=cap, flag=flag)
        if rc is not None:
            return None, None, rc
        if urls is not None:
            queue_media[field] = urls

    elements, rc = _resolve_elements(getattr(args, "element", None))
    if rc is not None:
        return None, None, rc
    if elements is not None:
        queue_media["elements"] = elements

    # Reference-video total duration -> quote only (R2V pricing tier).
    dur = getattr(args, "reference_video_duration", None)
    if dur is not None:
        quote_media["reference_video_total_duration"] = dur

    return quote_media, queue_media, None


def _run_generate(args) -> int:
    userconfig.apply_defaults(args, "video")
    if not args.prompt:
        print("video: prompt required (or use: venice video-status <id>)", file=sys.stderr)
        return 2

    client, rc = _queue.build_client()
    if rc != 0:
        return rc

    models = _models.catalog(client, "video")
    model, rc = _models.resolve_model(
        args.model, models, label="video", noun="video model"
    )
    if rc is not None:
        return rc

    quote_media, queue_media, rc = _collect_media(args)
    if rc is not None:
        return rc

    extra = _shared_params(args)
    quote_body = {"model": model, "duration": args.duration}
    quote_body.update(extra)
    quote_body.update(quote_media)
    try:
        quote = client.post_json("/video/quote", quote_body)
    except VeniceAPIError as e:
        print(f"quote rejected: {e}", file=sys.stderr)
        return _queue.status_to_exit(e)

    quote_value = quote.get("quote", quote)
    label = f"model={model}, duration={args.duration}"
    if args.resolution:
        label += f", {args.resolution}"
    _shared.print_estimate(quote_value, label)
    _shared.print_balance_and_remaining(client, quote_value, show=not args.no_balance)

    if _shared.over_budget(quote_value, args.max_spend):
        print(
            f"video: quote {billing.format_usd(quote_value)} exceeds "
            f"--max-spend {billing.format_usd(args.max_spend)}; aborting",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        return 0

    if not args.background:
        rc = _shared.confirm_or_exit(args.yes)
        if rc is not None:
            return rc

    queue_body = {"model": model, "prompt": args.prompt, "duration": args.duration}
    queue_body.update(extra)
    if args.negative_prompt:
        queue_body["negative_prompt"] = args.negative_prompt
    queue_body.update(queue_media)

    try:
        queued = client.post_json("/video/queue", queue_body)
    except VeniceAPIError as e:
        print(f"queue failed: {e}", file=sys.stderr)
        return _queue.status_to_exit(e)

    queue_id = queued.get("queue_id") or queued.get("id") or ""
    if not queue_id:
        print(f"queue response missing queue_id: {queued!r}", file=sys.stderr)
        return 5
    download_url = queued.get("download_url") or None

    if args.background:
        sys.stdout.write(queue_id + "\n")
        sys.stdout.flush()
        msg = f"queued as {queue_id}; fetch with: venice video-status {queue_id} --model {model}"
        if download_url:
            msg += f" --download-url {download_url}"
        print(msg, file=sys.stderr)
        return 0

    return _retrieve_and_save(
        client, model, queue_id, download_url,
        args.output, args.poll_interval, args.max_wait, args.no_cleanup,
    )


def _run_status(args) -> int:
    client, rc = _queue.build_client()
    if rc != 0:
        return rc
    model = args.model
    if not model:
        models = _models.catalog(client, "video")
        model, rc = _models.resolve_model(
            args.model, models, label="video", noun="video model"
        )
        if rc is not None:
            return rc
    return _retrieve_and_save(
        client, model, args.queue_id, args.download_url,
        args.output, args.poll_interval, args.max_wait, args.no_cleanup,
    )


class NoVideoStream(Exception):
    """A job reported COMPLETED but returned no video bytes and no `download_url`
    is known to fetch them from."""


def retrieve_bytes(
    client, model, queue_id, *,
    poll_interval, max_wait, download_url=None, on_tick=None,
) -> Tuple[str, bytes]:
    """Poll /video/retrieve until the video is ready and return (ctype, bytes).

    Print-free core of `_retrieve_and_save`: the bare poll plus the VPS-model
    download-url fallback, with no stderr, file I/O, or cleanup -- so callers
    that own stdout (e.g. the MCP stdio server) can reuse it without corrupting
    their transport. Non-VPS models stream the mp4 straight from /video/retrieve;
    VPS-backed models report COMPLETED (JSON) and the mp4 is fetched from the
    presigned `download_url`.

    Raises VeniceAPIError on a terminal API error, TimeoutError on `max_wait`,
    and NoVideoStream when a COMPLETED job yields no bytes and no download_url.
    """
    ctype, payload = client.poll_retrieve(
        "/video/retrieve",
        {"model": model, "queue_id": queue_id},
        interval=poll_interval,
        max_wait=max_wait,
        on_tick=on_tick,
        terminal_statuses=("COMPLETED",),
    )
    if isinstance(payload, (bytes, bytearray)):
        return ctype, bytes(payload)
    # VPS-backed model: retrieve reported COMPLETED (JSON) -- fetch the mp4.
    if not download_url:
        raise NoVideoStream(queue_id)
    return client.get_url_bytes(download_url)


def _retrieve_and_save(
    client, model, queue_id, download_url, out_arg,
    poll_interval, max_wait, no_cleanup,
) -> int:
    start = time.monotonic()
    try:
        ctype, data = retrieve_bytes(
            client, model, queue_id,
            poll_interval=poll_interval, max_wait=max_wait,
            download_url=download_url, on_tick=_queue.progress_tick(start),
        )
    except VeniceAPIError as e:
        sys.stderr.write("\n")
        print(f"retrieve failed: {e}", file=sys.stderr)
        return _queue.status_to_exit(e)
    except TimeoutError as e:
        sys.stderr.write("\n")
        print(
            f"{e}; check later with: venice video-status {queue_id} --model {model}",
            file=sys.stderr,
        )
        return 7
    except NoVideoStream:
        sys.stderr.write("\n")
        print(
            "video: job completed but returned no video stream and no "
            "download_url is known; re-run `venice video-status "
            f"{queue_id} --model {model} --download-url <url>` with the URL "
            "printed when the job was queued.",
            file=sys.stderr,
        )
        return 2
    sys.stderr.write("\n")

    ext, unknown = _queue.ext_for(ctype, VIDEO_EXT_BY_CTYPE, default=".mp4")
    if unknown:
        print(f"warning: unexpected content-type {ctype!r}; saving as .mp4", file=sys.stderr)
    out_path = _queue.resolve_output_path(out_arg, queue_id, ext, prefix="venice-video")

    try:
        out_path.write_bytes(data)
    except OSError as e:
        print(f"could not write {out_path}: {e}", file=sys.stderr)
        return 9

    abs_path = out_path.resolve()
    print(str(abs_path))
    print(f"wrote {len(data)} bytes to {abs_path}", file=sys.stderr)

    if not no_cleanup:
        try:
            client.post_json("/video/complete", {"model": model, "queue_id": queue_id})
        except VeniceAPIError as e:
            print(f"warning: cleanup call failed: {e}", file=sys.stderr)
    return 0
