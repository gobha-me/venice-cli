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

import sys
import time
from pathlib import Path
from typing import Optional

from .. import billing, config, userconfig
from ..client import VeniceAPIError
from . import _models, _queue, _shared

VIDEO_EXT_BY_CTYPE = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}

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

    extra = _shared_params(args)
    quote_body = {"model": model, "duration": args.duration}
    quote_body.update(extra)
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


def _retrieve_and_save(
    client, model, queue_id, download_url, out_arg,
    poll_interval, max_wait, no_cleanup,
) -> int:
    start = time.monotonic()
    try:
        ctype, payload = client.poll_retrieve(
            "/video/retrieve",
            {"model": model, "queue_id": queue_id},
            interval=poll_interval,
            max_wait=max_wait,
            on_tick=_queue.progress_tick(start),
            terminal_statuses=("COMPLETED",),
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
    sys.stderr.write("\n")

    if isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    else:
        # VPS-backed model: retrieve reported COMPLETED (JSON) -- fetch the mp4.
        if not download_url:
            print(
                "video: job completed but returned no video stream and no "
                "download_url is known; re-run `venice video-status "
                f"{queue_id} --model {model} --download-url <url>` with the URL "
                "printed when the job was queued.",
                file=sys.stderr,
            )
            return 2
        try:
            ctype, data = client.get_url_bytes(download_url)
        except VeniceAPIError as e:
            print(f"download failed: {e}", file=sys.stderr)
            return _queue.status_to_exit(e)

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
