"""Local audio mastering via `ffmpeg`/`ffprobe`. Shells out; never touches the API.

Produces a 48k/24-bit WAV master with 2-pass `loudnorm` (LUFS + true-peak) and
an optional seamless loop (crossfade tail->head). ffmpeg/ffprobe are external
dependencies, detected at call time; missing tools fail cleanly before any work.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_CODECS = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}

# Stand-in measured values for --dry-run display (real ones come from pass 1).
_PLACEHOLDER_MEASURED = {
    "input_i": "<I>", "input_tp": "<TP>", "input_lra": "<LRA>",
    "input_thresh": "<thresh>", "target_offset": "<offset>",
}


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def default_output(input_path: Path) -> Path:
    """`foo.mp3` -> `foo.mastered.wav` next to the input."""
    return input_path.with_name(input_path.stem + ".mastered.wav")


def add_master_flags(parser, *, include_toggle: bool) -> None:
    """Shared mastering flags for `master`, `music`, and `sfx`.

    `include_toggle` adds the `--master` on/off switch (music/sfx only; the
    standalone `master` command always masters)."""
    if include_toggle:
        parser.add_argument(
            "--master",
            action="store_true",
            help="After saving, master to WAV (48k/24-bit, LUFS/true-peak; needs ffmpeg).",
        )
    parser.add_argument("--lufs", type=float, default=-16.0, metavar="LUFS",
                        help="Integrated loudness target (default -16).")
    parser.add_argument("--true-peak", type=float, default=-1.0, dest="true_peak",
                        metavar="DBTP", help="True-peak ceiling in dBTP (default -1.0).")
    parser.add_argument("--sample-rate", type=int, default=48000, dest="sample_rate",
                        metavar="HZ", help="Output sample rate (default 48000).")
    parser.add_argument("--bit-depth", type=int, choices=(16, 24, 32), default=24,
                        dest="bit_depth", help="Output PCM bit depth (default 24).")
    parser.add_argument("--loop", action="store_true",
                        help="Make it seamlessly loopable (crossfade tail into head).")
    parser.add_argument("--loop-crossfade", type=float, default=2.0, dest="loop_crossfade",
                        metavar="SEC", help="Loop crossfade length in seconds (default 2).")


def master_kwargs(args) -> dict:
    """Pull the mastering knobs off a parsed namespace into master() kwargs."""
    return dict(
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        lufs=args.lufs,
        true_peak=args.true_peak,
        loop=args.loop,
        loop_crossfade=args.loop_crossfade,
    )


def master_hook(args):
    """Post-save callback (path -> exit code) for the music/sfx `--master` flag:
    masters the just-written file to `<file>.mastered.wav` in place."""
    def _run(path: Path) -> int:
        return master(path, default_output(path), **master_kwargs(args))
    return _run


def _n(v: float) -> str:
    return f"{v:g}"


def _loudnorm(lufs: float, tp: float, measured: Optional[dict]) -> str:
    """Build a loudnorm filter string. Pass 1 (measured=None) prints JSON stats;
    pass 2 feeds the measured values back for an accurate linear normalization."""
    base = f"loudnorm=I={_n(lufs)}:TP={_n(tp)}:LRA=11"
    if measured is None:
        return base + ":print_format=json"
    return (
        base
        + f":measured_I={measured['input_i']}"
        + f":measured_TP={measured['input_tp']}"
        + f":measured_LRA={measured['input_lra']}"
        + f":measured_thresh={measured['input_thresh']}"
        + f":offset={measured['target_offset']}"
        + ":linear=true:print_format=summary"
    )


def _loop_filter(src: str, dur: Optional[float], cf: float) -> str:
    """Filtergraph (from label `src` to `[out]`) that folds the last `cf`s of the
    stream, faded out, over the first `cf`s, faded in -- a click-free loop of
    length `dur - cf`. `dur=None` emits placeholders for dry-run display."""
    c = _n(cf)
    if dur is None:
        d, tail, body = "<DUR>", "<DUR-CF>", "<DUR-CF>"
    else:
        d, tail, body = _n(dur), _n(dur - cf), _n(dur - cf)
    return (
        f"[{src}]asplit=3[la][lb][lc];"
        f"[la]atrim=0:{c},asetpts=N/SR/TB,afade=t=in:st=0:d={c}[head];"
        f"[lb]atrim={tail}:{d},asetpts=N/SR/TB,afade=t=out:st=0:d={c}[tail];"
        f"[head][tail]amix=inputs=2:normalize=0[seam];"
        f"[lc]atrim={c}:{body},asetpts=N/SR/TB[mid];"
        f"[seam][mid]concat=n=2:v=0:a=1[out]"
    )


def _pass1_cmd(input_path: Path, lufs: float, tp: float) -> List[str]:
    return [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
        "-af", _loudnorm(lufs, tp, None), "-f", "null", "-",
    ]


def _pass2_cmd(input_path: Path, output_path: Path, *, lufs: float, tp: float,
               measured: Optional[dict], codec: str, sample_rate: int,
               loop: bool, dur: Optional[float], cf: float) -> List[str]:
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(input_path)]
    norm = _loudnorm(lufs, tp, measured)
    if loop:
        graph = f"[0:a]{norm}[m];" + _loop_filter("m", dur, cf)
        cmd += ["-filter_complex", graph, "-map", "[out]"]
    else:
        cmd += ["-af", norm]
    cmd += ["-ar", str(sample_rate), "-c:a", codec, str(output_path)]
    return cmd


def _parse_loudnorm_json(stderr: str) -> dict:
    """Extract the trailing JSON block ffmpeg's loudnorm prints on stderr."""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no loudnorm JSON found in ffmpeg output")
    return json.loads(stderr[start:end + 1])


def _run(cmd: List[str]):
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        print(f"failed to run {cmd[0]}: {e}", file=sys.stderr)
        return None


def _probe_duration(input_path: Path) -> Optional[float]:
    cp = _run_capture([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_path),
    ])
    if cp is None or cp.returncode != 0:
        return None
    try:
        return float((cp.stdout or "").strip())
    except (TypeError, ValueError):
        return None


def _run_capture(cmd: List[str]):
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        print(f"failed to run {cmd[0]}: {e}", file=sys.stderr)
        return None


def master(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 48000,
    bit_depth: int = 24,
    lufs: float = -16.0,
    true_peak: float = -1.0,
    loop: bool = False,
    loop_crossfade: float = 2.0,
    dry_run: bool = False,
) -> int:
    """Master `input_path` to a WAV at `output_path`. Returns an exit code
    (0 ok, 2 bad arg / missing ffmpeg, 5 ffmpeg failure)."""
    codec = _CODECS.get(bit_depth)
    if codec is None:
        print(f"master: unsupported --bit-depth {bit_depth}", file=sys.stderr)
        return 2

    if not dry_run and not has_ffmpeg():
        print("master: ffmpeg not found on PATH; install it (e.g. apt install ffmpeg)",
              file=sys.stderr)
        return 2

    dur: Optional[float] = None
    if loop and not dry_run:
        if not has_ffprobe():
            print("master: --loop needs ffprobe (ships with ffmpeg)", file=sys.stderr)
            return 2
        dur = _probe_duration(input_path)
        if dur is None:
            print(f"master: could not read duration of {input_path}", file=sys.stderr)
            return 5
        if dur <= 2 * loop_crossfade:
            print(
                f"master: input is {dur:.2f}s; too short to loop with a "
                f"{loop_crossfade:g}s crossfade (need > {2 * loop_crossfade:g}s)",
                file=sys.stderr,
            )
            return 2

    cmd1 = _pass1_cmd(input_path, lufs, true_peak)

    if dry_run:
        cmd2_template = _pass2_cmd(
            input_path, output_path, lufs=lufs, tp=true_peak,
            measured=_PLACEHOLDER_MEASURED, codec=codec, sample_rate=sample_rate,
            loop=loop, dur=dur, cf=loop_crossfade,
        )
        print("master: dry run -- would run:", file=sys.stderr)
        print("  pass 1 (measure): " + " ".join(cmd1), file=sys.stderr)
        print("  pass 2 (apply):   " + " ".join(cmd2_template), file=sys.stderr)
        print("  (pass 2's measured_* / <DUR> values are resolved at run time)", file=sys.stderr)
        return 0

    r1 = _run_capture(cmd1)
    if r1 is None or r1.returncode != 0:
        print("master: loudness analysis (pass 1) failed", file=sys.stderr)
        if r1 is not None and r1.stderr:
            print(r1.stderr.strip()[-500:], file=sys.stderr)
        return 5
    try:
        measured = _parse_loudnorm_json(r1.stderr or "")
    except (ValueError, json.JSONDecodeError):
        print("master: could not parse loudnorm stats from ffmpeg", file=sys.stderr)
        return 5

    cmd2 = _pass2_cmd(
        input_path, output_path, lufs=lufs, tp=true_peak, measured=measured,
        codec=codec, sample_rate=sample_rate, loop=loop, dur=dur, cf=loop_crossfade,
    )
    r2 = _run(cmd2)
    if r2 is None or r2.returncode != 0:
        print("master: encode (pass 2) failed", file=sys.stderr)
        if r2 is not None and r2.stderr:
            print(r2.stderr.strip()[-500:], file=sys.stderr)
        return 5

    print(str(output_path.resolve()))
    print(f"mastered -> {output_path.resolve()}", file=sys.stderr)
    return 0
