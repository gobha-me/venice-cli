"""Local audio mastering via `ffmpeg`/`ffprobe`. Shells out; never touches the API.

Produces a 48k/24-bit WAV master with 2-pass `loudnorm` (LUFS + true-peak) and
an optional seamless loop (crossfade tail->head). ffmpeg/ffprobe are external
dependencies, detected at call time; missing tools fail cleanly before any work.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import userconfig

_CODECS = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}

# The `--bit-depth` choices, DERIVED from the codec table rather than repeated,
# so a depth can never be offered that `master()` has no encoder for. Public
# because `userconfig._one_of` resolves it by name. (#57 Class C2)
BIT_DEPTHS = tuple(sorted(_CODECS))

# Built-in defaults for the mastering chain. These live here, not on the parser,
# so `userconfig.apply_defaults` can reach the dests -- it fills a dest only
# while it is still None, which a hardcoded argparse default makes impossible.
# `apply_master_literals` puts them back on last. (#57 Class C2)
DEFAULT_LUFS = -16.0
DEFAULT_TRUE_PEAK = -1.0
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BIT_DEPTH = 24
DEFAULT_LOOP_CROSSFADE = 2.0

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
            action=argparse.BooleanOptionalAction,
            default=None,
            help="After saving, master to WAV (48k/24-bit, LUFS/true-peak; needs "
            "ffmpeg). Config-backable via defaults.sfx.master / "
            "defaults.music.master; an explicit --master/--no-master still wins.",
        )
    # #57 Class C2: the five valued knobs default to None so config can reach
    # them; the literals go back on in `apply_master_literals`. They are GLOBAL
    # config keys, not per-command ones, because this one chain is shared by
    # three commands -- `defaults.<cmd>.<knob>` still overrides (see _GLOBAL_MAP).
    parser.add_argument("--lufs", type=float, default=None, metavar="LUFS",
                        help=f"Integrated loudness target (default {DEFAULT_LUFS:g}). "
                             "Config-backable via defaults.lufs.")
    parser.add_argument("--true-peak", type=float, default=None, dest="true_peak",
                        metavar="DBTP",
                        help=f"True-peak ceiling in dBTP (default {DEFAULT_TRUE_PEAK:g}). "
                             "Config-backable via defaults.true_peak.")
    parser.add_argument("--sample-rate", type=int, default=None, dest="sample_rate",
                        metavar="HZ",
                        help=f"Output sample rate (default {DEFAULT_SAMPLE_RATE}). "
                             "Config-backable via defaults.sample_rate.")
    parser.add_argument("--bit-depth", type=int, choices=BIT_DEPTHS, default=None,
                        dest="bit_depth",
                        help=f"Output PCM bit depth (default {DEFAULT_BIT_DEPTH}). "
                             "Config-backable via defaults.bit_depth.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=None,
                        help="Make it seamlessly loopable (crossfade tail into head). "
                             "Config-backable via defaults.{master,sfx,music}.loop; an "
                             "explicit --loop/--no-loop still wins.")
    parser.add_argument("--loop-crossfade", type=float, default=None,
                        dest="loop_crossfade", metavar="SEC",
                        help="Loop crossfade length in seconds (default "
                             f"{DEFAULT_LOOP_CROSSFADE:g}). Config-backable via "
                             "defaults.loop_crossfade.")


def apply_master_literals(args) -> None:
    """Put the mastering chain's built-in literals back on a parsed namespace.

    Call AFTER `userconfig.apply_defaults` and BEFORE anything evaluates
    `master_kwargs(args)` -- which reads all six dests unconditionally, so a
    leftover None reaches ffmpeg. The failure is not uniform and only one mode is
    loud: `sample_rate=None` stringifies to a literal `-ar None` in the pass-2
    argv, and because pass 1 succeeds first the media is already downloaded (and
    on sfx/music already PAID FOR) before it dies as a generic exit 5.

    Deliberately NOT `master=`/`loop=`: those are tri-state booleans whose None
    is meaningful (`if args.master`, `bool(args.loop)`), and `venice master`
    registers with include_toggle=False so its namespace has no `master` dest at
    all -- `apply_literals` raises on an unknown dest, so listing it here would
    break `venice master` outright. (#57 Class C2)
    """
    userconfig.apply_literals(
        args,
        lufs=DEFAULT_LUFS,
        true_peak=DEFAULT_TRUE_PEAK,
        sample_rate=DEFAULT_SAMPLE_RATE,
        bit_depth=DEFAULT_BIT_DEPTH,
        loop_crossfade=DEFAULT_LOOP_CROSSFADE,
    )


def master_kwargs(args) -> dict:
    """Pull the mastering knobs off a parsed namespace into master() kwargs."""
    return dict(
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        lufs=args.lufs,
        true_peak=args.true_peak,
        # bool(): `--loop` is tri-state (default None) so config can reach it,
        # but `master()` annotates loop as a plain bool. (#57 Class B)
        loop=bool(args.loop),
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
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bit_depth: int = DEFAULT_BIT_DEPTH,
    lufs: float = DEFAULT_LUFS,
    true_peak: float = DEFAULT_TRUE_PEAK,
    loop: bool = False,
    loop_crossfade: float = DEFAULT_LOOP_CROSSFADE,
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
