"""Local contact-sheet montage via ImageMagick `montage` or `ffmpeg`.

Shells out; never touches the API. Builds a grid (montage) of card art /
variant rolls for review at a glance. The external tool is auto-detected: use
`montage` when present (purpose-built, native labels), else `ffmpeg`'s tile
filter, else fail cleanly with an install hint before any work.
"""
from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

# Fonts to try for ffmpeg drawtext labels (montage finds its own).
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def has_montage() -> bool:
    return shutil.which("montage") is not None


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def select_engine(preferred: str) -> Optional[str]:
    """Resolve which external tool to use. `preferred` is auto|montage|ffmpeg.

    auto prefers `montage` (native labels), falling back to `ffmpeg`."""
    if preferred == "montage":
        return "montage" if has_montage() else None
    if preferred == "ffmpeg":
        return "ffmpeg" if has_ffmpeg() else None
    if has_montage():
        return "montage"
    if has_ffmpeg():
        return "ffmpeg"
    return None


def default_output() -> Path:
    return Path.cwd() / "contact-sheet.png"


def collect_inputs(inputs: List[str]) -> List[Path]:
    """Expand the positional arg(s) into a sorted list of image files.

    A single existing directory expands to its images; otherwise each arg is
    treated as a glob (or literal file). Non-image files are dropped."""
    found: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found += [q for q in p.iterdir() if q.is_file()]
        elif any(ch in item for ch in "*?[]"):
            found += [Path(m) for m in glob.glob(item)]
        elif p.is_file():
            found.append(p)
        # nonexistent literal -> silently skipped (reported as "no inputs")
    imgs = [p for p in found if p.suffix.lower() in _IMAGE_EXTS]
    # de-dup while sorting for a deterministic sheet order
    return sorted(set(imgs), key=lambda p: str(p))


def _parse_cell(cell: str) -> Optional[tuple]:
    m = re.fullmatch(r"(\d+)x(\d+)", cell.strip())
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    return w, h


def _find_font() -> Optional[str]:
    for f in _FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    return None


def _label_text(path: Path) -> str:
    """Filename stem, reduced to a filtergraph-safe charset for ffmpeg drawtext."""
    return re.sub(r"[^A-Za-z0-9 ._-]", "_", path.stem)


def _montage_cmd(images: List[Path], output: Path, *, cols: int, cell: tuple,
                 label: bool, background: str, padding: int) -> List[str]:
    w, h = cell
    cmd = ["montage", "-background", background]
    if label:
        cmd += ["-fill", "black", "-label", "%t"]
    cmd += [str(p) for p in images]
    cmd += [
        "-tile", f"{cols}x",
        "-geometry", f"{w}x{h}+{padding}+{padding}",
        str(output),
    ]
    return cmd


def _ffmpeg_cmd(images: List[Path], output: Path, *, cols: int, cell: tuple,
                label: bool, background: str, padding: int) -> List[str]:
    w, h = cell
    n = len(images)
    rows = math.ceil(n / cols)
    font = _find_font() if label else None

    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y"]
    for p in images:
        cmd += ["-i", str(p)]

    chains = []
    for i, p in enumerate(images):
        step = (
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={background},"
            f"setsar=1,format=rgba"
        )
        if label:
            txt = _label_text(p)
            draw = (
                f",drawtext=text='{txt}':x=(w-text_w)/2:y=h-text_h-4:"
                f"fontsize=16:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4"
            )
            if font:
                draw += f":fontfile={font}"
            step += draw
        chains.append(step + f"[v{i}]")

    concat_in = "".join(f"[v{i}]" for i in range(n))
    graph = (
        ";".join(chains)
        + f";{concat_in}concat=n={n}:v=1:a=0[cat]"
        + f";[cat]tile={cols}x{rows}:padding={padding}:color={background}[out]"
    )
    cmd += ["-filter_complex", graph, "-map", "[out]", "-frames:v", "1", str(output)]
    return cmd


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


def contact_sheet(
    inputs: List[str],
    output: Path,
    *,
    cols: int = 4,
    cell: str = "256x320",
    label: bool = False,
    background: str = "white",
    padding: int = 4,
    engine: str = "auto",
    dry_run: bool = False,
) -> int:
    """Build a contact-sheet montage. Returns an exit code
    (0 ok, 2 bad arg / no engine, 5 tool failure, 6 no inputs)."""
    if cols < 1:
        print(f"contact-sheet: --cols must be >= 1 (got {cols})", file=sys.stderr)
        return 2
    parsed_cell = _parse_cell(cell)
    if parsed_cell is None:
        print(f"contact-sheet: --cell must be WxH, e.g. 256x320 (got {cell!r})",
              file=sys.stderr)
        return 2

    images = collect_inputs(inputs)
    if not images:
        print(f"contact-sheet: no images found in {inputs}", file=sys.stderr)
        return 6

    eng = select_engine(engine)
    if eng is None and not dry_run:
        if engine == "montage":
            print("contact-sheet: montage (ImageMagick) not found on PATH; "
                  "install it (e.g. apt install imagemagick)", file=sys.stderr)
        elif engine == "ffmpeg":
            print("contact-sheet: ffmpeg not found on PATH; "
                  "install it (e.g. apt install ffmpeg)", file=sys.stderr)
        else:
            print("contact-sheet: needs ImageMagick (montage) or ffmpeg on PATH -- "
                  "install one (e.g. apt install imagemagick, or apt install ffmpeg)",
                  file=sys.stderr)
        return 2
    if eng is None:  # dry-run without either tool: show the preferred one
        eng = "ffmpeg" if engine == "ffmpeg" else "montage"

    builder = _montage_cmd if eng == "montage" else _ffmpeg_cmd
    cmd = builder(images, output, cols=cols, cell=parsed_cell, label=label,
                  background=background, padding=padding)

    if dry_run:
        print(f"contact-sheet: dry run -- would run ({eng}):", file=sys.stderr)
        print("  " + " ".join(cmd), file=sys.stderr)
        return 0

    r = _run(cmd)
    if r is None or r.returncode != 0:
        print(f"contact-sheet: {eng} failed", file=sys.stderr)
        if r is not None and r.stderr:
            print(r.stderr.strip()[-500:], file=sys.stderr)
        return 5

    print(str(output.resolve()))
    print(f"contact sheet ({len(images)} images, {eng}) -> {output.resolve()}",
          file=sys.stderr)
    return 0
