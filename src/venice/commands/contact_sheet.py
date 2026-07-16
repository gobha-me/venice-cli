"""`venice contact-sheet` -- montage a set of images into a review grid.

Pure local post-processing (no Venice API call): tiles card art / variant rolls
into one sheet via ImageMagick `montage` or `ffmpeg` (auto-detected). Optional
per-cell filename labels. Feeds the FRONTLINE review workflow.
"""
from __future__ import annotations

from pathlib import Path

from .. import image_montage


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "contact-sheet",
        help="Montage images into a review grid (needs ImageMagick montage or ffmpeg).",
        description=(
            "Tiles a directory/glob of images into a single contact sheet -- no "
            "Venice API call. Uses ImageMagick `montage` when present, else "
            "ffmpeg's tile filter (auto-detected). With --label, each cell is "
            "captioned with its filename."
        ),
    )
    p.add_argument("inputs", nargs="+", metavar="DIR_OR_GLOB",
                   help="A directory, glob, or list of image files.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output image path (default: ./contact-sheet.png).")
    p.add_argument("--cols", type=int, default=4, metavar="N",
                   help="Number of columns in the grid (default 4).")
    p.add_argument("--cell", default="256x320", metavar="WxH",
                   help="Cell (thumbnail) size, WxH (default 256x320).")
    p.add_argument("--label", action="store_true",
                   help="Caption each cell with its filename.")
    p.add_argument("--background", default="white", metavar="COLOR",
                   help="Background/pad color (default white).")
    p.add_argument("--padding", type=int, default=4, metavar="PX",
                   help="Gap between cells in pixels (default 4).")
    p.add_argument("--engine", choices=("auto", "montage", "ffmpeg"), default="auto",
                   help="Which tool to use (default auto: montage, else ffmpeg).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the montage/ffmpeg command without running it.")
    p.set_defaults(handler=_run)


def _run(args) -> int:
    out = args.output or image_montage.default_output()
    return image_montage.contact_sheet(
        args.inputs,
        out,
        cols=args.cols,
        cell=args.cell,
        label=args.label,
        background=args.background,
        padding=args.padding,
        engine=args.engine,
        dry_run=args.dry_run,
    )
