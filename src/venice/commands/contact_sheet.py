"""`venice contact-sheet` -- montage a set of images into a review grid.

Pure local post-processing (no Venice API call): tiles a batch of generated
images -- variant rolls, a whole set -- into one sheet via ImageMagick `montage`
or `ffmpeg` (auto-detected). Optional per-cell filename labels. Makes a large
batch reviewable at a glance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import image_montage, userconfig
from . import _shared


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
    # #57 Class D: every knob defaults to None so `apply_defaults` can reach it;
    # the literals go back on in `_run` via `apply_literals`.
    p.add_argument("--cols", type=int, default=None, metavar="N",
                   help="Number of columns in the grid (default "
                        f"{image_montage.DEFAULT_COLS}). Config-backable via "
                        "defaults.contact_sheet.cols.")
    p.add_argument("--cell", default=None, metavar="WxH",
                   help="Cell (thumbnail) size, WxH (default "
                        f"{image_montage.DEFAULT_CELL}). Config-backable via "
                        "defaults.contact_sheet.cell.")
    p.add_argument("--label", action=argparse.BooleanOptionalAction, default=None,
                   help="Caption each cell with its filename. Config-backable "
                        "via defaults.contact_sheet.label; an explicit "
                        "--label/--no-label still wins.")
    p.add_argument("--background", default=None, metavar="COLOR",
                   help="Background/pad color (default "
                        f"{image_montage.DEFAULT_BACKGROUND}). Config-backable "
                        "via defaults.contact_sheet.background.")
    p.add_argument("--padding", type=int, default=None, metavar="PX",
                   help="Gap between cells in pixels (default "
                        f"{image_montage.DEFAULT_PADDING}). Config-backable via "
                        "defaults.contact_sheet.padding.")
    p.add_argument("--engine", choices=image_montage.ENGINES, default=None,
                   help="Which tool to use (default "
                        f"{image_montage.DEFAULT_ENGINE}: montage, else ffmpeg). "
                        "Config-backable via defaults.contact_sheet.engine.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the montage/ffmpeg command without running it.")
    p.set_defaults(handler=_run)


def _run(args) -> int:
    # The section is `contact_sheet` with an UNDERSCORE even though the command
    # is `contact-sheet` -- the `image_edit` precedent. `apply_defaults` takes
    # the section as a literal string, and dotted config keys split on `.` only,
    # so a hyphen would be unreachable from `venice config set`. (#57 Class D)
    userconfig.apply_defaults(args, "contact_sheet")
    # Literals last, and before `image_montage.contact_sheet`, which owns the
    # "--cols must be >= 1" and "--cell must be WxH" exit-2 validators. A
    # config-set `cols: 0` must reach those rather than being rewritten to 4,
    # which is why `apply_literals` fills with `is not None` and never `or`.
    userconfig.apply_literals(
        args,
        cols=image_montage.DEFAULT_COLS,
        cell=image_montage.DEFAULT_CELL,
        background=image_montage.DEFAULT_BACKGROUND,
        padding=image_montage.DEFAULT_PADDING,
        engine=image_montage.DEFAULT_ENGINE,
    )
    # `--output` can now also arrive from the `defaults.output_dir` GLOBAL, which
    # is a DIRECTORY. Unresolved it goes straight into the montage/ffmpeg argv as
    # the output file (exit 5) and gets printed as the machine-readable path --
    # exactly the landmine `venice master` hit in Class B. Resolve through the
    # same shared helper every sibling command uses. (#57 Class D)
    out = _shared.resolve_output(args.output, image_montage.DEFAULT_OUTPUT_NAME)
    return image_montage.contact_sheet(
        args.inputs,
        out,
        cols=args.cols,
        cell=args.cell,
        # bool(): --label is tri-state (default None) so config can reach it,
        # but `contact_sheet()` annotates label as a plain bool. (#57 Class D)
        label=bool(args.label),
        background=args.background,
        padding=args.padding,
        engine=args.engine,
        dry_run=args.dry_run,
    )
