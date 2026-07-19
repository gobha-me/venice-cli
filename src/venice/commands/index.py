"""`venice index` -- build/update a project semantic index (issue #24).

Walks a project tree, chunks its text files, embeds the chunks (Venice or a local
OpenAI-compatible backend, #23), and writes vectors to ``<PATH>/.venice/index/``.
Incremental: a re-run only re-embeds files whose contents changed. The heavy
lifting lives in the print-free engine ``commands._index``; this is a thin CLI
wrapper that prints a human summary to stderr and the store path to stdout.
"""
from __future__ import annotations

import os
import sys

from .. import config, userconfig
from . import _index


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "index",
        help="Build/update a semantic index of a project tree.",
        description=(
            "Chunk a project tree, embed the chunks, and store vectors under "
            "PATH/.venice/index/ for `venice search`. Re-runs only re-embed "
            "changed files. Uses Venice embeddings by default, or a local "
            "OpenAI-compatible backend with --embed-base-url."
        ),
    )
    p.add_argument(
        "path", nargs="?", default=".",
        help="Project root to index (default: current directory).",
    )
    p.add_argument(
        "--model", "-m", default=None,
        help="Venice embedding model id (required unless --embed-base-url; "
             "Venice advertises no default embedding model).",
    )
    p.add_argument(
        "--embed-base-url", dest="embed_base_url", default=None, metavar="URL",
        help="Embed against an alternate OpenAI-compatible endpoint (local "
             "llama.cpp/Ollama/TEI) instead of Venice; also $VENICE_EMBED_BASE_URL.",
    )
    p.add_argument(
        "--embed-model", dest="embed_model", default=None, metavar="NAME",
        help="Model id for --embed-base-url (required with it).",
    )
    p.add_argument(
        "--dimensions", type=int, default=None,
        help="Truncate embedding vectors to this many dimensions (if supported).",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Discard any existing index and re-embed the whole tree.",
    )
    p.add_argument(
        "--exclude", action="append", default=None, metavar="GLOB",
        help="Skip files/dirs matching GLOB (repeatable), in addition to the "
             "built-in vcs/vendor/secret ignores and a simple .gitignore.",
    )
    p.add_argument(
        "--batch", type=int, default=None,
        help=f"Chunks per embeddings request (default: {_index.DEFAULT_BATCH}).",
    )
    p.add_argument(
        "--chunk-lines", dest="chunk_lines", type=int, default=None,
        help=f"Lines per chunk (default: {_index.DEFAULT_CHUNK_LINES}).",
    )
    p.add_argument(
        "--chunk-overlap", dest="chunk_overlap", type=int, default=None,
        help=f"Overlap between chunks in lines (default: {_index.DEFAULT_CHUNK_OVERLAP}).",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    # Backend flags follow CLI > env > config: layer env before apply_defaults
    # (which only fills a dest still None), same as `venice embed`.
    if args.embed_base_url is None:
        args.embed_base_url = os.environ.get(config.ENV_EMBED_BASE_URL)
    userconfig.apply_defaults(args, "index")

    def _progress(msg: str) -> None:
        print(f"index: {msg}", file=sys.stderr)

    try:
        summary = _index.build_index(
            args.path,
            model=args.model,
            embed_base_url=args.embed_base_url,
            embed_model=args.embed_model,
            dimensions=args.dimensions,
            rebuild=args.rebuild,
            excludes=args.exclude,
            batch=args.batch or _index.DEFAULT_BATCH,
            chunk_lines=args.chunk_lines or _index.DEFAULT_CHUNK_LINES,
            chunk_overlap=(args.chunk_overlap if args.chunk_overlap is not None
                           else _index.DEFAULT_CHUNK_OVERLAP),
            on_progress=_progress,
        )
    except _index.IndexingError as e:
        if str(e):
            print(f"index: {e}", file=sys.stderr)
        return e.exit_code

    dim = summary["dimensions"]
    print(
        f"index: {summary['indexed']} new, {summary['reused']} unchanged, "
        f"{summary['removed']} removed -- {summary['files']} files / "
        f"{summary['chunks']} chunks "
        f"[{summary['backend']}:{summary['model']}"
        f"{f', {dim}d' if dim else ''}]",
        file=sys.stderr,
    )
    print(summary["store"])
    return 0
