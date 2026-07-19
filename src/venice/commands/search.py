"""`venice search` -- semantic search over a project index (issue #24).

Embeds the query with the *same* backend/model the index was built with, cosine-
ranks the stored chunks, and prints the top matches as ``path:start-end`` with a
score and a short preview. Requires an index built by `venice index`. The engine
lives in the print-free ``commands._index``; this wraps it for the CLI.
"""
from __future__ import annotations

import json
import sys

from .. import userconfig
from . import _index


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "search",
        help="Semantic search over a project index (built by `venice index`).",
        description=(
            "Find the most relevant chunks for a natural-language query. Walks up "
            "from the current directory to locate a .venice/index (or use --index), "
            "embeds the query with the index's own model, and prints ranked "
            "path:line-range hits."
        ),
    )
    p.add_argument("query", help="Natural-language query.")
    p.add_argument(
        "--index", dest="index_path", default=None, metavar="PATH",
        help="Index location (a project root or a .venice/index dir); default: "
             "discovered by walking up from the current directory, or "
             "$VENICE_INDEX_DIR.",
    )
    p.add_argument(
        "-k", "--top-k", dest="top_k", type=int, default=None,
        help=f"Number of results to return (default: {_index.DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print results as a JSON object instead of text.",
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    userconfig.apply_defaults(args, "search")

    store_dir = _index.discover_store(args.index_path)
    if store_dir is None:
        where = args.index_path or "the current directory tree"
        print(f"search: no index found in {where}; run `venice index` first",
              file=sys.stderr)
        return 6

    try:
        results = _index.search_index(
            store_dir, args.query, k=args.top_k or _index.DEFAULT_TOP_K)
    except _index.IndexingError as e:
        if str(e):
            print(f"search: {e}", file=sys.stderr)
        return e.exit_code

    if args.json:
        json.dump({"query": args.query, "store": str(_index.store_file(store_dir)),
                   "results": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not results:
        print("search: no matches", file=sys.stderr)
        return 0

    for r in results:
        loc = f"{r['path']}:{r['start']}-{r['end']}"
        flag = " (changed since index)" if r.get("changed") else ""
        print(f"{r['score']:.4f}  {loc}{flag}")
        preview = r.get("preview")
        if preview:
            for line in preview.splitlines():
                print(f"    {line}")
    return 0
