"""`venice embed` -- one-shot /embeddings.

Built on the official `openai` SDK (Venice is OpenAI-compatible; the SDK is
lazy-imported so the rest of the stdlib-only CLI works without it). Mirrors the
shape of `venice chat`: the free `/models?type=embedding` catalog GET (via the
lean urllib client) validates `--model` and resolves a default before the paid
embeddings call.

Output: by default one embedding is printed per line as a JSON array
(newline-delimited JSON -- pipes cleanly to `jq`). `--json` dumps the full raw
response object (model, data, usage) instead.
"""
from __future__ import annotations

import json
import sys

from .. import auth
from ..client import build_client_from_auth
from . import _models, _openai


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "embed",
        help="Create embeddings (/embeddings).",
        description=(
            "Turn text into embedding vectors with a Venice embedding model. "
            "Reads the text from the argument, from stdin when it is '-' or "
            "piped, or one input per non-empty line with --from-file (batch). "
            "By default prints one vector per line as a JSON array."
        ),
    )
    p.add_argument(
        "text",
        nargs="?",
        help="Input text. Use '-' (or pipe stdin) to read from stdin.",
    )
    p.add_argument(
        "--from-file",
        dest="from_file",
        default=None,
        metavar="PATH",
        help="Batch: embed one input per non-empty line of PATH.",
    )
    p.add_argument(
        "--model",
        "-m",
        default=None,
        help="Embedding model id (default: the catalog's 'default'-trait model).",
    )
    p.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Truncate output vectors to this many dimensions (if supported).",
    )
    p.add_argument(
        "--encoding-format",
        choices=("float", "base64"),
        default=None,
        dest="encoding_format",
        help="Vector encoding to request (default: float).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the raw response object (model, data, usage).",
    )
    p.set_defaults(handler=_run)


def _resolve_inputs(args) -> tuple:
    """Return (inputs, exit_code). inputs is a str (single) or list[str] (batch).

    exit_code is None on success. Rejects giving both a positional/stdin text
    and --from-file.
    """
    if args.from_file is not None:
        if args.text:
            print(
                "embed: provide either text or --from-file, not both",
                file=sys.stderr,
            )
            return None, 2
        try:
            with open(args.from_file, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh]
        except OSError as e:
            print(f"embed: cannot read {args.from_file}: {e}", file=sys.stderr)
            return None, 2
        inputs = [ln for ln in lines if ln]
        if not inputs:
            print(f"embed: no non-empty lines in {args.from_file}", file=sys.stderr)
            return None, 2
        return inputs, None

    text = args.text
    if text == "-" or (text is None and not sys.stdin.isatty()):
        text = sys.stdin.read().strip() or None
    if not text:
        print(
            "embed: no input (pass text, pipe stdin, or use --from-file)",
            file=sys.stderr,
        )
        return None, 2
    return text, None


def _build_kwargs(args, model: str, inputs) -> dict:
    kwargs: dict = {"model": model, "input": inputs}
    if args.dimensions is not None:
        kwargs["dimensions"] = args.dimensions
    if args.encoding_format is not None:
        kwargs["encoding_format"] = args.encoding_format
    return kwargs


def _run(args) -> int:
    inputs, rc = _resolve_inputs(args)
    if rc is not None:
        return rc

    openai = _openai.import_openai("embed")
    if openai is None:
        return 2

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    models = _models.catalog(client, "embedding")
    model, rc = _models.resolve_model(
        args.model, models, label="embed", noun="embedding model"
    )
    if rc is not None:
        return rc

    oai = _openai.build_openai(openai, client)
    kwargs = _build_kwargs(args, model, inputs)

    try:
        resp = oai.embeddings.create(**kwargs)
    except openai.OpenAIError as e:
        return _openai.status_to_exit(openai, e, "embed")

    if args.json:
        json.dump(resp.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    for item in sorted(resp.data, key=lambda d: d.index):
        json.dump(item.embedding, sys.stdout)
        sys.stdout.write("\n")
    return 0
