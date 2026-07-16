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
from typing import List, Optional

from .. import auth
from ..client import VeniceAPIError, build_client_from_auth

OPENAI_MISSING_HINT = (
    "venice embed needs the openai package: pip install -r requirements.txt "
    "(or: pip install openai)"
)


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


def _embedding_models(client) -> Optional[List[dict]]:
    """Fetch the embedding-model catalog. None if the (free) GET is unavailable."""
    try:
        doc = client.get_json("/models", params={"type": "embedding"})
    except VeniceAPIError:
        return None
    data = doc.get("data") if isinstance(doc, dict) else None
    return list(data) if isinstance(data, list) else None


def _default_model(models: List[dict]) -> Optional[str]:
    for m in models:
        spec = m.get("model_spec") if isinstance(m, dict) else None
        traits = spec.get("traits") if isinstance(spec, dict) else None
        if isinstance(traits, list) and "default" in traits:
            return m.get("id")
    return None


def _resolve_model(args, models: Optional[List[dict]]) -> tuple:
    """Return (model_id, exit_code). exit_code is None on success."""
    if models is None:
        # Catalog unavailable: can't validate or pick a default.
        if args.model:
            return args.model, None
        print(
            "embed: could not fetch the model catalog; pass --model explicitly",
            file=sys.stderr,
        )
        return None, 2

    ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
    if args.model:
        if args.model in ids:
            return args.model, None
        print(f"embed: unknown embedding model {args.model!r}", file=sys.stderr)
        print("available: " + ", ".join(ids), file=sys.stderr)
        return None, 6

    default = _default_model(models)
    if default:
        return default, None
    print(
        "embed: no default embedding model advertised; pass --model. "
        "available: " + ", ".join(ids),
        file=sys.stderr,
    )
    return None, 6


def _build_kwargs(args, model: str, inputs) -> dict:
    kwargs: dict = {"model": model, "input": inputs}
    if args.dimensions is not None:
        kwargs["dimensions"] = args.dimensions
    if args.encoding_format is not None:
        kwargs["encoding_format"] = args.encoding_format
    return kwargs


def _openai_exit(oai, e) -> int:
    """Map an openai SDK exception to a venice exit code."""
    if isinstance(e, oai.APIConnectionError):
        print(f"embed: connection error: {e}", file=sys.stderr)
        return 8
    status = getattr(e, "status_code", None)
    print(f"embed: API error: {e}", file=sys.stderr)
    if status == 401:
        return 2
    if status == 404:
        return 6
    if status == 429:
        return 4
    if isinstance(status, int) and 500 <= status < 600:
        return 5
    if isinstance(status, int) and 400 <= status < 500:
        return 2
    return 5


def _run(args) -> int:
    inputs, rc = _resolve_inputs(args)
    if rc is not None:
        return rc

    try:
        import openai
    except ImportError:
        print(OPENAI_MISSING_HINT, file=sys.stderr)
        return 2

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    models = _embedding_models(client)
    model, rc = _resolve_model(args, models)
    if rc is not None:
        return rc

    oai = openai.OpenAI(api_key=client.api_key, base_url=client.base_url)
    kwargs = _build_kwargs(args, model, inputs)

    try:
        resp = oai.embeddings.create(**kwargs)
    except openai.OpenAIError as e:
        return _openai_exit(openai, e)

    if args.json:
        json.dump(resp.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    for item in sorted(resp.data, key=lambda d: d.index):
        json.dump(item.embedding, sys.stdout)
        sys.stdout.write("\n")
    return 0
