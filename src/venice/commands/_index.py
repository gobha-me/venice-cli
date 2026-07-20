"""Project semantic index + search engine (issue #24).

The reusable core behind `venice index`, `venice search`, and the `project_search`
agent tool. It walks a project tree, chunks text files, embeds the chunks (via the
same `openai`-SDK machinery as `venice embed` -- Venice **or** a local/pluggable
backend from #23), persists vectors to a project-local on-disk store, and answers
nearest-neighbour queries with a pure-Python cosine scan (no numpy -- the base CLI
stays stdlib-only; only the `[openai]` extra is needed, for embedding).

stdout discipline (mirrors ``commands._mcp``): every function here is **stdout-free**
so `project_search` is safe inside the `mcp-serve` JSON-RPC transport. Progress and
warnings go to *stderr* (fine -- not the transport) or an injected ``on_progress``
callback; the CLI wrappers own all stdout. Failures raise :class:`IndexingError`
carrying an exit code; reused helpers (`_models.resolve_model`, `status_to_exit`)
already print their own stderr detail, so those are re-raised with an empty message.

Store layout: a single atomically-rewritten ``<root>/.venice/index/index.json``
(house style from ``userconfig.save_config``). Vectors are base64-packed float32
(stdlib ``array``) -- a few-thousand-chunk repo is a handful of MB and decodes
C-fast. Incremental re-index keys each file on ``sha256(bytes)``: unchanged files
keep their vectors, changed/new files are re-embedded, vanished files are dropped.

Security: the walker's ignore set excludes credential/secret-shaped files
(``credentials``, ``.env``, ``*.pem``, ``*.key``, ``id_rsa*``) and prunes ``.git``/
``.venice``/vcs/vendor dirs; symlinks that resolve outside the tree are skipped. The
Venice API key is never read for indexing and never stored.
"""
from __future__ import annotations

import array
import base64
import fnmatch
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import auth, config
from ..client import build_client_from_auth
from . import _models, _openai

STORE_VERSION = 1

DEFAULT_CHUNK_LINES = 40
DEFAULT_CHUNK_OVERLAP = 5
DEFAULT_BATCH = 64
DEFAULT_TOP_K = 8

MAX_FILE_BYTES = 1_500_000  # skip files larger than this (generated/data blobs)
MAX_CHUNK_CHARS = 4000      # cap a single chunk's embed input (minified lines)
SNIFF_BYTES = 8192          # prefix scanned for NUL bytes (binary detection)

# Directory names pruned during the walk (never descended into).
_DIR_DENYLIST = frozenset({
    ".git", ".hg", ".svn", ".venice", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "node_modules", ".venv", "venv",
    "env", ".idea", ".vscode", "dist", "build", ".egg-info",
})

# Exact filenames never indexed (credential/secret-shaped -- CLAUDE.md invariant).
_SECRET_NAMES = frozenset({"credentials", ".env", ".netrc", ".pgpass"})
# Glob patterns (matched on basename) never indexed.
_SECRET_GLOBS = (
    "*.pem", "*.key", "*.pfx", "*.p12", "*.keystore", "*.crt",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    ".env.*", "*.env", "*.secret", "*secrets*",
)


class IndexingError(Exception):
    """An index/search failure carrying a venice exit code.

    ``message`` is empty when an underlying helper (e.g. ``_models.resolve_model``,
    ``_openai.status_to_exit``) already printed the detail to stderr; callers then
    just propagate ``exit_code`` without double-printing.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# --------------------------------------------------------------------------- #
# Store paths + discovery
# --------------------------------------------------------------------------- #
def store_dir_for_root(root: Path) -> Path:
    """The index directory for a project root: ``<root>/.venice/index``."""
    return Path(root) / config.INDEX_DIRNAME / config.INDEX_SUBDIR


def store_file(store_dir: Path) -> Path:
    return Path(store_dir) / config.INDEX_FILENAME


def _root_of(store_dir: Path) -> Path:
    """Best-effort project root for a store dir (``<root>/.venice/index`` -> root)."""
    sd = Path(store_dir)
    if sd.name == config.INDEX_SUBDIR and sd.parent.name == config.INDEX_DIRNAME:
        return sd.parent.parent
    return sd


def discover_store(explicit: Optional[str], start: Optional[Path] = None) -> Optional[Path]:
    """Locate an index store dir for `venice search`.

    Precedence: ``--index`` (as a store dir OR a project root) -> ``$VENICE_INDEX_DIR``
    -> walk up from `start` (default cwd) for a ``.venice/index/index.json``, stopping
    at the filesystem root and never climbing above ``$HOME``. None if not found.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if store_file(p).exists():
            return p
        nested = store_dir_for_root(p)
        if store_file(nested).exists():
            return nested
        return None

    env = os.environ.get(config.ENV_INDEX_DIR)
    if env:
        p = Path(env).expanduser()
        return p if store_file(p).exists() else None

    start = Path(start or os.getcwd()).resolve()
    home = config.HOME.resolve()
    cur = start
    while True:
        cand = store_dir_for_root(cur)
        if store_file(cand).exists():
            return cand
        if cur == cur.parent or cur == home:
            return None
        cur = cur.parent


# --------------------------------------------------------------------------- #
# Store I/O (atomic single-file, base64 float32 vectors)
# --------------------------------------------------------------------------- #
def _encode_vec(vec) -> str:
    return base64.b64encode(array.array("f", vec).tobytes()).decode("ascii")


def _decode_vec(s: str) -> array.array:
    a = array.array("f")
    a.frombytes(base64.b64decode(s))
    return a


def load_store(sfile: Path, *, strict: bool):
    """Read the index JSON. Missing -> None.

    ``strict`` (search): a present-but-malformed store raises IndexingError.
    Non-strict (index rebuild path): warn to stderr and return None so the caller
    re-indexes from scratch rather than crashing.
    """
    p = Path(sfile)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        if strict:
            raise IndexingError(f"unreadable index {p}: {e}", 6)
        print(f"index: ignoring unreadable store {p}: {e}", file=sys.stderr)
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("meta"), dict) \
            or not isinstance(doc.get("files"), dict):
        if strict:
            raise IndexingError(f"malformed index {p}", 6)
        print(f"index: ignoring malformed store {p}", file=sys.stderr)
        return None
    return doc


def save_store(store_dir: Path, doc: dict) -> Path:
    """Atomically write index.json at mode 0600 (mirrors userconfig.save_config).

    The tmp file lives inside `store_dir` so os.replace stays on one filesystem.
    Raises OSError on disk failure (callers map to exit 9).
    """
    sd = Path(store_dir)
    sd.mkdir(parents=True, exist_ok=True)
    _write_gitignore(sd)
    target = store_file(sd)
    tmp = target.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"), sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, target)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return target


def _write_gitignore(store_dir: Path) -> None:
    """Drop a ``.venice/.gitignore`` that ignores the whole store, so a user's repo
    never accidentally commits vectors. Best-effort; never fatal."""
    try:
        venice_dir = Path(store_dir).parent  # <root>/.venice
        gi = venice_dir / ".gitignore"
        if not gi.exists():
            gi.write_text("# Venice semantic index -- machine-generated, do not commit.\n*\n",
                          encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Tree walking + ignore matching
# --------------------------------------------------------------------------- #
def _is_secret_name(name: str) -> bool:
    if name in _SECRET_NAMES:
        return True
    return any(fnmatch.fnmatch(name, g) for g in _SECRET_GLOBS)


def _load_gitignore(root: Path) -> List[str]:
    """A *simple* subset of the top-level .gitignore: plain patterns matched by
    fnmatch on basename/relpath. Negations (`!`), anchoring and nesting are ignored
    (documented) -- full gitignore semantics are out of scope for v1."""
    pats: List[str] = []
    gi = Path(root) / ".gitignore"
    try:
        lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return pats
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        pats.append(line.rstrip("/"))
    return pats


def _gitignored(rel_posix: str, name: str, pats: List[str]) -> bool:
    for p in pats:
        if fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel_posix, p) \
                or fnmatch.fnmatch(rel_posix, p + "/*"):
            return True
        if "/" not in p and p in rel_posix.split("/"):
            return True
    return False


def walk_files(root: Path, excludes: Optional[List[str]] = None):
    """Yield indexable file Paths under `root`.

    Prunes vcs/vendor dirs and gitignored dirs; skips secret-shaped files,
    excluded globs, oversized files, and symlinks resolving outside `root`.
    """
    root = Path(root).resolve()
    excludes = list(excludes or [])
    gitpats = _load_gitignore(root)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        # prune directories in place
        kept = []
        for d in dirnames:
            if d in _DIR_DENYLIST:
                continue
            rel = (dp / d).relative_to(root).as_posix()
            if _gitignored(rel, d, gitpats):
                continue
            if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(d, g) for g in excludes):
                continue
            kept.append(d)
        dirnames[:] = kept

        for name in filenames:
            fp = dp / name
            rel = fp.relative_to(root).as_posix()
            if _is_secret_name(name):
                continue
            if _gitignored(rel, name, gitpats):
                continue
            if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in excludes):
                continue
            try:
                if fp.is_symlink() and not _resolves_inside(fp, root):
                    continue
                st = fp.stat()
            except OSError:
                continue
            if not (st.st_mode & 0o170000 == 0o100000):  # not a regular file
                continue
            if st.st_size > MAX_FILE_BYTES:
                continue
            yield fp


def _resolves_inside(path: Path, root: Path) -> bool:
    try:
        real = Path(os.path.realpath(path))
        real.relative_to(root)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Public path-safety API (reused by the coding toolset, commands._code)
#
# These bless the security-critical helpers above as a public API so the coding
# tools depend on a stable name rather than a private one -- a single source of
# truth for "is this path inside the project?" and "is this file secret-shaped?".
# --------------------------------------------------------------------------- #
def resolves_inside(path: Path, root: Path) -> bool:
    """True if `path` (symlinks resolved) is `root` or a descendant of it.

    The sandbox primitive for the coding tools. `root` must itself be realpath-
    resolved by the caller, or a symlinked root yields false negatives.
    """
    return _resolves_inside(path, root)


def is_secret_name(name: str) -> bool:
    """True if a *basename* is credential/secret-shaped (the index denylist)."""
    return _is_secret_name(name)


def is_secret_path(rel_posix: str) -> bool:
    """True if any segment of a relative POSIX path is secret-shaped.

    Broader than :func:`is_secret_name` (which is basename-only): catches a
    nested secret like ``config/secrets/app.key`` that a coding tool might be
    asked to read or write directly.
    """
    return any(is_secret_name(seg) for seg in rel_posix.split("/") if seg)


def is_protected_dir_path(rel_posix: str) -> bool:
    """True if any segment is a denylisted dir (``.git``/``.venice``/vendor/...).

    Lets the coding tools refuse to read or write repo internals and the local
    index, mirroring what :func:`walk_files` prunes during a tree walk.
    """
    return any(seg in _DIR_DENYLIST for seg in rel_posix.split("/") if seg)


# --------------------------------------------------------------------------- #
# Reading + chunking
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    """Return (raw_bytes, text). text is None for binary/undecodable files.

    Binary heuristic: a NUL byte in the first `SNIFF_BYTES`, or a UTF-8 decode
    failure. (UTF-16 and other non-UTF-8 encodings are skipped by design in v1.)
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    if b"\x00" in data[:SNIFF_BYTES]:
        return data, None
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError:
        return data, None


def chunk_text(text: str, *, chunk_lines: int, overlap: int,
               max_chars: int = MAX_CHUNK_CHARS) -> List[dict]:
    """Split into overlapping line windows carrying 1-based start/end line numbers.

    Empty/whitespace-only windows are dropped; a window's embed text is capped at
    `max_chars` so a minified line can't blow the model's input limit.
    """
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    step = max(1, chunk_lines - overlap)
    out: List[dict] = []
    i = 0
    while i < n:
        end = min(i + chunk_lines, n)
        body = "\n".join(lines[i:end])
        if body.strip():
            out.append({"start": i + 1, "end": end, "text": body[:max_chars]})
        if end >= n:
            break
        i += step
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Embedding backend resolution + batched embedding
# --------------------------------------------------------------------------- #
def _resolve_backend(openai, *, embed_base_url, embed_model, model):
    """Return (oai_client, model, backend_meta, exit_code). exit_code None on success.

    Mirrors ``embed._resolve_backend`` but stdout-free and primitive-arg. Local
    backend (--embed-base-url) skips Venice auth+catalog; Venice path validates
    --model against the free /models catalog.
    """
    if embed_base_url:
        if not embed_model:
            print("index: --embed-model is required with --embed-base-url", file=sys.stderr)
            return None, None, None, 2
        oai = _openai.build_openai(
            openai, base_url=embed_base_url,
            api_key=os.environ.get(config.ENV_EMBED_API_KEY),
        )
        meta = {"backend": "local", "model": embed_model, "base_url": embed_base_url}
        return oai, embed_model, meta, None

    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return None, None, None, 2
    models = _models.catalog(client, "embedding")
    resolved, rc = _models.resolve_model(model, models, label="index", noun="embedding model")
    if rc is not None:
        return None, None, None, rc
    return _openai.build_openai(openai, client), resolved, {"backend": "venice", "model": resolved}, None


def _backend_from_meta(openai, meta: dict):
    """Rebuild the SDK client for `venice search` from stored index meta.

    The query MUST be embedded with the same backend/model/dims as the index, so
    everything comes from meta -- never re-resolved against a (possibly changed)
    catalog. The local-backend key is re-supplied from $VENICE_EMBED_API_KEY.
    """
    if meta.get("backend") == "local":
        oai = _openai.build_openai(
            openai, base_url=meta.get("base_url"),
            api_key=os.environ.get(config.ENV_EMBED_API_KEY),
        )
        return oai, meta.get("model"), None
    try:
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return None, None, 2
    return _openai.build_openai(openai, client), meta.get("model"), None


def _embed_batches(oai, openai, model, texts: List[str], *, dimensions: Optional[int],
                   batch: int, label: str, on_progress: Optional[Callable[[int, int], None]] = None):
    """Embed `texts` in batches of `batch`. Return (vectors, exit_code).

    vectors is a list aligned to `texts` (None never appears on success). On an SDK
    error returns (partial_vectors, exit_code) so callers can persist completed work.
    """
    vectors: List[Optional[list]] = []
    total = len(texts)
    for i in range(0, total, batch):
        part = texts[i:i + batch]
        kwargs = {"model": model, "input": part}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        try:
            resp = oai.embeddings.create(**kwargs)
        except openai.OpenAIError as e:
            return vectors, _openai.status_to_exit(openai, e, label)
        for item in sorted(resp.data, key=lambda d: d.index):
            vectors.append(list(item.embedding))
        if on_progress is not None:
            on_progress(min(i + batch, total), total)
    return vectors, None


# --------------------------------------------------------------------------- #
# Build (index)
# --------------------------------------------------------------------------- #
def _meta_conflict(old_meta: dict, backend_meta: dict, dimensions: Optional[int]) -> Optional[str]:
    """Describe an embedding-space mismatch between an existing index and a re-index
    request, or None if compatible. Cosine across two spaces is meaningless, so a
    conflict must force --rebuild rather than silently mixing vectors."""
    if old_meta.get("backend") != backend_meta.get("backend"):
        return f"backend {old_meta.get('backend')!r} -> {backend_meta.get('backend')!r}"
    if old_meta.get("model") != backend_meta.get("model"):
        return f"model {old_meta.get('model')!r} -> {backend_meta.get('model')!r}"
    if old_meta.get("backend") == "local" and old_meta.get("base_url") != backend_meta.get("base_url"):
        return f"base_url {old_meta.get('base_url')!r} -> {backend_meta.get('base_url')!r}"
    if dimensions is not None and old_meta.get("dimensions") not in (None, dimensions):
        return f"dimensions {old_meta.get('dimensions')!r} -> {dimensions!r}"
    return None


def build_index(root, *, model=None, embed_base_url=None, embed_model=None,
                dimensions=None, rebuild=False, excludes=None, batch=DEFAULT_BATCH,
                chunk_lines=DEFAULT_CHUNK_LINES, chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                on_progress: Optional[Callable[[str], None]] = None) -> dict:
    """Build/update the index under `root`. Returns a summary dict on success;
    raises IndexingError(exit_code) on failure. stdout-free (progress -> on_progress
    callback / stderr)."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise IndexingError(f"not a directory: {root}", 2)
    batch = max(1, int(batch or DEFAULT_BATCH))
    chunk_lines = max(1, int(chunk_lines or DEFAULT_CHUNK_LINES))
    chunk_overlap = max(0, min(int(chunk_overlap or 0), chunk_lines - 1))

    openai = _openai.import_openai("index")
    if openai is None:
        raise IndexingError("", 2)

    oai, resolved_model, backend_meta, rc = _resolve_backend(
        openai, embed_base_url=embed_base_url, embed_model=embed_model, model=model)
    if rc is not None:
        raise IndexingError("", rc)

    sdir = store_dir_for_root(root)
    old = None if rebuild else load_store(store_file(sdir), strict=False)
    if old is not None:
        conflict = _meta_conflict(old["meta"], backend_meta, dimensions)
        if conflict:
            raise IndexingError(
                f"index was built with a different embedding space ({conflict}); "
                "re-run with --rebuild to replace it", 2)

    prev_files: Dict[str, dict] = dict(old["files"]) if old else {}
    locked_dim = old["meta"].get("dimensions") if old else None

    # Walk + diff. `keep` = unchanged files carried forward; `pending` = (rel, fmeta, chunks).
    keep: Dict[str, dict] = {}
    pending: List[Tuple[str, dict, List[dict]]] = []
    seen = set()
    for fp in walk_files(root, excludes):
        rel = fp.relative_to(root).as_posix()
        seen.add(rel)
        data, text = read_text(fp)
        if data is None:
            continue
        sha = _sha256(data)
        prev = prev_files.get(rel)
        if prev is not None and prev.get("sha256") == sha:
            keep[rel] = prev
            continue
        if text is None:  # binary/undecodable -> not indexed (drop any stale entry)
            continue
        chunks = chunk_text(text, chunk_lines=chunk_lines, overlap=chunk_overlap)
        if not chunks:
            continue
        try:
            mtime = int(fp.stat().st_mtime)
        except OSError:
            mtime = 0
        pending.append((rel, {"sha256": sha, "size": len(data), "mtime": mtime}, chunks))

    removed = [r for r in prev_files if r not in seen]

    if on_progress:
        on_progress(
            f"{len(pending)} file(s) to embed, {len(keep)} unchanged, "
            f"{len(removed)} removed")

    # Flatten chunk texts across pending files (cross-file batching).
    texts: List[str] = []
    for _, _, chunks in pending:
        texts.extend(c["text"] for c in chunks)

    dim = locked_dim
    files_out = dict(keep)

    if texts:
        def _tick(done, total):
            if on_progress:
                on_progress(f"embedding {done}/{total} chunks")

        vectors, erc = _embed_batches(
            oai, openai, resolved_model, texts, dimensions=dimensions,
            batch=batch, label="index", on_progress=_tick)

        # Assemble whichever files are fully covered by the vectors we got (all of
        # them on success; a prefix on mid-run SDK failure -> resumable next run).
        pos = 0
        for rel, fmeta, chunks in pending:
            n = len(chunks)
            if pos + n > len(vectors):
                break
            vecs = vectors[pos:pos + n]
            pos += n
            if dim is None and vecs:
                dim = len(vecs[0])
            if any(len(v) != dim for v in vecs):
                raise IndexingError(
                    f"backend returned inconsistent vector dimensions (expected {dim})", 5)
            fmeta = dict(fmeta)
            fmeta["chunks"] = [
                {"start": c["start"], "end": c["end"], "vec": _encode_vec(v)}
                for c, v in zip(chunks, vecs)
            ]
            files_out[rel] = fmeta

        if erc is not None:
            # Persist the completed prefix before surfacing the error, but only if
            # at least one new file finished embedding -- a total first-run failure
            # should not leave an empty index behind (and an unchanged prior store
            # is already on disk, so there is nothing new to write).
            if len(files_out) > len(keep):
                _persist(sdir, backend_meta, dim, chunk_lines, chunk_overlap, dimensions, files_out)
            raise IndexingError("", erc)

    _persist(sdir, backend_meta, dim, chunk_lines, chunk_overlap, dimensions, files_out)

    total_chunks = sum(len(f.get("chunks", [])) for f in files_out.values())
    return {
        "store": str(store_file(sdir)),
        "backend": backend_meta["backend"],
        "model": resolved_model,
        "dimensions": dim,
        "indexed": len(pending),
        "reused": len(keep),
        "removed": len(removed),
        "files": len(files_out),
        "chunks": total_chunks,
    }


def _persist(sdir, backend_meta, dim, chunk_lines, chunk_overlap, dimensions, files_out) -> dict:
    meta = {
        "version": STORE_VERSION,
        "backend": backend_meta["backend"],
        "model": backend_meta["model"],
        "dimensions": dim,
        "chunk_lines": chunk_lines,
        "chunk_overlap": chunk_overlap,
        "updated": int(time.time()),
    }
    if backend_meta.get("base_url"):
        meta["base_url"] = backend_meta["base_url"]
    if dimensions is not None:
        meta["requested_dimensions"] = dimensions
    try:
        save_store(sdir, {"meta": meta, "files": files_out})
    except OSError as e:
        raise IndexingError(f"cannot write index: {e}", 9)
    return meta


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _cosine(q: List[float], qnorm: float, v: array.array) -> float:
    dot = 0.0
    vn = 0.0
    for a, b in zip(q, v):
        dot += a * b
        vn += b * b
    if qnorm == 0.0 or vn == 0.0:
        return 0.0
    return dot / (qnorm * math.sqrt(vn))


def _preview(root: Path, rel: str, start: int, end: int, stored_sha: Optional[str],
             max_lines: int = 3, max_chars: int = 160) -> dict:
    """Best-effort snippet for a hit: the first few lines of the chunk, plus a
    'changed' flag if the file's current sha differs from the indexed one."""
    fp = Path(root) / rel
    try:
        data = fp.read_bytes()
    except OSError:
        return {"text": None, "changed": True}
    changed = stored_sha is not None and _sha256(data) != stored_sha
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return {"text": None, "changed": changed}
    snippet = lines[start - 1:min(end, start - 1 + max_lines)]
    text = "\n".join(ln[:max_chars] for ln in snippet)
    return {"text": text, "changed": changed}


def search_index(store_dir, query: str, *, k: int = DEFAULT_TOP_K,
                 with_preview: bool = True) -> List[dict]:
    """Embed `query` with the index's own backend/model and return the top-`k`
    chunks as dicts ``{path, start, end, score[, preview, changed]}``. stdout-free;
    raises IndexingError(exit_code) on failure."""
    if not query or not query.strip():
        raise IndexingError("empty query", 2)
    sdir = Path(store_dir)
    store = load_store(store_file(sdir), strict=True)
    if store is None:
        raise IndexingError(f"no index at {sdir}; run `venice index` first", 6)
    meta = store["meta"]

    openai = _openai.import_openai("search")
    if openai is None:
        raise IndexingError("", 2)
    oai, model, rc = _backend_from_meta(openai, meta)
    if rc is not None:
        raise IndexingError("", rc)

    qvecs, erc = _embed_batches(
        oai, openai, model, [query.strip()],
        dimensions=meta.get("requested_dimensions"), batch=1, label="search")
    if erc is not None:
        raise IndexingError("", erc)
    q = qvecs[0]
    qnorm = math.sqrt(sum(a * a for a in q))

    scored: List[dict] = []
    for rel, fmeta in store["files"].items():
        sha = fmeta.get("sha256")
        for ch in fmeta.get("chunks", []):
            try:
                v = _decode_vec(ch["vec"])
            except (KeyError, ValueError):
                continue
            scored.append({
                "path": rel, "start": ch.get("start"), "end": ch.get("end"),
                "score": _cosine(q, qnorm, v), "_sha": sha,
            })
    scored.sort(key=lambda r: (-r["score"], r["path"], r["start"] or 0))
    top = scored[:max(1, int(k or DEFAULT_TOP_K))]

    root = _root_of(sdir)
    results = []
    for r in top:
        out = {"path": r["path"], "start": r["start"], "end": r["end"],
               "score": round(r["score"], 6)}
        if with_preview:
            pv = _preview(root, r["path"], r["start"], r["end"], r.get("_sha"))
            out["preview"] = pv["text"]
            out["changed"] = pv["changed"]
        results.append(out)
    return results
