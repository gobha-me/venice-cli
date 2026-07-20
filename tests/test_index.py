"""Unit tests for `venice index` and the semantic-index engine (commands/_index.py).

Uses the local/pluggable backend path (--embed-base-url) so no Venice key or
catalog GET is needed; the fake OpenAI client returns deterministic vectors keyed
on three marker words so ranking/incrementality are checkable. No network.
The openai package must be importable (pip install -e ".[openai]").
"""
import argparse
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _index


# --- deterministic fake embeddings: vector = marker-word counts + bias ---

def marker_vec(text):
    t = text.lower()
    return [float(t.count("alpha")), float(t.count("beta")), float(t.count("gamma")), 1.0]


class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _Resp:
    def __init__(self, inputs):
        self.data = [_Item(i, marker_vec(t)) for i, t in enumerate(inputs)]


def fake_openai():
    """Return (fake_client, calls). create() records every input text it sees."""
    calls = {"n": 0, "texts": [], "kwargs": []}
    fake = mock.MagicMock()

    def _create(**kwargs):
        inp = kwargs["input"]
        inp = [inp] if isinstance(inp, str) else list(inp)
        calls["n"] += 1
        calls["texts"].extend(inp)
        calls["kwargs"].append(kwargs)
        return _Resp(inp)

    fake.embeddings.create.side_effect = _create
    return fake, calls


def _clean_doc():
    return {"version": 1, "mcpServers": {}, "defaults": {}}


def write(root, rel, content, *, binary=False):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


def _no_key_env():
    return {k: v for k, v in os.environ.items()
            if k not in ("VENICE_API_KEY", "VENICE_EMBED_BASE_URL",
                         "VENICE_EMBED_API_KEY", "VENICE_EMBED_CA_BUNDLE")}


class _EngineBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def build(self, fake, **ov):
        """Run build_index on self.root via the local backend, no Venice key."""
        kw = dict(embed_base_url="http://local/v1", embed_model="m")
        kw.update(ov)
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            return _index.build_index(self.root, **kw)

    def store(self):
        return _index.store_dir_for_root(self.root)


class TestChunking(unittest.TestCase):
    def test_line_windows_with_overlap(self):
        text = "\n".join(f"line{i}" for i in range(1, 11))  # 10 lines
        chunks = _index.chunk_text(text, chunk_lines=4, overlap=1)
        # step = 3 -> windows [1-4],[4-7],[7-10]; the last already reaches EOF so
        # there is no redundant trailing chunk, and every line (1..10) is covered.
        spans = [(c["start"], c["end"]) for c in chunks]
        self.assertEqual(spans, [(1, 4), (4, 7), (7, 10)])

    def test_empty_and_whitespace_dropped(self):
        self.assertEqual(_index.chunk_text("", chunk_lines=40, overlap=5), [])
        self.assertEqual(_index.chunk_text("   \n\n  \n", chunk_lines=40, overlap=5), [])

    def test_char_cap(self):
        text = "x" * 10000
        chunks = _index.chunk_text(text, chunk_lines=40, overlap=5, max_chars=100)
        self.assertTrue(all(len(c["text"]) <= 100 for c in chunks))


class TestVectorCodec(unittest.TestCase):
    def test_roundtrip(self):
        vec = [1.5, -2.0, 0.0, 3.25]
        enc = _index._encode_vec(vec)
        self.assertIsInstance(enc, str)
        dec = list(_index._decode_vec(enc))
        self.assertEqual(dec, vec)


class TestBuild(_EngineBase):
    def test_build_writes_atomic_0600_store(self):
        write(self.root, "src/a.py", "alpha alpha\ncode\n")
        write(self.root, "src/b.py", "beta\nmore\n")
        fake, calls = fake_openai()
        summary = self.build(fake)
        self.assertEqual(summary["indexed"], 2)
        self.assertEqual(summary["reused"], 0)
        self.assertEqual(summary["backend"], "local")
        self.assertEqual(summary["dimensions"], 4)
        sfile = _index.store_file(self.store())
        self.assertTrue(sfile.exists())
        self.assertEqual(os.stat(sfile).st_mode & 0o777, 0o600)
        self.assertFalse(sfile.with_suffix(".tmp").exists())
        # a self-ignoring .venice/.gitignore is dropped
        self.assertTrue((Path(self.root) / ".venice" / ".gitignore").exists())

    def test_local_backend_needs_no_venice_key(self):
        write(self.root, "a.txt", "alpha\n")
        fake, _ = fake_openai()
        # _no_key_env strips VENICE_API_KEY; a clean run proves none was required.
        summary = self.build(fake)
        self.assertEqual(summary["indexed"], 1)

    def test_embed_model_required_with_base_url(self):
        write(self.root, "a.txt", "alpha\n")
        fake, _ = fake_openai()
        with self.assertRaises(_index.IndexingError) as cm:
            self.build(fake, embed_model=None)
        self.assertEqual(cm.exception.exit_code, 2)

    def test_vectors_roundtrip_in_store(self):
        write(self.root, "a.txt", "alpha alpha beta\n")
        fake, _ = fake_openai()
        self.build(fake)
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        chunk = doc["files"]["a.txt"]["chunks"][0]
        self.assertEqual(list(_index._decode_vec(chunk["vec"])), [2.0, 1.0, 0.0, 1.0])


class TestIgnores(_EngineBase):
    def test_secret_and_binary_and_vcs_skipped(self):
        write(self.root, "keep.py", "alpha\n")
        write(self.root, ".env", "SECRET=xyz\n")
        write(self.root, "id_rsa", "PRIVATE KEY\n")
        write(self.root, "cert.pem", "----\n")
        write(self.root, "credentials", "topsecret\n")
        write(self.root, "blob.bin", b"\x00\x01\x02binary", binary=True)
        write(self.root, ".git/config", "gitguts\n")
        write(self.root, "node_modules/pkg/index.js", "alpha\n")
        fake, _ = fake_openai()
        summary = self.build(fake)
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        self.assertEqual(list(doc["files"]), ["keep.py"])
        self.assertEqual(summary["indexed"], 1)

    def test_exclude_glob(self):
        write(self.root, "keep.py", "alpha\n")
        write(self.root, "skip.md", "alpha\n")
        fake, _ = fake_openai()
        self.build(fake, excludes=["*.md"])
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        self.assertEqual(list(doc["files"]), ["keep.py"])

    def test_gitignore_subset(self):
        write(self.root, ".gitignore", "build/\n*.log\n")
        write(self.root, "keep.py", "alpha\n")
        write(self.root, "out.log", "alpha\n")
        write(self.root, "build/generated.py", "alpha\n")
        fake, _ = fake_openai()
        self.build(fake)
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        # the .gitignore patterns exclude the log and the build/ dir; keep.py stays
        self.assertIn("keep.py", doc["files"])
        self.assertNotIn("out.log", doc["files"])
        self.assertNotIn("build/generated.py", doc["files"])

    def test_oversize_file_skipped(self):
        write(self.root, "keep.py", "alpha\n")
        write(self.root, "huge.py", "x\n" * (_index.MAX_FILE_BYTES // 2 + 10))
        fake, _ = fake_openai()
        self.build(fake)
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        self.assertEqual(list(doc["files"]), ["keep.py"])


class TestIncremental(_EngineBase):
    def test_only_changed_files_reembedded(self):
        write(self.root, "a.py", "alpha\n")
        write(self.root, "b.py", "beta\n")
        fake, calls = fake_openai()
        self.build(fake)
        self.assertEqual(len(calls["texts"]), 2)

        # change b.py only; add c.py; remove nothing
        calls["texts"].clear()
        write(self.root, "b.py", "beta beta\n")
        write(self.root, "c.py", "gamma\n")
        summary = self.build(fake)
        self.assertEqual(summary["reused"], 1)   # a.py
        self.assertEqual(summary["indexed"], 2)  # b.py + c.py
        self.assertEqual(sorted(calls["texts"]), ["beta beta", "gamma"])

    def test_removed_file_dropped(self):
        write(self.root, "a.py", "alpha\n")
        b = write(self.root, "b.py", "beta\n")
        fake, _ = fake_openai()
        self.build(fake)
        b.unlink()
        summary = self.build(fake)
        self.assertEqual(summary["removed"], 1)
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        self.assertEqual(list(doc["files"]), ["a.py"])

    def test_rebuild_reembeds_all(self):
        write(self.root, "a.py", "alpha\n")
        write(self.root, "b.py", "beta\n")
        fake, calls = fake_openai()
        self.build(fake)
        calls["texts"].clear()
        summary = self.build(fake, rebuild=True)
        self.assertEqual(summary["reused"], 0)
        self.assertEqual(len(calls["texts"]), 2)

    def test_param_drift_requires_rebuild(self):
        write(self.root, "a.py", "alpha\n")
        fake, _ = fake_openai()
        self.build(fake, embed_model="m")
        with self.assertRaises(_index.IndexingError) as cm:
            self.build(fake, embed_model="different-model")
        self.assertEqual(cm.exception.exit_code, 2)
        self.assertIn("rebuild", str(cm.exception))


class TestBuildFailure(_EngineBase):
    def test_sdk_error_persists_completed_prefix(self):
        # Two files; the embeddings call fails on the SECOND batch (batch=1).
        write(self.root, "a.py", "alpha\n")
        write(self.root, "b.py", "beta\n")
        import openai as _oai

        fake = mock.MagicMock()
        seq = {"n": 0}

        def _create(**kwargs):
            seq["n"] += 1
            if seq["n"] == 2:
                raise _oai.APIConnectionError(request=None)
            return _Resp([kwargs["input"]] if isinstance(kwargs["input"], str)
                         else list(kwargs["input"]))

        fake.embeddings.create.side_effect = _create
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(_index.IndexingError) as cm:
                _index.build_index(self.root, embed_base_url="http://local/v1",
                                   embed_model="m", batch=1)
        self.assertEqual(cm.exception.exit_code, 8)  # APIConnectionError -> 8
        # the first file was persisted before the failure -> resumable
        import json
        doc = json.loads(_index.store_file(self.store()).read_text())
        self.assertEqual(len(doc["files"]), 1)

    def test_total_failure_writes_no_empty_store(self):
        # The very first batch fails -> nothing completed -> no stray empty index.
        write(self.root, "a.py", "alpha\n")
        import openai as _oai

        fake = mock.MagicMock()

        def _create(**kwargs):
            raise _oai.APIConnectionError(request=None)

        fake.embeddings.create.side_effect = _create
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(_index.IndexingError) as cm:
                _index.build_index(self.root, embed_base_url="http://local/v1",
                                   embed_model="m", batch=1)
        self.assertEqual(cm.exception.exit_code, 8)
        self.assertFalse(_index.store_file(self.store()).exists())


class TestMissingOpenai(_EngineBase):
    def test_missing_openai_exits_2(self):
        write(self.root, "a.py", "alpha\n")
        with mock.patch("venice.commands._openai.import_openai", return_value=None), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(_index.IndexingError) as cm:
                _index.build_index(self.root, embed_base_url="http://local/v1",
                                   embed_model="m")
        self.assertEqual(cm.exception.exit_code, 2)


class TestIndexCLI(unittest.TestCase):
    def _args(self, root, **ov):
        base = dict(path=root, model=None, embed_base_url="http://local/v1",
                    embed_model="m", embed_ca_bundle=None, embed_insecure=False,
                    dimensions=None, rebuild=False, exclude=None,
                    batch=None, chunk_lines=None, chunk_overlap=None)
        base.update(ov)
        return argparse.Namespace(**base)

    def test_run_prints_store_path(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write(tmp.name, "a.py", "alpha\n")
        from venice.commands import index
        fake, _ = fake_openai()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("venice.userconfig.load_config", return_value=_clean_doc()), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stdout", out), \
             mock.patch.object(sys, "stderr", err):
            rc = index._run(self._args(tmp.name))
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(),
                         str(_index.store_file(_index.store_dir_for_root(tmp.name))))

    # --- #42: TLS override reaches the SDK through CLI env/config layering ---

    def _run_tls(self, *, env=None, doc=None, **arg_ov):
        """Drive index._run against the local backend with httpx.Client patched to
        a sentinel; returns (rc, Hx_mock) so a test can assert the verify value."""
        from venice.commands import index
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write(tmp.name, "a.py", "alpha\n")
        base_env = _no_key_env()
        if env:
            base_env.update(env)
        fake, _ = fake_openai()
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx, \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("venice.userconfig.load_config",
                        return_value=doc or _clean_doc()), \
             mock.patch.dict(os.environ, base_env, clear=True), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = index._run(self._args(tmp.name, **arg_ov))
        return rc, Hx

    def test_ca_bundle_from_flag(self):
        rc, Hx = self._run_tls(embed_ca_bundle="/flag-ca.pem")
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/flag-ca.pem")

    def test_ca_bundle_from_env(self):
        rc, Hx = self._run_tls(env={"VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"})
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/env-ca.pem")

    def test_ca_bundle_from_config(self):
        doc = {"version": 1, "mcpServers": {}, "defaults": {
            "index": {"embed_ca_bundle": "/cfg-ca.pem"}}}
        rc, Hx = self._run_tls(doc=doc)
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/cfg-ca.pem")

    def test_flag_beats_env(self):
        rc, Hx = self._run_tls(embed_ca_bundle="/flag-ca.pem",
                               env={"VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"})
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/flag-ca.pem")


class TestIndexTLS(_EngineBase):
    """#42: TLS escape hatch on the local embedding backend (engine level)."""

    def _build(self, *, stderr=None, **ov):
        """build_index on self.root via the local backend, httpx.Client patched to
        a sentinel. Returns (Hx_mock, OAI_mock); propagates IndexingError."""
        write(self.root, "a.txt", "alpha\n")
        fake, _ = fake_openai()
        kw = dict(embed_base_url="http://local/v1", embed_model="m")
        kw.update(ov)
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx, \
             mock.patch("openai.OpenAI", return_value=fake) as OAI, \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            _index.build_index(self.root, **kw)
        return Hx, OAI

    def test_ca_bundle_reaches_sdk_on_alt_path(self):
        Hx, OAI = self._build(ca_bundle="/ca.pem")
        Hx.assert_called_once_with(verify="/ca.pem")
        self.assertEqual(OAI.call_args.kwargs["http_client"], "httpx-sentinel")

    def test_insecure_disables_verification_and_warns(self):
        err = io.StringIO()
        Hx, OAI = self._build(insecure=True, stderr=err)
        Hx.assert_called_once_with(verify=False)
        self.assertEqual(OAI.call_args.kwargs["http_client"], "httpx-sentinel")
        self.assertIn("TLS verification disabled", err.getvalue())

    def test_no_override_builds_no_http_client(self):
        Hx, OAI = self._build()
        Hx.assert_not_called()
        self.assertNotIn("http_client", OAI.call_args.kwargs)

    def test_insecure_without_base_url_is_rejected(self):
        err = io.StringIO()
        with self.assertRaises(_index.IndexingError) as cm:
            self._build(embed_base_url=None, insecure=True, stderr=err)
        self.assertEqual(cm.exception.exit_code, 2)
        self.assertIn("only apply with --embed-base-url", err.getvalue())

    def test_ca_bundle_without_base_url_is_rejected(self):
        err = io.StringIO()
        with self.assertRaises(_index.IndexingError) as cm:
            self._build(embed_base_url=None, ca_bundle="/ca.pem", stderr=err)
        self.assertEqual(cm.exception.exit_code, 2)
        self.assertIn("only apply with --embed-base-url", err.getvalue())

    def test_ca_bundle_and_insecure_are_mutually_exclusive(self):
        from venice.commands import index
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        index.register(sub)
        with mock.patch.object(sys, "stderr", io.StringIO()), \
             self.assertRaises(SystemExit):
            parser.parse_args(
                ["index", "--embed-ca-bundle", "/ca.pem", "--embed-insecure"])


if __name__ == "__main__":
    unittest.main()
