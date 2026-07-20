"""Unit tests for `venice search`, the search engine, and the `project_search`
agent tool. Reuses the deterministic fake backend from tests.test_index (marker-
word vectors) so ranking is checkable. Local backend -> no Venice key/catalog.
"""
import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _index
from tests.test_index import fake_openai, write, _clean_doc, _no_key_env


def build(root, fake, **ov):
    kw = dict(embed_base_url="http://local/v1", embed_model="m")
    kw.update(ov)
    with mock.patch("openai.OpenAI", return_value=fake), \
         mock.patch.dict(os.environ, _no_key_env(), clear=True), \
         mock.patch.object(sys, "stderr", io.StringIO()):
        return _index.build_index(root, **kw)


class _Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        write(self.root, "alpha_file.py", "alpha alpha alpha\ncode here\n")
        write(self.root, "beta_file.py", "beta beta\nmore code\n")
        write(self.root, "gamma_file.py", "gamma\nstuff\n")
        fake, _ = fake_openai()
        build(self.root, fake)
        self.store = _index.store_dir_for_root(self.root)

    def search(self, query, **ov):
        fake, calls = fake_openai()
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            res = _index.search_index(self.store, query, **ov)
        return res, calls


class TestSearchEngine(_Base):
    def test_ranks_by_marker(self):
        res, _ = self.search("alpha")
        self.assertEqual(res[0]["path"], "alpha_file.py")
        self.assertEqual(res[0]["start"], 1)
        # scores are monotonically non-increasing
        scores = [r["score"] for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k(self):
        res, _ = self.search("beta", k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["path"], "beta_file.py")

    def test_query_uses_stored_model(self):
        _, calls = self.search("gamma")
        self.assertEqual(calls["kwargs"][0]["model"], "m")  # meta model, not re-resolved

    def test_preview_and_changed_flag(self):
        res, _ = self.search("alpha")
        hit = res[0]
        self.assertIn("alpha", hit["preview"])
        self.assertFalse(hit["changed"])
        # mutate the file on disk -> next search flags it changed
        write(self.root, "alpha_file.py", "alpha alpha alpha\nEDITED\n")
        res2, _ = self.search("alpha")
        self.assertTrue(res2[0]["changed"])

    def test_missing_index_raises_6(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(_index.IndexingError) as cm:
                _index.search_index(_index.store_dir_for_root(empty), "x")
        self.assertEqual(cm.exception.exit_code, 6)

    def test_empty_query_raises_2(self):
        with self.assertRaises(_index.IndexingError) as cm:
            self.search("   ")
        self.assertEqual(cm.exception.exit_code, 2)


class TestDiscovery(_Base):
    def test_walk_up_from_subdir(self):
        sub = Path(self.root) / "src" / "deep"
        sub.mkdir(parents=True)
        found = _index.discover_store(None, start=sub)
        self.assertEqual(found, self.store)

    def test_explicit_root_or_store_dir(self):
        self.assertEqual(_index.discover_store(self.root), self.store)
        self.assertEqual(_index.discover_store(str(self.store)), self.store)

    def test_env_override(self):
        with mock.patch.dict(os.environ, {_index.config.ENV_INDEX_DIR: str(self.store)}):
            self.assertEqual(_index.discover_store(None), self.store)

    def test_not_found(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(_index.discover_store(None, start=Path(empty)))


class TestSearchCLI(_Base):
    def _args(self, **ov):
        base = dict(query="alpha", index_path=self.root, top_k=None, json=False,
                    embed_ca_bundle=None, embed_insecure=False)
        base.update(ov)
        return argparse.Namespace(**base)

    def _run(self, args):
        from venice.commands import search
        fake, _ = fake_openai()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("venice.userconfig.load_config", return_value=_clean_doc()), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stdout", out), \
             mock.patch.object(sys, "stderr", err):
            rc = search._run(args)
        return rc, out.getvalue(), err.getvalue()

    def test_text_output(self):
        rc, out, _ = self._run(self._args(top_k=1))
        self.assertEqual(rc, 0)
        self.assertIn("alpha_file.py:1-2", out)

    def test_json_output(self):
        rc, out, _ = self._run(self._args(json=True))
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertEqual(doc["query"], "alpha")
        self.assertEqual(doc["results"][0]["path"], "alpha_file.py")

    def test_no_index_exits_6(self):
        with tempfile.TemporaryDirectory() as empty:
            rc, out, err = self._run(self._args(index_path=empty))
        self.assertEqual(rc, 6)
        self.assertIn("no index", err)


class TestSearchTool(_Base):
    def _in_root(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.root)

    def test_returns_ok_dict(self):
        self._in_root()
        fake, _ = fake_openai()
        with mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.dict(os.environ, _no_key_env(), clear=True), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            from venice.commands import _mcp
            res = _mcp.search_tool(object(), "alpha", k=2)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["results"][0]["path"], "alpha_file.py")

    def test_no_index_returns_error_dict(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        with tempfile.TemporaryDirectory() as empty:
            os.chdir(empty)
            from venice.commands import _mcp
            with mock.patch.object(sys, "stderr", io.StringIO()):
                res = _mcp.search_tool(object(), "alpha")
        self.assertEqual(res["status"], "error")
        self.assertIn("venice index", res["message"])

    def test_empty_query_error(self):
        from venice.commands import _mcp
        res = _mcp.search_tool(object(), "  ")
        self.assertEqual(res["status"], "error")


class TestSearchTLS(_Base):
    """#42: TLS override supplied at query time reaches the SDK even though the
    backend base_url comes from stored index meta (a local-backed index)."""

    def _search_tls(self, *, env=None, stderr=None, **ov):
        """search_index against the local-backed store with httpx.Client patched
        to a sentinel; returns (Hx_mock, OAI_mock)."""
        fake, _ = fake_openai()
        base_env = _no_key_env()
        if env:
            base_env.update(env)
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx, \
             mock.patch("openai.OpenAI", return_value=fake) as OAI, \
             mock.patch.dict(os.environ, base_env, clear=True), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            _index.search_index(self.store, "alpha", **ov)
        return Hx, OAI

    def test_ca_bundle_reaches_sdk_at_query_time(self):
        Hx, OAI = self._search_tls(ca_bundle="/ca.pem")
        Hx.assert_called_once_with(verify="/ca.pem")
        self.assertEqual(OAI.call_args.kwargs["http_client"], "httpx-sentinel")

    def test_insecure_disables_verification_and_warns(self):
        err = io.StringIO()
        Hx, _ = self._search_tls(insecure=True, stderr=err)
        Hx.assert_called_once_with(verify=False)
        self.assertIn("TLS verification disabled", err.getvalue())

    def test_no_override_builds_no_http_client(self):
        Hx, OAI = self._search_tls()
        Hx.assert_not_called()
        self.assertNotIn("http_client", OAI.call_args.kwargs)

    def test_env_ca_bundle_picked_up_when_no_arg(self):
        # The no-CLI project_search path: search_index falls back to the env var so
        # the agent tool also reaches a self-signed embedder.
        Hx, _ = self._search_tls(env={"VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"})
        Hx.assert_called_once_with(verify="/env-ca.pem")

    def test_explicit_ca_bundle_beats_env(self):
        Hx, _ = self._search_tls(ca_bundle="/arg-ca.pem",
                                 env={"VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"})
        Hx.assert_called_once_with(verify="/arg-ca.pem")

    def test_project_search_tool_uses_env_ca_bundle(self):
        # End-to-end for the agent path: _mcp.search_tool -> search_index picks up
        # the env CA bundle. Discovery is cwd-based, so chdir into the indexed root.
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.root)
        fake, _ = fake_openai()
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx, \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.dict(os.environ, {**_no_key_env(),
                                          "VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"},
                             clear=True), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            from venice.commands import _mcp
            res = _mcp.search_tool(object(), "alpha", k=1)
        self.assertEqual(res["status"], "ok")
        Hx.assert_called_once_with(verify="/env-ca.pem")


class TestBackendFromMetaGate(unittest.TestCase):
    """#42: on a Venice-built index the TLS flags don't apply. Explicit flags are
    rejected (exit 2); a globally-set env CA bundle is ignored, never tripping the
    gate -- so project_search over a Venice index keeps working."""

    def test_explicit_flags_rejected_on_venice_index(self):
        import openai
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            oai, model, rc = _index._backend_from_meta(
                openai, {"backend": "venice", "model": "m"}, ca_bundle="/ca.pem")
        self.assertEqual(rc, 2)
        self.assertIsNone(oai)
        self.assertIn("only apply to a local-backend index", err.getvalue())

    def test_env_ca_bundle_does_not_trip_venice_gate(self):
        import openai
        from venice import auth
        err = io.StringIO()
        # env set + no explicit flag: the venice path never consults env, so the
        # TLS gate is NOT tripped -- we fall through to auth (stubbed to fail).
        with mock.patch.dict(os.environ, {**_no_key_env(),
                                          "VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"},
                             clear=True), \
             mock.patch("venice.commands._index.build_client_from_auth",
                        side_effect=auth.AuthError("no key")), \
             mock.patch.object(sys, "stderr", err):
            oai, model, rc = _index._backend_from_meta(
                openai, {"backend": "venice", "model": "m"})
        self.assertEqual(rc, 2)
        self.assertNotIn("only apply to a local-backend index", err.getvalue())
        self.assertIn("no key", err.getvalue())


if __name__ == "__main__":
    unittest.main()
