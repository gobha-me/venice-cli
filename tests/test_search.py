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
        base = dict(query="alpha", index_path=self.root, top_k=None, json=False)
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


if __name__ == "__main__":
    unittest.main()
