"""Tests for the `venice_web_search` rail (#77).

Covers the core `_agent.run_web_search` completion helper + model resolution, the
`_code.web_search_tool` Tool factory (metadata + invoke envelopes), the "docs scout"
integration (scout carries the read-only web tool), the worker-exclusion guarantee
(category `web` is never granted to a spawn worker), the rail invariants (not in
`_REGISTRY` / not a `code_tools` default), and the `venice code --web-search` wiring.

Reuses the fake OpenAI helpers from `tests.test_chat`.
"""
import argparse
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_chat import FakeToolCompletion, _fake_openai_seq, _urlopen_ok
from venice.commands import _agent, _code


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _catalog(*, coder_ws=None, extra=None, pricing=None):
    """A text catalog with a coding model `coder-x`.

    `coder_ws` sets `coder-x`'s supportsWebSearch (None => the field is absent).
    `pricing` attaches a `model_spec.pricing` block. `extra` appends more models.
    """
    caps = {"supportsFunctionCalling": True}
    if coder_ws is not None:
        caps["supportsWebSearch"] = coder_ws
    spec = {"capabilities": caps}
    if pricing is not None:
        spec["pricing"] = pricing
    data = [{"id": "coder-x", "type": "text", "model_spec": spec}]
    if extra:
        data.extend(extra)
    return data


def _resp(answer="the answer", citations=None, usage=None):
    vp = {"web_search_citations": citations} if citations is not None else None
    return FakeToolCompletion(answer, venice_parameters=vp, usage=usage)


# --------------------------------------------------------------------------- #
# _agent.run_web_search
# --------------------------------------------------------------------------- #
class TestRunWebSearch(unittest.TestCase):
    def test_request_carries_web_search_venice_parameters(self):
        fake, calls = _fake_openai_seq([_resp()])
        out = _agent.run_web_search(fake, "coder-x", "how do I X?")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["model"], "coder-x")
        self.assertEqual(calls[0]["model"], "coder-x")
        self.assertEqual(calls[0]["messages"][-1]["content"], "how do I X?")
        vp = calls[0]["extra_body"]["venice_parameters"]
        self.assertEqual(vp["enable_web_search"], "on")
        self.assertTrue(vp["enable_web_citations"])

    def test_mode_auto_passthrough(self):
        fake, calls = _fake_openai_seq([_resp()])
        _agent.run_web_search(fake, "coder-x", "q", mode="auto")
        vp = calls[0]["extra_body"]["venice_parameters"]
        self.assertEqual(vp["enable_web_search"], "auto")

    def test_parses_answer_and_citations(self):
        cites = [
            {"title": "T", "url": "http://e/x", "date": "2026-01-01", "content": "body"},
            {"title": "no url dropped"},          # url-less -> skipped
            {"url": "http://e/y"},                 # title defaults to ""
        ]
        fake, _ = _fake_openai_seq([_resp("hello", cites)])
        out = _agent.run_web_search(fake, "coder-x", "q")
        self.assertEqual(out["answer"], "hello")
        self.assertEqual(out["citations"], [
            {"title": "T", "url": "http://e/x", "date": "2026-01-01"},  # content dropped
            {"title": "", "url": "http://e/y"},
        ])

    def test_absent_citations_is_empty_list(self):
        fake, _ = _fake_openai_seq([_resp("hi", citations=None)])
        self.assertEqual(_agent.run_web_search(fake, "coder-x", "q")["citations"], [])

    def test_empty_query_makes_no_completion(self):
        fake, calls = _fake_openai_seq([_resp()])
        out = _agent.run_web_search(fake, "coder-x", "   ")
        self.assertEqual(out["status"], "error")
        self.assertEqual(calls, [])  # never hit the API

    def test_cost_none_when_pricing_unknown(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        fake, _ = _fake_openai_seq([_resp(usage=usage)])
        out = _agent.run_web_search(fake, "coder-x", "q", models=_catalog())
        self.assertIsNone(out["cost_estimate_usd"])  # catalog has no pricing block

    def test_cost_none_when_usage_absent(self):
        # Priced catalog but the response carried no usage block -> honest None, not $0.00.
        priced = _catalog(pricing={"input": {"usd": 2.0}, "output": {"usd": 6.0}})
        fake, _ = _fake_openai_seq([_resp(usage=None)])
        out = _agent.run_web_search(fake, "coder-x", "q", models=priced)
        self.assertIsNone(out["cost_estimate_usd"])

    def test_cost_estimated_when_priced(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        priced = _catalog(pricing={"input": {"usd": 2.0}, "output": {"usd": 6.0}})
        fake, _ = _fake_openai_seq([_resp(usage=usage)])
        out = _agent.run_web_search(fake, "coder-x", "q", models=priced)
        self.assertAlmostEqual(out["cost_estimate_usd"], 2.0)  # 1M input @ $2/1M


# --------------------------------------------------------------------------- #
# _agent.resolve_web_search_model
# --------------------------------------------------------------------------- #
class TestResolveWebSearchModel(unittest.TestCase):
    def test_explicit_override_wins(self):
        self.assertEqual(
            _agent.resolve_web_search_model(_catalog(coder_ws=False), "picked", "coder-x"),
            "picked")

    def test_coding_model_when_capable(self):
        self.assertEqual(
            _agent.resolve_web_search_model(_catalog(coder_ws=True), None, "coder-x"),
            "coder-x")

    def test_coding_model_when_capability_unknown(self):
        # No supportsWebSearch field (None) -> attempt anyway with the coder.
        self.assertEqual(
            _agent.resolve_web_search_model(_catalog(coder_ws=None), None, "coder-x"),
            "coder-x")

    def test_auto_picks_capable_catalog_model_when_coder_incapable(self):
        extra = [{"id": "searcher", "type": "text",
                  "model_spec": {"capabilities": {"supportsWebSearch": True}}}]
        got = _agent.resolve_web_search_model(
            _catalog(coder_ws=False, extra=extra), None, "coder-x")
        self.assertEqual(got, "searcher")

    def test_none_when_no_capable_model(self):
        self.assertIsNone(
            _agent.resolve_web_search_model(_catalog(coder_ws=False), None, "coder-x"))


# --------------------------------------------------------------------------- #
# _code.web_search_tool
# --------------------------------------------------------------------------- #
class TestWebSearchTool(unittest.TestCase):
    def test_metadata(self):
        t = _code.web_search_tool(None, "coder-x", models=_catalog(coder_ws=True))
        self.assertEqual(t.name, _agent.WEB_SEARCH_TOOL_NAME)
        self.assertEqual(t.name, "venice_web_search")
        self.assertFalse(t.paid)             # required so a scout can carry it
        self.assertEqual(t.category, "web")  # same rail category as browser tools
        self.assertIn("read", t.tags)
        self.assertIn("network", t.tags)

    def test_ok_envelope(self):
        fake, _ = _fake_openai_seq([_resp("ans", [{"url": "http://e/a"}])])
        t = _code.web_search_tool(fake, "coder-x", models=_catalog(coder_ws=True))
        out = t.invoke({"query": "find docs"})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["answer"], "ans")
        self.assertEqual(out["citations"], [{"title": "", "url": "http://e/a"}])

    def test_empty_query_errors_without_calling_api(self):
        fake, calls = _fake_openai_seq([_resp()])
        t = _code.web_search_tool(fake, "coder-x", models=_catalog(coder_ws=True))
        out = t.invoke({"query": "  "})
        self.assertEqual(out["status"], "error")
        self.assertEqual(calls, [])

    def test_no_capable_model_errors(self):
        # coder incapable + no other capable model -> resolved None -> actionable error.
        t = _code.web_search_tool(None, "coder-x", models=_catalog(coder_ws=False))
        out = t.invoke({"query": "q"})
        self.assertEqual(out["status"], "error")
        self.assertIn("web-search-capable", out["message"])

    def test_nested_exception_becomes_error_envelope(self):
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("boom")
        t = _code.web_search_tool(fake, "coder-x", models=_catalog(coder_ws=True))
        out = t.invoke({"query": "q"})
        self.assertEqual(out["status"], "error")
        self.assertIn("web_search failed", out["message"])

    def test_search_model_override_used(self):
        fake, calls = _fake_openai_seq([_resp()])
        t = _code.web_search_tool(fake, "coder-x", models=_catalog(coder_ws=False),
                                  search_model="my-search-model")
        t.invoke({"query": "q"})
        self.assertEqual(calls[0]["model"], "my-search-model")


# --------------------------------------------------------------------------- #
# Rail invariants: not in the registry, not a code_tools default
# --------------------------------------------------------------------------- #
class TestWebSearchRailInvariants(unittest.TestCase):
    def test_not_in_registry(self):
        names = {s.name for s in _agent._REGISTRY}
        self.assertNotIn(_agent.WEB_SEARCH_TOOL_NAME, names)
        self.assertEqual(len(_agent._REGISTRY), 16)  # unchanged

    def test_web_category_not_in_registry_taxonomy(self):
        self.assertNotIn("web", _agent.list_categories())

    def test_not_a_code_tools_rail(self):
        tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        names = {t.name for t in _code.code_tools(tmp, None)}
        self.assertNotIn(_agent.WEB_SEARCH_TOOL_NAME, names)


# --------------------------------------------------------------------------- #
# Docs scout: the scout carries the read-only web tool
# --------------------------------------------------------------------------- #
class TestDocsScout(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        with open(os.path.join(self.tmp, "a.py"), "w") as fh:
            fh.write("x = 1\n")

    def test_scout_inner_set_includes_web_tool(self):
        ws = _code.web_search_tool(None, "coder-x", models=_catalog(coder_ws=True))
        seen = {}

        def _rec(oai, model, task, tools, base_kwargs, **kw):
            seen["names"] = {t.name for t in tools}
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        with mock.patch.object(_agent, "run_scout", _rec):
            _code.scout_tool(None, "coder-x", self.tmp, None, {},
                             web_tool=ws).invoke({"task": "find the docs"})
        self.assertIn(_agent.WEB_SEARCH_TOOL_NAME, seen["names"])
        self.assertIn("read_file", seen["names"])  # still read-only base tools

    def test_run_scout_accepts_the_free_web_tool(self):
        # The real guard: run_scout rejects paid tools; the web tool is paid=False,
        # so a docs scout is valid (no ValueError).
        ws = _code.web_search_tool(None, "coder-x", models=_catalog(coder_ws=True))
        fake, _ = _fake_openai_seq([FakeToolCompletion("FINDINGS: ok")])
        out = _code.scout_tool(fake, "coder-x", self.tmp, None, {},
                               web_tool=ws).invoke({"task": "t"})
        self.assertEqual(out["status"], "ok")

    def test_no_web_tool_scout_omits_it(self):
        seen = {}

        def _rec(oai, model, task, tools, base_kwargs, **kw):
            seen["names"] = {t.name for t in tools}
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        with mock.patch.object(_agent, "run_scout", _rec):
            _code.scout_tool(None, "coder-x", self.tmp, None, {}).invoke({"task": "t"})
        self.assertNotIn(_agent.WEB_SEARCH_TOOL_NAME, seen["names"])


# --------------------------------------------------------------------------- #
# Worker exclusion: a spawn worker never gets the web tool
# --------------------------------------------------------------------------- #
class TestWorkerExcludesWeb(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_web_tool_never_granted_to_a_code_worker(self):
        ws = _code.web_search_tool(None, "coder-x", models=_catalog(coder_ws=True))
        parent = list(_code.code_tools(self.tmp, None)) + [ws]
        rec = {}

        def _rec(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            rec["names"] = {t.name for t in tools}
            rec["cats"] = {t.category for t in tools}
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        with mock.patch.object(_agent, "run_spawn", _rec):
            _code.spawn_tool(None, "coder-x", {}, parent).invoke(
                {"task": "x", "role": "code"})
        self.assertNotIn(_agent.WEB_SEARCH_TOOL_NAME, rec["names"])
        self.assertNotIn("web", rec["cats"])

    def test_web_tool_not_granted_even_if_web_category_requested(self):
        ws = _code.web_search_tool(None, "coder-x", models=_catalog(coder_ws=True))
        parent = list(_code.code_tools(self.tmp, None)) + [ws]
        rec = {}

        def _rec(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            rec["names"] = {t.name for t in tools}
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        with mock.patch.object(_agent, "run_spawn", _rec):
            _code.spawn_tool(None, "coder-x", {}, parent).invoke(
                {"task": "x", "categories": ["fs", "web"]})
        self.assertNotIn(_agent.WEB_SEARCH_TOOL_NAME, rec["names"])


# --------------------------------------------------------------------------- #
# `venice code --web-search` wiring
# --------------------------------------------------------------------------- #
def _code_args(**ov):
    base = dict(
        task=None, root=None, model=None, system=None, temperature=None,
        max_tokens=None, json=False, auto=None, manual=None, yes=None,
        plan_only=False, no_plan=False, no_verify=False, max_tool_calls=None,
        exec_timeout=None, interactive=False, resume=None, assets=None,
        scout=None, spawn=None, spawn_max_spend=None, planner=None, memory=None,
        auto_compact=None, compact_threshold=None, compact_keep_turns=None,
        session_max_spend=None, cont=None, ephemeral=None,
        web_search=None, web_search_model=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class TestWebSearchWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def _run(self, args, seq, config=None):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False
        doc = {"version": 1, "mcpServers": {}, "defaults": config or {}}
        cfg = mock.patch("venice.userconfig.load_config", lambda *a, **k: doc)
        cfg.start()
        self.addCleanup(cfg.stop)
        sess = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(sess, ignore_errors=True))
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": sess}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = code._run(args)
        return rc, calls

    _PLAN = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]

    def test_flag_advertises_venice_web_search(self):
        rc, calls = self._run(
            _code_args(task="do x", root=self.tmp, plan_only=True, web_search=True),
            list(self._PLAN))
        self.assertEqual(rc, 0)
        self.assertIn(_agent.WEB_SEARCH_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_no_flag_omits_the_tool(self):
        rc, calls = self._run(
            _code_args(task="do x", root=self.tmp, plan_only=True), list(self._PLAN))
        self.assertEqual(rc, 0)
        self.assertNotIn(_agent.WEB_SEARCH_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_config_default_enables_the_rail(self):
        rc, calls = self._run(
            _code_args(task="do x", root=self.tmp, plan_only=True), list(self._PLAN),
            config={"code": {"web_search": True}})
        self.assertEqual(rc, 0)
        self.assertIn(_agent.WEB_SEARCH_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_web_search_with_scout_makes_a_docs_scout(self):
        # Both the parent web tool AND the scout are advertised; the scout carries the
        # web tool in its inner set (checked structurally in TestDocsScout).
        rc, calls = self._run(
            _code_args(task="do x", root=self.tmp, plan_only=True,
                       web_search=True, scout=True), list(self._PLAN))
        self.assertEqual(rc, 0)
        system_msg = calls[0]["messages"][0]["content"]
        self.assertIn(_agent.WEB_SEARCH_TOOL_NAME, system_msg)
        self.assertIn(_agent.SCOUT_TOOL_NAME, system_msg)


if __name__ == "__main__":
    unittest.main()
