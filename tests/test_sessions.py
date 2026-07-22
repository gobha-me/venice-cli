"""Tests for the session store + `venice sessions` CLI (#47).

Redirects the store to a throwaway dir via ``$VENICE_SESSIONS_DIR`` (which also
exercises the env override) so nothing touches ~/.config/venice/sessions.
"""
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice import cli
from venice.commands import _session as S
from venice.commands._agent import CostLedger


def _capture(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
    return rc, out.getvalue(), err.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zone = Path(self.tmp.name) / "sessions"
        env = mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": str(self.zone)})
        env.start()
        self.addCleanup(env.stop)


class TestStore(_Base):
    def _mk(self, command="chat", **ov):
        kw = dict(model="m-1", gen_kwargs={"temperature": 0.5}, max_tool_calls=8,
                  messages=[{"role": "user", "content": "hi"}])
        kw.update(ov)
        return S.new_session(command, **kw)

    def test_save_roundtrip_and_modes(self):
        sess = self._mk(gen_kwargs={
            "temperature": 0.7,
            "extra_body": {"venice_parameters": {"enable_web_search": "on"}},
        })
        sess.usage = {"total": 1.5, "prompt_tokens": 20, "cache_read_tokens": 4}
        path = S.save(sess)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(path.parent).st_mode), 0o700)

        r = S.load(sess.id, "chat")
        self.assertEqual(r.id, sess.id)              # in-place keeps the id
        self.assertEqual(r.model, "m-1")
        self.assertEqual(r.max_tool_calls, 8)
        self.assertEqual(r.gen_kwargs["temperature"], 0.7)
        self.assertEqual(
            r.gen_kwargs["extra_body"]["venice_parameters"]["enable_web_search"], "on")
        self.assertEqual(r.usage["cache_read_tokens"], 4)
        self.assertEqual([m["content"] for m in r.messages], ["hi"])

    def test_envelope_has_version_and_no_key(self):
        sess = self._mk()
        path = S.save(sess)
        doc = json.loads(path.read_text())
        self.assertEqual(doc["venice_session"], S.SESSION_VERSION)
        # Hygiene: the envelope carries only messages/settings/usage, never a key.
        self.assertNotIn("VENICE_API_KEY", path.read_text())
        self.assertNotIn("api_key", doc)

    def test_new_id_sortable(self):
        a = S.new_id()
        b = S.new_id()
        self.assertNotEqual(a, b)
        self.assertRegex(a, r"^\d{8}T\d{6}-[0-9a-f]{6}$")

    def test_load_bare_list_mints_new_id_and_leaves_file(self):
        f = Path(self.tmp.name) / "old.json"
        payload = [{"role": "system", "content": "brief"},
                   {"role": "user", "content": "hi"}]
        f.write_text(json.dumps(payload))
        before = f.read_bytes()
        s = S.load(str(f), "chat")
        self.assertTrue(s.id)
        self.assertEqual(s.command, "chat")           # command from caller
        self.assertEqual(len(s.messages), 2)
        self.assertEqual(f.read_bytes(), before)       # original untouched

    def test_load_envelope_file_mints_new_id(self):
        sess = self._mk(model="mm")
        path = S.save(sess)
        imported = S.load(str(path), "code")
        self.assertNotEqual(imported.id, sess.id)      # imported -> fresh identity
        self.assertEqual(imported.model, "mm")

    def test_load_unknown_id_raises(self):
        with self.assertRaises(S.SessionError):
            S.load("nope", "chat")

    def test_load_traversal_rejected(self):
        with self.assertRaises(S.SessionError):
            S.load("../credentials", "chat")

    def test_load_missing_path_reports_transcript(self):
        with self.assertRaises(S.SessionError) as cm:
            S.load("/no/such/file.json", "chat")
        self.assertIn("transcript", str(cm.exception))

    def test_load_malformed_dict_reports_list(self):
        f = Path(self.tmp.name) / "bad.json"
        f.write_text('{"not": "a list"}')
        with self.assertRaises(S.SessionError) as cm:
            S.load(str(f), "chat")
        self.assertIn("list of message objects", str(cm.exception))

    def test_most_recent_filters_by_command(self):
        c = self._mk("chat"); S.save(c)
        k = self._mk("code"); S.save(k)
        self.assertEqual(S.most_recent("chat").id, c.id)
        self.assertEqual(S.most_recent("code").id, k.id)

    def test_most_recent_none(self):
        self.assertIsNone(S.most_recent("chat"))

    def test_list_and_delete(self):
        a = self._mk(); S.save(a)
        b = self._mk(); S.save(b)
        rows = S.list_sessions()
        self.assertEqual(len(rows), 2)
        # newest first (b saved last)
        self.assertEqual(rows[0][0], b.id)
        self.assertTrue(S.delete(a.id))
        self.assertFalse(S.delete(a.id))               # already gone
        self.assertEqual(len(S.list_sessions()), 1)

    def test_apply_to_args_only_fills_none(self):
        sess = self._mk(model="saved-m",
                        gen_kwargs={"temperature": 0.9, "max_tokens": 128},
                        max_tool_calls=17)
        ns = mock.Mock(model=None, temperature=0.1, max_tokens=None, max_tool_calls=None)
        S.apply_to_args(ns, sess, "chat")
        self.assertEqual(ns.model, "saved-m")          # was None -> restored
        self.assertEqual(ns.temperature, 0.1)          # explicit -> kept
        self.assertEqual(ns.max_tokens, 128)           # was None -> restored
        self.assertEqual(ns.max_tool_calls, 17)

    def test_merge_gen_kwargs_deep_merges_vp(self):
        saved = {"temperature": 0.5,
                 "extra_body": {"venice_parameters": {"enable_web_search": "on",
                                                      "character_slug": "x"}}}
        fresh = {"extra_body": {"venice_parameters": {"character_slug": "y"}}}
        merged = S.merge_gen_kwargs(saved, fresh)
        vp = merged["extra_body"]["venice_parameters"]
        self.assertEqual(vp["enable_web_search"], "on")   # from saved
        self.assertEqual(vp["character_slug"], "y")       # fresh wins
        self.assertEqual(merged["temperature"], 0.5)

    def test_resolve_from_args(self):
        c = self._mk("chat"); S.save(c)
        # --resume <id>
        ns = mock.Mock(resume=c.id, cont=None)
        self.assertEqual(S.resolve_from_args(ns, "chat").id, c.id)
        # --continue
        ns2 = mock.Mock(resume=None, cont=True)
        self.assertEqual(S.resolve_from_args(ns2, "chat").id, c.id)
        # neither
        ns3 = mock.Mock(resume=None, cont=None)
        self.assertIsNone(S.resolve_from_args(ns3, "chat"))
        # empty --continue for a fresh command
        ns4 = mock.Mock(resume=None, cont=True)
        with self.assertRaises(S.SessionError):
            S.resolve_from_args(ns4, "code")


class TestLedgerSnapshot(unittest.TestCase):
    def test_to_dict_restore_roundtrip(self):
        led = CostLedger()
        led.bind_pricing({"input": {"usd": 1.0}, "output": {"usd": 2.0}})
        led.record({"prompt_tokens": 100, "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 30}})
        snap = led.to_dict()
        self.assertEqual(snap["cache_read_tokens"], 30)
        # A fresh ledger seeded from the snapshot carries the totals forward.
        led2 = CostLedger()
        led2.restore(snap)
        self.assertEqual(led2.prompt_tokens, 100)
        self.assertEqual(led2.cache_read_tokens, 30)
        self.assertAlmostEqual(led2.total, led.total)

    def test_restore_tolerates_partial_and_raw_names(self):
        led = CostLedger()
        led.restore({"cached_tokens": 5, "cache_creation_input_tokens": 2})
        self.assertEqual(led.cache_read_tokens, 5)      # raw #75 name accepted
        self.assertEqual(led.cache_write_tokens, 2)
        led.restore({})                                 # empty -> no crash
        led.restore("garbage")                          # non-dict -> no crash


class TestSessionsCLI(_Base):
    def _mk_saved(self, command="chat", model="m-1"):
        s = S.new_session(command, model=model,
                          messages=[{"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "yo"}])
        S.save(s)
        return s

    def test_ls_empty(self):
        rc, out, err = _capture(cli.main, ["sessions", "ls"])
        self.assertEqual(rc, 0)
        self.assertIn("no saved sessions", err)

    def test_ls_lists(self):
        s = self._mk_saved()
        rc, out, err = _capture(cli.main, ["sessions", "ls"])
        self.assertEqual(rc, 0)
        self.assertIn(s.id, out)
        self.assertIn("[chat]", out)
        self.assertIn("m-1", out)

    def test_show(self):
        s = self._mk_saved()
        rc, out, err = _capture(cli.main, ["sessions", "show", s.id])
        self.assertEqual(rc, 0)
        self.assertIn(s.id, out)
        self.assertIn("messages: 2", out)

    def test_show_missing(self):
        rc, out, err = _capture(cli.main, ["sessions", "show", "nope"])
        self.assertEqual(rc, 1)

    def test_rm(self):
        s = self._mk_saved()
        rc, out, err = _capture(cli.main, ["sessions", "rm", s.id])
        self.assertEqual(rc, 0)
        self.assertIsNone(S.most_recent("chat"))
        rc, out, err = _capture(cli.main, ["sessions", "rm", s.id])
        self.assertEqual(rc, 1)                          # already gone

    def test_bare_sessions_prints_help(self):
        rc, out, err = _capture(cli.main, ["sessions"])
        self.assertEqual(rc, 2)
        self.assertIn("ACTION", err)


if __name__ == "__main__":
    unittest.main()
