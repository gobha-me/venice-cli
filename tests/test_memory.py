"""Tests for the memory + task store, the free tools, and the `venice memory` CLI (#49).

Redirects the global tier to a throwaway dir via ``$VENICE_MEMORY_DIR`` (which also
exercises the env override) and chdirs into a throwaway project so the project tier
(``<cwd>/.venice/memory``) never touches a real repo or ~/.config.
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
from venice.commands import _agent, _mcp
from venice.commands import _memory as M


def _capture(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
        rc = fn(*args)
    return rc, out.getvalue(), err.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.glob = root / "global"
        self.proj = root / "project"
        self.proj.mkdir()
        env = mock.patch.dict(os.environ, {"VENICE_MEMORY_DIR": str(self.glob)})
        env.start()
        self.addCleanup(env.stop)
        # Project tier resolves from cwd -- drive tools/CLI from inside the tmp project.
        self._cwd = os.getcwd()
        os.chdir(self.proj)
        self.addCleanup(lambda: os.chdir(self._cwd))


# --------------------------------------------------------------------------- #
# Store (direct _memory calls; project tier pinned via start= for hermeticity)
# --------------------------------------------------------------------------- #
class TestStore(_Base):
    def test_write_read_roundtrip_and_modes(self):
        meta = M.write_entry("style", "use tabs", scope="project", start=self.proj,
                             type="feedback", description="indent")
        self.assertEqual(meta["scope"], "project")
        self.assertEqual(meta["type"], "feedback")
        self.assertNotIn("content", meta)  # write returns metadata, not the body

        path = self.proj / ".venice" / "memory" / "memory.json"
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(path.parent).st_mode), 0o700)
        # project tier is local-by-default (.venice/.gitignore ignores it)
        self.assertTrue((self.proj / ".venice" / ".gitignore").exists())

        got = M.read_entry("style", start=self.proj)
        self.assertEqual(got["content"], "use tabs")
        self.assertEqual(got["scope"], "project")

    def test_envelope_version_and_shape(self):
        M.write_entry("a", "x", start=self.proj)
        doc = json.loads((self.proj / ".venice" / "memory" / "memory.json").read_text())
        self.assertEqual(doc["venice_memory"], M.MEMORY_VERSION)
        self.assertIn("a", doc["entries"])

    def test_global_tier_uses_env_dir_and_no_gitignore(self):
        M.write_entry("g", "world", scope="global", start=self.proj)
        gpath = self.glob / "memory.json"
        self.assertTrue(gpath.exists())
        self.assertEqual(stat.S_IMODE(os.stat(gpath).st_mode), 0o600)
        # global lives under ~/.config/venice-like dir, not .venice -> no gitignore drop
        self.assertFalse((self.glob / ".gitignore").exists())

    def test_two_tier_isolation(self):
        M.write_entry("only-proj", "p", scope="project", start=self.proj)
        M.write_entry("only-glob", "g", scope="global", start=self.proj)
        self.assertIsNone(M.read_entry("only-glob", scope="project", start=self.proj))
        self.assertIsNone(M.read_entry("only-proj", scope="global", start=self.proj))
        # default read tries project then global
        self.assertEqual(M.read_entry("only-glob", start=self.proj)["scope"], "global")

    def test_overwrite_preserves_created(self):
        first = M.write_entry("k", "v1", start=self.proj)
        second = M.write_entry("k", "v2", start=self.proj)
        self.assertEqual(first["created"], second["created"])
        self.assertEqual(M.read_entry("k", start=self.proj)["content"], "v2")

    def test_name_validation_rejects_unsafe_and_secret(self):
        for bad in ["../x", "a/b", ".", "..", "", "has space",
                    "credentials", "id_rsa", "x.key", "mysecrets", ".env"]:
            with self.assertRaises(M.MemStoreError):
                M.write_entry(bad, "x", start=self.proj)

    def test_content_cap(self):
        with self.assertRaises(M.MemStoreError):
            M.write_entry("big", "x" * (M.MAX_CONTENT_CHARS + 1), start=self.proj)

    def test_list_is_metadata_only_and_both_tiers(self):
        M.write_entry("p", "pbody", scope="project", start=self.proj)
        M.write_entry("g", "gbody", scope="global", start=self.proj)
        rows = M.list_entries(start=self.proj)
        self.assertEqual({(r["name"], r["scope"]) for r in rows},
                         {("p", "project"), ("g", "global")})
        for r in rows:
            self.assertNotIn("content", r)

    def test_search_both_tiers_with_preview(self):
        M.write_entry("p", "the quick brown fox", scope="project", start=self.proj)
        M.write_entry("g", "lazy dog sleeps", scope="global", start=self.proj)
        hits = M.search_entries("quick", start=self.proj)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["scope"], "project")
        self.assertIn("quick", hits[0]["preview"])
        # description + name are searchable too
        self.assertEqual(len(M.search_entries("dog", start=self.proj)), 1)
        with self.assertRaises(M.MemStoreError):
            M.search_entries("   ", start=self.proj)

    def test_delete(self):
        M.write_entry("d", "x", start=self.proj)
        self.assertTrue(M.delete_entry("d", start=self.proj))
        self.assertFalse(M.delete_entry("d", start=self.proj))

    def test_unknown_scope(self):
        with self.assertRaises(M.MemStoreError):
            M.write_entry("x", "y", scope="nope", start=self.proj)

    def test_malformed_store_read_is_tolerant_write_is_strict(self):
        path = self.proj / ".venice" / "memory" / "memory.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ not json")
        # tolerant read -> empty (warns to stderr), swallowed here
        with mock.patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(M.list_entries(scope="project", start=self.proj), [])
        # strict write refuses to clobber a corrupt store
        with self.assertRaises(M.MemStoreError):
            M.write_entry("x", "y", scope="project", start=self.proj)


class TestTasks(_Base):
    def test_add_update_list_filter(self):
        t1 = M.add_task("ship", start=self.proj)
        t2 = M.add_task("docs", start=self.proj)
        self.assertEqual((t1["id"], t1["status"]), ("1", "pending"))
        self.assertEqual(t2["id"], "2")  # monotonic ids
        upd = M.update_task("1", status="in_progress", start=self.proj)
        self.assertEqual(upd["status"], "in_progress")
        self.assertEqual([t["id"] for t in M.list_tasks(start=self.proj)], ["1", "2"])
        self.assertEqual(
            [t["id"] for t in M.list_tasks(status="in_progress", start=self.proj)], ["1"])

    def test_tasks_are_project_only_at_the_expected_path(self):
        M.add_task("x", start=self.proj)
        self.assertTrue((self.proj / ".venice" / "memory" / "tasks.json").exists())

    def test_update_text(self):
        M.add_task("old", start=self.proj)
        self.assertEqual(M.update_task("1", text="new", start=self.proj)["text"], "new")

    def test_rejects(self):
        with self.assertRaises(M.MemStoreError):      # empty text
            M.add_task("  ", start=self.proj)
        M.add_task("t", start=self.proj)
        with self.assertRaises(M.MemStoreError):      # unknown id
            M.update_task("99", status="done", start=self.proj)
        with self.assertRaises(M.MemStoreError):      # bad status
            M.update_task("1", status="bogus", start=self.proj)
        with self.assertRaises(M.MemStoreError):      # nothing to update
            M.update_task("1", start=self.proj)
        with self.assertRaises(M.MemStoreError):      # bad filter
            M.list_tasks(status="bogus", start=self.proj)


# --------------------------------------------------------------------------- #
# Free tools (called directly, like test_search.py's TestSearchTool)
# --------------------------------------------------------------------------- #
class TestTools(_Base):
    def test_memory_write_read_search_list(self):
        w = _mcp.memory_write_tool(object(), name="conv", content="use tabs",
                                   scope="project", type="feedback")
        self.assertEqual(w["status"], "ok")
        self.assertEqual(w["scope"], "project")

        r = _mcp.memory_read_tool(object(), name="conv")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["content"], "use tabs")

        miss = _mcp.memory_read_tool(object(), name="nope")
        self.assertEqual(miss["status"], "error")

        s = _mcp.memory_search_tool(object(), query="tabs")
        self.assertEqual(s["status"], "ok")
        self.assertEqual(s["count"], 1)
        self.assertEqual(s["results"][0]["scope"], "project")

        li = _mcp.memory_list_tool(object())
        self.assertEqual(li["count"], 1)
        self.assertNotIn("content", li["entries"][0])

    def test_memory_write_rejects_secret_name(self):
        res = _mcp.memory_write_tool(object(), name="credentials", content="x")
        self.assertEqual(res["status"], "error")
        self.assertIn("secret-shaped", res["message"])

    def test_task_tools_nest_under_task_keys(self):
        add = _mcp.task_add_tool(object(), text="do it")
        self.assertEqual(add["status"], "ok")            # envelope status
        self.assertEqual(add["task"]["status"], "pending")  # task's own status (nested)

        upd = _mcp.task_update_tool(object(), id="1", status="done")
        self.assertEqual(upd["status"], "ok")
        self.assertEqual(upd["task"]["status"], "done")

        bad = _mcp.task_update_tool(object(), id="99", status="done")
        self.assertEqual(bad["status"], "error")

        lst = _mcp.task_list_tool(object())
        self.assertEqual(lst["status"], "ok")
        self.assertEqual(lst["tasks"][0]["status"], "done")


class TestBuiltinWiring(_Base):
    def test_memory_tools_shape(self):
        tools = _agent.memory_tools()
        self.assertEqual(len(tools), 7)
        self.assertEqual({t.category for t in tools}, {"memory", "tasks"})
        self.assertTrue(all(not t.paid for t in tools))
        names = {t.name for t in tools}
        self.assertEqual(names, {"memory_write", "memory_read", "memory_search",
                                 "memory_list", "task_add", "task_update", "task_list"})

    def test_builtin_tools_gate(self):
        base = _agent.builtin_tools(object())
        withmem = _agent.builtin_tools(object(), memory=True)
        self.assertEqual(len(withmem) - len(base), 7)
        # opt-out default keeps memory tools out of the advertised set
        self.assertNotIn("memory_write", {t.name for t in base})

    def test_registry_pins_untouched(self):
        # memory/tasks are rails, NOT registry rows -> the #50 taxonomy is unchanged.
        self.assertEqual(len(_agent._REGISTRY), 16)
        self.assertNotIn("memory", _agent.list_categories())
        self.assertNotIn("tasks", _agent.list_categories())

    def test_invoke_closures_end_to_end(self):
        tools = {t.name: t for t in _agent.memory_tools()}
        self.assertEqual(
            tools["memory_write"].invoke(
                {"name": "x", "content": "hello"})["status"], "ok")
        self.assertEqual(
            tools["memory_search"].invoke({"query": "hello"})["count"], 1)


# --------------------------------------------------------------------------- #
# CLI (drives cli.main like test_sessions.py's TestSessionsCLI)
# --------------------------------------------------------------------------- #
class TestMemoryCLI(_Base):
    def test_ls_empty(self):
        rc, out, err = _capture(cli.main, ["memory", "ls"])
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("no saved memory notes", err)

    def test_ls_show_rm_flow(self):
        M.write_entry("style", "use tabs", scope="project", start=self.proj,
                     type="feedback", description="indent")
        rc, out, _ = _capture(cli.main, ["memory", "ls"])
        self.assertEqual(rc, 0)
        self.assertIn("style", out)
        self.assertIn("[project/feedback]", out)

        rc, out, _ = _capture(cli.main, ["memory", "show", "style"])
        self.assertEqual(rc, 0)
        self.assertIn("use tabs", out)

        rc, _, err = _capture(cli.main, ["memory", "show", "nope"])
        self.assertEqual(rc, 1)

        rc, _, err = _capture(cli.main, ["memory", "rm", "style"])
        self.assertEqual(rc, 0)
        rc, _, err = _capture(cli.main, ["memory", "rm", "style"])
        self.assertEqual(rc, 1)

    def test_scope_filter(self):
        M.write_entry("gnote", "x", scope="global", start=self.proj)
        M.write_entry("pnote", "y", scope="project", start=self.proj)
        # global filter shows only the global note
        rc, out, _ = _capture(cli.main, ["memory", "ls", "--scope", "global"])
        self.assertIn("gnote", out)
        self.assertNotIn("pnote", out)
        # project filter shows only the project note (out is non-empty, so this
        # actually exercises the exclusion rather than passing vacuously)
        rc, out, _ = _capture(cli.main, ["memory", "ls", "--scope", "project"])
        self.assertIn("pnote", out)
        self.assertNotIn("gnote", out)

    def test_tasks(self):
        M.add_task("ship", start=self.proj)
        M.add_task("docs", start=self.proj)
        M.update_task("2", status="in_progress", start=self.proj)
        rc, out, _ = _capture(cli.main, ["memory", "tasks"])
        self.assertEqual(rc, 0)
        self.assertIn("ship", out)
        self.assertIn("[in_progress]", out)
        rc, out, _ = _capture(cli.main, ["memory", "tasks", "--status", "in_progress"])
        self.assertIn("docs", out)
        self.assertNotIn("ship", out)

    def test_bare_prints_help_rc2(self):
        rc, out, err = _capture(cli.main, ["memory"])
        self.assertEqual(rc, 2)
        self.assertIn("ACTION", err)


# --------------------------------------------------------------------------- #
# #52 --parallel: the in-process store lock serializes the read-modify-write so
# concurrent subagents can't drop one another's change.
# --------------------------------------------------------------------------- #
class TestStoreConcurrency(_Base):
    def test_concurrent_add_task_loses_nothing(self):
        import threading
        n = 40
        errors = []

        def add(i):
            try:
                M.add_task(f"unit {i}", start=self.proj)
            except Exception as e:  # pragma: no cover - would signal a lock bug
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        tasks = M.list_tasks(start=self.proj)
        self.assertEqual(len(tasks), n)                      # no lost write
        self.assertEqual(len({t["id"] for t in tasks}), n)   # unique ids (racy next_id)

    def test_concurrent_write_entry_loses_nothing(self):
        import threading
        n = 40

        def put(i):
            M.write_entry(f"k{i}", f"v{i}", start=self.proj)

        threads = [threading.Thread(target=put, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rows = M.list_entries(scope="project", start=self.proj)
        self.assertEqual(len(rows), n)                        # every entry survived
        # And the file is still valid JSON (an interleaved _save would corrupt it).
        path = self.proj / ".venice" / "memory" / "memory.json"
        json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
