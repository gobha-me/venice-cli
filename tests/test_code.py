"""Unit tests for the built-in coding toolset (`venice.commands._code`, #29).

Hermetic: a throwaway project tree under a tmpdir, no network, no real key. The
engine is stdlib-only + mcp-free, so this whole file runs on the 3.9 floor.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _code


def _tools(root, **kw):
    return {t.name: t for t in _code.code_tools(root, **kw)}


_ASSET_NAMES = {
    "venice_image", "venice_image_edit", "venice_sfx", "venice_music",
    "venice_tts", "venice_upscale", "venice_bg_remove", "venice_video",
}


class TestCodeTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)
        self.tools = _tools(self.root, exec_timeout=5)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel, text):
        p = Path(self.root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    # --- read_file ---
    def test_read_file_and_offset_limit(self):
        self._write("a.txt", "l1\nl2\nl3\nl4\n")
        r = self.tools["read_file"].invoke({"path": "a.txt"})
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["content"], "l1\nl2\nl3\nl4")
        self.assertEqual(r["total_lines"], 4)
        r2 = self.tools["read_file"].invoke({"path": "a.txt", "offset": 2, "limit": 2})
        self.assertEqual(r2["content"], "l2\nl3")
        self.assertTrue(r2["truncated"])

    def test_read_missing_and_dir(self):
        self.assertEqual(self.tools["read_file"].invoke({"path": "nope"})["status"], "error")
        os.mkdir(os.path.join(self.root, "d"))
        self.assertIn("directory", self.tools["read_file"].invoke({"path": "d"})["message"])

    def test_read_binary_rejected(self):
        (Path(self.root) / "b.bin").write_bytes(b"\x00\x01\x02ELF")
        r = self.tools["read_file"].invoke({"path": "b.bin"})
        self.assertEqual(r["status"], "error")
        self.assertIn("binary", r["message"])

    def test_read_oversize_rejected(self):
        big = "x" * (_code.MAX_READ_BYTES + 10)
        self._write("big.txt", big)
        r = self.tools["read_file"].invoke({"path": "big.txt"})
        self.assertEqual(r["status"], "error")
        self.assertIn("too large", r["message"])

    # --- sandbox + secret + protected ---
    def test_sandbox_escape_rejected(self):
        for tool, args in [
            ("read_file", {"path": "../../etc/passwd"}),
            ("read_file", {"path": "/etc/passwd"}),
            ("list_dir", {"path": ".."}),
        ]:
            r = self.tools[tool].invoke(args)
            self.assertEqual(r["status"], "error", (tool, args))
            self.assertIn("escape", r["message"])

    def test_secret_file_rejected_even_with_confirm(self):
        self._write("credentials", "secret")
        r = self.tools["read_file"].invoke({"path": "credentials"})
        self.assertEqual(r["status"], "error")
        w = self.tools["write_file"].invoke({"path": "x.pem", "content": "k"}, confirm=True)
        self.assertEqual(w["status"], "error")
        self.assertIn("protected", w["message"])

    def test_protected_dir_rejected(self):
        os.makedirs(os.path.join(self.root, ".git"))
        self._write(".git/config", "[core]\n")
        r = self.tools["read_file"].invoke({"path": ".git/config"})
        self.assertEqual(r["status"], "error")
        self.assertIn("protected", r["message"])

    def test_symlink_escape_rejected(self):
        outside = tempfile.mkdtemp()
        try:
            (Path(outside) / "secret.txt").write_text("boo", encoding="utf-8")
            os.symlink(os.path.join(outside, "secret.txt"),
                       os.path.join(self.root, "link.txt"))
            r = self.tools["read_file"].invoke({"path": "link.txt"})
            self.assertEqual(r["status"], "error")
            self.assertIn("escape", r["message"])
        finally:
            import shutil
            shutil.rmtree(outside, ignore_errors=True)

    # --- list_dir ---
    def test_list_dir_hides_secrets(self):
        self._write("a.py", "x")
        self._write("credentials", "s")
        os.mkdir(os.path.join(self.root, "sub"))
        r = self.tools["list_dir"].invoke({})
        names = {e["name"]: e["type"] for e in r["entries"]}
        self.assertIn("a.py", names)
        self.assertEqual(names["sub"], "dir")
        self.assertNotIn("credentials", names)

    # --- grep ---
    def test_grep_matches_and_glob(self):
        self._write("a.py", "TODO one\nnope\n")
        self._write("b.js", "TODO two\n")
        r = self.tools["grep"].invoke({"pattern": "TODO"})
        self.assertEqual(r["count"], 2)
        r2 = self.tools["grep"].invoke({"pattern": "TODO", "glob": "*.py"})
        self.assertEqual([m["path"] for m in r2["matches"]], ["a.py"])

    def test_grep_bad_regex(self):
        r = self.tools["grep"].invoke({"pattern": "("})
        self.assertEqual(r["status"], "error")
        self.assertIn("regex", r["message"])

    def test_grep_max_matches_truncates(self):
        self._write("many.txt", "\n".join("hit" for _ in range(10)))
        r = self.tools["grep"].invoke({"pattern": "hit", "max_matches": 3})
        self.assertEqual(r["count"], 3)
        self.assertTrue(r["truncated"])

    # --- write_file ---
    def test_write_gate_then_confirm(self):
        gate = self.tools["write_file"].invoke({"path": "n.py", "content": "x=1\n"})
        self.assertEqual(gate["status"], "confirmation_required")
        self.assertFalse(os.path.exists(os.path.join(self.root, "n.py")))
        ok = self.tools["write_file"].invoke({"path": "n.py", "content": "x=1\n"}, confirm=True)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["action"], "created")
        self.assertEqual(Path(self.root, "n.py").read_text(), "x=1\n")

    def test_write_creates_parent_dirs_and_overwrites(self):
        self.tools["write_file"].invoke(
            {"path": "pkg/mod.py", "content": "a\n"}, confirm=True)
        self.assertEqual(Path(self.root, "pkg/mod.py").read_text(), "a\n")
        r = self.tools["write_file"].invoke(
            {"path": "pkg/mod.py", "content": "b\n"}, confirm=True)
        self.assertEqual(r["action"], "overwrote")
        self.assertEqual(Path(self.root, "pkg/mod.py").read_text(), "b\n")

    # --- edit_file ---
    def test_edit_unique_replace(self):
        self._write("e.py", "def f():\n    return 1\n")
        gate = self.tools["edit_file"].invoke(
            {"path": "e.py", "old": "return 1", "new": "return 2"})
        self.assertEqual(gate["status"], "confirmation_required")
        ok = self.tools["edit_file"].invoke(
            {"path": "e.py", "old": "return 1", "new": "return 2"}, confirm=True)
        self.assertEqual(ok["status"], "ok")
        self.assertIn("return 2", Path(self.root, "e.py").read_text())

    def test_edit_not_found_and_not_unique(self):
        self._write("e.py", "x\nx\n")
        self.assertIn("not found", self.tools["edit_file"].invoke(
            {"path": "e.py", "old": "zzz", "new": "y"})["message"])
        self.assertIn("not unique", self.tools["edit_file"].invoke(
            {"path": "e.py", "old": "x", "new": "y"})["message"])

    # --- apply_patch (#63) ---
    def test_apply_patch_gate_then_confirm(self):
        self._write("a.py", "x = 1\n")
        gate = self.tools["apply_patch"].invoke(
            {"patches": [{"path": "a.py", "edits": [{"old": "1", "new": "2"}]}]})
        self.assertEqual(gate["status"], "confirmation_required")
        self.assertEqual(Path(self.root, "a.py").read_text(), "x = 1\n")  # untouched
        ok = self.tools["apply_patch"].invoke(
            {"patches": [{"path": "a.py", "edits": [{"old": "1", "new": "2"}]}]},
            confirm=True)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["total_edits"], 1)
        self.assertEqual(Path(self.root, "a.py").read_text(), "x = 2\n")

    def test_apply_patch_multi_hunk_in_order(self):
        self._write("a.py", "foo\nbar\nbaz\n")
        ok = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [
                {"old": "foo", "new": "FOO"},
                {"old": "baz", "new": "BAZ"},
            ]}]}, confirm=True)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(Path(self.root, "a.py").read_text(), "FOO\nbar\nBAZ\n")

    def test_apply_patch_occurrence_resolves_non_unique(self):
        self._write("a.py", "x\nx\nx\n")
        ok = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [{"old": "x", "new": "MID", "occurrence": 2}]}]},
            confirm=True)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(Path(self.root, "a.py").read_text(), "x\nMID\nx\n")

    def test_apply_patch_non_unique_without_occurrence_errors(self):
        self._write("a.py", "x\nx\n")
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [{"old": "x", "new": "y"}]}]}, confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("not unique", r["message"])
        self.assertIn("occurrence", r["message"])
        self.assertEqual(Path(self.root, "a.py").read_text(), "x\nx\n")  # unchanged

    def test_apply_patch_atomic_per_file_on_failure(self):
        # Second hunk fails -> the whole file is left untouched (no partial edit).
        self._write("a.py", "keep\n")
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [
                {"old": "keep", "new": "CHANGED"},
                {"old": "MISSING", "new": "y"},
            ]}]}, confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("not found", r["message"])
        self.assertEqual(Path(self.root, "a.py").read_text(), "keep\n")

    def test_apply_patch_multiple_files(self):
        self._write("a.py", "a1\n")
        self._write("b.py", "b1\n")
        ok = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [{"old": "a1", "new": "a2"}]},
            {"path": "b.py", "edits": [{"old": "b1", "new": "b2"}]},
        ]}, confirm=True)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["total_edits"], 2)
        self.assertEqual(len(ok["files"]), 2)
        self.assertEqual(Path(self.root, "a.py").read_text(), "a2\n")
        self.assertEqual(Path(self.root, "b.py").read_text(), "b2\n")

    def test_apply_patch_sandbox_and_shape_errors(self):
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "../escape.py", "edits": [{"old": "a", "new": "b"}]}]},
            confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("escapes", r["message"])
        self.assertEqual(self.tools["apply_patch"].invoke(
            {"patches": []})["status"], "error")
        self.assertEqual(self.tools["apply_patch"].invoke(
            {"patches": [{"path": "a.py", "edits": []}]})["status"], "error")

    def test_apply_patch_occurrence_out_of_range(self):
        self._write("a.py", "x\nx\n")
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [{"old": "x", "new": "y", "occurrence": 5}]}]},
            confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("out of range", r["message"])

    def test_apply_patch_missing_file(self):
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "nope.py", "edits": [{"old": "a", "new": "b"}]}]}, confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("no such file", r["message"])

    def test_apply_patch_dry_run_previews_without_writing(self):
        # dry_run reports the per-hunk old->new for every file and writes nothing,
        # without needing confirmation.
        self._write("a.py", "a1\nx\nx\n")
        self._write("b.py", "b1\n")
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [
                {"old": "a1", "new": "A1"},
                {"old": "x", "new": "MID", "occurrence": 2},
            ]},
            {"path": "b.py", "edits": [{"old": "b1", "new": "B1"}]},
        ], "dry_run": True})  # note: no confirm=True
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["action"], "dry_run")
        self.assertEqual(r["total_edits"], 3)
        self.assertEqual(len(r["files"]), 2)
        first = r["files"][0]
        self.assertEqual(first["path"], "a.py")
        self.assertEqual(first["edits"][0], {"index": 0, "old": "a1", "new": "A1"})
        self.assertEqual(first["edits"][1],
                         {"index": 1, "old": "x", "new": "MID", "occurrence": 2})
        # Nothing written to disk.
        self.assertEqual(Path(self.root, "a.py").read_text(), "a1\nx\nx\n")
        self.assertEqual(Path(self.root, "b.py").read_text(), "b1\n")

    def test_apply_patch_dry_run_reports_validation_error(self):
        # dry_run still validates -- a non-matching hunk is an error, not a preview.
        self._write("a.py", "keep\n")
        r = self.tools["apply_patch"].invoke({"patches": [
            {"path": "a.py", "edits": [{"old": "MISSING", "new": "y"}]}],
            "dry_run": True})
        self.assertEqual(r["status"], "error")
        self.assertIn("not found", r["message"])
        self.assertEqual(Path(self.root, "a.py").read_text(), "keep\n")

    def test_apply_patch_cross_file_atomic_on_write_failure(self):
        # If a later file fails to stage, no earlier file is committed -- the batch
        # is all-or-nothing across files (#67), not just per file.
        self._write("a.py", "a1\n")
        self._write("b.py", "b1\n")
        real_stage = _code._stage_write
        calls = {"n": 0}

        def flaky_stage(target, text):
            calls["n"] += 1
            if calls["n"] == 2:  # second file fails to write
                raise OSError("disk full (simulated)")
            return real_stage(target, text)

        with mock.patch.object(_code, "_stage_write", flaky_stage):
            r = self.tools["apply_patch"].invoke({"patches": [
                {"path": "a.py", "edits": [{"old": "a1", "new": "a2"}]},
                {"path": "b.py", "edits": [{"old": "b1", "new": "b2"}]},
            ]}, confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("write failed", r["message"])
        # Both files unchanged -- file 1 was staged but never committed.
        self.assertEqual(Path(self.root, "a.py").read_text(), "a1\n")
        self.assertEqual(Path(self.root, "b.py").read_text(), "b1\n")
        # No temp files left behind.
        leftover = [p.name for p in Path(self.root).iterdir()
                    if p.name.endswith(".venice-tmp")]
        self.assertEqual(leftover, [])

    # --- run ---
    def test_run_gate_then_exec_cwd_and_scrub(self):
        gate = self.tools["run"].invoke({"command": "echo hi"})
        self.assertEqual(gate["status"], "confirmation_required")
        self.assertIn("echo hi", gate["message"])
        os.environ["VENICE_API_KEY"] = "test-fake-key"
        try:
            r = self.tools["run"].invoke(
                {"command": "pwd; echo key=[${VENICE_API_KEY:-EMPTY}]"}, confirm=True)
        finally:
            os.environ.pop("VENICE_API_KEY", None)
        self.assertEqual(r["exit_code"], 0)
        self.assertIn(self.root, r["stdout"])          # cwd forced to root
        self.assertIn("key=[EMPTY]", r["stdout"])       # Venice key scrubbed

    def test_run_timeout(self):
        r = self.tools["run"].invoke({"command": "sleep 5"}, confirm=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("timed out", r["message"])

    def test_run_nonzero_exit(self):
        r = self.tools["run"].invoke({"command": "exit 3"}, confirm=True)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["exit_code"], 3)

    # --- git ---
    def test_git_readonly_ok_mutation_refused(self):
        r = self.tools["git"].invoke({"subcommand": "status"})
        # git may or may not be a repo here; either way it's not the guard error
        self.assertIn(r["status"], ("ok", "error"))
        bad = self.tools["git"].invoke({"subcommand": "commit", "args": ["-m", "x"]})
        self.assertEqual(bad["status"], "error")
        self.assertIn("read-only", bad["message"])


class TestCodeFactory(unittest.TestCase):
    def test_paid_flags(self):
        by = {t.name: t for t in _code.code_tools("/tmp")}
        self.assertFalse(by["read_file"].paid)
        self.assertFalse(by["grep"].paid)
        self.assertFalse(by["git"].paid)
        self.assertTrue(by["write_file"].paid)
        self.assertTrue(by["edit_file"].paid)
        self.assertTrue(by["run"].paid)

    def test_schemas_exclude_control_kwargs(self):
        for t in _code.code_tools("/tmp"):
            props = t.parameters.get("properties", {})
            for banned in ("confirm", "max_spend", "output_dir"):
                self.assertNotIn(banned, props, f"{t.name} leaks {banned}")

    def test_project_search_absent_without_index(self):
        names = {t.name for t in _code.code_tools("/tmp", client=object(),
                                                  include_search=True)}
        self.assertNotIn("project_search", names)  # no .venice index discoverable

    def test_assets_absent_by_default(self):
        names = {t.name for t in _code.code_tools("/tmp", client=object())}
        self.assertEqual(names & _ASSET_NAMES, set())

    def test_assets_need_a_client(self):
        # the flag alone (no client) folds nothing in
        names = {t.name for t in _code.code_tools("/tmp", assets=True)}
        self.assertEqual(names & _ASSET_NAMES, set())

    def test_assets_present_when_enabled(self):
        names = {t.name for t in _code.code_tools("/tmp", client=object(),
                                                  assets=True)}
        self.assertTrue(_ASSET_NAMES <= names)   # all 8 folded in
        self.assertNotIn("venice_chat", names)   # excluded by design
        self.assertIn("venice_video", names)

    def test_models_tool_present_with_client(self):
        by = {t.name: t for t in _code.code_tools("/tmp", client=object())}
        self.assertIn("venice_models", by)   # free model-catalog lookup for the agent
        self.assertIn("venice_model_details", by)   # cost/context-limit lookup
        self.assertIn("venice_vision", by)   # the agent's eyes (#60)
        self.assertIn("venice_job_status", by)   # #62 async render poll
        self.assertIn("venice_job_result", by)   # #62 async render fetch
        self.assertFalse(by["venice_models"].paid)   # read-only, not spend-gated
        self.assertFalse(by["venice_model_details"].paid)
        self.assertFalse(by["venice_vision"].paid)
        self.assertFalse(by["venice_job_status"].paid)   # charged at queue time
        self.assertFalse(by["venice_job_result"].paid)

    def test_models_tool_absent_without_client(self):
        names = {t.name for t in _code.code_tools("/tmp")}
        self.assertNotIn("venice_models", names)   # needs a client for the /models GET
        self.assertNotIn("venice_model_details", names)
        self.assertNotIn("venice_vision", names)
        self.assertNotIn("venice_job_status", names)
        self.assertNotIn("venice_job_result", names)

    def test_asset_tools_are_paid(self):
        by = {t.name: t for t in _code.code_tools("/tmp", client=object(),
                                                  assets=True)}
        for n in _ASSET_NAMES:
            self.assertTrue(by[n].paid, f"{n} should be paid")

    def test_asset_schemas_exclude_control_kwargs(self):
        for t in _code.code_tools("/tmp", client=object(), assets=True):
            props = t.parameters.get("properties", {})
            for banned in ("confirm", "max_spend", "output_dir"):
                self.assertNotIn(banned, props, f"{t.name} leaks {banned}")

    def test_controlled_kwargs_stripped_from_model_args(self):
        # a model that smuggles confirm=True must not self-approve a paid tool
        with tempfile.TemporaryDirectory() as d:
            tools = {t.name: t for t in _code.code_tools(os.path.realpath(d))}
            r = tools["write_file"].invoke(
                {"path": "x.py", "content": "1", "confirm": True})
            self.assertEqual(r["status"], "confirmation_required")


if __name__ == "__main__":
    unittest.main()
