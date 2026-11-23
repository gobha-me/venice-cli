"""Exercise issue #266 through the real, offline ``venice memory`` CLI.

Fixtures use a disposable Git project, HOME, and global memory directory. The
command and memory store are never mocked; malformed stores are real files.
"""

import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from venice import cli
from venice.commands import _memory as memory_store


class TestMemoryUnhappyPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="venice-memory-unhappy-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.project = self.root / "project"
        self.project.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.global_memory = self.root / "global-memory"
        self.project_notes = self.project / ".venice" / "memory" / "memory.json"
        self.global_notes = self.global_memory / "memory.json"
        self.tasks = self.project_notes.with_name("tasks.json")
        environment = patch.dict(
            os.environ,
            {
                "HOME": str(self.home),
                "XDG_CONFIG_HOME": str(self.home / ".config"),
                "XDG_CACHE_HOME": str(self.root / "cache"),
                "TMPDIR": str(self.root),
                "VENICE_MEMORY_DIR": str(self.global_memory),
                "VENICE_SESSIONS_DIR": str(self.root / "sessions"),
                "VENICE_API_KEY": "test-fake-key",
                "PATH": os.defpath,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)
        # A real nested repo prevents an unrelated ancestor checkout from
        # choosing the fixture's project-memory location.
        subprocess.run(
            ["git", "init", "--quiet", str(self.project)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        original_directory = Path.cwd()
        self.addCleanup(os.chdir, original_directory)
        os.chdir(self.project)

    def invoke(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                result = cli.main(["memory", *arguments])
            except SystemExit as exited:
                result = exited.code
        return result, output.getvalue(), errors.getvalue()

    def snapshot(self):
        """Check store bytes and visible fixture paths without reading Git data."""
        stores = {
            str(path.relative_to(self.root)): (
                path.read_bytes() if path.exists() else None
            )
            for path in (self.project_notes, self.global_notes, self.tasks)
        }
        paths = sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if ".git" not in path.relative_to(self.root).parts
        )
        return stores, paths

    def seed_notes(self):
        for scope, content in (
            ("project", "project note stays intact"),
            ("global", "global note stays intact"),
        ):
            memory_store.write_entry(
                "kept-note", content, scope=scope, start=self.project
            )

    def corrupt_note_store(self, scope):
        path = self.project_notes if scope == "project" else self.global_notes
        path.write_bytes(b"{ not json")
        return path

    @staticmethod
    def malformed_warning(path):
        return (
            f"venice: ignoring malformed memory store {path}: "
            "Expecting property name enclosed in double quotes: "
            "line 1 column 3 (char 2)\n"
        )

    def test_unknown_scope_rejects_each_note_action_without_changes(self):
        self.seed_notes()
        original = self.snapshot()
        for action, arguments, positional in (
            ("ls", [], ""),
            ("show", ["missing-note"], " name"),
            ("rm", ["missing-note"], " name"),
        ):
            with self.subTest(action=action):
                expected = (
                    f"usage: venice memory {action} [-h] "
                    f"[--scope {{project,global}}]{positional}\n"
                    f"venice memory {action}: error: argument --scope: "
                    "invalid choice: 'elsewhere' "
                    "(choose from 'project', 'global')\n"
                )
                self.assertEqual(
                    self.invoke(action, *arguments, "--scope", "elsewhere"),
                    (2, "", expected),
                )
                self.assertEqual(self.snapshot(), original)

    def test_show_missing_note_in_populated_stores_preserves_both_tiers(self):
        self.seed_notes()
        original = self.snapshot()
        for scope in (None, "project", "global"):
            with self.subTest(scope=scope):
                options = [] if scope is None else ["--scope", scope]
                self.assertEqual(
                    self.invoke("show", "missing-note", *options),
                    (1, "", "memory: no note named 'missing-note'\n"),
                )
                self.assertEqual(self.snapshot(), original)

    def test_show_missing_note_does_not_create_an_empty_store(self):
        original = self.snapshot()
        self.assertEqual(
            self.invoke("show", "missing-note"),
            (1, "", "memory: no note named 'missing-note'\n"),
        )
        self.assertEqual(self.snapshot(), original)

    def test_rm_missing_note_in_populated_stores_preserves_both_tiers(self):
        self.seed_notes()
        original = self.snapshot()
        for scope in (None, "project", "global"):
            with self.subTest(scope=scope):
                options = [] if scope is None else ["--scope", scope]
                selected = "project" if scope is None else scope
                self.assertEqual(
                    self.invoke("rm", "missing-note", *options),
                    (1, "", f"memory: no note named 'missing-note' in {selected}\n"),
                )
                self.assertEqual(self.snapshot(), original)

    def test_rm_missing_note_does_not_create_an_empty_store(self):
        original = self.snapshot()
        self.assertEqual(
            self.invoke("rm", "missing-note"),
            (1, "", "memory: no note named 'missing-note' in project\n"),
        )
        self.assertEqual(self.snapshot(), original)

    def test_ls_malformed_note_store_warns_and_preserves_original_bytes(self):
        for scope in ("project", "global"):
            with self.subTest(scope=scope):
                self.seed_notes()
                path = self.corrupt_note_store(scope)
                original = self.snapshot()
                try:
                    expected = self.malformed_warning(path) + (
                        f"no saved memory notes ({scope}). The agent writes them with "
                        "`venice chat --memory` / `venice code --memory`.\n"
                    )
                    self.assertEqual(
                        self.invoke("ls", "--scope", scope), (0, "", expected)
                    )
                    self.assertEqual(self.snapshot(), original)
                finally:
                    # Restore only our fixture, including after a failed subtest.
                    path.unlink(missing_ok=True)

    def test_show_malformed_scope_does_not_read_the_other_tier(self):
        for scope in ("project", "global"):
            with self.subTest(scope=scope):
                self.seed_notes()
                path = self.corrupt_note_store(scope)
                original = self.snapshot()
                try:
                    self.assertEqual(
                        self.invoke("show", "kept-note", "--scope", scope),
                        (
                            1,
                            "",
                            self.malformed_warning(path)
                            + "memory: no note named 'kept-note'\n",
                        ),
                    )
                    self.assertEqual(self.snapshot(), original)
                finally:
                    path.unlink(missing_ok=True)

    def test_rm_malformed_note_store_refuses_to_clobber_either_tier(self):
        for scope in ("project", "global"):
            with self.subTest(scope=scope):
                self.seed_notes()
                path = self.corrupt_note_store(scope)
                original = self.snapshot()
                try:
                    self.assertEqual(
                        self.invoke("rm", "kept-note", "--scope", scope),
                        (
                            1,
                            "",
                            (
                                f"memory: malformed memory store {path}: "
                                "Expecting property name enclosed in double quotes: "
                                "line 1 column 3 (char 2)\n"
                            ),
                        ),
                    )
                    self.assertEqual(self.snapshot(), original)
                finally:
                    path.unlink(missing_ok=True)

    def test_tasks_without_any_store_reports_empty_without_creating_files(self):
        original = self.snapshot()
        for status in (None, "pending", "in_progress", "done"):
            with self.subTest(status=status):
                options = [] if status is None else ["--status", status]
                self.assertEqual(
                    self.invoke("tasks", *options),
                    (0, "", "no tasks in this project.\n"),
                )
                self.assertEqual(self.snapshot(), original)

    def test_tasks_without_task_store_preserves_existing_note_stores(self):
        self.seed_notes()
        self.assertFalse(self.tasks.exists())
        original = self.snapshot()
        self.assertEqual(
            self.invoke("tasks"),
            (0, "", "no tasks in this project.\n"),
        )
        self.assertEqual(self.snapshot(), original)

    def test_unknown_task_status_rejects_without_creating_files(self):
        original = self.snapshot()
        self.assertEqual(
            self.invoke("tasks", "--status", "elsewhere"),
            (
                2,
                "",
                (
                    "usage: venice memory tasks [-h] "
                    "[--status {pending,in_progress,done}]\n"
                    "venice memory tasks: error: argument --status: "
                    "invalid choice: 'elsewhere' "
                    "(choose from 'pending', 'in_progress', 'done')\n"
                ),
            ),
        )
        self.assertEqual(self.snapshot(), original)


if __name__ == "__main__":
    unittest.main()
