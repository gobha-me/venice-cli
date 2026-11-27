"""Offline tests for the on-demand tool-fit harness (#280)."""
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tool_fit import run as tool_fit


TASKS_YAML = """\
- id: images
  description: Export product images in batches.
  success_hint: Save each view with a transparent background.
- id: documents
  description: Find the design archive.
  success_hint: Identify a searchable project folder.
- id: code
  description: Rename a shared helper.
  success_hint: Update every caller.
"""


def answer(can_do, *, missing=(), time_savers=()):
    return {
        "can_do": can_do,
        "missing_tools": [{"key": key, "summary": summary} for key, summary in missing],
        "time_savers": [{"key": key, "summary": summary} for key, summary in time_savers],
    }


class TestToolFitHarness(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.tasks = Path(workspace.name) / "tasks.yaml"
        self.tasks.write_text(TASKS_YAML, encoding="utf-8")
        self.output = Path(workspace.name) / "reports"
        self.answers = {
            "code": answer("yes"),
            "images": answer(
                "no",
                missing=[("batch-image-export", "Export each view in one request.")],
                time_savers=[("offline-preview", "Inspect the output without uploading it.")],
            ),
            "documents": answer(
                "limp",
                missing=[("archive-reader", "Read the original design files.")],
                time_savers=[
                    ("batch-image-export", "Export archived views together."),
                    ("project-index", "Find the archive in the project index."),
                ],
            ),
        }
        self.content = None
        self.issues = []
        self.commands = []
        self.failure = None

    def run_command(self, command, *, input=None, **kwargs):
        self.commands.append((command, input))
        program = Path(command[0]).name
        if program == self.failure:
            raise subprocess.CalledProcessError(17, command)
        if program == "gh" and command[1] == "api":
            body = "\n".join(json.dumps(issue) for issue in self.issues)
        elif program == "venice" and command[1] == "chat":
            body = json.dumps({
                "final": self.content if self.content is not None else json.dumps(self.answers),
                "model": "test-model",
                "tools": ["read_file", "mcp_docs_search"],
                "cost_upper_bound_usd": 0.20,
                "usage": {"prompt_tokens": 200, "completion_tokens": 50},
            })
        else:
            self.fail(f"unexpected subprocess command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=body)

    def run_harness(self, *options, tasks=True):
        args = ["--output-dir", str(self.output), "--max-spend", "0.25"]
        if tasks:
            args.extend(["--tasks", str(self.tasks)])
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(tool_fit.subprocess, "run", side_effect=self.run_command), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = tool_fit.main([*args, *options])
        return status, out.getvalue().strip(), err.getvalue()

    def read_report(self, *options, tasks=True):
        status, out, err = self.run_harness(*options, tasks=tasks)
        self.assertEqual(status, 0, err)
        path = Path(out)
        self.assertEqual(path.parent, self.output)
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        return path, report

    def test_writes_reports_and_one_local_draft_per_new_gap(self):
        self.issues = [
            {"title": "Access the archive", "body": "<!-- tool-fit-gap: archive-reader -->"},
            {"title": "[TOOL-GAP] Offline Preview", "body": None},
        ]
        path, report = self.read_report(
            "--model", "test-model", "--max-tokens", "128", "--mcp", "docs", "--mcp", "search"
        )

        self.assertEqual(len(self.commands), 2)
        (issues, _), (chat, prompt) = self.commands
        self.assertIn("--paginate", issues)
        self.assertTrue(any("state=open" in arg for arg in issues))
        self.assertNotIn("--method", issues)
        self.assertIn("--tool-fit", chat)
        self.assertIn("--json", chat)
        for flag, value in (("--max-spend", "0.25"), ("--max-tokens", "128"),
                            ("--model", "test-model")):
            self.assertEqual(chat[chat.index(flag) + 1], value)
        self.assertEqual([chat[i + 1] for i, arg in enumerate(chat) if arg == "--mcp"],
                         ["docs", "search"])
        self.assertEqual(chat[-1], "-")
        self.assertIn("save time", prompt)
        self.assertIn("Save each view with a transparent background.", prompt)

        self.assertEqual(report["model"], "test-model")
        self.assertEqual(report["tools"], ["read_file", "mcp_docs_search"])
        self.assertEqual(report["max_spend_usd"], 0.25)
        self.assertEqual(report["cost_upper_bound_usd"], 0.20)
        self.assertEqual(report["usage"]["completion_tokens"], 50)
        self.assertEqual([(row["id"], row["can_do"]) for row in report["results"]],
                         [("images", "no"), ("documents", "limp"), ("code", "yes")])
        self.assertEqual(report["results"][0]["success_hint"],
                         "Save each view with a transparent background.")
        self.assertEqual(report["results"][1]["time_savers"], self.answers["documents"]["time_savers"])

        drafts = report["draft_issues"]
        self.assertCountEqual([draft["key"] for draft in drafts],
                              ["batch-image-export", "project-index"])
        self.assertCountEqual([file.name for file in (path / "drafts").iterdir()],
                              ["batch-image-export.md", "project-index.md"])
        for draft in drafts:
            self.assertEqual(draft["labels"], ["tool-gap"])
            local = (path / "drafts" / (draft["key"] + ".md")).read_text(encoding="utf-8")
            self.assertIn('labels: ["tool-gap"]', local)
            self.assertIn(draft["body"], local)
            self.assertTrue(local.rstrip().endswith("Agent: Codex (gpt-6.2-sol)"))
        export = next(draft for draft in drafts if draft["key"] == "batch-image-export")
        self.assertIn("Export each view in one request.", export["body"])
        self.assertIn("Export archived views together.", export["body"])
        markdown = (path / "report.md").read_text(encoding="utf-8")
        self.assertIn("Save each view with a transparent background.", markdown)
        self.assertIn("Find the archive in the project index.", markdown)
        self.assertIn("drafts/batch-image-export.md", markdown)

    def test_existing_issue_with_same_draft_title_suppresses_new_draft(self):
        self.issues = [{"title": "Tool gap: batch image export", "body": None}]

        path, report = self.read_report()

        self.assertNotIn("batch-image-export", [draft["key"] for draft in report["draft_issues"]])
        self.assertFalse((path / "drafts" / "batch-image-export.md").exists())

    def test_default_seed_tasks_cover_code_assets_mcp_and_search(self):
        tasks = tool_fit.load_tasks(tool_fit.HERE / "tasks.yaml")
        self.answers = {task["id"]: answer("yes") for task in tasks}

        path, report = self.read_report(tasks=False)

        self.assertEqual(len(report["results"]), 10)
        self.assertEqual({row["id"].split("-", 1)[0] for row in report["results"]},
                         {"code", "asset", "mcp", "search"})
        self.assertTrue(all(row["description"] and row["success_hint"] for row in report["results"]))
        self.assertEqual(report["draft_issues"], [])
        self.assertFalse((path / "drafts").exists())
        self.assertIn("--no-mcp", self.commands[1][0])

    def test_invalid_task_lists_stop_before_subprocesses(self):
        cases = {
            "empty": "[]",
            "malformed": "- id: [\n",
            "duplicate": TASKS_YAML + TASKS_YAML,
            "missing hint": "- id: images\n  description: Export product images.\n",
        }
        for name, bad_tasks in cases.items():
            with self.subTest(name=name):
                self.tasks.write_text(bad_tasks, encoding="utf-8")
                status, out, err = self.run_harness()
                self.assertEqual(status, 1)
                self.assertEqual(out, "")
                self.assertIn("tool-fit:", err)
                self.assertEqual(self.commands, [])
                self.assertFalse(self.output.exists())

    def test_invalid_budget_is_rejected_before_subprocesses(self):
        for options in (("--max-spend", "0"), ("--max-spend", "nan"),
                        ("--max-spend", "inf"), ("--max-tokens", "0")):
            with self.subTest(options=options), self.assertRaises(SystemExit) as caught:
                self.run_harness(*options)
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.commands, [])
        self.assertFalse(self.output.exists())

    def test_invalid_model_answers_do_not_publish_reports(self):
        valid = json.dumps(self.answers)
        cases = {
            "truncated JSON": valid[:-1],
            "missing task": json.dumps({"code": self.answers["code"], "images": self.answers["images"]}),
            "duplicate task": '{"code": {}, ' + valid[1:],
            "invalid status": json.dumps({**self.answers, "code": answer("maybe")}),
            "invalid capability key": json.dumps({
                **self.answers, "code": answer("no", missing=[("../outside", "Missing capability.")]),
            }),
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                self.content = content
                status, out, err = self.run_harness()
                self.assertEqual(status, 1)
                self.assertEqual(out, "")
                self.assertIn("tool-fit:", err)
                self.assertEqual(list(self.output.iterdir()), [])

    def test_failed_subprocess_does_not_publish_reports_or_retry(self):
        for program, expected_calls in (("gh", ["gh"]), ("venice", ["gh", "venice"])):
            with self.subTest(program=program):
                self.failure = program
                self.commands.clear()
                status, out, err = self.run_harness()
                self.assertEqual(status, 1)
                self.assertEqual(out, "")
                self.assertIn(program, err)
                self.assertEqual([Path(command[0]).name for command, _ in self.commands],
                                 expected_calls)
                self.assertEqual(list(self.output.iterdir()), [])
