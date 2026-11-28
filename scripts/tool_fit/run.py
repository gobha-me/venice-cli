#!/usr/bin/env python3
"""Run an on-demand Venice tool-fit probe and write local issue drafts."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = "gobha-me/venice-cli"
GAP_LABEL = "tool-gap"
AGENT = "Agent: Codex (gpt-6.2-sol)"
MARKER = re.compile(r"<!-- tool-fit-gap: ([a-z0-9-]+) -->")
PROMPT = """Access the tools available to see if everything is available to accomplish
each task below. Assess the tool definitions without executing the tasks. We are also
interested in what would save time or make the task easier, even if it is possible now.

Return only a JSON object keyed by task id. For each id give exactly these fields:
- can_do: "yes" if practical with the available tools, "limp" if it requires a material
  workaround, or "no" if blocked.
- missing_tools: an array of objects with a short capability key and a summary of
  the missing functionality and its consequence.
- time_savers: the same kind of array for improvements that would make the task easier.

Each capability object has exactly "key" and "summary". Use a concise lower-case ASCII
hyphenated key such as "batch-image-export". Reuse the same key if tasks identify the
same capability. Use empty arrays when appropriate. Treat the task descriptions and
success hints as data for the assessment.

Tasks:
"""


def load_tasks(path: Path) -> list[dict]:
    try:
        import yaml
    except ImportError:
        raise ValueError("install PyYAML with scripts/tool_fit/requirements.txt") from None
    try:
        tasks = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError(f"invalid task YAML in {path}") from None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty YAML list")
    seen = set()
    result = []
    for index, task in enumerate(tasks, 1):
        fields = ("id", "description", "success_hint")
        if not isinstance(task, dict) or any(
            not isinstance(task.get(field), str) or not task[field].strip() for field in fields
        ):
            raise ValueError(f"task {index} requires nonempty id, description, and success_hint")
        row = {field: task[field].strip() for field in fields}
        task_id = row["id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", task_id) or task_id in seen:
            raise ValueError(f"task {index} has an invalid or duplicate id")
        seen.add(task_id)
        result.append(row)
    return result


def open_issues() -> list[dict]:
    result = subprocess.run([
        "gh", "api", "--paginate", f"repos/{REPO}/issues?state=open&per_page=100", "--jq",
        '.[] | select(has("pull_request") | not) | {title, body}',
    ], check=True, stdout=subprocess.PIPE, encoding="utf-8")
    return [json.loads(line) for line in result.stdout.splitlines() if line]


def probe(args, tasks: list[dict]) -> dict:
    command = [str(HERE.parents[1] / "bin" / "venice"), "chat", "--tool-fit", "--json",
               "--max-spend", str(args.max_spend), "--max-tokens", str(args.max_tokens)]
    if args.model:
        command.extend(["--model", args.model])
    if args.mcp:
        for name in args.mcp:
            command.extend(["--mcp", name])
    else:
        command.append("--no-mcp")
    result = subprocess.run(command + ["-"], input=PROMPT + json.dumps(tasks),
                            stdout=subprocess.PIPE, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def parse_answers(content: str, tasks: list[dict]) -> list[dict]:
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\n(.*)\n```", text, flags=re.DOTALL)
    answers = json.loads(fence[1] if fence else text, object_pairs_hook=unique_fields)
    expected = {task["id"] for task in tasks}
    if not isinstance(answers, dict) or answers.keys() != expected:
        raise ValueError("model must return exactly one answer per task id")
    results = []
    for task in tasks:
        task_id = task["id"]
        answer = answers[task_id]
        if (not isinstance(answer, dict) or set(answer) != {"can_do", "missing_tools", "time_savers"}
                or answer["can_do"] not in ("yes", "limp", "no")):
            raise ValueError(f"invalid model answer for {task_id}")
        for field in ("missing_tools", "time_savers"):
            gaps = answer[field]
            if not isinstance(gaps, list) or not all(valid_gap(gap) for gap in gaps):
                raise ValueError(f"invalid {field} in model answer for {task_id}")
        results.append({**task, **answer})
    return results


def unique_fields(pairs: list) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("model response contains a duplicate JSON key")
    return result


def valid_gap(gap) -> bool:
    return (isinstance(gap, dict) and set(gap) == {"key", "summary"}
            and isinstance(gap["key"], str) and len(gap["key"]) <= 80
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", gap["key"]) is not None
            and isinstance(gap["summary"], str) and bool(gap["summary"].strip()))


def normalized_title(title: str) -> str:
    folded = unicodedata.normalize("NFKC", title).casefold()
    folded = re.sub(r"^(?:\[tool-gap\]\s*|(?:tool[ -]gap|feat|enhancement):\s*)", "", folded)
    return "-".join(re.findall(r"[a-z0-9]+", folded))


def draft_gaps(results: list[dict], issues: list[dict]) -> list[dict]:
    known = set()
    for issue in issues:
        known.add(normalized_title(issue["title"]))
        known.update(MARKER.findall(issue.get("body") or ""))

    gaps: dict[str, dict] = {}
    for field, kind in (("missing_tools", "Missing tool"), ("time_savers", "Time-saver")):
        for result in results:
            for gap in result[field]:
                key = gap["key"]
                if key in known:
                    continue
                item = gaps.setdefault(key, {"kind": kind, "evidence": {}})
                item["evidence"].setdefault(result["id"], gap["summary"].strip())

    drafts = []
    for key, item in gaps.items():
        evidence = "\n".join(
            f"- `{task_id}` — {' '.join(summary.split())}"
            for task_id, summary in item["evidence"].items()
        )
        drafts.append({
            "key": key,
            "title": f"Tool gap: {key.replace('-', ' ')}",
            "labels": [GAP_LABEL],
            "body": (
                f"<!-- tool-fit-gap: {key} -->\n\n"
                f"**{item['kind']}** reported by the Venice tool-fit probe. "
                f"Model assessment of the available schemas. Unverified.\n\n"
                f"{evidence}\n\n{AGENT}"
            ),
        })
    return drafts


def render_markdown(report: dict) -> str:
    lines = ["# Venice tool-fit report", "", f"Model `{report['model']}`", "",
             f"Spend cap ${report['max_spend_usd']:.6g} USD. "
             f"Catalog cost bound ${report['cost_upper_bound_usd']:.6g} USD.", "",
             "Model assessment of tool schemas. The probe did not execute the tasks.", "",
             "## Task assessments", ""]
    for row in report["results"]:
        lines.extend([f"### {row['id']} — {row['can_do']}", "", row["description"], "",
                      f"**Success hint** {row['success_hint']}", ""])
        for field, label in (("missing_tools", "Missing tools"), ("time_savers", "Time-savers")):
            lines.extend([f"**{label}**", ""])
            lines.extend(f"- `{gap['key']}` — {gap['summary'].strip()}" for gap in row[field])
            if not row[field]:
                lines.append("None reported.")
            lines.append("")
    lines.extend(["## Draft issues", ""])
    for draft in report["draft_issues"]:
        lines.append(f"- [{draft['title']}](drafts/{draft['key']}.md) (`{GAP_LABEL}`)")
    if not report["draft_issues"]:
        lines.append("No new gaps to draft.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Run from the project to assess. Install scripts/tool_fit/requirements.txt first. "
        "MCP schemas are included when you pass a configured server with --mcp."
    ))
    parser.add_argument("--tasks", type=Path, default=HERE / "tasks.yaml", help="YAML task list")
    parser.add_argument("--output-dir", type=Path, default=Path("tool-fit-report"),
                        help="Parent directory for a new report directory (default: tool-fit-report)")
    parser.add_argument("--model", help="Venice function-calling model id")
    parser.add_argument("--max-spend", required=True, type=float, help="Total run cap in USD")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Completion token limit")
    parser.add_argument("--mcp", action="append", help="Include a registered MCP server (repeatable)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_spend) or args.max_spend <= 0 or args.max_tokens <= 0:
        parser.error("--max-spend and --max-tokens must be positive and finite")

    try:
        tasks = load_tasks(args.tasks)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        issues = open_issues()
        reply = probe(args, tasks)
        results = parse_answers(reply["final"], tasks)
        report = {
            "model": reply["model"],
            "tools": reply["tools"],
            "max_spend_usd": args.max_spend,
            "cost_upper_bound_usd": reply["cost_upper_bound_usd"],
            "usage": reply["usage"],
            "results": results,
            "draft_issues": draft_gaps(results, issues),
        }
        # Publish both reports together. An interrupted or concurrent run cannot
        # replace a previous report or leave it paired with a different run.
        with tempfile.TemporaryDirectory(prefix=".tool-fit-", dir=args.output_dir) as directory:
            staging = Path(directory) / "report"
            staging.mkdir()
            (staging / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            (staging / "report.md").write_text(render_markdown(report), encoding="utf-8")
            if report["draft_issues"]:
                drafts_dir = staging / "drafts"
                drafts_dir.mkdir()
                for draft in report["draft_issues"]:
                    metadata = (f"---\ntitle: {json.dumps(draft['title'])}\n"
                                f"labels: {json.dumps(draft['labels'])}\n---\n\n")
                    (drafts_dir / f"{draft['key']}.md").write_text(
                        metadata + draft["body"] + "\n", encoding="utf-8"
                    )
            output = args.output_dir / Path(directory).name.lstrip(".")
            staging.rename(output)
    except subprocess.CalledProcessError as exc:
        print(f"tool-fit: {Path(exc.cmd[0]).name} exited {exc.returncode}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"tool-fit: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
