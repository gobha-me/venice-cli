#!/usr/bin/env python3
"""Fail closed unless a release tag has successful exact-SHA CI.

This runs before a production artifact is built.  It intentionally uses only
the standard library and reports no response bodies or authorization values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, List, Mapping, Optional

from venice import __version__


API_VERSION = "2022-11-28"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class ReleaseGateError(RuntimeError):
    """A production release invariant was not satisfied."""


def verify_tag(tag: str, version: str = __version__) -> None:
    expected = f"v{version}"
    if tag != expected:
        raise ReleaseGateError(
            f"tag {tag!r} does not match imported package version {expected!r}"
        )


def _workflow_runs(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, dict):
        raise ReleaseGateError("GitHub Actions response is not an object")
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise ReleaseGateError("GitHub Actions response has no workflow_runs list")
    if not all(isinstance(run, dict) for run in runs):
        raise ReleaseGateError("GitHub Actions response contains an invalid run")
    return runs


def require_successful_run(payload: Any, *, sha: str, branch: str) -> Mapping[str, Any]:
    if not SHA_RE.fullmatch(sha):
        raise ReleaseGateError(f"invalid commit SHA {sha!r}")

    runs = _workflow_runs(payload)
    matching = [
        run
        for run in runs
        if run.get("head_sha") == sha
        and run.get("event") == "push"
        and run.get("head_branch") == branch
    ]
    successful = [
        run
        for run in matching
        if run.get("status") == "completed" and run.get("conclusion") == "success"
    ]
    if successful:
        return successful[0]

    states = sorted(
        {
            f"{run.get('status', 'unknown')}/{run.get('conclusion', 'none')}"
            for run in matching
        }
    )
    detail = ", ".join(states) if states else "no matching master push run"
    raise ReleaseGateError(
        f"test.yml has no completed successful {branch} push for exact SHA {sha}; "
        f"observed: {detail}"
    )


def fetch_workflow_runs(
    *,
    repository: str,
    workflow: str,
    sha: str,
    token: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Any:
    try:
        owner, name = repository.split("/", 1)
    except ValueError as exc:
        raise ReleaseGateError(f"invalid GitHub repository {repository!r}") from exc
    if not owner or not name or "/" in name:
        raise ReleaseGateError(f"invalid GitHub repository {repository!r}")
    if not token:
        raise ReleaseGateError("GITHUB_TOKEN is unavailable")
    if not SHA_RE.fullmatch(sha):
        raise ReleaseGateError(f"invalid commit SHA {sha!r}")

    repo_path = "/".join(urllib.parse.quote(part, safe="") for part in (owner, name))
    workflow_path = urllib.parse.quote(workflow, safe="")
    query = urllib.parse.urlencode({"head_sha": sha, "per_page": 100})
    url = (
        f"https://api.github.com/repos/{repo_path}/actions/workflows/"
        f"{workflow_path}/runs?{query}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with opener(request, timeout=30) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ReleaseGateError(f"GitHub Actions query failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ReleaseGateError("GitHub Actions query failed") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ReleaseGateError("GitHub Actions response exceeded the size limit")
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("GitHub Actions response was not valid JSON") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", default="test.yml")
    parser.add_argument("--branch", default="master")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verify_tag(args.tag)
        payload = fetch_workflow_runs(
            repository=args.repository,
            workflow=args.workflow,
            sha=args.sha,
            token=os.environ.get("GITHUB_TOKEN", ""),
        )
        run = require_successful_run(payload, sha=args.sha, branch=args.branch)
    except ReleaseGateError as exc:
        print(f"release gate: {exc}", file=sys.stderr)
        return 1

    print(
        f"release gate: {args.tag} is pinned to {args.sha} and exact-SHA CI "
        f"run {run.get('id')} succeeded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
