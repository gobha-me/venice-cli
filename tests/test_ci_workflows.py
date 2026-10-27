"""Guard the Python 3.9 CI leg's runner image (#249).

GitHub moved ``ubuntu-latest`` to an image setup-python has no 3.9 build for,
so the 3.9 leg must stay pinned to ``ubuntu-24.04`` (or an explicitly older
image). This parses the real workflow files -- no fixtures -- and fails if 3.9
drops out of the test matrix, if its runner resolves to ``*-latest`` or a newer
image, or if a step is gated so 3.9 would run a different set of steps.
"""
from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - CI installs PyYAML on every leg
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PY_KEY = "python-version"
# The newest image known to carry a setup-python 3.9 build.
NEWEST_39_IMAGE = (24, 4)

_EXPR = re.compile(r"^\$\{\{\s*(?P<body>.*?)\s*\}\}$", re.S)
_MATRIX_REF = re.compile(r"^matrix\.(?P<key>[\w-]+)$")
# The ternary idiom: matrix.K == 'V' && 'A' || 'B'  (or !=).
_TERNARY = re.compile(
    r"^matrix\.(?P<key>[\w-]+)\s*(?P<op>==|!=)\s*'(?P<value>[^']*)'"
    r"\s*&&\s*'(?P<then>[^']*)'\s*\|\|\s*'(?P<other>[^']*)'$"
)


def _load_workflows():
    loaded = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        loaded[path.name] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _combos(matrix):
    """Every matrix combination that names a Python version, as dicts."""
    versions = [str(v) for v in matrix.get(PY_KEY, [])]
    excludes = [
        {k: str(v) for k, v in entry.items()}
        for entry in matrix.get("exclude", []) or []
    ]
    combos = []
    for version in versions:
        combo = {PY_KEY: version}
        if any(all(combo.get(k) == v for k, v in ex.items()) for ex in excludes):
            continue
        combos.append(combo)
    for entry in matrix.get("include", []) or []:
        entry = {k: str(v) for k, v in entry.items()}
        if PY_KEY in entry:
            combos.append(entry)
    return combos


def _resolve_runs_on(runs_on, combo):
    """Resolve a job's ``runs-on`` for one matrix combination.

    Only the shapes this repo uses are understood; anything else fails loudly
    so the guard is taught the new shape rather than silently passing.
    """
    if isinstance(runs_on, list):
        return [_resolve_runs_on(label, combo) for label in runs_on]
    runs_on = str(runs_on)
    match = _EXPR.match(runs_on.strip())
    if match is None:
        if "${{" in runs_on:
            raise AssertionError(f"unrecognised runs-on expression: {runs_on!r}")
        return runs_on
    body = match.group("body")
    ref = _MATRIX_REF.match(body)
    if ref is not None:
        return combo[ref.group("key")]
    ternary = _TERNARY.match(body)
    if ternary is not None:
        equal = combo.get(ternary.group("key")) == ternary.group("value")
        if ternary.group("op") == "!=":
            equal = not equal
        return ternary.group("then") if equal else ternary.group("other")
    raise AssertionError(f"unrecognised runs-on expression: {runs_on!r}")


def _image_supports_39(label):
    found = re.fullmatch(r"ubuntu-(\d+)\.(\d+)", label)
    return found is not None and (int(found[1]), int(found[2])) <= NEWEST_39_IMAGE


def _python_matrix_jobs(workflows):
    for name, workflow in workflows.items():
        for job_id, job in (workflow.get("jobs") or {}).items():
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            if isinstance(matrix, dict) and (
                PY_KEY in matrix
                or any(PY_KEY in e for e in matrix.get("include", []) or [])
            ):
                yield f"{name}:{job_id}", job, matrix


@unittest.skipIf(
    yaml is None and not os.environ.get("CI"),
    "PyYAML not installed (pip install -r scripts/openapi-requirements.txt)",
)
class Python39RunnerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if yaml is None:
            raise AssertionError("CI must install PyYAML so this guard runs")
        cls.jobs = list(_python_matrix_jobs(_load_workflows()))

    def test_a_workflow_has_a_python_test_matrix(self):
        self.assertTrue(self.jobs, "no workflow job has a python-version matrix")

    def test_matrix_still_includes_python_39(self):
        legs = [
            (job_name, combo)
            for job_name, _, matrix in self.jobs
            for combo in _combos(matrix)
            if combo[PY_KEY] == "3.9"
        ]
        self.assertTrue(legs, "Python 3.9 dropped out of every CI test matrix")

    def test_python_39_leg_runs_on_an_image_with_39(self):
        for job_name, job, matrix in self.jobs:
            for combo in _combos(matrix):
                if combo[PY_KEY] != "3.9":
                    continue
                resolved = _resolve_runs_on(job.get("runs-on"), combo)
                labels = resolved if isinstance(resolved, list) else [resolved]
                with self.subTest(job=job_name, runs_on=labels):
                    self.assertTrue(
                        any(_image_supports_39(label) for label in labels),
                        f"{job_name}: Python 3.9 resolves to {labels}; it must "
                        "be pinned to ubuntu-24.04 or older, not *-latest",
                    )

    def test_python_39_leg_runs_every_step(self):
        # A per-version `if:` would let 3.9 go green while skipping tests.
        for job_name, job, _ in self.jobs:
            for step in job.get("steps") or []:
                condition = str(step.get("if", ""))
                with self.subTest(job=job_name, step=step.get("name", step.get("uses"))):
                    self.assertNotIn(PY_KEY, condition)


if __name__ == "__main__":
    unittest.main()
