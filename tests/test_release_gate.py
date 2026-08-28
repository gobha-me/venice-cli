import importlib.util
import io
import json
import pathlib
import unittest
import urllib.error
from contextlib import redirect_stderr


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "verify_release.py"
SPEC = importlib.util.spec_from_file_location("verify_release", SCRIPT)
verify_release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_release)

SHA = "a" * 40


def run(*, event="push", branch="master", status="completed", conclusion="success", sha=SHA):
    return {
        "id": 123,
        "head_sha": sha,
        "event": event,
        "head_branch": branch,
        "status": status,
        "conclusion": conclusion,
    }


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.body


class ReleaseGateTests(unittest.TestCase):
    def test_tag_must_match_imported_version(self):
        verify_release.verify_tag("v1.2.3", "1.2.3")
        with self.assertRaisesRegex(verify_release.ReleaseGateError, "imported package version"):
            verify_release.verify_tag("v1.2.2", "1.2.3")

    def test_accepts_only_completed_successful_master_push_for_exact_sha(self):
        payload = {
            "workflow_runs": [
                run(event="pull_request"),
                run(branch="other"),
                run(sha="b" * 40),
                run(status="in_progress", conclusion=None),
                run(),
            ]
        }
        accepted = verify_release.require_successful_run(payload, sha=SHA, branch="master")
        self.assertEqual(accepted["id"], 123)

    def test_rejects_missing_pending_failed_and_non_push_runs(self):
        cases = (
            [],
            [run(status="in_progress", conclusion=None)],
            [run(conclusion="failure")],
            [run(event="workflow_dispatch")],
            [run(event="pull_request")],
            [run(branch="release")],
            [run(sha="b" * 40)],
        )
        for runs in cases:
            with self.subTest(runs=runs):
                with self.assertRaises(verify_release.ReleaseGateError):
                    verify_release.require_successful_run(
                        {"workflow_runs": runs}, sha=SHA, branch="master"
                    )

    def test_rejects_malformed_payload_and_sha(self):
        for payload in (None, [], {}, {"workflow_runs": {}}, {"workflow_runs": [None]}):
            with self.subTest(payload=payload):
                with self.assertRaises(verify_release.ReleaseGateError):
                    verify_release.require_successful_run(payload, sha=SHA, branch="master")
        with self.assertRaisesRegex(verify_release.ReleaseGateError, "invalid commit SHA"):
            verify_release.require_successful_run(
                {"workflow_runs": []}, sha="not-a-sha", branch="master"
            )

    def test_fetch_uses_bounded_authenticated_exact_sha_query(self):
        seen = {}

        def opener(request, timeout):
            seen["request"] = request
            seen["timeout"] = timeout
            return _Response(json.dumps({"workflow_runs": [run()]}).encode())

        payload = verify_release.fetch_workflow_runs(
            repository="owner/repo",
            workflow="test.yml",
            sha=SHA,
            token="test-token",
            opener=opener,
        )
        self.assertEqual(payload["workflow_runs"][0]["head_sha"], SHA)
        self.assertIn(f"head_sha={SHA}", seen["request"].full_url)
        self.assertIn("test.yml", seen["request"].full_url)
        self.assertEqual(seen["request"].get_header("Authorization"), "Bearer test-token")
        self.assertEqual(seen["timeout"], 30)

    def test_fetch_fails_closed_without_echoing_token(self):
        token = "test-secret-never-print"

        def opener(_request, timeout):
            self.assertEqual(timeout, 30)
            raise urllib.error.HTTPError("https://api.github.invalid", 403, "no", {}, None)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaisesRegex(verify_release.ReleaseGateError, "HTTP 403"):
                verify_release.fetch_workflow_runs(
                    repository="owner/repo",
                    workflow="test.yml",
                    sha=SHA,
                    token=token,
                    opener=opener,
                )
        self.assertNotIn(token, stderr.getvalue())

    def test_fetch_rejects_invalid_json_and_oversized_response(self):
        for body in (b"not-json", b"x" * (verify_release.MAX_RESPONSE_BYTES + 1)):
            with self.subTest(size=len(body)):
                with self.assertRaises(verify_release.ReleaseGateError):
                    verify_release.fetch_workflow_runs(
                        repository="owner/repo",
                        workflow="test.yml",
                        sha=SHA,
                        token="test-token",
                        opener=lambda *_args, **_kwargs: _Response(body),
                    )


if __name__ == "__main__":
    unittest.main()
