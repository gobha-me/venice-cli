"""Adversarial regressions for #247; all HTTP is mocked and keys are fake."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

from tests.test_client import FakeResp
from tests.test_image import _resolved_args
from tests.test_image_edit import _args as _edit_args
from tests.test_openapi_contract import LOCK_PATH, MANIFEST_PATH, contract
from tests.test_video import _build_args as _video_args
from tests.test_video import _catalog
from venice.client import VeniceClient
from venice.commands import image, image_edit, video


FIXTURES = Path(__file__).parent / "fixtures" / "openapi_247"


class TestOfflineDriftGate(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def _check_simulated_upstream(self, live_lock):
        """Replace only fetching/normalization; exercise the real CLI gate."""
        stdout = io.StringIO()
        original_lock = LOCK_PATH.read_bytes()
        with mock.patch.object(contract, "fetch_spec", return_value=b"offline fixture"), \
             mock.patch.object(contract, "build_lock", return_value=live_lock), \
             mock.patch("urllib.request.urlopen", side_effect=AssertionError(
                 "offline drift tests must never use the network"
             )) as http, redirect_stdout(stdout):
            rc = contract.main([
                "check-live", "--lock", str(LOCK_PATH),
                "--manifest", str(MANIFEST_PATH),
            ])
        http.assert_not_called()
        self.assertEqual(LOCK_PATH.read_bytes(), original_lock)
        return rc, stdout.getvalue()

    def test_committed_lock_older_than_simulated_upstream_fails(self):
        addition = json.loads(
            (FIXTURES / "future-video-queue-field.json").read_text(encoding="utf-8")
        )
        newer = copy.deepcopy(self.baseline)
        properties = newer["implemented_operations"][addition["operation"]][
            "request_bodies"
        ][addition["media_type"]]["properties"]
        self.assertNotIn(addition["property"], properties)
        properties[addition["property"]] = addition["schema"]
        newer["source"]["version"] += "-simulated-next"
        newer["source"]["sha256"] = "f" * 64

        report = contract.compare_locks(self.baseline, newer)
        self.assertEqual(report["implemented_changes"], [(
            "review",
            "/POST /video/queue/request_bodies/application/json/properties/future_optional",
            None,
            {"type": "boolean"},
        )])
        self.assertTrue(report["source_changed"])
        rc, rendered = self._check_simulated_upstream(newer)
        self.assertEqual(rc, 1, rendered)
        self.assertIn("future_optional", rendered)

    def test_equal_committed_and_simulated_upstream_locks_pass(self):
        equal = copy.deepcopy(self.baseline)
        self.assertEqual(contract.compare_locks(self.baseline, equal), {
            "implemented_changes": [],
            "unsupported_added": [],
            "unsupported_removed": [],
            "source_changed": False,
        })
        rc, rendered = self._check_simulated_upstream(equal)
        self.assertEqual(rc, 0, rendered)
        self.assertIn("No implemented-operation drift detected.", rendered)


class TestTransparentBackgroundSerialization(unittest.TestCase):
    def test_true_false_and_omitted_on_every_image_route(self):
        client = VeniceClient(api_key="test-fake-key", base_url="https://api.invalid")
        # A missing attribute covers older programmatic callers too. Explicit
        # None covers argparse/MCP's normal representation of an omitted flag.
        for route in ("generate", "edit", "multi-edit"):
            for state in ("true", "false", "none", "missing"):
                with self.subTest(route=route, state=state):
                    options = {} if state == "missing" else {
                        "transparent_background": {
                            "true": True, "false": False, "none": None,
                        }[state]
                    }
                    if route == "generate":
                        endpoint = "/image/generate"
                        body = image._build_body("a glass bird", _resolved_args(**options))
                    else:
                        endpoint, body = image_edit._build_body(
                            _edit_args(**options), "base-image",
                            ["layer-image"] if route == "multi-edit" else [],
                        )

                    with mock.patch("venice.client.urllib.request.urlopen",
                                    return_value=FakeResp(200, b'{"ok": true}')) as http:
                        self.assertEqual(client.post_json(endpoint, body), {"ok": True})
                    http.assert_called_once()
                    request = http.call_args.args[0]
                    self.assertEqual(urlsplit(request.full_url).path, f"/image/{route}")
                    self.assertEqual(request.get_method(), "POST")
                    serialized = json.loads(request.data.decode("utf-8"))
                    if state in ("none", "missing"):
                        self.assertNotIn("transparent_background", serialized)
                    else:
                        self.assertIs(serialized["transparent_background"], state == "true")


class TestVideoQueueOnlyFields(unittest.TestCase):
    def setUp(self):
        # No auth loading, user config, saved jobs, polling, or downloads.
        client = VeniceClient(api_key="test-fake-key", base_url="https://api.invalid")
        self.client = mock.patch("venice.commands.video._queue.build_client",
                                 return_value=(client, 0))
        self.client.start()
        self.addCleanup(self.client.stop)
        defaults = mock.patch("venice.userconfig.load_config", return_value={
            "version": 1, "defaults": {}, "mcpServers": {},
        })
        defaults.start()
        self.addCleanup(defaults.stop)
        retrieve = mock.patch("venice.commands.video._retrieve_and_save", return_value=0)
        retrieve.start()
        self.addCleanup(retrieve.stop)

    def _run(self, **options):
        requests = {}

        def fake_urlopen(request, timeout=None):
            path = urlsplit(request.full_url).path
            if request.get_method() == "GET" and path == "/models":
                return _catalog("video-edit-model")
            self.assertEqual(request.get_method(), "POST")
            self.assertIn(path, ("/video/quote", "/video/queue"))
            requests[path] = json.loads(request.data.decode("utf-8"))
            if path == "/video/quote":
                return FakeResp(200, b'{"quote": 0.5}')
            return FakeResp(200, b'{"queue_id": "offline-job"}')

        with mock.patch("venice.client.urllib.request.urlopen", side_effect=fake_urlopen) as http:
            rc = video._run_generate(_video_args(**options))
        return rc, requests, http

    def test_zero_and_nonzero_image_strength_only_reach_queue(self):
        for strength in (0, 0.65, 1):
            with self.subTest(image_strength=strength):
                rc, requests, http = self._run(
                    image="data:image/png;base64,eA==", image_strength=strength,
                )
                self.assertEqual(rc, 0)
                self.assertEqual(http.call_count, 3)
                self.assertEqual(set(requests), {"/video/quote", "/video/queue"})
                quote, queued = requests["/video/quote"], requests["/video/queue"]
                self.assertNotIn("image_strength", quote)
                self.assertNotIn("image_url", quote)
                self.assertEqual(queued["image_strength"], strength)
                self.assertIsInstance(queued["image_strength"], (int, float))
                self.assertNotIsInstance(queued["image_strength"], bool)
                self.assertEqual(queued["image_url"], "data:image/png;base64,eA==")

    def test_omitted_image_strength_is_absent_from_quote_and_queue(self):
        rc, requests, _ = self._run(image="data:image/png;base64,eA==")
        self.assertEqual(rc, 0)
        for body in requests.values():
            self.assertNotIn("image_strength", body)

    def test_32000_character_prompt_is_accepted_unchanged(self):
        prompt = "p" * 32000
        rc, requests, _ = self._run(prompt=prompt)
        self.assertEqual(rc, 0)
        self.assertEqual(requests["/video/queue"]["prompt"], prompt)
        self.assertNotIn("prompt", requests["/video/quote"])

    def test_32001_character_prompt_is_rejected_before_http(self):
        rc, requests, http = self._run(prompt="p" * 32001)
        self.assertEqual(rc, 2)
        self.assertEqual(requests, {})
        http.assert_not_called()

    def test_request_builder_enforces_the_same_prompt_boundary(self):
        self.assertEqual(video.queue_body("m", "p" * 32000, {}, {})["prompt"],
                         "p" * 32000)
        with self.assertRaisesRegex(ValueError, "prompt exceeds 32000 chars"):
            video.queue_body("m", "p" * 32001, {}, {})


if __name__ == "__main__":
    unittest.main()
