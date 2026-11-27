"""Exercise the tool-fit spend gate through the CLI with a fake transport."""

import contextlib
import io
import json
import unittest
from unittest import mock

from venice.cli import main
from venice.client import VeniceAPIError, VeniceClient
from venice.commands import chat


class TestToolFitBudget(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "traits": ["default"],
            "availableContextTokens": 32_768,
            "capabilities": {"supportsFunctionCalling": True},
            "pricing": {"input": {"usd": 4}, "output": {"usd": 20}},
        }
        self.client = mock.Mock(spec=VeniceClient)
        self.client.api_key = "test-fake-key"
        self.client.get_json.return_value = {
            "data": [{"id": "test-model", "model_spec": self.spec}],
        }
        self.client.post_json.return_value = {
            "choices": [{"message": {"content": "assessment"}, "finish_reason": "stop"}],
        }
        self.build_client = self._patch(chat, "build_client_from_auth", self.client)
        self._patch(chat.userconfig, "load_config", {})

    def _patch(self, target, name, result):
        patch = mock.patch.object(target, name, return_value=result)
        self.addCleanup(patch.stop)
        return patch.start()

    def invoke(self, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = main(["chat", "--tool-fit", "--no-mcp", "--json", *options,
                               "Assess tools for a task"])
            except SystemExit as exc:
                status = exc.code
        return status, stdout.getvalue(), stderr.getvalue()

    def test_bound_below_cap_permits_completion_and_is_reported(self):
        status, output, error = self.invoke("--max-spend", "0.22", "--max-tokens", "4096")

        self.assertEqual(status, 0, error)
        self.client.get_json.assert_called_once_with("/models", params={"type": "text"})
        self.client.post_json.assert_called_once()
        self.assertEqual(self.client.post_json.call_args.args[1]["max_tokens"], 4096)
        self.assertAlmostEqual(json.loads(output)["cost_upper_bound_usd"], 0.212992)

    def test_exact_cap_permits_completion(self):
        # (32,768 x $4 + 4,096 x $20) / 1,000,000 is exactly $0.212992.
        status, output, error = self.invoke("--max-spend", "0.212992", "--max-tokens", "4096")

        self.assertEqual(status, 0, error)
        self.client.post_json.assert_called_once()
        self.assertLessEqual(json.loads(output)["cost_upper_bound_usd"], 0.212992)

    def test_bound_above_cap_blocks_even_with_yes(self):
        status, output, error = self.invoke(
            "--max-spend", "0.212991", "--max-tokens", "4096", "--yes",
        )

        self.assertNotEqual(status, 0)
        self.assertEqual(output, "")
        self.assertIn("--max-spend", error)
        self.client.post_json.assert_not_called()

    def test_higher_cache_rates_are_included_in_bound(self):
        for rate in ("cache_input", "cache_write"):
            with self.subTest(rate=rate), mock.patch.dict(self.spec["pricing"], {rate: {"usd": 8}}):
                status, output, error = self.invoke("--max-spend", "0.30", "--max-tokens", "4096")

                self.assertNotEqual(status, 0)
                self.assertEqual(output, "")
                self.assertIn("--max-spend", error)
                self.client.post_json.assert_not_called()

    def test_unusable_catalog_data_blocks_before_completion(self):
        cases = (
            {"availableContextTokens": None},
            {"availableContextTokens": 0},
            {"availableContextTokens": True},
            {"pricing": {"input": {"usd": 4}}},
            {"pricing": {"input": {"usd": -1}, "output": {"usd": 20}}},
            {"pricing": {"input": {"usd": 4}, "output": {"usd": "NaN"}}},
        )
        for fields in cases:
            with self.subTest(fields=fields), mock.patch.dict(self.spec, fields):
                status, output, error = self.invoke("--max-spend", "1", "--max-tokens", "4096")

                self.assertNotEqual(status, 0)
                self.assertEqual(output, "")
                self.assertIn("pricing", error)
                self.client.post_json.assert_not_called()

    def test_catalog_outage_blocks_even_with_explicit_model(self):
        self.client.get_json.side_effect = VeniceAPIError(503, "/models", "Unavailable")
        status, output, error = self.invoke(
            "--model", "test-model", "--max-spend", "1", "--max-tokens", "4096",
        )

        self.assertNotEqual(status, 0)
        self.assertEqual(output, "")
        self.assertIn("catalog", error)
        self.client.post_json.assert_not_called()

    def test_missing_or_invalid_limits_fail_before_authentication(self):
        for options in (
            ("--max-tokens", "4096"),
            ("--max-spend", "1"),
            ("--max-spend", "1", "--max-tokens", "0"),
            ("--max-spend", "1", "--max-tokens", "-1"),
            ("--max-spend", "-1", "--max-tokens", "4096"),
            ("--max-spend", "nan", "--max-tokens", "4096"),
            ("--max-spend", "inf", "--max-tokens", "4096"),
        ):
            with self.subTest(options=options):
                status, output, _ = self.invoke(*options)

                self.assertEqual(status, 2)
                self.assertEqual(output, "")
                self.build_client.assert_not_called()
