"""Tests for the controlled live cache diagnostic (#97); no real API calls."""
import argparse
import io
import json
import sys
import unittest
from unittest import mock

from venice.client import VeniceAPIError
from venice.commands import cache_probe


def _model(mid, *, default=False, family_cache=True, input_price=1.0,
           output_price=2.0):
    pricing = {
        "input": {"usd": input_price},
        "output": {"usd": output_price},
    }
    if family_cache:
        pricing["cache_input"] = {"usd": input_price / 10}
    return {
        "id": mid,
        "model_spec": {
            "traits": ["default"] if default else [],
            "pricing": pricing,
        },
    }


MODELS = [
    _model("qwen3-primary", default=True),
    _model("qwen-2.5-sibling"),
    _model("llama-control"),
]


def _args(**overrides):
    base = dict(
        model=["qwen3-primary"], prefix_tokens=[8], repeat=2,
        json=False, yes=True, max_spend=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _response(cached, *, details_extra=None, prompt=12):
    details = None if cached is None else {"cached_tokens": cached}
    if details is not None and details_extra:
        details.update(details_extra)
    return {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": 1,
            "prompt_tokens_details": details,
        }
    }


class FakeClient:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.posts = []

    def get_json(self, path, params=None):
        self.get = (path, params)
        return {"data": MODELS}

    def post_json(self, path, body):
        self.posts.append((path, body))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestArguments(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        cache_probe.register(parser.add_subparsers(dest="command"))
        return parser

    def test_repeatable_models_and_sizes_preserve_order(self):
        args = self.parser().parse_args([
            "cache-probe", "-m", "a", "-m", "b",
            "--prefix-tokens", "8192", "--prefix-tokens", "120000",
            "--repeat", "3", "--json", "--yes", "--max-spend", "1.25",
        ])
        self.assertEqual(args.model, ["a", "b"])
        self.assertEqual(args.prefix_tokens, [8192, 120000])
        self.assertEqual(args.repeat, 3)
        self.assertTrue(args.json)
        self.assertTrue(args.yes)
        self.assertEqual(args.max_spend, 1.25)

    def test_defaults_are_small_and_two_calls(self):
        args = self.parser().parse_args(["cache-probe"])
        self.assertIsNone(args.model)
        self.assertIsNone(args.prefix_tokens)
        self.assertEqual(args.repeat, 2)

    def test_prefix_and_repeat_bounds_fail_at_parse_time(self):
        for argv in (
            ["cache-probe", "--prefix-tokens", "0"],
            ["cache-probe", "--prefix-tokens", "1000001"],
            ["cache-probe", "--repeat", "1"],
            ["cache-probe", "--repeat", "11"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as ctx:
                self.parser().parse_args(argv)
            self.assertEqual(ctx.exception.code, 2)


class TestSelectionAndEstimation(unittest.TestCase):
    def test_auto_selection_uses_configured_primary_and_different_family_control(self):
        args = _args(model=None)
        with mock.patch(
            "venice.commands.cache_probe.userconfig.resolve_default",
            return_value="qwen3-primary",
        ):
            resolved, rc = cache_probe._resolve_models(args, MODELS)
        self.assertIsNone(rc)
        self.assertEqual(resolved, ["qwen3-primary", "llama-control"])

    def test_explicit_selection_deduplicates_and_adds_no_control(self):
        resolved, rc = cache_probe._resolve_models(
            _args(model=["llama-control", "llama-control"]), MODELS
        )
        self.assertIsNone(rc)
        self.assertEqual(resolved, ["llama-control"])

    def test_no_control_warns_and_keeps_primary(self):
        err = io.StringIO()
        with mock.patch(
            "venice.commands.cache_probe.userconfig.resolve_default",
            return_value=None,
        ), mock.patch.object(sys, "stderr", err):
            resolved, rc = cache_probe._resolve_models(
                _args(model=None), MODELS[:2]
            )
        self.assertIsNone(rc)
        self.assertEqual(resolved, ["qwen3-primary"])
        self.assertIn("probing only the primary", err.getvalue())

    def test_estimate_is_all_uncached_and_unknown_pricing_fails_closed(self):
        known = cache_probe._estimate(MODELS, ["qwen3-primary"], [8], 2)
        self.assertTrue(known["pricing_complete"])
        self.assertGreater(known["usd_upper_bound"], 0)
        self.assertEqual(known["calls"], 2)

        broken = [_model("broken", default=True, output_price=float("nan"))]
        unknown = cache_probe._estimate(broken, ["broken"], [8], 2)
        self.assertFalse(unknown["pricing_complete"])
        self.assertIsNone(unknown["usd_upper_bound"])
        self.assertTrue(cache_probe._shared.over_budget(None, 1.0))


class TestResults(unittest.TestCase):
    def test_three_state_verdict(self):
        cases = (
            ([_response(0), _response(10)], "warms"),
            ([_response(0), _response(0)], "never warms"),
            ([_response(0), _response(None)], "field absent"),
            ([_response(None), _response(None)], "field absent"),
        )
        for responses, expected in cases:
            calls = [cache_probe._call_row(r, i + 1) for i, r in enumerate(responses)]
            with self.subTest(expected=expected):
                self.assertEqual(cache_probe._verdict(calls), expected)

    def test_raw_extension_is_preserved(self):
        response = _response(
            8, details_extra={"provider_extension": {"write": 4, "name": "raw"}}
        )
        row = cache_probe._call_row(response, 1)
        self.assertEqual(row["prompt_tokens_details"], {
            "cached_tokens": 8,
            "provider_extension": {"write": 4, "name": "raw"},
        })

    def test_non_finite_raw_usage_fails_closed(self):
        response = _response(float("nan"))
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            cache_probe._call_row(response, 1)

    def test_prefix_is_stable_within_size_and_distinct_across_sizes(self):
        first = cache_probe._build_prefix(8)
        self.assertEqual(first, cache_probe._build_prefix(8))
        self.assertNotEqual(first, cache_probe._build_prefix(9))
        self.assertTrue(first.startswith("CACHE-PROBE-V1 SIZE=8\n"))


class TestRun(unittest.TestCase):
    def run_command(self, args, responses):
        client = FakeClient(responses)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch(
            "venice.commands.cache_probe.build_client_from_auth", return_value=client
        ), mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            rc = cache_probe._run(args)
        return rc, client, out.getvalue(), err.getvalue()

    def test_json_matrix_preserves_order_and_raw_calls(self):
        args = _args(
            model=["qwen3-primary", "llama-control"],
            prefix_tokens=[8, 16, 8], repeat=2, json=True,
        )
        responses = []
        for _model_id in range(2):
            responses.extend([
                _response(0), _response(8),
                _response(0), _response(0),
            ])
        rc, client, out, err = self.run_command(args, responses)
        self.assertEqual(rc, 0, err)
        doc = json.loads(out)
        self.assertEqual(doc["models"], ["qwen3-primary", "llama-control"])
        self.assertEqual(doc["prefix_tokens"], [8, 16])
        self.assertEqual([row["verdict"] for row in doc["results"]], [
            "warms", "never warms", "warms", "never warms",
        ])
        self.assertEqual(len(client.posts), 8)
        self.assertEqual(client.posts[0][1], client.posts[1][1])
        self.assertNotEqual(client.posts[1][1], client.posts[2][1])

    def test_over_cap_aborts_before_any_post(self):
        rc, client, _out, err = self.run_command(
            _args(max_spend=0), [_response(0), _response(0)]
        )
        self.assertEqual(rc, 1)
        self.assertEqual(client.posts, [])
        self.assertIn("cannot be kept within --max-spend", err)

    def test_noninteractive_without_yes_aborts_before_any_post(self):
        client = FakeClient([_response(0), _response(0)])
        err = io.StringIO()
        with mock.patch(
            "venice.commands.cache_probe.build_client_from_auth", return_value=client
        ), mock.patch(
            "venice.commands.cache_probe.sys.stdin.isatty", return_value=False
        ), mock.patch.object(sys, "stderr", err):
            rc = cache_probe._run(_args(yes=False))
        self.assertEqual(rc, 1)
        self.assertEqual(client.posts, [])
        self.assertIn("pass --yes", err.getvalue())

    def test_json_interactive_prompt_stays_off_stdout(self):
        client = FakeClient([_response(0), _response(8)])
        out, err = io.StringIO(), io.StringIO()

        def confirm(prompt):
            print(prompt, end="")
            return "y"

        with mock.patch(
            "venice.commands.cache_probe.build_client_from_auth", return_value=client
        ), mock.patch(
            "venice.commands.cache_probe.sys.stdin.isatty", return_value=True
        ), mock.patch("builtins.input", side_effect=confirm) as input_mock, \
             mock.patch.object(sys, "stdout", out), \
             mock.patch.object(sys, "stderr", err):
            rc = cache_probe._run(_args(json=True, yes=False))
        self.assertEqual(rc, 0, err.getvalue())
        self.assertEqual(input_mock.call_args.args, ("Proceed? [y/N] ",))
        self.assertNotIn("Proceed?", out.getvalue())
        self.assertIn("Proceed?", err.getvalue())
        self.assertEqual(json.loads(out.getvalue())["results"][0]["verdict"], "warms")

    def test_api_error_stops_remaining_paid_calls(self):
        error = VeniceAPIError(503, "/chat/completions", "unavailable")
        rc, client, _out, err = self.run_command(
            _args(repeat=3), [_response(0), error, _response(8)]
        )
        self.assertEqual(rc, 5)
        self.assertEqual(len(client.posts), 2)
        self.assertIn("cache-probe: HTTP 503", err)


if __name__ == "__main__":
    unittest.main()
