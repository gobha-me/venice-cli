"""Exercise the schema-only tool-fit probe through the chat CLI."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from venice import cli
from venice.client import VeniceAPIError, VeniceClient
from venice.commands import chat


ASSESSMENT = json.dumps({
    "docs": {"can_do": "yes", "missing_tools": [], "time_savers": []},
})


class TestToolFitChat(unittest.TestCase):
    def setUp(self):
        server = {"url": "https://docs.example.test/mcp"}
        self.config = {
            "version": 1,
            "mcpServers": {"docs": server},
            "defaults": {"chat": {"mcp": ["docs"]}},
        }
        self.capabilities = {"supportsFunctionCalling": True, "supportsWebSearch": True}
        self.client = mock.create_autospec(VeniceClient, instance=True)
        self.client.api_key = "test-fake-key"
        self.client.get_json.return_value = {"data": [{
            "id": "probe-model",
            "model_spec": {
                "traits": ["default"],
                "capabilities": self.capabilities,
                "availableContextTokens": 8192,
                "pricing": {"input": {"usd": 1}, "output": {"usd": 2}},
            },
        }]}
        self.client.post_json.return_value = {
            "choices": [{"message": {"content": ASSESSMENT}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 23},
        }
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.object(
            chat, "build_client_from_auth", return_value=self.client,
        ))
        patches.enter_context(mock.patch.object(
            chat.userconfig, "load_config", return_value=self.config,
        ))
        self.index = patches.enter_context(mock.patch.object(
            chat._code._index, "discover_store", return_value=None,
        ))
        patches.enter_context(mock.patch.object(chat._mcp, "import_mcp", return_value=object()))

        self.mcp_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        self.mcp_invoke = mock.Mock(return_value={"status": "ok"})
        remote = chat._agent.Tool(
            "docs_find", "Find a document in the connected workspace.", self.mcp_schema,
            self.mcp_invoke,
        )
        self.attach = patches.enter_context(mock.patch.object(
            chat._mcp_client, "attach", return_value=contextlib.nullcontext([remote]),
        ))

    def run_probe(self, *args, prompt="Assess the available schemas."):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(prompt)), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = cli.main([
                "chat", "--tool-fit", "--json", "--max-spend", "0.10",
                "--max-tokens", "128", *args, "-",
            ])
        return rc, out.getvalue(), err.getvalue()

    def test_sends_one_completion_with_available_code_asset_search_and_mcp_schemas(self):
        self.index.return_value = Path("test-index")
        prompt = "Assess whether you can find the official documentation."
        rc, out, err = self.run_probe(
            "--mcp", "docs", "--web-search", "on", "--web-scraping", "--x-search",
            prompt=prompt,
        )

        self.assertEqual(rc, 0, err)
        self.attach.assert_called_once_with([("docs", self.config["mcpServers"]["docs"])])
        self.client.post_json.assert_called_once()
        self.mcp_invoke.assert_not_called()
        route, payload = self.client.post_json.call_args.args
        self.assertEqual(route, "/chat/completions")
        self.assertEqual(payload["model"], "probe-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": prompt}])
        self.assertEqual(payload["max_tokens"], 128)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["tool_choice"], "none")
        params = payload["venice_parameters"]
        self.assertEqual(params["enable_web_search"], "off")
        self.assertFalse(params["enable_web_scraping"])
        self.assertFalse(params["enable_x_search"])

        schemas = {row["function"]["name"]: row["function"] for row in payload["tools"]}
        self.assertEqual(len(schemas), len(payload["tools"]))
        self.assertLessEqual(
            {"read_file", "edit_file", "venice_image", "venice_video", "project_search",
             "venice_web_search", "docs_find"},
            schemas.keys(),
        )
        self.assertEqual(schemas["docs_find"]["parameters"], self.mcp_schema)

        reply = json.loads(out)
        self.assertEqual(reply["final"], ASSESSMENT)
        self.assertEqual(reply["model"], "probe-model")
        self.assertCountEqual(reply["tools"], schemas.keys())
        self.assertEqual(reply["usage"]["prompt_tokens"], 41)
        self.assertEqual(reply["usage"]["completion_tokens"], 23)
        self.assertEqual(reply["usage"]["tools"], {})

    def test_excludes_unavailable_web_search_and_respects_no_mcp_over_config(self):
        self.capabilities["supportsWebSearch"] = False

        rc, out, err = self.run_probe("--no-mcp")

        self.assertEqual(rc, 0, err)
        self.attach.assert_not_called()
        tools = json.loads(out)["tools"]
        self.assertNotIn("venice_web_search", tools)
        self.assertNotIn("docs_find", tools)

    def test_mcp_attach_failure_prevents_completion(self):
        self.attach.side_effect = OSError("server unavailable")

        rc, out, err = self.run_probe("--mcp", "docs")

        self.attach.assert_called_once()
        self.client.post_json.assert_not_called()
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("server unavailable", err)

    def test_tool_call_returned_by_provider_is_never_dispatched(self):
        self.client.post_json.return_value["choices"][0]["message"]["tool_calls"] = [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "docs_find", "arguments": '{"query":"onboarding"}'},
        }]

        rc, out, err = self.run_probe("--mcp", "docs")

        self.client.post_json.assert_called_once()
        self.mcp_invoke.assert_not_called()
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("complete text answer", err)

    def test_rejects_truncated_blank_or_missing_assessment(self):
        for choices in (
            [{"message": {"content": ASSESSMENT}, "finish_reason": "length"}],
            [{"message": {"content": "   "}, "finish_reason": "stop"}],
            [],
        ):
            with self.subTest(choices=choices):
                self.client.post_json.reset_mock()
                self.client.post_json.return_value = {"choices": choices}

                rc, out, err = self.run_probe("--no-mcp")

                self.client.post_json.assert_called_once()
                self.assertNotEqual(rc, 0)
                self.assertEqual(out, "")
                self.assertIn("complete text answer", err)

    def test_rejects_model_without_function_calling_without_spending(self):
        self.capabilities["supportsFunctionCalling"] = False

        rc, out, err = self.run_probe("--no-mcp")

        self.client.post_json.assert_not_called()
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("function calling", err)

    def test_rejects_interactive_mode_and_empty_input_before_spending(self):
        for options, prompt in [(("--interactive",), "Assess tools."), ((), "  ")]:
            with self.subTest(options=options):
                rc, out, err = self.run_probe(*options, prompt=prompt)

                self.client.post_json.assert_not_called()
                self.assertNotEqual(rc, 0)
                self.assertEqual(out, "")
                self.assertIn("one-shot message", err)

    def test_provider_error_does_not_trigger_a_second_completion(self):
        self.client.post_json.side_effect = VeniceAPIError(
            500, "https://venice.example.test/chat/completions", "Unavailable",
        )

        rc, out, err = self.run_probe("--no-mcp")

        self.client.post_json.assert_called_once()
        self.assertNotEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("500", err)
