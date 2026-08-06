"""Tests for the drive suite's fake Venice API (tests/_venice_fake_server.py).

The fake is the seam the #80 drive tests depend on, so it gets its own direct
coverage: if a drive test goes red we want to know immediately whether the CLI
broke or the fixture did. These cases need no pty and no pexpect -- they drive
the fake with `VeniceClient` (the same urllib path every stdlib command uses)
and with the OpenAI SDK, so they keep running for contributors without the
`[test]` extra.

No network, no real key: the server binds 127.0.0.1 on an ephemeral port.
"""
import importlib.util
import json
import os
import unittest
import urllib.error
import urllib.request
from unittest import mock

from tests import _venice_fake_server as fake
from venice.client import VeniceAPIError, VeniceClient

_HAS_OPENAI = importlib.util.find_spec("openai") is not None

# urllib and httpx both honor these, and neither auto-bypasses loopback. The
# drive tests get this for free (the child's env is built from scratch), but
# these cases call the fake from *this* process -- so without clearing them a
# developer behind a proxy or on a VPN sees the whole fixture suite go red, and
# on a proxy that resolves rather than refuses, loopback traffic leaves the box.
_PROXY_VARS = {
    v: "" for v in (
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    )
}


class _NoProxyCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, _PROXY_VARS)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestFakeServerViaClient(_NoProxyCase):
    """Drive the fake through VeniceClient -- the real urllib path."""

    def setUp(self):
        super().setUp()
        self.api = fake.FakeVenice().start()
        self.addCleanup(self.api.stop)
        self.client = VeniceClient(api_key="test-fake-key", base_url=self.api.base_url)

    def test_models_catalog_carries_a_default_trait(self):
        from venice.commands import _models

        doc = self.client.get_json("/models", params={"type": "text"})
        models = doc["data"]
        self.assertEqual(_models.default_model(models), "llama-3.3-70b")
        # Prefix-stripped path and the query both land in the request log.
        self.assertEqual(self.api.paths, ["/models"])
        self.assertEqual(self.api.requests[0]["query"], {"type": ["text"]})

    def test_image_models_price_resolves_to_one_cent(self):
        from venice.commands import image

        price = image._fetch_image_price(self.client, "venice-sd35")
        self.assertEqual(price, 0.01)

    def test_balance_block_sums_usd_and_diem(self):
        from venice import billing

        info = billing.fetch_balance(self.client)
        self.assertEqual(info["total"], 12.5)
        self.assertEqual(info["tier"], "explorer")

    def test_image_generate_returns_decodable_bytes(self):
        from venice.commands import image

        doc = self.client.post_json("/image/generate", {"prompt": "a cat"})
        self.assertEqual(image._decode_images(doc), [fake.PNG_BYTES])
        # The POST body is recorded for later assertions.
        self.assertEqual(self.api.bodies("/image/generate"), [{"prompt": "a cat"}])

    def test_authorization_presence_is_recorded_without_the_value(self):
        self.client.get_json("/models", params={"type": "text"})
        entry = self.api.requests[0]
        self.assertTrue(entry["has_auth"])
        # The key must not be retrievable from the request log (AGENTS.md).
        self.assertNotIn("test-fake-key", json.dumps(entry))

    def test_unknown_route_is_404_not_5xx(self):
        with self.assertRaises(VeniceAPIError) as ctx:
            self.client.get_json("/nope")
        self.assertEqual(ctx.exception.status, 404)

    def test_reset_clears_the_request_log(self):
        self.client.get_json("/models", params={"type": "text"})
        self.api.reset()
        self.assertEqual(self.api.paths, [])


@unittest.skipUnless(_HAS_OPENAI, "openai SDK not installed")
class TestFakeServerViaOpenAI(_NoProxyCase):
    """The SDK path (`venice chat`/`code`/`embed`) talks to the same fake."""

    def setUp(self):
        super().setUp()
        self.api = fake.FakeVenice().start()
        self.addCleanup(self.api.stop)
        import openai

        self.oai = openai.OpenAI(api_key="test-fake-key", base_url=self.api.base_url)

    def test_non_streamed_completion(self):
        self.api.reply("HELLO-FROM-FAKE")
        out = self.oai.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(out.choices[0].message.content, "HELLO-FROM-FAKE")
        self.assertEqual(out.usage.total_tokens, 14)

    def test_streamed_completion_yields_deltas_then_usage(self):
        self.api.reply_chunks("HELLO-", "FROM-", "FAKE")
        stream = self.oai.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        pieces, usage = [], None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                pieces.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                usage = chunk.usage
        self.assertEqual("".join(pieces), "HELLO-FROM-FAKE")
        self.assertIsNotNone(usage, "the final usage frame must reach the SDK")
        self.assertEqual(usage.total_tokens, 14)

    def test_replies_are_queued_in_order_and_bodies_recorded(self):
        self.api.reply("FIRST")
        self.api.reply("SECOND")
        for model in ("llama-3.3-70b", "venice-uncensored"):
            self.oai.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "hi"}]
            )
        bodies = self.api.bodies("/chat/completions")
        self.assertEqual([b["model"] for b in bodies],
                         ["llama-3.3-70b", "venice-uncensored"])

    def test_unscripted_turn_is_loud_rather_than_hanging(self):
        out = self.oai.chat.completions.create(
            model="llama-3.3-70b", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertEqual(out.choices[0].message.content, "UNSCRIPTED-REPLY")


class TestFakeServerLifecycle(_NoProxyCase):
    def test_stop_closes_the_port(self):
        api = fake.FakeVenice().start()
        url = api.base_url + "/models?type=text"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
        api.stop()
        # URLError specifically, not bare Exception: a broad assertRaises would
        # also pass on a proxy or DNS error, i.e. for the wrong reason.
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen(url, timeout=5)


if __name__ == "__main__":
    unittest.main()
