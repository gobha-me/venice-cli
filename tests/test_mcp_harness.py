"""Every advertised MCP tool, called through real MCP SDK sessions (r1 audit).

Test author of record: Claude (production change 1a844f2 is Codex's).

Profiles and transports:

* ``stdio``        -- ``python -m venice mcp-serve`` as a subprocess (real stdio
                      framing, real CLI auth via ``$VENICE_API_KEY``, temp HOME);
* ``stdio``        -- the same server in-process (``Client(server)``);
* ``http_media``   -- ``build_http_server(media_dir=...)`` served through its real
                      streamable-HTTP ASGI app with a bearer-token verifier, so the
                      principal comes from the auth middleware, not a patch;
* ``http_minimal`` -- ``build_http_server()`` in-process.

The tool set each session *lists* must equal the committed
contracts/openapi/mcp-surface.json profile, and the set the test *calls* must
equal the listed set -- so an added or dropped tool fails twice. The only fake
is the Venice HTTP peer (tests/_r1_loopback.py); no handler is stubbed. Keys are
synthetic and HOME is a temp dir.

Requires the ``mcp`` and ``openai`` extras (CI installs ``.[all,test]``).
"""
import asyncio
import base64
import importlib.util
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import venice
from tests._r1_loopback import (
    MP3,
    MP4,
    PNG,
    LoopbackVenice,
    bytes_reply,
    echo_bearer,
    json_reply,
)
from venice import config
from venice.client import VeniceClient

HAS_EXTRAS = all(importlib.util.find_spec(n) is not None for n in ("mcp", "openai"))
SRC = Path(venice.__file__).resolve().parents[1]
SURFACE = json.loads((Path(__file__).resolve().parents[1] / "contracts" / "openapi"
                      / "mcp-surface.json").read_text(encoding="utf-8"))
KEY = "r1-harness-synthetic-credential-Lp3"
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()
DEFAULT_IDS = {
    "text": "r1-text", "video": "r1-video", "image": "venice-sd35",
    "inpaint": "firered-image-edit", "tts": "tts-kokoro", "music": "elevenlabs-music",
    "sfx": "elevenlabs-sound-effects-v2",
}


def _entry(identifier, default):
    return {"id": identifier, "type": "any", "model_spec": {
        "traits": ["default"] if default else [],
        "capabilities": {"supportsVision": True, "supportsFunctionCalling": True},
        "pricing": {"input": {"usd": 0.001}, "output": {"usd": 0.001},
                    "image": {"usd": 0.01}},
        "supported_formats": ["mp3"], "default_format": "mp3",
        "constraints": {"durations": ["5s"], "resolutions": ["720p"],
                        "aspect_ratios": ["16:9"]},
    }}


def _catalog(seen, _bearer):
    kind = (seen.query.get("type") or ["text"])[0]
    first = DEFAULT_IDS.get(kind, "r1-" + kind)
    rest = [i for i in DEFAULT_IDS.values() if i != first]
    return json_reply({"data": [_entry(first, True)] + [_entry(i, False) for i in rest]})


def _chat(seen, _bearer):
    return json_reply({
        "id": "r1", "object": "chat.completion", "created": 0, "model": seen.body["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "r1 reply"}}],
    })


def script_venice(api):
    """Answer every endpoint the media/chat tools use, and nothing else."""
    counter = itertools.count(1)

    def queued(kind):
        return lambda seen, _b: json_reply({"queue_id": f"{kind}-q{next(counter)}",
                                            "model": seen.body.get("model")})

    api.on("GET", "/models", _catalog)
    api.on("POST", "/chat/completions", _chat)
    api.on("POST", "/image/generate", json_reply({"images": [base64.b64encode(PNG).decode()]}))
    for path in ("/image/upscale", "/image/background-remove", "/image/edit",
                 "/image/multi-edit"):
        api.on("POST", path, bytes_reply(PNG, "image/png"))
    api.on("POST", "/audio/speech", bytes_reply(MP3, "audio/mpeg"))
    for kind in ("audio", "video"):
        api.on("POST", f"/{kind}/quote", json_reply({"quote": 0.01}))
        api.on("POST", f"/{kind}/queue", queued(kind))
        api.on("POST", f"/{kind}/complete", json_reply({"success": True}))
    api.on("POST", "/audio/retrieve", bytes_reply(MP3, "audio/mpeg"))
    api.on("POST", "/video/retrieve", bytes_reply(MP4, "video/mp4"))
    return api


def _payload(result):
    return json.loads(result.content[0].text)


@unittest.skipUnless(HAS_EXTRAS, "MCP and OpenAI extras required for protocol harness")
class _Harness(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.image = self.root / "input.png"
        self.image.write_bytes(PNG)
        self.out = self.root / "out"
        self.out.mkdir()
        self.env = {
            "HOME": str(self.root),
            "XDG_CONFIG_HOME": str(self.root / ".config"),
            "VENICE_MCP_OUTPUT_DIR": str(self.out),
            config.ENV_VIDEO_JOBS_FILE: str(self.root / "jobs.json"),
        }
        saved = {k: os.environ.get(k) for k in (*self.env, "VENICE_API_KEY")}
        os.environ.update(self.env)
        os.environ["VENICE_API_KEY"] = KEY

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)

    async def ok(self, session, name, arguments, status="ok"):
        result = await session.call_tool(name, arguments)
        payload = _payload(result)
        self.assertFalse(result.is_error, (name, payload))
        self.assertEqual(payload["status"], status, (name, payload))
        return payload

    async def rejected(self, session, name, arguments):
        """A protocol- or tool-level refusal; never a silent success."""
        from mcp.shared.exceptions import MCPError

        try:
            result = await session.call_tool(name, arguments)
        except MCPError:  # the SDK raises MCPError for protocol errors
            return None
        self.assertTrue(result.is_error, (name, arguments, result))
        return result

    def local_args(self):
        paid = {"confirm": True, "output_dir": str(self.out)}
        return {
            "venice_image": {"prompt": "a tree", **paid},
            "venice_tts": {"text": "hello", **paid},
            "venice_sfx": {"prompt": "wind", **paid},
            "venice_music": {"prompt": "rain", **paid},
            "venice_video": {"prompt": "a tree", "image_strength": 0.5, **paid},
            "venice_upscale": {"input_path": str(self.image), "scale": 2, **paid},
            "venice_bg_remove": {"input_path": str(self.image), **paid},
            "venice_image_edit": {"prompt": "leaves", "input_path": str(self.image), **paid},
            "venice_chat": {"message": "hello"},
            "venice_vision": {"input_path": str(self.image)},
        }

    async def exercise_local(self, session):
        listed = {t.name for t in (await session.list_tools()).tools}
        self.assertEqual(listed, set(SURFACE["stdio"]))
        args = self.local_args()
        self.assertEqual(set(args), listed, "harness must call every listed tool")
        for name, arguments in args.items():
            with self.subTest(tool=name):
                payload = await self.ok(session, name, arguments)
                written = payload.get("paths") or ([payload["path"]] if "path" in payload else [])
                if name not in ("venice_chat", "venice_vision"):
                    self.assertTrue(written, payload)
                for path in written:
                    self.assertTrue(Path(path).is_file(), path)
                    self.assertTrue(Path(path).resolve().is_relative_to(self.out.resolve()))
        # Unhappy paths through the same session.
        await self.rejected(session, "venice_not_a_tool", {})
        await self.rejected(session, "venice_image", {})                 # missing prompt
        await self.rejected(session, "venice_upscale",
                            {"input_path": str(self.image), "scale": 3})  # not 2|4
        gated = await session.call_tool("venice_upscale", {"input_path": str(self.image)})
        self.assertEqual(_payload(gated)["status"], "confirmation_required")

    def assert_cleanup_contract(self, api):
        self.assertEqual(api.unexpected, [])
        video_ids = [b["queue_id"] for b in api.bodies("/video/retrieve")]
        self.assertTrue(video_ids)
        self.assertTrue(all(set(b) == {"model", "queue_id"} for b in api.bodies("/video/retrieve")))
        self.assertEqual(api.bodies("/video/complete"),
                         [{"queue_id": q} for q in dict.fromkeys(video_ids)])
        audio = api.bodies("/audio/complete")
        self.assertTrue(audio)
        self.assertTrue(all(set(b) == {"model", "queue_id"} and b["model"] for b in audio))
        self.assertTrue(all(s.authorized for s in api.seen))


class LocalProfile(_Harness):
    def test_every_stdio_tool_over_a_real_subprocess(self):
        from mcp import Client, StdioServerParameters

        async def run(api):
            env = {**os.environ, "VENICE_BASE_URL": api.base_url, "PYTHONPATH": str(SRC)}
            params = StdioServerParameters(command=sys.executable,
                                           args=["-m", "venice", "mcp-serve"],
                                           env=env, cwd=str(self.root))
            async with Client(params) as session:
                await self.exercise_local(session)

        with script_venice(LoopbackVenice()) as api:
            asyncio.run(run(api))
        self.assert_cleanup_contract(api)

    def test_every_stdio_tool_in_process(self):
        from mcp import Client

        from venice.mcp_server import build_server

        async def run(api):
            server = build_server(VeniceClient(KEY, base_url=api.base_url), doc={},
                                  root=str(self.root))
            async with Client(server) as session:
                await self.exercise_local(session)

        with script_venice(LoopbackVenice()) as api:
            asyncio.run(run(api))
        self.assert_cleanup_contract(api)

    def test_upstream_credential_echo_never_reaches_the_tool_result(self):
        from mcp import Client, StdioServerParameters

        async def run(api):
            env = {**os.environ, "VENICE_BASE_URL": api.base_url, "PYTHONPATH": str(SRC)}
            params = StdioServerParameters(command=sys.executable,
                                           args=["-m", "venice", "mcp-serve"],
                                           env=env, cwd=str(self.root))
            async with Client(params) as session:
                results = []
                for name, arguments in (
                    ("venice_image", {"prompt": "x", "confirm": True,
                                      "output_dir": str(self.out)}),
                    ("venice_video", {"prompt": "x", "confirm": True,
                                      "output_dir": str(self.out)}),
                    ("venice_chat", {"message": "x"}),
                ):
                    result = await session.call_tool(name, arguments)
                    results.append((name, result))
                return results

        with LoopbackVenice() as api:
            api.on("GET", "/models", _catalog)
            hostile = echo_bearer(401, '{"error": {"code": "{key}", "message": "bad {key}"}}')
            for path in ("/image/generate", "/video/quote", "/chat/completions"):
                api.on("POST", path, hostile)
            results = asyncio.run(run(api))
        for name, result in results:
            with self.subTest(tool=name):
                text = "".join(getattr(block, "text", "") for block in result.content)
                self.assertEqual(_payload(result)["status"], "error", text)
                self.assertNotIn(KEY, text)
                self.assertNotIn(KEY[:12], text)
        self.assertNotIn("/video/queue", api.paths())


class RemoteProfiles(_Harness):
    PUBLIC = "https://mcp.example.test/mcp"

    class Verifier:
        async def verify_token(self, token):
            from mcp.server.auth.provider import AccessToken
            if token != "r1-mcp-token":
                return None
            return AccessToken(token=token, client_id="r1-client", subject="r1-user",
                               scopes=["venice:mcp"])

    def build(self, api, **extra):
        from venice.mcp_server import build_http_server
        return build_http_server(
            VeniceClient(KEY, base_url=api.base_url), doc={}, public_url=self.PUBLIC,
            issuer_url="https://auth.example.test", jwks_url="https://auth.example.test/jwks",
            audience="r1", scopes=["venice:mcp"], token_verifier=self.Verifier(), **extra,
        )

    async def over_http(self, server, body):
        """Run ``body(session)`` through the server's real authenticated ASGI app."""
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.server.transport_security import TransportSecuritySettings

        app = server.streamable_http_app(
            stateless_http=True, json_response=True, host="mcp.example.test",
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True, allowed_hosts=["mcp.example.test"],
                allowed_origins=["https://allowed.example"]),
        )
        async with server.session_manager.run(), httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url=self.PUBLIC,
            headers={"Authorization": "Bearer r1-mcp-token"},
        ) as http, Client(streamable_http_client(self.PUBLIC, http_client=http)) as session:
            return await body(session)

    def test_every_media_tool_and_job_lifecycle(self):
        async def body(session):
            listed = {t.name for t in (await session.list_tools()).tools}
            self.assertEqual(listed, set(SURFACE["http_media"]))
            called = set()

            async def call(name, arguments, status="ok"):
                called.add(name)
                return await self.ok(session, name, arguments, status)

            uri = (await call("venice_media_import", {"url": DATA_URL}))["media"][0]["uri"]
            await call("venice_chat", {"message": "hello"})
            await call("venice_vision", {"media_uri": uri})
            # No spend without confirm: the gate answers before any paid request.
            gated = await call("venice_image", {"prompt": "a tree"}, "confirmation_required")
            self.assertNotIn("media", gated)
            for name, arguments in (
                ("venice_image", {"prompt": "a tree"}),
                ("venice_tts", {"text": "hello"}),
                ("venice_upscale", {"media_uri": uri, "scale": 2}),
                ("venice_bg_remove", {"media_uri": uri}),
                ("venice_image_edit", {"prompt": "leaves", "media_uri": uri}),
            ):
                stored = await call(name, {**arguments, "confirm": True})
                self.assertTrue(stored["media"], name)
                self.assertNotIn("path", stored)
                self.assertNotIn("paths", stored)
            for name in ("venice_sfx", "venice_music", "venice_video"):
                queued = await call(name, {"prompt": "rain", "confirm": True}, "queued")
                self.assertNotIn("queue_id", queued)
                status = await call("venice_job_status", {"job_id": queued["job_id"]})
                result = await call("venice_job_result", {"job_id": queued["job_id"]})
                self.assertTrue(result["media"])
                self.assertEqual(status["media"], result["media"])
            # Unhappy paths.
            both = await session.call_tool("venice_vision",
                                           {"media_uri": uri, "image_url": DATA_URL})
            self.assertTrue(both.is_error)
            missing = await session.call_tool("venice_job_status", {"job_id": "no-such-job"})
            self.assertTrue(missing.is_error)
            await call("venice_media_delete", {"media_uri": uri})
            again = await session.call_tool("venice_media_delete", {"media_uri": uri})
            self.assertEqual(_payload(again)["status"], "error")
            self.assertEqual(called, listed, "harness must call every listed tool")

        with script_venice(LoopbackVenice()) as api:
            server = self.build(api, media_dir=str(self.root / "media"), remote_max_spend=1,
                                allow_dynamic_spend=True)
            asyncio.run(self.over_http(server, body))
        self.assert_cleanup_contract(api)
        self.assertEqual(len(api.bodies("/image/generate")), 1)  # the gated call spent nothing

    def test_minimal_profile_lists_and_calls_both_tools(self):
        from mcp import Client

        async def run(server):
            async with Client(server) as session:
                listed = {t.name for t in (await session.list_tools()).tools}
                self.assertEqual(listed, set(SURFACE["http_minimal"]))
                await self.ok(session, "venice_chat", {"message": "hello"})
                await self.ok(session, "venice_vision", {"image_url": DATA_URL})
                for bad in ("https://user:pw@images.example.test/x.png", str(self.image),
                            "ftp://images.example.test/x.png", "data:image/png,notbase64"):
                    with self.subTest(image_url=bad):
                        result = await session.call_tool("venice_vision", {"image_url": bad})
                        self.assertTrue(result.is_error)
                        self.assertEqual(_payload(result)["status"], "error")
                await self.rejected(session, "venice_image", {"prompt": "x"})
                return listed

        with script_venice(LoopbackVenice()) as api:
            asyncio.run(run(self.build(api)))
        self.assertEqual(api.unexpected, [])
        self.assertEqual(len(api.bodies("/chat/completions")), 2)


if __name__ == "__main__":
    unittest.main()
