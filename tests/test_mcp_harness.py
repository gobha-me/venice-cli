"""Exercise every advertised MCP tool through SDK sessions and a loopback API.

No handlers are stubbed. The REST fixture records request paths and bodies, never
credentials. A subprocess session also exercises real stdio framing. Every key
and HOME is synthetic; there is no live mode or credential-file lookup.
"""
import asyncio
import base64
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from venice.client import VeniceClient


HAS_EXTRAS = all(importlib.util.find_spec(name) for name in ("mcp", "openai"))
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZQAAAABJRU5ErkJggg=="
)
AUDIO = b"ID3" + bytes(40)
VIDEO = b"\x00\x00\x00\x18ftypisom" + bytes(40)
IMAGE_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()
LOCAL_TOOLS = {
    "venice_image", "venice_tts", "venice_sfx", "venice_music", "venice_video",
    "venice_upscale", "venice_bg_remove", "venice_image_edit", "venice_chat", "venice_vision",
}
REMOTE_TOOLS = LOCAL_TOOLS | {
    "venice_media_import", "venice_media_delete", "venice_job_status", "venice_job_result",
}


def _model(kind):
    """Return a synthetic, capability-bearing catalog entry for each modality."""
    identifier = {
        "text": "harness-text", "video": "harness-video", "image": "venice-sd35",
        "tts": "tts-kokoro", "music": "elevenlabs-music", "inpaint": "firered-image-edit",
    }.get(kind, "harness-model")
    return {"id": identifier, "type": kind, "model_spec": {
        "traits": ["default"],
        "capabilities": {"supportsVision": True, "supportsFunctionCalling": True},
        "pricing": {"input": {"usd": 0.01}, "image": {"usd": 0.01}},
        "supported_formats": ["mp3"], "default_format": "mp3",
        "constraints": {"durations": ["5s"], "resolutions": ["720p"],
                        "aspect_ratios": ["16:9"]},
    }}


class _API:
    """An allowlisted loopback HTTP API; unscripted routes fail the harness."""

    def __init__(self):
        self.requests = []
        self.unexpected = []

    def __enter__(self):
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, body, content_type="application/json"):
                if isinstance(body, dict):
                    body = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urllib.parse.urlsplit(self.path)
                fixture.requests.append(("GET", parsed.path, None))
                if parsed.path == "/api/v1/models":
                    kind = urllib.parse.parse_qs(parsed.query).get("type", ["text"])[0]
                    self.reply({"data": [_model(kind)]})
                else:
                    fixture.unexpected.append(self.path)
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                path = urllib.parse.urlsplit(self.path).path
                fixture.requests.append(("POST", path, body))
                if path == "/api/v1/chat/completions":
                    self.reply({"id": "harness", "object": "chat.completion", "created": 0,
                                "model": body["model"], "choices": [{"index": 0,
                                "message": {"role": "assistant", "content": "harness response"},
                                "finish_reason": "stop"}]})
                elif path == "/api/v1/image/generate":
                    self.reply({"images": [base64.b64encode(PNG).decode()]})
                elif path in {"/api/v1/image/upscale", "/api/v1/image/background-remove",
                              "/api/v1/image/edit", "/api/v1/image/multi-edit"}:
                    self.reply(PNG, "image/png")
                elif path == "/api/v1/audio/speech" or path == "/api/v1/audio/retrieve":
                    self.reply(AUDIO, "audio/mpeg")
                elif path == "/api/v1/video/retrieve":
                    self.reply(VIDEO, "video/mp4")
                elif path in {"/api/v1/audio/quote", "/api/v1/video/quote"}:
                    self.reply({"quote": 0.01})
                elif path in {"/api/v1/audio/queue", "/api/v1/video/queue"}:
                    self.reply({"queue_id": "job-" + str(len(fixture.requests))})
                elif path in {"/api/v1/audio/complete", "/api/v1/video/complete"}:
                    self.reply({"ok": True})
                else:
                    fixture.unexpected.append(path)
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/api/v1"
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@unittest.skipUnless(HAS_EXTRAS, "MCP and OpenAI extras required for protocol harness")
class MCPHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "input.png"
        self.image.write_bytes(PNG)
        self.output = self.root / "output"
        self.output.mkdir()
        env = mock.patch.dict(os.environ, {
            "HOME": str(self.root), "VENICE_API_KEY": "test-fake-key",
            "VENICE_MCP_OUTPUT_DIR": str(self.output),
            "VENICE_VIDEO_JOBS_FILE": str(self.root / "jobs.json"),
        })
        env.start()
        self.addCleanup(env.stop)

    async def call(self, session, name, arguments, expected="ok"):
        result = await session.call_tool(name, arguments)
        self.assertFalse(result.is_error, name)
        payload = json.loads(result.content[0].text)
        self.assertEqual(payload["status"], expected, (name, payload))
        return payload

    async def exercise_local(self, session):
        listed = {tool.name for tool in (await session.list_tools()).tools}
        self.assertEqual(listed, LOCAL_TOOLS)
        args = {
            "venice_image": {"prompt": "a tree", "transparent_background": False},
            "venice_tts": {"text": "hello"},
            "venice_sfx": {"prompt": "wind"},
            "venice_music": {"prompt": "gentle rain"},
            "venice_video": {"prompt": "a tree", "image_strength": 0.5},
            "venice_upscale": {"input_path": str(self.image), "scale": 2},
            "venice_bg_remove": {"input_path": str(self.image)},
            "venice_image_edit": {"prompt": "add leaves", "input_path": str(self.image)},
            "venice_chat": {"message": "hello"},
            "venice_vision": {"input_path": str(self.image)},
        }
        for name, arguments in args.items():
            if name not in {"venice_chat", "venice_vision"}:
                arguments.update(confirm=True, output_dir=str(self.output))
            with self.subTest(tool=name):
                payload = await self.call(session, name, arguments)
                for path in payload.get("paths", []) + ([payload["path"]] if "path" in payload else []):
                    self.assertTrue(Path(path).is_file(), name)
        self.assertEqual(set(args), listed)

    def test_every_local_tool_over_in_process_protocol(self):
        from mcp import Client
        from venice.mcp_server import build_server

        async def run(api):
            server = build_server(VeniceClient("test-fake-key", base_url=api.base_url),
                                  doc={}, root=str(self.root))
            async with Client(server) as session:
                await self.exercise_local(session)

        with _API() as api:
            asyncio.run(run(api))
        self.assertEqual(api.unexpected, [])
        cleanups = [body for _, path, body in api.requests if path == "/api/v1/video/complete"]
        self.assertTrue(cleanups)
        self.assertTrue(all(set(body) == {"queue_id"} for body in cleanups))

    def test_every_local_tool_over_real_stdio(self):
        from mcp import Client, StdioServerParameters

        async def run(api):
            env = {**os.environ, "VENICE_BASE_URL": api.base_url,
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
            params = StdioServerParameters(command=sys.executable,
                                          args=["-m", "venice", "mcp-serve"],
                                          env=env, cwd=str(self.root))
            async with Client(params) as session:
                await self.exercise_local(session)

        with _API() as api:
            asyncio.run(run(api))
        self.assertEqual(api.unexpected, [])

    def test_every_remote_media_tool_and_job_lifecycle(self):
        from mcp import Client
        from mcp.server.auth.provider import AccessToken
        from venice import mcp_server

        access = AccessToken(token="fake-mcp-token", client_id="harness-client",
                             subject="harness-user", scopes=["venice:mcp"])

        async def run(api):
            server = mcp_server.build_http_server(
                VeniceClient("test-fake-key", base_url=api.base_url), doc={},
                public_url="https://mcp.example.test/mcp", issuer_url="https://auth.example.test",
                jwks_url="https://auth.example.test/jwks", audience="harness", scopes=["venice:mcp"],
                media_dir=str(self.root / "media"), remote_max_spend=1,
                allow_dynamic_spend=True,
            )
            async with Client(server) as session:
                listed = {tool.name for tool in (await session.list_tools()).tools}
                self.assertEqual(listed, REMOTE_TOOLS)
                called = set()

                async def call(name, arguments, expected="ok"):
                    called.add(name)
                    return await self.call(session, name, arguments, expected)

                imported = await call("venice_media_import", {"url": IMAGE_URL})
                uri = imported["media"][0]["uri"]
                await call("venice_chat", {"message": "hello"})
                await call("venice_vision", {"media_uri": uri})
                for name, arguments in {
                    "venice_image": {"prompt": "a tree"},
                    "venice_tts": {"text": "hello"},
                    "venice_upscale": {"media_uri": uri, "scale": 2},
                    "venice_bg_remove": {"media_uri": uri},
                    "venice_image_edit": {"prompt": "leaves", "media_uri": uri},
                }.items():
                    result = await call(name, {**arguments, "confirm": True})
                    self.assertTrue(result["media"], name)
                for name in ("venice_sfx", "venice_music", "venice_video"):
                    queued = await call(name, {"prompt": "gentle rain", "confirm": True}, "queued")
                    job_id = queued["job_id"]
                    status = await call("venice_job_status", {"job_id": job_id})
                    result = await call("venice_job_result", {"job_id": job_id})
                    self.assertEqual(status["media"], result["media"])
                    self.assertTrue(result["media"])
                await call("venice_media_delete", {"media_uri": uri})
                self.assertEqual(called, listed)

        with _API() as api, mock.patch.object(mcp_server, "get_access_token", return_value=access):
            asyncio.run(run(api))
        self.assertEqual(api.unexpected, [])

    def test_minimal_remote_profile_calls_both_tools(self):
        from mcp import Client
        from venice.mcp_server import build_http_server

        async def run(api):
            server = build_http_server(
                VeniceClient("test-fake-key", base_url=api.base_url), doc={},
                public_url="https://mcp.example.test/mcp", issuer_url="https://auth.example.test",
                jwks_url="https://auth.example.test/jwks", audience="harness", scopes=["venice:mcp"],
            )
            async with Client(server) as session:
                self.assertEqual({t.name for t in (await session.list_tools()).tools},
                                 {"venice_chat", "venice_vision"})
                await self.call(session, "venice_chat", {"message": "hello"})
                await self.call(session, "venice_vision", {"image_url": IMAGE_URL})

        with _API() as api:
            asyncio.run(run(api))
        self.assertEqual(api.unexpected, [])
