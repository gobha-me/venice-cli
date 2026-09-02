"""Authenticated Streamable HTTP coverage for issue #31."""
import argparse
import asyncio
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from venice.commands import mcp_serve


try:
    _HAS_MCP2 = importlib.util.find_spec("mcp.server.mcpserver") is not None
except ModuleNotFoundError:
    _HAS_MCP2 = False


def _args(**overrides):
    values = dict(
        http=True,
        host="127.0.0.1",
        port=8000,
        host_image_content=False,
        public_url="https://mcp.example.test/mcp",
        oauth_issuer="https://auth.example.test",
        oauth_jwks_url="https://auth.example.test/.well-known/jwks.json",
        oauth_audience="https://mcp.example.test",
        oauth_scope=["venice:mcp"],
        allowed_origin=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestRemoteConfig(unittest.TestCase):
    def test_parser_registers_http_surface_and_bounds_port(self):
        from venice import cli

        args = cli.build_parser().parse_args([
            "mcp-serve", "--http", "--port", "8443",
            "--public-url", "https://mcp.example.test/mcp",
            "--oauth-issuer", "https://auth.example.test",
            "--oauth-jwks-url", "https://auth.example.test/jwks",
            "--oauth-audience", "venice-api",
            "--oauth-scope", "venice:mcp",
            "--oauth-scope", "profile",
        ])
        self.assertTrue(args.http)
        self.assertEqual(args.port, 8443)
        self.assertEqual(args.oauth_scope, ["venice:mcp", "profile"])
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["mcp-serve", "--port", "0"])

    def test_cli_values_override_environment(self):
        cfg = mcp_serve.resolve_remote_config(
            _args(allowed_origin=["https://claude.example"]),
            {
                "VENICE_MCP_PUBLIC_URL": "https://wrong.example/mcp",
                "VENICE_MCP_OAUTH_SCOPES": "wrong",
                "VENICE_MCP_ALLOWED_ORIGINS": "https://wrong.example",
            },
        )
        self.assertEqual(cfg.public_url, "https://mcp.example.test/mcp")
        self.assertEqual(cfg.scopes, ("venice:mcp",))
        self.assertEqual(cfg.allowed_origins, ("https://claude.example",))

    def test_environment_fallback_is_complete_and_deduplicated(self):
        cfg = mcp_serve.resolve_remote_config(
            _args(
                public_url=None,
                oauth_issuer=None,
                oauth_jwks_url=None,
                oauth_audience=None,
                oauth_scope=None,
            ),
            {
                "VENICE_MCP_PUBLIC_URL": "https://mcp.example.test/mcp",
                "VENICE_MCP_OAUTH_ISSUER": "https://auth.example.test",
                "VENICE_MCP_OAUTH_JWKS_URL": "https://auth.example.test/jwks",
                "VENICE_MCP_OAUTH_AUDIENCE": "venice-api",
                "VENICE_MCP_OAUTH_SCOPES": "venice:mcp profile venice:mcp",
                "VENICE_MCP_ALLOWED_ORIGINS": (
                    "https://one.example https://two.example"
                ),
            },
        )
        self.assertEqual(cfg.scopes, ("venice:mcp", "profile"))
        self.assertEqual(
            cfg.allowed_origins,
            ("https://one.example", "https://two.example"),
        )

    def test_remote_config_rejects_missing_or_unsafe_values(self):
        cases = (
            _args(public_url=None),
            _args(oauth_scope=[]),
            _args(oauth_scope=["not a scope"]),
            _args(public_url="http://mcp.example.test/mcp"),
            _args(public_url="https://mcp.example.test/other"),
            _args(oauth_issuer="https://user:pass@auth.example.test"),
            _args(oauth_jwks_url="https://auth.example.test/jwks?key=1"),
            _args(allowed_origin=["https://browser.example/path"]),
        )
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(mcp_serve.RemoteConfigError):
                    mcp_serve.resolve_remote_config(args, {})

    def test_media_config_is_bounded_and_dynamic_spend_is_fail_closed(self):
        cfg = mcp_serve.resolve_remote_config(
            _args(media_dir="/srv/venice-media", remote_max_spend=2.5),
            {"VENICE_MCP_MEDIA_TTL_SECONDS": "3600"},
        )
        self.assertEqual(cfg.media_dir, "/srv/venice-media")
        self.assertEqual(cfg.media_ttl_seconds, 3600)
        self.assertEqual(cfg.remote_max_spend, 2.5)
        self.assertFalse(cfg.allow_dynamic_spend)
        with self.assertRaises(mcp_serve.RemoteConfigError):
            mcp_serve.resolve_remote_config(
                _args(media_dir="/srv/venice-media", allow_dynamic_spend=True), {}
            )
        with self.assertRaises(mcp_serve.RemoteConfigError):
            mcp_serve.resolve_remote_config(
                _args(media_dir=None), {"VENICE_MCP_MEDIA_MAX_OBJECTS": "0"}
            )
        with self.assertRaises(mcp_serve.RemoteConfigError):
            mcp_serve.resolve_remote_config(_args(media_dir="relative/media"), {})

    def test_http_native_content_combination_fails_before_auth(self):
        with mock.patch.object(mcp_serve._mcp, "import_mcp", return_value=object()), \
             mock.patch.object(mcp_serve._openai, "import_openai", return_value=object()), \
             mock.patch.object(mcp_serve, "build_client_from_auth") as build, \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = mcp_serve._run(_args(host_image_content=True))
        self.assertEqual(rc, 2)
        build.assert_not_called()

    def test_http_flags_without_http_are_rejected(self):
        args = _args(http=False, public_url=None, oauth_issuer=None,
                     oauth_jwks_url=None, oauth_audience=None, oauth_scope=None,
                     host="0.0.0.0")
        with mock.patch.object(mcp_serve._mcp, "import_mcp", return_value=object()), \
             mock.patch.object(mcp_serve, "build_client_from_auth") as build, \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = mcp_serve._run(args)
        self.assertEqual(rc, 2)
        build.assert_not_called()

    @unittest.skipUnless(_HAS_MCP2, "MCP SDK 2.x is required")
    def test_http_run_forwards_only_validated_remote_settings(self):
        from venice import mcp_server

        client = object()
        doc = {"defaults": {}}
        with mock.patch.object(mcp_serve._mcp, "import_mcp", return_value=object()), \
             mock.patch.object(mcp_serve._openai, "import_openai", return_value=object()), \
             mock.patch.object(
                 mcp_serve, "build_client_from_auth", return_value=client
             ), \
             mock.patch.object(
                 mcp_serve.userconfig, "load_config", return_value=doc
             ), \
             mock.patch.object(mcp_server, "serve_http") as serve_http, \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = mcp_serve._run(_args(host="0.0.0.0", port=8080))
        self.assertEqual(rc, 0)
        serve_http.assert_called_once_with(
            client,
            doc=doc,
            host="0.0.0.0",
            port=8080,
            public_url="https://mcp.example.test/mcp",
            issuer_url="https://auth.example.test",
            jwks_url="https://auth.example.test/.well-known/jwks.json",
            audience="https://mcp.example.test",
            scopes=("venice:mcp",),
            allowed_origins=(),
            media_dir=None,
            media_ttl_seconds=24 * 60 * 60,
            media_max_objects=100,
            media_principal_max_bytes=1024 * 1024 * 1024,
            media_global_max_bytes=4 * 1024 * 1024 * 1024,
            media_max_pending_jobs=4,
            media_global_max_pending_jobs=16,
            media_mcp_read_max_bytes=32 * 1024 * 1024,
            remote_max_spend=0.10,
            allow_dynamic_spend=False,
        )


@unittest.skipUnless(_HAS_MCP2, "MCP SDK 2.x is required")
class TestJWTVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cryptography.hazmat.primitives.asymmetric import rsa

        cls.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        cls.public_key = cls.private_key.public_key()

    def _verifier(self):
        from venice.mcp_server import JWKSJWTVerifier

        verifier = JWKSJWTVerifier(
            jwks_url="https://auth.example.test/jwks",
            issuer="https://auth.example.test",
            audience="venice-api",
            required_scopes=["venice:mcp"],
        )
        verifier._jwks.get_signing_key_from_jwt = mock.Mock(
            return_value=SimpleNamespace(key=self.public_key)
        )
        return verifier

    def _token(self, **overrides):
        import jwt

        claims = {
            "iss": "https://auth.example.test",
            "aud": "venice-api",
            "sub": "user-1",
            "client_id": "claude-client",
            "scope": "openid venice:mcp",
            "exp": int(time.time()) + 300,
        }
        claims.update(overrides)
        return jwt.encode(
            claims,
            self.private_key,
            algorithm="RS256",
            headers={"kid": "key-1"},
        )

    def test_valid_token_maps_only_bounded_identity_claims(self):
        verifier = self._verifier()
        access = asyncio.run(verifier.verify_token(self._token(secret="do-not-copy")))
        self.assertEqual(access.client_id, "claude-client")
        self.assertEqual(access.subject, "user-1")
        self.assertEqual(access.resource, "venice-api")
        self.assertEqual(access.claims, {"iss": "https://auth.example.test"})
        self.assertEqual(access.scopes, ["openid", "venice:mcp"])

    def test_azp_and_list_scp_are_supported(self):
        verifier = self._verifier()
        token = self._token(client_id=None, azp="claude", scope=None, scp=["venice:mcp"])
        access = asyncio.run(verifier.verify_token(token))
        self.assertEqual(access.client_id, "claude")

    def test_bad_claims_and_jwks_fail_closed(self):
        now = int(time.time())
        cases = (
            self._token(iss="https://other.example"),
            self._token(aud="other-api"),
            self._token(scope="openid"),
            self._token(exp=now - 120),
            self._token(sub=""),
            self._token(client_id=None),
        )
        for token in cases:
            with self.subTest(token_length=len(token)):
                self.assertIsNone(asyncio.run(self._verifier().verify_token(token)))

        verifier = self._verifier()
        verifier._jwks.get_signing_key_from_jwt.side_effect = OSError("offline")
        marker = self._token()
        with mock.patch.object(sys, "stdout", io.StringIO()) as out, \
             mock.patch.object(sys, "stderr", io.StringIO()) as err:
            self.assertIsNone(asyncio.run(verifier.verify_token(marker)))
        self.assertNotIn(marker, out.getvalue() + err.getvalue())

    def test_symmetric_missing_key_and_oversized_tokens_are_rejected_pre_jwks(self):
        import jwt

        verifier = self._verifier()
        symmetric = jwt.encode(
            {"exp": int(time.time()) + 300},
            "test-secret-that-is-at-least-32-bytes-long",
            algorithm="HS256",
            headers={"kid": "key-1"},
        )
        no_kid = jwt.encode(
            {"exp": int(time.time()) + 300},
            self.private_key,
            algorithm="RS256",
        )
        for token in (symmetric, no_kid, "x" * (16 * 1024 + 1)):
            self.assertIsNone(asyncio.run(verifier.verify_token(token)))
        verifier._jwks.get_signing_key_from_jwt.assert_not_called()


@unittest.skipUnless(_HAS_MCP2, "MCP SDK 2.x is required")
class TestHTTPServer(unittest.TestCase):
    class FakeClient:
        api_key = "fake"
        base_url = "https://api.venice.ai/api/v1"

    class StaticVerifier:
        async def verify_token(self, token):
            from mcp.server.auth.provider import AccessToken

            if token not in ("good-token", "other-token"):
                return None
            return AccessToken(
                token=token,
                client_id="claude-client",
                scopes=["venice:mcp"],
                subject="user-2" if token == "other-token" else "user-1",
            )

    def _server(self):
        from venice.mcp_server import build_http_server

        return build_http_server(
            self.FakeClient(),
            doc={},
            public_url="https://mcp.example.test/mcp",
            issuer_url="https://auth.example.test",
            jwks_url="https://auth.example.test/jwks",
            audience="venice-api",
            scopes=["venice:mcp"],
            token_verifier=self.StaticVerifier(),
        )

    def _media_server(self, media_dir, **overrides):
        from venice.mcp_server import build_http_server

        values = dict(
            doc={}, public_url="https://mcp.example.test/mcp",
            issuer_url="https://auth.example.test",
            jwks_url="https://auth.example.test/jwks", audience="venice-api",
            scopes=["venice:mcp"], token_verifier=self.StaticVerifier(),
            media_dir=media_dir,
        )
        values.update(overrides)
        return build_http_server(self.FakeClient(), **values)

    @staticmethod
    def _app(server):
        from mcp.server.transport_security import TransportSecuritySettings

        return server.streamable_http_app(
            stateless_http=True,
            json_response=True,
            host="mcp.example.test",
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["mcp.example.test"],
                allowed_origins=["https://allowed.example"],
            ),
        )

    def test_remote_profile_exposes_only_path_independent_tools(self):
        server = self._server()
        tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
        self.assertEqual(set(tools), {"venice_chat", "venice_vision"})
        vision = tools["venice_vision"].parameters
        self.assertEqual(
            set(vision["properties"]),
            {"image_url", "prompt", "model", "max_tokens"},
        )
        self.assertEqual(vision["required"], ["image_url"])
        for tool in tools.values():
            self.assertIs(tool.annotations.read_only_hint, False)
            self.assertIs(tool.annotations.destructive_hint, False)
            self.assertIs(tool.annotations.open_world_hint, True)

    def test_media_profile_exposes_only_uri_based_media_tools(self):
        with tempfile.TemporaryDirectory() as media_dir:
            server = self._media_server(media_dir)
            tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
        self.assertEqual(set(tools), {
            "venice_chat", "venice_vision", "venice_media_import",
            "venice_media_delete", "venice_image", "venice_tts", "venice_sfx",
            "venice_music", "venice_upscale", "venice_bg_remove",
            "venice_image_edit", "venice_video", "venice_job_status",
            "venice_job_result",
        })
        forbidden = {"input_path", "output_dir", "max_spend", "queue_id"}
        for tool in tools.values():
            self.assertTrue(forbidden.isdisjoint(tool.parameters["properties"]))
        self.assertEqual(
            set(tools["venice_job_result"].parameters["properties"]), {"job_id"}
        )

    def test_media_http_upload_range_owner_binding_and_delete(self):
        import httpx2

        png = b"\x89PNG\r\n\x1a\n" + b"x" * 64
        with tempfile.TemporaryDirectory() as media_dir:
            server = self._media_server(media_dir)
            app = self._app(server)

            async def exercise():
                transport = httpx2.ASGITransport(app=app)
                async with server.session_manager.run():
                    async with httpx2.AsyncClient(
                        transport=transport, base_url="https://mcp.example.test"
                    ) as client:
                        unauth = await client.post("/media", content=png)
                        uploaded = await client.post(
                            "/media", content=png,
                            headers={
                                "Authorization": "Bearer good-token",
                                "Content-Type": "image/png",
                                "X-Venice-Filename": "frame.png",
                            },
                        )
                        uri = uploaded.json()["uri"]
                        path = urllib.parse.urlsplit(uri).path
                        full = await client.get(
                            path, headers={"Authorization": "Bearer good-token"}
                        )
                        ranged = await client.get(
                            path, headers={
                                "Authorization": "Bearer good-token", "Range": "bytes=0-7",
                            },
                        )
                        other = await client.get(
                            path, headers={"Authorization": "Bearer other-token"}
                        )
                        deleted = await client.delete(
                            path, headers={"Authorization": "Bearer good-token"}
                        )
                        gone = await client.get(
                            path, headers={"Authorization": "Bearer good-token"}
                        )
                        return unauth, uploaded, full, ranged, other, deleted, gone

            results = asyncio.run(exercise())
        unauth, uploaded, full, ranged, other, deleted, gone = results
        self.assertEqual(unauth.status_code, 401)
        self.assertEqual(uploaded.status_code, 201)
        self.assertEqual(uploaded.json()["name"], "frame.png")
        self.assertEqual(full.content, png)
        self.assertEqual(full.headers["cache-control"], "private, no-store")
        self.assertEqual(ranged.status_code, 206)
        self.assertEqual(ranged.content, png[:8])
        self.assertEqual(other.status_code, 404)
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(gone.status_code, 404)

    def test_media_resource_read_and_paid_tool_confirmation(self):
        import base64
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.types import ResourceLink
        from venice.commands import _mcp

        png = b"\x89PNG\r\n\x1a\n" + b"generated"
        with tempfile.TemporaryDirectory() as media_dir:
            server = self._media_server(media_dir)
            app = self._app(server)

            def fake_image(_client, prompt, **kwargs):
                self.assertTrue(kwargs["require_confirmation"])
                self.assertEqual(kwargs["max_spend"], 0.10)
                self.assertEqual(kwargs["hard_max_spend"], 0.10)
                if not kwargs["confirm"]:
                    return {
                        "status": "confirmation_required",
                        "estimated_cost_usd": 0.01,
                    }
                path = Path(kwargs["output_dir"]) / "generated.png"
                path.write_bytes(png)
                return {
                    "status": "ok", "paths": [str(path)], "count": 1,
                    "bytes": len(png), "cost_estimate_usd": 0.01,
                }

            async def exercise():
                url = "https://mcp.example.test/mcp"
                transport = httpx2.ASGITransport(app=app)
                headers = {"Authorization": "Bearer good-token"}
                async with server.session_manager.run():
                    async with (
                        httpx2.AsyncClient(
                            transport=transport, base_url=url, headers=headers,
                        ) as http_client,
                        Client(streamable_http_client(url, http_client=http_client)) as client,
                    ):
                        gated = await client.call_tool(
                            "venice_image", {"prompt": "test", "confirm": False}
                        )
                        made = await client.call_tool(
                            "venice_image", {"prompt": "test", "confirm": True}
                        )
                        link = next(
                            block for block in made.content if isinstance(block, ResourceLink)
                        )
                        read = await client.read_resource(link.uri)
                        return gated, made, link, read

            with mock.patch.object(_mcp, "image_tool", side_effect=fake_image):
                gated, made, link, read = asyncio.run(exercise())
        self.assertEqual(json.loads(gated.content[0].text)["status"], "confirmation_required")
        self.assertEqual(json.loads(made.content[0].text)["status"], "ok")
        self.assertEqual(link.mime_type, "image/png")
        self.assertEqual(base64.b64decode(read.contents[0].blob), png)

    def test_queued_media_uses_owner_bound_job_handle_and_stores_result(self):
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.types import ResourceLink
        from venice.commands import _mcp

        mp3 = b"ID3" + b"audio"
        with tempfile.TemporaryDirectory() as media_dir:
            server = self._media_server(media_dir)
            app = self._app(server)

            def fake_sfx(_client, prompt, **kwargs):
                if not kwargs["confirm"]:
                    return {
                        "status": "confirmation_required", "estimated_cost_usd": 0.02,
                    }
                self.assertTrue(kwargs["background"])
                return {
                    "status": "queued", "queue_id": "private-backend-id",
                    "type": "sfx", "model": "sfx-model", "cost_estimate_usd": 0.02,
                }

            def fake_result(_client, **kwargs):
                self.assertEqual(kwargs["queue_id"], "private-backend-id")
                self.assertEqual(kwargs["type"], "sfx")
                self.assertFalse(kwargs["complete"])
                path = Path(kwargs["output_dir"]) / "sound.mp3"
                path.write_bytes(mp3)
                return {"status": "ok", "path": str(path), "bytes": len(mp3)}

            async def exercise():
                url = "https://mcp.example.test/mcp"
                transport = httpx2.ASGITransport(app=app)
                headers = {"Authorization": "Bearer good-token"}
                async with server.session_manager.run():
                    async with (
                        httpx2.AsyncClient(
                            transport=transport, base_url=url, headers=headers,
                        ) as http_client,
                        Client(streamable_http_client(url, http_client=http_client)) as client,
                    ):
                        gated = await client.call_tool(
                            "venice_sfx", {"prompt": "boom", "confirm": False}
                        )
                        queued = await client.call_tool(
                            "venice_sfx", {"prompt": "boom", "confirm": True}
                        )
                        queued_doc = json.loads(queued.content[0].text)
                        ready = await client.call_tool(
                            "venice_job_result", {"job_id": queued_doc["job_id"]}
                        )
                        return gated, queued_doc, ready

            with mock.patch.object(_mcp, "sfx_tool", side_effect=fake_sfx), \
                 mock.patch.object(_mcp, "job_result_tool", side_effect=fake_result), \
                 mock.patch.object(_mcp, "complete_job") as complete:
                gated, queued, ready = asyncio.run(exercise())
        self.assertEqual(json.loads(gated.content[0].text)["status"], "confirmation_required")
        self.assertEqual(queued["status"], "queued")
        self.assertNotIn("queue_id", queued)
        link = next(block for block in ready.content if isinstance(block, ResourceLink))
        self.assertEqual(link.mime_type, "audio/mpeg")
        complete.assert_called_once()

    def test_remote_vision_forces_delegation_even_with_native_config(self):
        from venice.commands import _mcp
        from venice.mcp_server import build_http_server

        server = build_http_server(
            self.FakeClient(),
            doc={"defaults": {"vision": {"mode": "native"}}},
            public_url="https://mcp.example.test/mcp",
            issuer_url="https://auth.example.test",
            jwks_url="https://auth.example.test/jwks",
            audience="venice-api",
            scopes=["venice:mcp"],
            token_verifier=self.StaticVerifier(),
        )
        delegated = {"status": "ok", "content": "seen"}
        with mock.patch.object(_mcp, "vision_tool", return_value=delegated) as impl:
            result = server._tool_manager.get_tool("venice_vision").fn(
                image_url="https://images.example/frame.png"
            )
        self.assertEqual(json.loads(result.content[0].text), delegated)
        self.assertIsNone(impl.call_args.kwargs["input_path"])
        self.assertEqual(impl.call_args.kwargs["mode"], "delegate")

    def test_remote_vision_rejects_local_or_credentialed_urls(self):
        from venice.commands import _mcp

        tool = self._server()._tool_manager.get_tool("venice_vision").fn
        with mock.patch.object(_mcp, "vision_tool") as impl:
            results = [
                tool(image_url="frame.png"),
                tool(image_url="file:///etc/passwd"),
                tool(image_url="https://user:pass@images.example/frame.png"),
                tool(image_url="data:text/plain;base64,SGVsbG8="),
            ]
        self.assertTrue(all(result.is_error for result in results))
        impl.assert_not_called()

    def test_metadata_health_and_auth_failures(self):
        import httpx2

        server = self._server()
        app = self._app(server)

        async def exercise():
            transport = httpx2.ASGITransport(app=app)
            async with server.session_manager.run():
                async with httpx2.AsyncClient(
                    transport=transport,
                    base_url="https://mcp.example.test",
                ) as client:
                    health = await client.get("/healthz")
                    metadata = await client.get(
                        "/.well-known/oauth-protected-resource/mcp"
                    )
                    missing = await client.post("/mcp", json={})
                    invalid = await client.post(
                        "/mcp",
                        json={},
                        headers={"Authorization": "Bearer bad-token"},
                    )
                    wrong_host = await client.post(
                        "/mcp",
                        json={},
                        headers={
                            "Authorization": "Bearer good-token",
                            "Host": "wrong.example",
                        },
                    )
                    wrong_origin = await client.post(
                        "/mcp",
                        json={},
                        headers={
                            "Authorization": "Bearer good-token",
                            "Origin": "https://denied.example",
                        },
                    )
                    return health, metadata, missing, invalid, wrong_host, wrong_origin

        health, metadata, missing, invalid, wrong_host, wrong_origin = asyncio.run(
            exercise()
        )
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(metadata.json()["resource"], "https://mcp.example.test/mcp")
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(missing.json(), invalid.json())
        self.assertIn("resource_metadata=", missing.headers["www-authenticate"])
        self.assertEqual(wrong_host.status_code, 421)
        self.assertEqual(wrong_origin.status_code, 403)

    def test_authenticated_client_lists_and_calls_chat(self):
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from venice.commands import _mcp

        server = self._server()
        app = self._app(server)

        async def exercise():
            url = "https://mcp.example.test/mcp"
            transport = httpx2.ASGITransport(app=app)
            headers = {"Authorization": "Bearer good-token"}
            async with server.session_manager.run():
                async with (
                    httpx2.AsyncClient(
                        transport=transport,
                        base_url=url,
                        headers=headers,
                    ) as http_client,
                    Client(streamable_http_client(url, http_client=http_client)) as client,
                ):
                    listed = await client.list_tools()
                    result = await client.call_tool(
                        "venice_chat", {"message": "ping"}
                    )
                    return {tool.name for tool in listed.tools}, result

        with mock.patch.object(
            _mcp,
            "chat_tool",
            return_value={"status": "ok", "content": "pong"},
        ) as impl:
            names, result = asyncio.run(exercise())
        self.assertEqual(names, {"venice_chat", "venice_vision"})
        self.assertEqual(
            json.loads(result.content[0].text),
            {"status": "ok", "content": "pong"},
        )
        impl.assert_called_once()

    def test_serve_http_pins_transport_settings(self):
        from venice import mcp_server

        built = mock.Mock()
        with mock.patch.object(
            mcp_server, "build_http_server", return_value=built
        ) as build:
            mcp_server.serve_http(
                self.FakeClient(),
                doc={},
                host="0.0.0.0",
                port=8080,
                public_url="https://mcp.example.test/mcp",
                issuer_url="https://auth.example.test",
                jwks_url="https://auth.example.test/jwks",
                audience="venice-api",
                scopes=["venice:mcp"],
                allowed_origins=["https://allowed.example"],
            )
        build.assert_called_once()
        kwargs = built.run.call_args.kwargs
        self.assertEqual(kwargs["transport"], "streamable-http")
        self.assertTrue(kwargs["stateless_http"])
        self.assertTrue(kwargs["json_response"])
        self.assertEqual(kwargs["streamable_http_path"], "/mcp")
        self.assertEqual(
            kwargs["transport_security"].allowed_hosts,
            ["mcp.example.test"],
        )


if __name__ == "__main__":
    unittest.main()
