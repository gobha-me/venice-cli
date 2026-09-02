"""Authenticated Streamable HTTP coverage for issue #31."""
import argparse
import asyncio
import importlib.util
import io
import json
import sys
import time
import unittest
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

            if token != "good-token":
                return None
            return AccessToken(
                token=token,
                client_id="claude-client",
                scopes=["venice:mcp"],
                subject="user-1",
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
