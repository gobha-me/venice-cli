# Findings for lead (r1 test re-make)

No source under `src/` or `scripts/` was edited. Everything below comes from
reading 1a844f2 while writing tests. None is asserted by a committed test,
because each would be a failing test against current behaviour.

## Code observations (Codex's lane)

1. **Low: the MCP error envelope only knows the environment key.**
   `_mcp._err` calls `_redact.redact_text(message)` with no explicit secret, so
   it redacts only `$VENICE_API_KEY`. When `mcp-serve` authenticates from the
   credentials file (no env var), a key reflected through a tool argument is
   *not* redacted. Example: `job_status_tool(type=<key>)` renders
   `unknown type '<key>'`. Upstream echoes are still covered, because
   `VeniceAPIError` redacts with the client's key. The only path is a host
   that already holds the key, which limits the impact. A fix could pass
   `client.api_key` into `_err`, or bind it when the server is built.
   `test_reflected_argument_is_redacted_in_tool_error` covers only the
   env-key case.

2. **Low: encodings `redact_text` does not cover.** Lowercase percent-escapes
   (`%2f`), JSON encoders that escape `/` as `\/`, base64 or other wrappings,
   and JSON with `ensure_ascii=False` for non-ASCII keys are all missed. Venice
   keys are probably `[A-Za-z0-9_-]`, which makes most of these moot. Worth a
   sentence in `_redact`'s docstring about which forms are handled.

3. **Low: no minimum secret length.** A one-character `$VENICE_API_KEY`
   (misconfiguration) would replace every occurrence of that character in all
   diagnostics. The empty string is guarded; short strings are not.

4. **Info: surface.json and surface.md use different operation orders.**
   `surface.json` is written with sorted keys. `surface.md` follows the lock's
   upstream order (path, then method). So `render_surface(json.load(surface.json))`
   does **not** reproduce the committed `.md`; you must re-order by
   `lock.all_operations` first. This looks intentional. The test documents it.

## Environment blockers in this session (not code defects)

5. **`mcp` and `openai` SDKs are not installed**, and installing them or creating
   a venv needs approval this session cannot grant. Consequences:
   * `tests/test_mcp_harness.py` (5 tests) and
     `test_openapi_surface.LiveMcpInventory` (2 tests) skip. They have **never
     been executed**, so they must be run under `pip install -e ".[all,test]"`
     before merge. The SDK calls mirror existing passing tests
     (`tests/test_mcp_serve.py`, `tests/test_mcp_http.py`). The catalog fixture is
     my best reading of what each tool validates, and is the most likely
     place for a first-run failure.
   * `make test`: 2138 run, **2 failures + 330 errors**, all in modules this
     branch does not touch (`test_chat`, `test_repl`, `test_code_command`,
     `test_embed`, ...). 330 errors plus `test_shared_openai` are
     `ModuleNotFoundError: No module named 'openai'`. The other failure is
     `test_install_scripts.test_bsd_readlink_supports_symlinked_install_and_uninstall`
     (file mode `1472 != 448`, an environment umask/permission artefact). All new
     and edited tests pass.
6. **`flock` is refused** by the command policy, so mutmut was not run (see
   MUTATION-RESULTS.md).
