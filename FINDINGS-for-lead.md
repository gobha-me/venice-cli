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

## Follow-up (option 1 ruling)

### Redaction gaps now pinned by red-first tests

Findings 1 and 2 above are now executable as `@unittest.expectedFailure`
tests in `tests/test_r1_redact_boundary.py::KnownRedactionGaps`. Each one was
run once without its decorator against unmodified `src/` and failed at the
leak assertion (see BROKEN-COPY-RESULTS.md, "Red-first evidence"). When Codex
fixes one, its test will report "unexpected success". Remove that decorator
in the same change.

* **F-fu-1** (= finding 1): a credentials-file key with no `$VENICE_API_KEY`,
  reflected through `job_status_tool(type=...)`, reaches the `_mcp._err`
  envelope unredacted. Suggested fix: thread `client.api_key` into `_err`.
* **F-fu-2** (part of finding 2): lowercase percent-escapes (`%2f`, `%2b`) of
  the key survive into `VeniceAPIError.body`. Suggested fix: match `%XX` case-
  insensitively (for example, also add the lowercased-hex variant).
* **F-fu-3** (part of finding 2): JSON `\/` escaping of `/` survives. Suggested
  fix: add `json_variant.replace("/", "\\/")`.

### SDK-gated tests (step 4)

With `runs/test-r1-fu/venv`, `tests.test_mcp_harness` and
`tests.test_openapi_surface` (including `LiveMcpInventory`) ran with no
skips: **34 tests, OK**. None of my 7de7a07 SDK-gated tests needed a fix, and
no code defect showed up. Blocker 5 above is resolved.

### Blockers in this session

7. **`cp -r src ../../runs/test-r1-fu/broken/` was refused** ("cp with flags
   requires manual approval"), although it is the brief's exact allowed form.
   Step 3 (broken copies) stopped there. The copy was not recreated any other
   way. Planned breaks are in BROKEN-COPY-RESULTS.md.
8. **mutmut 3.8.0 is incompatible with the venv's pytest 9.2.0**: the stats
   phase dies on `item._nodeid` (removed in pytest 9.2). The lead's 9.1.1 run
   was unaffected. No score was produced. See MUTATION-RESULTS.md.
9. Two exploratory commands needed approval and were dropped, not retried
   another way: a `python3 -c` one-liner to print the synthetic key's
   encodings (the tests compute them instead), and `cat` of the lead's
   mutation wrapper (not needed to run it).
10. `make test`: 2154 run, **1 failure** (expected failures=3, skipped=1). The
    failure is the pre-existing `test_install_scripts.test_bsd_readlink_...`
    file-mode artefact (`1472 != 448`, item 5 above). That module is not
    touched by this branch. The earlier `openai`-missing errors are gone with
    the venv. `make lint` and `make scan` pass.
