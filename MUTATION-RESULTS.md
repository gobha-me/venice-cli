# Mutation results (r1 test re-make)

## Status: NOT RUN -- blocked in this session. Floor (70% killed) NOT demonstrated.

The brief requires every mutmut run to be wrapped as
`flock /config/m0/mutation.lock mutmut run ...`. In this session the command
policy refuses `flock` outright ("What 'flock' starts can't be checked before it
runs"). Running mutmut *without* the shared lock would break the host's
serialization contract (2 CPUs, other lanes), so I did not run it. I also did
not run it indirectly (for example, through a test shim), because that would
bypass the same refusal.

| module | killed / total | score | status |
|---|---|---|---|
| src/venice/_redact.py | -- | -- | not run (lock refused) |
| src/venice/commands/video.py | -- | -- | not run (lock refused) |
| scripts/openapi_surface.py | -- | -- | not run (lock refused) |
| src/venice/_mcp_common.py | -- | -- | not runnable here: module-scope `from mcp import types`; `mcp` SDK absent |
| scripts/mcp_surface.py | -- | -- | not runnable here: module-scope `from mcp import Client`; `mcp` SDK absent |

## Prepared configuration (scoping only; unvalidated)

`pyproject.toml` gained a `[tool.mutmut]` section:

* `source_paths = ["src", "scripts"]`, `only_mutate` = the three runnable units above;
* `mutate_only_covered_lines = true`, which scopes `video.py` (~830 lines) to the
  lines the r1 suites execute rather than the whole command module;
* `also_copy = ["contracts/"]`, because the surface tests read the committed snapshots
  relative to the copied tree;
* test selection = `tests/test_r1_audit.py`, `tests/test_openapi_surface.py`,
  `tests/test_openapi_contract.py`.

The config has never been executed, so treat it as a starting point.
In an environment with `pip install -e ".[all,test]"`, add
`src/venice/_mcp_common.py` and `scripts/mcp_surface.py` to `only_mutate` and
`tests/test_mcp_harness.py` to the selection.

Command for the lead:

```
flock /config/m0/mutation.lock mutmut run --max-children 2
mutmut results
```

## Interim evidence (not a substitute)

`BROKEN-COPY-RESULTS.md` records 65 hand-written single-edit breaks: 9 in
`_redact.py`, 5 in `video.py`, 15 in `openapi_surface.py` (plus 6 that tamper with
its committed outputs), and the rest in `client.py`, `_openai.py`, `_mcp.py`, the
MCP servers, `openapi_contract.py`, and one test-oracle break. All 65 were killed. These are targeted, so they say nothing about
the unbiased operator-level mutant population mutmut would generate, and they
do not satisfy the 70% requirement.
