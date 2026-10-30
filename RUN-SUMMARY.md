# R1 audit findings fix - run summary

Date: 2026-10-30. Branch: `fix/r1-audit-findings`, based on `cab81ce`.
Findings source: unchanged `REPORT.md`, SHA-256
`1f8470b851802178613ffa3551dc3340e43a692a030455eb074df0f9e4014069`.
All six backlog items are handled in order, one commit per item.

## Commits and per-item validation

Every full suite below ran with a temporary HOME and `VENICE_API_KEY=test-fake-key`.
Each final full suite ran 2,062 tests: 2,061 passed and one expected browser-test
skip, with zero failures or errors. `make lint` means the repository's compileall
syntax check; broad Ruff is reported separately and is not claimed green.

| Item | Commit | Change | Final validation before commit |
| --- | --- | --- | --- |
| 1 - Ruff regression | `0f22bf3` | Preserve compatibility aliases through assignments; sort imports, replace deprecated container aliases, and give each SDK annotation exception a specific `FA100` reason. No lint configuration edits. | `make lint`: zero syntax errors; `make scan`: zero findings; full suite: 2,062 tests, one skip, 486.890 s. Isolated F: zero diagnostics on all 13 requested files. MCP snapshot byte-identical. |
| 2 - SDK warning | `17301ec` | Explicit `validate_token_resource=False` preserves verifier-managed audience checks. Guard with `AuthSettings.model_fields` because MCP 2.1.1 lacks the field. | Lint/scan pass; full suite: 2,062 tests, one skip, 84.135 s. MCP 2.1.1 HTTP/harness: 26 passed. MCP 2.3.0 schema generation passes with deprecations treated as errors; both SDK surfaces byte-identical. |
| 3 - Python 3.9 boundary | `7da46dc` | Existing syntax already meets the requirement. Document the base import graph and intentional 3.9-compatible Optional hints. | Lint/scan pass; full suite: 2,062 tests, one skip, 137.510 s. Five extracted modules parse using Python 3.9 grammar; no PEP 604 union expressions. Base CLI/parser/registry/redaction imports pass with site packages disabled. Actual Python 3.9 was unavailable. |
| 4 - CostLedger | `4106a01` | Move accounting and its cohesive helpers into `commands/_cost_ledger.py`, with a module docstring and README section. Preserve `_agent.CostLedger` and existing accounting aliases. | Lint/scan pass; focused agent/REPL/session suite: 430 passed; final full suite: 2,062 tests, one skip, 461.547 s. All 15 moved executable bodies match; only container annotation spelling changes. Alias identity and SDK-free base imports pass. |
| 5 - Remote MCP handlers | `1c55bb4` | Move 24 named handler/helper functions, four utilities, and the job adapter into `_mcp_http.py`. Pass storage, quotas, spend/defaults, and request-time principal resolution explicitly. Keep typed registration wrappers and public entry points. | Lint/scan pass; focused suite: 105 passed; MCP 2.1.1 HTTP/harness: 26 passed; full suite: 2,062 tests, one skip, 293.754 s. All 19 registered signatures, return annotations, docstrings, and decorators match. Moved executable bodies match after dependency binding. Both SDK snapshots byte-identical. |
| 6 - Wrap-up | This documentation commit; resolve with `git log -1 --format=%H -- RUN-SUMMARY.md` | Final offline validation and this handoff. Source validation is against `1c55bb4`; this commit changes only the summary. | `make lint`: zero syntax errors; `make scan`: zero findings; `make openapi-check`: 18 implemented operations verified offline; final full suite: 2,062 tests, one skip, 344.499 s; isolated F: zero diagnostics on all requested and new modules. |

The wrap-up commit cannot embed its own hash; its exact SHA is returned by the
command above and supplied in the final handoff.

## Ruff accounting

Ruff 0.17.3 was used with this workspace's inherited default profile. The report's
historical 1,518/1,574 whole-tree totals were not reproduced in this environment.
On the exact 13-file list requested, the parent of `1a844f2` has 461 diagnostics
and `1a844f2` has 510. Item 1 reduces that to 289, with no added diagnostic count
or category relative to the parent. This includes removing the introduced
redundant aliases, duplicate deprecated typing imports, import ordering, and
unnecessary dict construction. Existing moved FA100 warnings on SDK wrappers have
narrow, explained exceptions because the annotations must remain concrete.

Whole-tree `ruff check src scripts` after item 1 reports 854 inherited diagnostics.
The CostLedger extraction reduces that to 840; remote handler extraction retains
840. This optional broad check still fails on inherited modernization/style debt.
Isolated `--select F` passes on the requested files and both new modules.

The requested `--target-version py39 --select UP,F` inspection reports three
`UP045` suggestions in `_tool_registry.py`, and no F findings. Optional hints are
retained so consumers can evaluate them on Python 3.9; no 3.10-only syntax was
introduced. Only Python 3.12 is installed under `/usr/bin/python3.*`. Compileall
and grammar/import checks are fallback evidence, not an actual 3.9 interpreter run.

## Preserved contracts and module boundaries

`contracts/openapi/mcp-surface.json` was never edited. Regeneration used
`PYTHONPATH=src python3 scripts/mcp_surface.py <temporary-output>` and `cmp`, with
MCP 2.3.0 and MCP 2.1.1. The committed and generated files all have SHA-256
`36546e335a07325d380b8ff8f501b2e1f0fe64a50b5d6c6d09fba4c324a7dd23`.
The profiles remain 10 stdio tools, two minimal HTTP tools, and 14 media HTTP tools.
`build_server`, `serve`, and `JWKSJWTVerifier` retain their original alias identities.
Principal identity is resolved per request, never cached on the shared context.
The job adapter receives its store and owner directly; storage retains reservation,
quota, publication, and cleanup authority.

`commands/_agent.py` is now 2,151 lines; `_cost_ledger.py` is 1,795 lines, including
its cohesive accounting helpers. `mcp_server.py` is 511 lines; `_mcp_http.py` is
862 lines of named functions and explicit context rather than a large registration
closure. Executable-body comparisons and the existing suites found no behavioral
changes. Public schemas are byte-identical.

## Environment issue and limits

The first item-1 full run executed 2,062 tests in 196.658 s and had one failure,
plus the expected skip. The install-script mode assertion observed 02700 instead
of 0700 because temporary directories inherited setgid from their parent. A
first attempted numeric chmod retained that bit; explicit `chmod g-s` on a newly
created private runtime temp directory resolved it. All ten existing install-script
tests then passed, followed by every final full suite. No source/test workaround
was made for this filesystem issue.

No tests or fixtures were edited, added, or deleted, and no test-author changes
were needed. The dedicated real-browser test remains the expected normal-suite
skip. Actual Python 3.9 and the separate browser CI job were not run. No backlog
items were skipped for time or disk space; `/config` remained above 5 GB free
(88 GB observed before each item).

No live Venice calls, public-spec refreshes, SSH, keys, forge operations, pushes,
fetches, issue changes, history rewrites, or verification bypasses were performed.
No real credential files were inspected or real credential values printed.
Task-created `.r1-work/`, `tmp/r1-venv`, and `tmp/r1-runtime` directories are
removed at handoff. The core limit is unlimited; no core dumps were found.
