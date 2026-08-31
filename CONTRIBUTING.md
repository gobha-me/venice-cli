# Contributing

Thanks for taking a look. This is a small project with a simple rhythm.

## Development setup

Clone, and run from the repo:

```sh
git clone https://github.com/gobha-me/venice-cli.git
cd venice-cli
PYTHONPATH=src python3 -m venice --help
```

That needs no install at all. For an editable install with everything the test
suite can exercise:

```sh
pip install -e ".[all,test]"
```

`[openai]` alone is enough for day-to-day model-backed work: chat, code, embed,
index/search, model-backed review, and their delegated model tools. `[mcp]`
adds MCP server/client transport; a model-delegating MCP tool can need both.
`[test]` adds `pexpect` for the drive suite below; without it those tests report
as **skipped**, not failed. The authoritative feature inventory is in README's
Dependencies section.

`./install.sh` symlinks `venice` onto your PATH if you want the real command
without pip. Don't mix the two: both own `~/.local/bin/venice`.

## Before opening a PR

```sh
make test    # unittest, no network, no API key required
make lint    # compileall syntax check
make scan    # invisible-character scan over every tracked file
make drive   # the drive suite + its fake-API fixture (a subset of `make test`)
make openapi-check  # offline implemented-operation inventory check
```

`make test`, `make lint` and `make scan` must be green. Tests are hermetic: `urlopen` is
mocked, `subprocess` and `shutil.which` are patched, `HOME` is redirected to a
tmpdir, and the OpenAI SDK is mocked. **No test should ever make a real API call
or need a real key.** If you find yourself wanting to hit the live API in a test,
that's a sign the seam is in the wrong place.

### OpenAPI contract inventory

`contracts/openapi/venice.lock.json` is the deterministic subset of the
official Venice Swagger that this curated CLI actually implements. It records
the upstream version and raw SHA-256 plus the normalized request schemas for
those operations; `implementation.json` records each CLI/MCP/agent consumer,
implemented request fields, deliberate omissions, and model-catalog-backed
constraints. The full third-party Swagger remains untracked.

Ordinary tests run `make openapi-check` offline. To compare the lock with the
current official schema, install the script-only parser and run:

```sh
python3 -m pip install -r scripts/openapi-requirements.txt
make openapi-live
```

The live check fails for any change to an implemented operation and labels the
change breaking or review-required. New or removed operations outside the
curated surface are informational. After reviewing and implementing relevant
drift, refresh the lock explicitly with `make openapi-refresh`, review the
generated diff, then rerun the offline check. A scheduled/manual GitHub Actions
workflow performs the same live comparison without making PR CI depend on
upstream availability; it writes the actionable report to the job summary and
an artifact, but does not open issues automatically.

### The invisible-character scan

`make scan` (and a regression test that runs it under `make test`) rejects
source containing characters a reviewer cannot see: zero-width joiners, the
bidi overrides behind "Trojan Source", Unicode tag characters, variation
selectors, private-use codepoints, and non-ASCII identifiers such as a Cyrillic
`a` in `api_key`. Reading the diff is how PRs here get reviewed, and a diff is
exactly what this class of payload is built to survive.

**Visible punctuation is fine** -- em dashes, arrows, `…`, `≤` and friends are
all in use already. The rule is drawn at invisibility, not at non-ASCII, so if
the scan ever flags something with a glyph, that's a bug in the scan.

### The drive suite

`tests/test_drive_cli.py` is hermetic by a *different mechanism* than the rest,
because it can't use the same one: it spawns the real `python -m venice` as a
child process, so there is no `urlopen` to patch. Instead it drives the CLI on a
pty (`pexpect`) against a stdlib fake API bound to `127.0.0.1:0`
(`tests/_venice_fake_server.py`), with `$HOME` redirected to a tmpdir and
`VENICE_API_KEY=test-fake-key`. Same rules hold: no network, no real key,
nothing written outside the tmpdir. The child's environment is built from
scratch rather than inherited, so a developer's `VENICE_*` or `http_proxy`
can't leak in and desync their machine from CI.

It exists because patching `builtins.input` proves a *branch* runs; it can't
prove the prompt reached a terminal in a usable order. Read the rules in
`tests/_drive.py`'s docstring before adding a case — each one (pty CRLF, echoed
input, the agent spinner) is a flake source already paid for once.

The child's import path mirrors the parent's effective site configuration:
user-site packages are included only when the parent interpreter has user-site
imports enabled, and the child disables HOME-derived user-site discovery. That
distinction is invisible until you run the suite from a venv, where
`site.ENABLE_USER_SITE` is `False` (#107).

## House style

- **No required dependencies in the base.** OpenAI-backed commands and tools
  lazy-import the SDK at their feature boundaries; MCP transport does the same
  with its SDK. A missing extra must degrade to a scoped hint/error rather than
  breaking `venice --help` or the complete command graph. Keep the README
  inventory and its regression guard current when adding a new lazy-import
  call site. New third-party dependencies need a good reason and must not be
  imported at module scope in base commands.
- **Shared plumbing lives in `src/venice/commands/_*.py`** (`_shared`, `_queue`,
  `_models`, `_openai`). These take primitive args -- a label, a model type, a
  cost -- rather than an argparse namespace, so they stay independent of any one
  command's argument shape. If you're copy-pasting a helper into a second
  command, extract it instead.
- **Exit codes are part of the interface.** See the table in the README; don't
  change what an existing condition returns without saying so.
- **Interactive surfaces need a drive test.** If you add or change a prompt, a
  slash-command, a confirm gate, or a signal handler, add a case to
  `tests/test_drive_cli.py`. A unit test that patches `input()` proves the
  branch runs; it doesn't prove the prompt is usable from a terminal.
- **Never log, print, or embed the API key**, including in error messages, test
  fixtures, or partial/redacted form.
- Adding a subcommand is one import and one entry in
  `src/venice/commands/__init__.py`.

## Commits and branches

One `feat/<name>` branch per issue. Commit subjects follow
`vX.Y: short description (#issue)`. Merges into `master` are `--no-ff`.

## Releases

Production releases are made only from `v<version>` tags on `master`. Merge the
release-version PR, then wait for the complete `test` workflow to succeed for
that exact merge SHA before creating and publishing the GitHub Release. The
`publish` workflow independently verifies the tag against the imported package
version and requires that exact successful `master` run before it builds or can
mint PyPI credentials. A missing, pending, failed, branch-equivalent, PR, or
manually dispatched test run never satisfies the gate.

Manual dispatch of `publish` is TestPyPI-only. If a GitHub Release is published
before exact-SHA CI finishes, production publication fails closed; after CI is
green, rerun the failed release workflow rather than moving or recreating the
tag.

## Reporting bugs

Include the command you ran, what you expected, what happened, the exit code,
and your Python version. **Never paste your API key** -- not even partially.

For security issues see [SECURITY.md](SECURITY.md); don't use the public
issue tracker.
