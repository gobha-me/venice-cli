# Contributing

Thanks for taking a look. This is a small project with a simple rhythm.

## Development setup

No build step. Clone, and run from the repo:

```sh
git clone https://github.com/gobha-me/venice-cli.git
cd venice-cli
pip install -r requirements.txt   # only needed for `venice chat` / `venice embed`
PYTHONPATH=src python3 -m venice --help
```

`./install.sh` symlinks `venice` onto your PATH if you want the real command.

## Before opening a PR

```sh
make test    # unittest, no network, no API key required
make lint    # compileall syntax check
```

Both must be green. Tests are hermetic: `urlopen` is mocked, `subprocess` and
`shutil.which` are patched, `HOME` is redirected to a tmpdir, and the OpenAI
SDK is mocked. **No test should ever make a real API call or need a real key.**
If you find yourself wanting to hit the live API in a test, that's a sign the
seam is in the wrong place.

## House style

- **Stdlib-only in the base.** `venice chat` and `venice embed` use the OpenAI
  SDK, lazy-imported inside the handler so a missing `openai` degrades to a
  hint and exit 2 rather than breaking `venice --help`. Keep it that way: new
  third-party deps need a good reason and must not be imported at module scope
  in the base commands.
- **Shared plumbing lives in `src/venice/commands/_*.py`** (`_shared`, `_queue`,
  `_models`, `_openai`). These take primitive args -- a label, a model type, a
  cost -- rather than an argparse namespace, so they stay independent of any one
  command's argument shape. If you're copy-pasting a helper into a second
  command, extract it instead.
- **Exit codes are part of the interface.** See the table in the README; don't
  change what an existing condition returns without saying so.
- **Never log, print, or embed the API key**, including in error messages, test
  fixtures, or partial/redacted form.
- Adding a subcommand is one import and one entry in
  `src/venice/commands/__init__.py`.

## Commits and branches

One `feat/<name>` branch per issue. Commit subjects follow
`vX.Y: short description (#issue)`. Merges into `master` are `--no-ff`.

## Reporting bugs

Include the command you ran, what you expected, what happened, the exit code,
and your Python version. **Never paste your API key** -- not even partially.

For security issues see [SECURITY.md](SECURITY.md); don't use the public
issue tracker.
