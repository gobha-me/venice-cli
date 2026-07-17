# Working in this repo as Claude Code

This project wraps the Venice.ai API. There is a credential file at
`~/.config/venice/credentials` (chmod 0600, plaintext). Treat its
contents as confidential by convention.

## Hard rules

- **Never** read, `cat`, `head`, `tail`, `less`, or use the Read tool on
  `~/.config/venice/credentials`. If you need to know whether it exists
  or what its mode is, use `ls -l ~/.config/venice/credentials` (which
  shows metadata, not contents). Same rule for any file matching
  `**/venice/credentials` or `**/.venice/key` anywhere under `$HOME`.
- **Never** echo, print, log, or include the value of `$VENICE_API_KEY`
  in any output -- not in command output, not in code you write, not in
  commit messages, not in test fixtures, not partially redacted ("first
  4 chars are..."). The key is plaintext on disk by design; the guard
  is what keeps it out of session transcripts and screen-shares.
- When testing API code, use a placeholder key in commands: e.g.
  `VENICE_API_KEY=test-fake-key python -m unittest tests.test_client`.
  Do **not** invoke real Venice endpoints with the user's real key
  unless the user explicitly asks for a live test. Tests mock
  `urlopen`.
- If a command would emit the key (e.g. `env | grep VENICE`, `cat the
  file`, printing `os.environ` in Python), **don't run it**. Suggest
  an alternative that doesn't leak: `env | grep -c VENICE_API_KEY`
  (count, not value).
- If the user pastes their key into a chat message by accident, do not
  repeat it back in your response -- even to confirm receipt.
  Acknowledge with the length only ("got a 40-char key").

## Why the rules exist

The key sits plaintext on the filesystem because there's no OS keychain
integration (see the security note in the README). The threat these rules
address isn't the local filesystem -- it's the *session transcript*, which
can be pasted into a bug report, screenshotted, or shipped to log storage.
Anything an assistant prints becomes part of that transcript. Convention is
the only barrier here; please honor it.

## What's safe to do

- Read source code (`src/venice/**`).
- Run `venice login` interactively (getpass hides input).
- Run `venice sfx --dry-run "..."` to verify quoting logic.
- Run tests: `make test`.
- Modify code, write tests, refactor.
