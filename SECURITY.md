# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately via GitHub's
[private vulnerability reporting](https://github.com/gobha-me/venice-cli/security/advisories/new)
(Security tab -> Report a vulnerability). Include what you found, how to
reproduce it, and what an attacker could do with it.

Expect an acknowledgement within a week. This is a small
community-maintained project with no SLA, but credential-handling bugs are
taken seriously and will be prioritized.

## Scope

In scope: anything in this repository -- the CLI, the API client, credential
storage and handling, and the shell installers.

Out of scope: the Venice.ai service and API itself. This is an unofficial,
unaffiliated client; report service-side issues to Venice.ai directly.

## What this tool does with your API key

Understanding the design will tell you whether a finding is a bug or a
documented limitation:

- The key is stored **plaintext** at `~/.config/venice/credentials`, mode
  0600, in a 0700 directory. There is no OS keychain integration. File
  permissions are the only protection -- this is a known and documented
  limitation, not a vulnerability.
- `$VENICE_API_KEY` overrides the file and is the recommended path for CI and
  shared environments.
- The key is sent only to the Venice API base URL (`$VENICE_BASE_URL` overrides
  it, for proxies and testing).
- The key is never written to logs, printed to stdout/stderr, or included in
  error messages.

**Genuinely in scope, and worth reporting:** any path where the key leaks into
output, logs, an error message, a saved file, a crash traceback, or a request
to any host other than the configured base URL. Also: the credentials file or
its directory being created with permissions looser than 0600/0700.

If you believe your key has been exposed, revoke and rotate it immediately at
<https://venice.ai/settings/api>.
