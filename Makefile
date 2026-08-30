.PHONY: install uninstall test drive lint scan openapi-check openapi-live openapi-refresh build clean

install:
	@./install.sh

uninstall:
	@./uninstall.sh

test:
	@PYTHONPATH=src python3 -m unittest discover -v -s tests

# The drive suite and its fake-API fixture, for fast iteration. `make test`
# already runs both (and degrades to skips without pexpect) -- but someone who
# explicitly typed `make drive` must not be told OK when nothing ran, so this
# target hard-fails on a missing dep instead. Both are checked: without pexpect
# every pty case skips, and without openai the two interleaved dialogues -- the
# whole point of the suite -- skip while the target still reports OK.
drive:
	@python3 -c "import pexpect" 2>/dev/null || \
	  { echo 'drive: pexpect missing -- pip install -e ".[test]"'; exit 1; }
	@python3 -c "import openai" 2>/dev/null || \
	  { echo 'drive: openai missing -- pip install -e ".[all,test]"'; exit 1; }
	@PYTHONPATH=src python3 -m unittest -v tests.test_drive_cli tests.test_fake_server

lint:
	@python3 -m compileall -q src tests
	@echo "syntax OK"

# Invisible-character scan (#109) -- the review pass a human diff-read can't
# do. `make test` runs the same scan as a regression test and *skips* it
# outside a git checkout; this target hard-fails there instead. A security
# guard that silently scans nothing is worse than no guard, because it prints
# OK. Same split as test/drive: the suite degrades, the explicit target won't.
scan:
	@python3 -m tests._hygiene

openapi-check:
	@python3 scripts/openapi_contract.py check

openapi-live:
	@python3 scripts/openapi_contract.py check-live

openapi-refresh:
	@python3 scripts/openapi_contract.py refresh

build:
	@rm -rf dist
	@python3 -m build
	@python3 -m twine check --strict dist/*

clean:
	@rm -rf dist build
	@find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.pyc' -delete 2>/dev/null || true
