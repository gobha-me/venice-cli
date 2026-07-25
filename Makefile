.PHONY: install uninstall test drive lint build clean

install:
	@./install.sh

uninstall:
	@./uninstall.sh

test:
	@PYTHONPATH=src python3 -m unittest discover -v -s tests

# The pty drive suite on its own, for fast iteration. `make test` already runs
# it (and degrades to skips without pexpect) -- but someone who explicitly typed
# `make drive` must not be told OK when nothing ran, so this one hard-fails.
drive:
	@python3 -c "import pexpect" 2>/dev/null || \
	  { echo 'drive: pexpect missing -- pip install -e ".[test]"'; exit 1; }
	@PYTHONPATH=src python3 -m unittest -v tests.test_drive_cli tests.test_fake_server

lint:
	@python3 -m compileall -q src tests
	@echo "syntax OK"

build:
	@rm -rf dist
	@python3 -m build
	@python3 -m twine check --strict dist/*

clean:
	@rm -rf dist build
	@find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.pyc' -delete 2>/dev/null || true
