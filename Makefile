.PHONY: install uninstall test lint build clean

install:
	@./install.sh

uninstall:
	@./uninstall.sh

test:
	@PYTHONPATH=src python3 -m unittest discover -v -s tests

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
