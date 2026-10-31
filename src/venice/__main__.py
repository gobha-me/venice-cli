"""Run the command dispatcher with ``python -m venice``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
