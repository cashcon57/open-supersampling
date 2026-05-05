"""Entry point for ``python -m oss.capture.tray``."""
from __future__ import annotations

import sys

from oss.capture.tray.app import main

if __name__ == "__main__":
    sys.exit(main())
