#!/usr/bin/env python3
"""Run the Phase 2 modular baseline from the repository root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Select a non-interactive Matplotlib backend before importing the legacy
# simulation module when the CLI is asked to run headlessly.
if "--headless" in sys.argv:
    os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confmate.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
