"""Audit the pinned historical capsule with the repository helper."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "tools/verify_issue1598_history.py"),
        run_name="__main__",
    )
