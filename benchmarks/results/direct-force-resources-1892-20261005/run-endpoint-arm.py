"""Run retained complete endpoints from an explicitly sealed source snapshot.

Source archives deliberately omit Git metadata. Only the HF harness's two Git
identity commands are mapped to verified snapshot receipts; scientific owners,
SCF stopping criteria, force options, and endpoint timers remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    """Select the existing moved-geometry protocol without normalizing iterations."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=("hf", "pbe0"), required=True)
    arguments, remaining = parser.parse_known_args()
    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        if arguments.method == "pbe0":
            from benchmarks.readme_pbe0 import main as endpoint

            endpoint()
            return

        from benchmarks import compare_df_direct_endpoint as harness

        identity = json.loads(Path(os.environ["P0B_SNAPSHOT_IDENTITY"]).read_text())
        original_check_output = subprocess.check_output

        def snapshot_output(
            command: list[str], *values: object, **options: object
        ) -> str | bytes:
            """Supply truthful archive provenance only for the known Git queries."""
            if command == ["git", "rev-parse", "HEAD"]:
                result = identity["base_commit"] + "\n"
            elif command == ["git", "status", "--porcelain"]:
                result = "".join(
                    f" M {relative}\n" for relative in identity["changed_files"]
                )
            else:
                return original_check_output(command, *values, **options)
            return result if options.get("text") else result.encode()

        with patch.object(harness.subprocess, "check_output", snapshot_output):
            harness.main()


if __name__ == "__main__":
    main()
