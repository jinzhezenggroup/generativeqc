"""Run the pinned upstream Array API suite without dtype or test exclusions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Verify suite/spec provenance and execute the complete CPU eager gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite", type=Path, help="checkout of data-apis/array-api-tests"
    )
    arguments = parser.parse_args()
    suite = arguments.suite.resolve()
    contract = json.loads((ROOT / "manifests/array_api_conformance.json").read_text())
    for checkout, expected in (
        (suite, contract["tests_revision"]),
        (suite / "array-api", contract["spec_revision"]),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        if actual != expected:
            parser.error(f"{checkout}: expected revision {expected}, found {actual}")
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(checkout),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            timeout=10,
        ).strip()
        if dirty:
            parser.error(f"{checkout}: qualification requires an unmodified suite/spec")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ARRAY_API_TESTS_")
    }
    environment.update(
        {
            "ARRAY_API_TESTS_MODULE": contract["module"],
            "ARRAY_API_TESTS_VERSION": contract["api_version"],
            "PYTHONPATH": os.pathsep.join((str(ROOT / "python"), str(suite))),
        }
    )
    # Do not inherit repository test selection, skip/xfail files or caller pytest
    # options: this gate must exercise every supported standard dtype and extension.
    environment.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite / "array_api_tests"),
            "-q",
            "-rs",
            "--hypothesis-max-examples",
            str(contract["max_examples"]),
            "--hypothesis-disable-deadline",
            "--hypothesis-derandomize",
            "--skips-file",
            os.devnull,
            "--xfails-file",
            os.devnull,
        ],
        cwd=suite,
        env=environment,
        check=False,
        timeout=600,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
