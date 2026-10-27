"""Recheck retained endpoint receipts and independent gates without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

from benchmarks.readme_defaults_qualification import qualify


def main() -> None:
    """Recompute every population, including the recorded fitting failures."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory", type=Path, default=root / ".artifacts/cold-opt"
    )
    raw = parser.parse_args().raw_directory
    summary = json.loads(lzma.decompress((directory / "summary.json.xz").read_bytes()))
    for name, expected in summary["references"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed retained reference: {name}")
    reference = directory.parent / "readme-final-20261022"
    oracles = json.loads(lzma.decompress((reference / "oracles.json.xz").read_bytes()))
    reuse = json.loads((reference / "reference-reuse.json").read_text())
    calls = failed = 0
    for population in summary["populations"]:
        payload = (raw / population["file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != population["sha256"]:
            raise ValueError(f"Changed receipt: {population['file']}")
        receipt = json.loads(payload)
        result = qualify([receipt], oracles, reference_reuse=reuse)[0]
        if result != population["qualification"]:
            raise ValueError(f"Changed qualification: {population['file']}")
        calls += len(receipt["samples"])
        failed += result["status"] == "failed"
    if calls != 96 or failed != 6:
        raise ValueError(f"Unexpected complete calls/failing populations: {calls}/{failed}")
    print(f"Rechecked {calls} complete calls; {failed} known DF HF populations still fail.")


if __name__ == "__main__":
    main()
