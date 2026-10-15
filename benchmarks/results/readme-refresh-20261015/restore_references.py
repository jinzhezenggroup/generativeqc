"""Restore historical geometry oracles without executing either reference engine."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from pathlib import Path


def restore(root: Path) -> None:
    """Reuse retained arrays; original reference timing reductions stay untouched."""
    results = root / "source/benchmarks/results"
    for mode, name in (("none", "summary.json.xz"), ("cuda", "df-summary.json.xz")):
        baseline_path = results / "readme-dft-20261013" / name
        update_path = results / "readme-wb97mv-20261014" / name
        baseline = json.loads(lzma.decompress(baseline_path.read_bytes()))
        update = json.loads(lzma.decompress(update_path.read_bytes()))
        baseline["methods"]["wb97m-v"] = update["methods"]["wb97m-v"]
        for method, points in baseline["methods"].items():
            for point in points:
                source = update_path if method == "wb97m-v" else baseline_path
                record = {
                    "schema": "generativeqc.readme-dft.v1",
                    "engine": "reference",
                    "status": "accuracy_only_historical_reconstruction",
                    "protocol": point["protocol"],
                    "records": point.get("independent_references", []),
                    "historical_binding": {
                        "publication": str(source.relative_to(root / "source")),
                        "publication_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "original_raw_sha256": point["engines"]["reference"].get(
                            "raw_sha256"
                        ),
                        "reference_not_reexecuted": True,
                    },
                }
                for population in ("measurements", "priming"):
                    directory = root / population / mode / method / str(point["atoms"])
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "reference.json").write_text(
                        json.dumps(record, allow_nan=False) + "\n"
                    )
                    for index, receipt in enumerate(
                        point.get("independent_accuracy_receipts", [])
                    ):
                        (directory / f"accuracy-reference-{index}.json").write_text(
                            json.dumps(receipt, allow_nan=False) + "\n"
                        )


if __name__ == "__main__":
    restore(Path(sys.argv[1]).resolve())
