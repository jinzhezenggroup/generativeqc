"""Validate and render PBE0 complete-endpoint evidence without partial timings."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from benchmarks.readme_omol25 import SIZES
from benchmarks.readme_pbe0 import SCHEMA
from tools.render_omol25_benchmarks import collect, figure


def main() -> None:
    """Keep raw oracle forces alongside compact, hash-bound public evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    points = [collect(args.raw_directory, atoms, schema=SCHEMA) for atoms in SIZES]
    for point in points:
        (args.destination / f"water{point['atoms']}.json").write_text(
            json.dumps(point, indent=2, allow_nan=False) + "\n"
        )
        for engine in ("native", "reference"):
            source = args.raw_directory / str(point["atoms"]) / f"{engine}.json"
            if source.exists():
                (
                    args.destination / f"water{point['atoms']}-{engine}.json.gz"
                ).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    target = args.destination / "def2-svp-ho.json"
    if target.resolve() != args.basis_file.resolve():
        shutil.copyfile(args.basis_file, target)
    figure(points, args.destination, title="PBE0 / def2-SVP", filename="pbe0.svg")


if __name__ == "__main__":
    main()
