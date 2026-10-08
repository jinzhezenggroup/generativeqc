"""Generate candidate-selected ordered-history CPU/CUDA algebra fragments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.method.gfn2_history_lowering import (
    emit_gfn2_history_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for backend in ("cpu", "cuda") if args.backend == "both" else (args.backend,):
        for name, source in emit_gfn2_history_artifacts(backend).items():
            (args.output_directory / name).write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
