"""Generate candidate-selected ordered-history CPU/CUDA algebra fragments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for backend in ("cpu", "cuda") if args.backend == "both" else (args.backend,):
        if backend == "cpu":
            from generativeqc_compiler.tensor.broyden_cpu_lowering import (
                emit_broyden_cpu_artifacts,
            )

            artifacts = emit_broyden_cpu_artifacts()
        else:
            from generativeqc_compiler.method.gfn2_history_lowering import (
                emit_gfn2_history_artifacts,
            )

            artifacts = emit_gfn2_history_artifacts("cuda")
        for name, source in artifacts.items():
            with (args.output_directory / name).open(
                "w", encoding="utf-8", newline="\n"
            ) as output:
                output.write(source)


if __name__ == "__main__":
    main()
