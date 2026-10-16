"""Build-only entry point for the native, numerically all-CUDA GMRES schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.tensor.cuda_gmres import emit_cuda_gmres

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, emit_cuda_gmres())


if __name__ == "__main__":
    main()
