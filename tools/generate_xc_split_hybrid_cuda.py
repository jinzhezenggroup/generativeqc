"""Command-line entry point for compiler-owned Libxc CUDA lowering."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.xc.split_hybrid_codegen import main

if __name__ == "__main__":
    main()
