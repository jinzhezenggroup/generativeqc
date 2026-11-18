"""Build-only finite CPU s/p/d/f mixed S/T/V primitive lowering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.one_electron_mixed import (
    emit_one_electron_mixed_cpu,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Generate without importing the runtime or loading a reference provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, emit_one_electron_mixed_cpu())


if __name__ == "__main__":
    main()
