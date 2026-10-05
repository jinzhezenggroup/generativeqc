"""Emit canonical resident Coulomb contraction bindings without a runtime."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor.vector_lowering import coulomb_header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    paths = (
        "src/tensor/cuda_vector_contraction.cpp",
        "src/tensor/cuda_vector_contraction.hpp",
        "src/tensor/native_contraction.hpp",
        "src/runtime/lowering_binding.hpp",
        "python/generativeqc_compiler/tensor/df_coulomb.py",
        "python/generativeqc_compiler/tensor/vector_lowering.py",
        "python/generativeqc_compiler/tensor/native_lowering.py",
        "python/generativeqc_compiler/tensor/lowering.py",
        "python/generativeqc_compiler/common/native_lowering.py",
    )
    identity = canonical_hash({path: (ROOT / path).read_text() for path in paths})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(coulomb_header(identity))


if __name__ == "__main__":
    main()
