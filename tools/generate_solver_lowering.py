"""Generate the shared ordinary symmetric eigensolver portfolio without CUDA."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.solver_lowering import ordinary_eigh_header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    # Logical paths and source bytes only. Runtime versions/device identity are
    # separately retained by the native owner, never guessed during generation.
    sources = (
        "src/scf/cuda/eigensolver.cpp",
        "src/solver/cuda/symmetric_eigen_provider.hpp",
        "src/solver/cuda/cusolver_compat.hpp",
        "src/solver/cuda/symmetric_eigen_provider.cpp",
        "src/scf/cuda_eigensolver_policy.hpp",
        "src/scf/cuda/eigensolver.hpp",
        "src/scf/cuda/eigensolver_kernels.cu",
        "src/scf/cuda/eigensolver_types.hpp",
        "src/scf/cuda/eigensolver_kernels.hpp",
        "src/scf/cuda/matrix_index.cuh",
        "src/scf/eigensolver_workspace.hpp",
        "src/runtime/lowering_binding.hpp",
        "src/runtime/execution_precision.hpp",
        "python/generativeqc_compiler/common/solver_lowering.py",
        "python/generativeqc_compiler/common/library.py",
        "python/generativeqc_compiler/common/native_lowering.py",
        "python/generativeqc_compiler/common/lowering_provider.py",
        "python/generativeqc_compiler/common/lowering_contract.py",
        "python/generativeqc_compiler/common/precision.py",
    )
    identity = canonical_hash({path: (ROOT / path).read_text() for path in sources})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ordinary_eigh_header(identity))


if __name__ == "__main__":
    main()
