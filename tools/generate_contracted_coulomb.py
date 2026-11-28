"""Link bounded contracted Coulomb pair and quartet leaves into the CUDA package."""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.contracted_coulomb import (
    emit_pair_jet,
    emit_quartet_jet,
    quartet_launch_shape,
)

from tools.generate_df_kernels import write_if_changed


def generate(kind: str, left: int, right: int) -> str:
    """Emit one device leaf and a typed, non-owning CUDA launch adapter."""
    key = f"{left}_{right}"
    if kind in {"pair", "aux"}:
        auxiliary = kind == "aux"
        leaf = emit_pair_jet(left, right, auxiliary=auxiliary)
        states = comb(left + right + 3, 3)
        lanes = (3 if auxiliary else 9) * states
        if auxiliary:
            key = f"aux_{key}"
        adapter = f"""
cudaError_t prepare_{key}(const double* records, unsigned count, PairJets fields, int* status,
                           cudaStream_t stream) noexcept {{
  if (!count || count > (UINT_MAX - 127) / {lanes}) return cudaErrorInvalidValue;
  ::prepare_{key}<<<(count * {lanes} + 127) / 128, 128, 0, stream>>>(
      records, count, fields.geometry, fields.weighted, fields.unweighted, status);
  return cudaGetLastError();
}}
"""
    elif kind in {"quartet", "fitted2", "fitted3"}:
        auxiliary_legs = {"quartet": 0, "fitted2": 2, "fitted3": 1}[kind]
        leaf = emit_quartet_jet(left, right, auxiliary_legs=auxiliary_legs)
        threads, shared = quartet_launch_shape(left, right, auxiliary_legs)
        if auxiliary_legs:
            key = f"{kind}_{key}"
        adapter = f"""
cudaError_t configure_{key}() noexcept {{
  return cudaFuncSetAttribute(contracted_detail::contract_{key},
                              cudaFuncAttributeMaxDynamicSharedMemorySize, {shared});
}}

cudaError_t launch_{key}(PairJets left, PairJets right, const unsigned* pairs, unsigned count,
                          double* window, int* status, cudaStream_t stream) noexcept {{
  if (!count || count > INT_MAX) return cudaErrorInvalidValue;
  contracted_detail::contract_{key}<<<count, {threads}, {shared}, stream>>>(
      left.geometry, right.geometry, left.weighted, right.weighted, left.unweighted,
      right.unweighted, pairs, window, count, status);
  return cudaGetLastError();
}}
"""
    else:
        raise ValueError("expected pair or quartet")
    return (
        '#include <climits>\n#include "dft/contracted_coulomb.hpp"\n'
        + leaf
        + "\nnamespace generativeqc::dft::hessian_aot::contracted {\n"
        + adapter
        + "}\n"
    )


def main() -> None:
    """Select a single finite compilation unit at build time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("pair", "aux", "quartet", "fitted2", "fitted3"),
        required=True,
    )
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_if_changed(args.output, generate(args.kind, args.left, args.right))


if __name__ == "__main__":
    main()
