"""Build-only CPU factorized ERI value, first, mixed and matrix libraries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.factorized_matrix import emit_factorized_matrix_cpu
from generativeqc_compiler.integral.factorized_second import emit_factorized_second

from tools.generate_df_kernels import write_if_changed


def emit_first_cpu_source() -> str:
    """Bind canonical value/gradient lowering without selecting by numeric motion.

    The consumer requests this operation only when it needs no mixed integral
    field. It still receives every independent mathematical-center gradient.
    Keep this specialization in its own translation unit, not a second algebra.
    """
    return (
        emit_factorized_second(backend="cpu", output="first")
        + r"""
#include "integrals/host_factorized_primitives.hpp"

namespace generativeqc::integrals {
bool full_range_primitive_first_cpu(const unsigned* angular, unsigned component,
                                    const double* record, double* output) {
  const double direction[12]{};
  return scf::factorized_second::primitive(0, 0, angular, component, record, direction, output);
}
}  // namespace generativeqc::integrals
"""
    )


def emit_value_cpu_source() -> str:
    """Expose canonical primal algebra without calculating independent partials."""
    return (
        emit_factorized_second(backend="cpu", output="value")
        + r"""
#include "integrals/host_factorized_primitives.hpp"

namespace generativeqc::integrals {
bool full_range_primitive_value_cpu(const unsigned* angular, unsigned component,
                                    const double* record, double* output) {
  const double direction[12]{};
  return scf::factorized_second::primitive(0, 0, angular, component, record, direction, output);
}
}  // namespace generativeqc::integrals
"""
    )


def main() -> None:
    """Lower canonical factor mathematics without a runtime or reference load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-output", required=True, type=Path)
    parser.add_argument("--value-output", required=True, type=Path)
    parser.add_argument("--matrix-output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(
        arguments.output, emit_factorized_second(backend="cpu", output="full")
    )
    arguments.first_output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.first_output, emit_first_cpu_source())
    arguments.value_output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.value_output, emit_value_cpu_source())
    arguments.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.matrix_output, emit_factorized_matrix_cpu())


if __name__ == "__main__":
    main()
