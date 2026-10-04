"""Generate native CPU GFN2 electronic scalar kernels from #505 TensorIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.method.gfn2_electronic_runtime import (
    GFN2_ELECTRONIC_RUNTIME_VERSION,
    build_gfn2_core_energy_update_program,
    build_gfn2_density_update_program,
    build_gfn2_energy_weight_program,
    build_gfn2_multipole_hamiltonian_update_program,
    build_gfn2_multipole_integral_vjp_program,
    build_gfn2_population_update_program,
    build_gfn2_restricted_population_publish_program,
    build_gfn2_scalar_hamiltonian_update_program,
    build_gfn2_scalar_integral_vjp_program,
    build_gfn2_spin_population_publish_program,
    build_gfn2_weighted_coefficient_program,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

POPULATION_INPUTS = ("density", "integral", "accumulator")
CORE_ENERGY_INPUTS = ("density", "h0", "accumulator")
ENERGY_WEIGHT_INPUTS = ("occupation", "eigenvalue")
WEIGHTED_COEFFICIENT_INPUTS = ("coefficient", "weight")
DENSITY_UPDATE_INPUTS = ("weighted_coefficient", "coefficient", "accumulator")
RESTRICTED_POPULATION_PUBLISH_INPUTS = ("electronic", "reference")
SPIN_POPULATION_PUBLISH_INPUTS = ("alpha", "beta", "reference")
SCALAR_HAMILTONIAN_INPUTS = (
    "overlap",
    "row_vat",
    "row_vsh",
    "column_vat",
    "column_vsh",
    "accumulator",
)
MULTIPOLE_HAMILTONIAN_INPUTS = (
    "forward_integral",
    "reverse_integral",
    "row_potential",
    "column_potential",
    "accumulator",
)
SCALAR_VJP_INPUTS = ("row_vat", "row_vsh", "column_vat", "column_vsh", "bar_updated")
MULTIPOLE_VJP_INPUTS = ("row_potential", "column_potential", "bar_updated")


def native_header() -> str:
    population = prepare_for_backend(
        build_gfn2_population_update_program(), backend="cpu"
    )
    core_energy = prepare_for_backend(
        build_gfn2_core_energy_update_program(), backend="cpu"
    )
    energy_weight = prepare_for_backend(
        build_gfn2_energy_weight_program(), backend="cpu"
    )
    weighted_coefficient = prepare_for_backend(
        build_gfn2_weighted_coefficient_program(), backend="cpu"
    )
    density_update = prepare_for_backend(
        build_gfn2_density_update_program(), backend="cpu"
    )
    restricted_publish = prepare_for_backend(
        build_gfn2_restricted_population_publish_program(), backend="cpu"
    )
    spin_publish = prepare_for_backend(
        build_gfn2_spin_population_publish_program(), backend="cpu"
    )
    scalar_h = prepare_for_backend(
        build_gfn2_scalar_hamiltonian_update_program(), backend="cpu"
    )
    multipole_h = prepare_for_backend(
        build_gfn2_multipole_hamiltonian_update_program(), backend="cpu"
    )
    scalar_vjp = build_gfn2_scalar_integral_vjp_program()
    scalar_vjp_program = prepare_for_backend(scalar_vjp.program, backend="cpu")
    multipole_vjp = build_gfn2_multipole_integral_vjp_program()
    multipole_vjp_program = prepare_for_backend(multipole_vjp.program, backend="cpu")

    bodies = (
        emit_scalar_cpp(
            population,
            fused_accumulation=True,
            function_name="gfn2_population_update_tensor",
            input_order=POPULATION_INPUTS,
            output_order=("updated",),
        ),
        emit_scalar_cpp(
            core_energy,
            fused_accumulation=True,
            function_name="gfn2_core_energy_update_tensor",
            input_order=CORE_ENERGY_INPUTS,
            output_order=("updated",),
        ),
        emit_scalar_cpp(
            energy_weight,
            function_name="gfn2_energy_weight_tensor",
            input_order=ENERGY_WEIGHT_INPUTS,
            output_order=("energy_weight",),
        ),
        emit_scalar_cpp(
            weighted_coefficient,
            function_name="gfn2_weighted_coefficient_tensor",
            input_order=WEIGHTED_COEFFICIENT_INPUTS,
            output_order=("weighted_coefficient",),
        ),
        emit_scalar_cpp(
            density_update,
            fused_accumulation=True,
            function_name="gfn2_density_update_tensor",
            input_order=DENSITY_UPDATE_INPUTS,
            output_order=("updated",),
            ordered_native_sums=True,
        ),
        emit_scalar_cpp(
            restricted_publish,
            function_name="gfn2_restricted_population_publish_tensor",
            input_order=RESTRICTED_POPULATION_PUBLISH_INPUTS,
            output_order=("charge",),
            ordered_native_sums=True,
        ),
        emit_scalar_cpp(
            spin_publish,
            function_name="gfn2_spin_population_publish_tensor",
            input_order=SPIN_POPULATION_PUBLISH_INPUTS,
            output_order=("charge", "magnetization"),
            ordered_native_sums=True,
        ),
        emit_scalar_cpp(
            scalar_h,
            fused_accumulation=True,
            function_name="gfn2_scalar_hamiltonian_update_tensor",
            input_order=SCALAR_HAMILTONIAN_INPUTS,
            output_order=("updated",),
        ),
        emit_scalar_cpp(
            multipole_h,
            fused_accumulation=True,
            function_name="gfn2_multipole_hamiltonian_update_tensor",
            input_order=MULTIPOLE_HAMILTONIAN_INPUTS,
            output_order=("updated",),
        ),
        emit_scalar_cpp(
            scalar_vjp_program,
            fused_accumulation=True,
            function_name="gfn2_scalar_integral_vjp_tensor",
            input_order=SCALAR_VJP_INPUTS,
            output_order=("bar_overlap",),
        ),
        emit_scalar_cpp(
            multipole_vjp_program,
            fused_accumulation=True,
            function_name="gfn2_multipole_integral_vjp_tensor",
            input_order=MULTIPOLE_VJP_INPUTS,
            output_order=("bar_forward_integral", "bar_reverse_integral"),
        ),
    )
    return f"""// Generated by tools/generate_gfn2_electronic_native.py from #505 TensorIR; do not edit.
#pragma once

#include <cmath>

namespace generativeqc::xtb::generated {{

inline constexpr const char* gfn2_electronic_runtime_version =
    "{GFN2_ELECTRONIC_RUNTIME_VERSION}";
inline constexpr const char* gfn2_population_update_logical_hash =
    "{population.logical_hash}";
inline constexpr const char* gfn2_core_energy_update_logical_hash =
    "{core_energy.logical_hash}";
inline constexpr const char* gfn2_energy_weight_logical_hash =
    "{energy_weight.logical_hash}";
inline constexpr const char* gfn2_weighted_coefficient_logical_hash =
    "{weighted_coefficient.logical_hash}";
inline constexpr const char* gfn2_density_update_logical_hash =
    "{density_update.logical_hash}";
inline constexpr const char* gfn2_restricted_population_publish_logical_hash =
    "{restricted_publish.logical_hash}";
inline constexpr const char* gfn2_spin_population_publish_logical_hash =
    "{spin_publish.logical_hash}";
inline constexpr const char* gfn2_scalar_hamiltonian_update_logical_hash =
    "{scalar_h.logical_hash}";
inline constexpr const char* gfn2_multipole_hamiltonian_update_logical_hash =
    "{multipole_h.logical_hash}";
inline constexpr const char* gfn2_scalar_integral_vjp_hash =
    "{scalar_vjp.derivative_hash}";
inline constexpr const char* gfn2_multipole_integral_vjp_hash =
    "{multipole_vjp.derivative_hash}";
inline constexpr const char* gfn2_scalar_integral_vjp_program_hash =
    "{scalar_vjp_program.logical_hash}";
inline constexpr const char* gfn2_multipole_integral_vjp_program_hash =
    "{multipole_vjp_program.logical_hash}";

{"".join(bodies)}
}}  // namespace generativeqc::xtb::generated
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(native_header(), encoding="utf-8")


if __name__ == "__main__":
    main()
