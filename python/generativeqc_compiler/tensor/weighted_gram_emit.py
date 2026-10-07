"""AOT source emission for admitted, incumbent weighted-Gram schedules.

All contraction loops and arithmetic below are shared compiler ownership.
Method envelopes bind physical names and failure destinations only. Retained
spelling is deliberate: complete generated function/kernel body hashes are gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scalar_cpp import emit_scalar_cpp
from .weighted_gram import (
    WeightedGram,
    canonical_regions,
    checked_scalar_programs,
    require_candidate,
    require_pair,
    retained_candidate,
    template_hash,
)

_OCCUPIED_CPU = r"""inline void density_from_orbitals(double* output, const double* coefficients,
                                  std::size_t nbf, std::size_t coefficient_stride,
                                  std::size_t occupied, double occupation_weight) {
  // Only canonical occupations authorize mirroring the legacy product order.
  const bool symmetric = occupation_weight == 1.0 || occupation_weight == 2.0;
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = symmetric ? mu : 0; nu < nbf; ++nu) {
      double value = 0.0;
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
        // Preserve the legacy FP64 product/accumulation order for the unique
        // AO pair. Physical SCF callers use exact occupation weights 1 or 2,
        // so the mirrored pair is bitwise-equivalent to the swapped product.
        value += occupation_weight * coefficients[mu * coefficient_stride + orbital] *
                 coefficients[nu * coefficient_stride + orbital];
      }
      output[mu * nbf + nu] = value;
      if (symmetric && mu != nu) output[nu * nbf + mu] = value;
    }
  }
}

inline void weighted_density_from_orbitals(double* output, const double* coefficients,
                                           const double* orbital_energies, std::size_t nbf,
                                           std::size_t coefficient_stride, std::size_t occupied,
                                           double occupation_weight) {
  for (std::size_t mu = 0; mu < nbf; ++mu) {
    for (std::size_t nu = 0; nu < nbf; ++nu) {
      double value = 0.0;
      for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
        value += occupation_weight * orbital_energies[orbital] *
                 coefficients[mu * coefficient_stride + orbital] *
                 coefficients[nu * coefficient_stride + orbital];
      }
      output[mu * nbf + nu] = value;
    }
  }
}"""

_OCCUPIED_CUDA = r"""// Generated from python/generativeqc_compiler/tensor/scf.py.
#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::scf::generated {

inline constexpr const char* cuda_density_tensor_template_hash = "@DENSITY_HASH@";
inline constexpr const char* cuda_weighted_density_tensor_template_hash = "@WEIGHTED_HASH@";
inline constexpr const char* cuda_density_schedule =
    "dense-upper-triangle-mirror-v1";

template <int OccupationWeight>
__global__ void occupied_density_kernel(
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf,
    const std::int32_t* occupied, const double* coefficients,
    const std::uint8_t* active, double* density) {
  static_assert(OccupationWeight == 1 || OccupationWeight == 2);
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(spin_count);
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;

  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  if (row > column) return;

  const std::size_t offset = state * matrix_size;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {
    const std::size_t orbital_index = static_cast<std::size_t>(orbital) * n;
    if constexpr (OccupationWeight == 2) {
      value += 2.0 * coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    } else {
      value += coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    }
  }
  density[element] = value;
  if (row != column)
    density[offset + column + row * n] = value;
}

template <int OccupationWeight>
__global__ void occupied_weighted_density_kernel(
    std::int32_t batch_size, std::int32_t spin_count, std::int32_t nbf,
    const std::int32_t* occupied, const double* coefficients,
    const double* orbital_energies, const std::uint8_t* active,
    double* weighted_density) {
  static_assert(OccupationWeight == 1 || OccupationWeight == 2);
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(spin_count);
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;

  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = state * matrix_size;
  const std::size_t eigen_offset = state * n;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {
    const std::size_t orbital_index = static_cast<std::size_t>(orbital) * n;
    if constexpr (OccupationWeight == 2) {
      value += 2.0 * orbital_energies[eigen_offset + orbital] *
               coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    } else {
      value += orbital_energies[eigen_offset + orbital] *
               coefficients[offset + row + orbital_index] *
               coefficients[offset + column + orbital_index];
    }
  }
  weighted_density[element] = value;
}

}  // namespace generativeqc::scf::generated
"""

_CHECKED_PAIR = r"""  for (std::int64_t pair = std::int64_t{blockIdx.z} * blockDim.x + threadIdx.x; pair < pair_count;
       pair += std::int64_t{gridDim.z} * blockDim.x) {
    const MatrixPair indices = matrix_pair(pair);
    double density = 0.0;
    double weighted_density = 0.0;
    bool finite = true;
    for (std::int64_t local = 0; local < count; ++local) {
      const double first = @COEFFICIENTS@[matrix_begin + indices.row * count + local];
      const double second = @COEFFICIENTS@[matrix_begin + indices.column * count + local];
      double density_left = 0.0;
      double density_contribution = 0.0;
      double density_updated = 0.0;
      if (!@SCALE@(
              first, @WEIGHTS@[orbital_begin + local], density_left) ||
          !@CONTRIBUTION@(
              density_left, second, density_contribution) ||
          !@UPDATE@(
              density_left, second, density, density_updated)) {
        @DENSITY_FAILURE@
        finite = false;
        break;
      }
      double weighted_left = 0.0;
      double weighted_contribution = 0.0;
      double weighted_updated = 0.0;
      if (!@SCALE@(
              first, @ENERGY_WEIGHTS@[orbital_begin + local], weighted_left) ||
          !@CONTRIBUTION@(
              weighted_left, second, weighted_contribution) ||
          !@UPDATE@(
              weighted_left, second, weighted_density, weighted_updated)) {
        @WEIGHTED_FAILURE@
        finite = false;
        break;
      }
      density = density_updated;
      weighted_density = weighted_updated;
    }
    if (finite) {
      const std::int64_t first = matrix_begin + indices.row * count + indices.column;
      const std::int64_t second = matrix_begin + indices.column * count + indices.row;
      @DENSITY_OUTPUT@[first] = density;
      @WEIGHTED_OUTPUT@[first] = weighted_density;
      @DENSITY_OUTPUT@[second] = density;
      @WEIGHTED_OUTPUT@[second] = weighted_density;
    }
  }"""


def emit_occupied(plain: WeightedGram, weighted: WeightedGram, *, backend: str) -> str:
    """Emit the selected occupied-prefix schedule with its original source ABI."""
    require_pair(plain, weighted)
    if backend not in ("cpu", "cuda"):
        raise ValueError("unsupported occupied weighted Gram backend")
    schedule = "occupied-cpu" if backend == "cpu" else "occupied-cuda"
    selected = tuple(
        retained_candidate(region, schedule) for region in (plain, weighted)
    )
    for region, candidate in zip((plain, weighted), selected, strict=True):
        require_candidate(region, candidate, schedule)
    execution = selected[0].execution
    assert execution is not None
    algorithm = execution.algorithm
    if algorithm == "occupied-cpu":
        return _OCCUPIED_CPU
    if algorithm == "occupied-cuda":
        return _OCCUPIED_CUDA.replace(
            "@DENSITY_HASH@", template_hash(plain.adapter.program)
        ).replace("@WEIGHTED_HASH@", template_hash(weighted.adapter.program))
    raise ValueError("no qualified occupied weighted Gram emitter")


@dataclass(frozen=True)
class CheckedPairBindings:
    """Physical names and two failure effects owned by the method envelope.

    The enclosing kernel provides count, pair_count, orbital_begin, matrix_begin,
    and the existing MatrixPair/matrix_pair index binding. No mathematical body
    or whole-method callback crosses this boundary.
    """

    coefficients: str
    weights: str
    energy_weights: str
    density_output: str
    weighted_output: str
    scale: str
    contribution: str
    update: str
    density_failure: str
    weighted_failure: str


def emit_checked_pair(
    plain: WeightedGram, weighted: WeightedGram, bindings: CheckedPairBindings
) -> str:
    require_pair(plain, weighted)
    for region in (plain, weighted):
        candidate = retained_candidate(region, "checked-pair-cuda")
        require_candidate(region, candidate, "checked-pair-cuda")
        if (
            candidate.execution is None
            or candidate.execution.algorithm != "checked-pair-cuda"
        ):
            raise ValueError("no qualified paired weighted Gram emitter")
    result = _CHECKED_PAIR
    for name in bindings.__dataclass_fields__:
        result = result.replace("@" + name.upper() + "@", getattr(bindings, name))
    return result


def emit_scalar_stages(backend: str, symbols: dict[str, str]) -> dict[str, str]:
    """Checked factorization of the original ternary einsum, preserving bodies."""
    programs = checked_scalar_programs(backend)
    roles = {
        "energy_weight": ("occupation", "eigenvalue"),
        "weighted_coefficient": ("coefficient", "weight"),
        "contribution": ("weighted_coefficient", "coefficient"),
        "updated": ("weighted_coefficient", "coefficient", "accumulator"),
    }
    if set(symbols) != set(roles):
        raise ValueError("weighted Gram requires every checked scalar stage")
    result = {}
    for stage, inputs in roles.items():
        source = emit_scalar_cpp(
            programs[stage],
            function_name=symbols[stage],
            input_order=inputs,
            output_order=(stage,),
            fused_accumulation=stage == "updated",
            ordered_native_sums=stage == "updated",
        )
        if backend == "cuda":
            source = source.replace(
                "inline bool " + symbols[stage] + "(",
                "__device__ inline bool " + symbols[stage] + "(",
                1,
            )
        result[stage] = source
    return result


def emit_native_header() -> str:
    # The selected candidate names the native execution specialization, not
    # merely a diagnostic identity emitted beside an independent implementation.
    selected = tuple(
        retained_candidate(region, "column-scaled-cpu")
        for region in canonical_regions("cpu", orbital_count=3)
    )
    for region, candidate in zip(
        canonical_regions("cpu", orbital_count=3), selected, strict=True
    ):
        require_candidate(region, candidate, "column-scaled-cpu")
        if (
            candidate.execution is None
            or candidate.execution.algorithm != "column-scaled-cpu"
        ):
            raise ValueError("no qualified native weighted Gram executor")
    native_type = {"column-scaled-cpu": "ColumnScaleDgemmLp64"}[
        selected[0].implementation
    ]
    symbols = {
        stage: symbol
        for stage, symbol in (
            ("energy_weight", "energy_weight"),
            ("weighted_coefficient", "weighted_coefficient"),
            ("contribution", "density_contribution"),
            ("updated", "density_update"),
        )
    }
    bodies = emit_scalar_stages("cpu", symbols)
    return (
        """// Generated from tensor.scf through its retained weighted-Gram lowering.
#pragma once
#include <cmath>
namespace generativeqc::tensor::weighted_gram::generated {
"""
        + "struct ColumnScaleDgemmLp64 {};\nusing CpuExecution = "
        + native_type
        + ";\n"
        + "".join(bodies.values())
        + "}  // namespace generativeqc::tensor::weighted_gram::generated\n"
    )
