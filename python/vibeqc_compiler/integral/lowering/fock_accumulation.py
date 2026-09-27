"""Emit the shared Coulomb and RHF/UHF Direct-Fock scatter contractions."""

from __future__ import annotations


def emit_fock_accumulation_cuda(
    *,
    function_name: str,
    parameters: str,
    setup: str,
    permutation_setup: str,
    matrix_index: str,
    density_offset: str,
    spin_offset: str,
    description: str,
    coulomb_scale: str,
    restricted_exchange_scale: str,
    unrestricted_exchange_scale: str,
    unroll_permutations: bool = True,
) -> str:
    """Emit one canonical-ERI scatter with shared J/K spin semantics."""

    unroll_directive = "#pragma unroll\n" if unroll_permutations else ""
    contribution_name = f"{function_name}_contribution"
    return f"""/** Scale one density-by-integral product into a Fock contribution.
 *
 * MixedProduct narrows only the density-by-integral product. Scale evaluation,
 * density storage, the returned contribution, and global Fock accumulation
 * remain FP64 so the mixed candidate does not widen its numerical scope.
 */
template <bool MixedProduct, typename Integral>
__device__ __forceinline__ double {contribution_name}(
    double scale, double density_value, Integral integral) {{
  if constexpr (MixedProduct) {{
    return scale * static_cast<double>(
        static_cast<float>(density_value) * static_cast<float>(integral));
  }}
  return scale * density_value * static_cast<double>(integral);
}}

/** {description} */
template <bool Unrestricted, bool MixedProduct = false, typename Integral = double>
__device__ __forceinline__ void {function_name}(
{parameters}) {{
{setup}
{unroll_directive}  for (unsigned permutation = 0; permutation < 8U; ++permutation) {{
{permutation_setup}
    const std::size_t ab = {matrix_index}(a, b, n);
    const std::size_t ac = {matrix_index}(a, c, n);
    const std::size_t cd = {matrix_index}(c, d, n);
    const std::size_t bd = {matrix_index}(b, d, n);
    const double j_scale = {coulomb_scale};
    if constexpr (Unrestricted) {{
      const double k_scale = {unrestricted_exchange_scale};
      const double alpha_cd = density[{spin_offset} + cd];
      const double beta_cd = density[{spin_offset} + matrix_size + cd];
      const double total_cd = alpha_cd + beta_cd;
      if (j_scale != 0.0 && total_cd != 0.0) {{
        atomicAdd(fock + {spin_offset} + ab, {contribution_name}<MixedProduct>(j_scale, total_cd, integral));
        atomicAdd(
            fock + {spin_offset} + matrix_size + ab,
            {contribution_name}<MixedProduct>(j_scale, total_cd, integral));
      }}
      if (k_scale != 0.0) {{
        const double alpha_bd = density[{spin_offset} + bd];
        const double beta_bd = density[{spin_offset} + matrix_size + bd];
        if (alpha_bd != 0.0) {{
          atomicAdd(fock + {spin_offset} + ac, {contribution_name}<MixedProduct>(k_scale, alpha_bd, integral));
        }}
        if (beta_bd != 0.0) {{
          atomicAdd(
              fock + {spin_offset} + matrix_size + ac,
              {contribution_name}<MixedProduct>(k_scale, beta_bd, integral));
        }}
      }}
    }} else {{
      const double k_scale = {restricted_exchange_scale};
      const double density_cd = density[{density_offset} + cd];
      if (j_scale != 0.0 && density_cd != 0.0) {{
        atomicAdd(fock + {density_offset} + ab, {contribution_name}<MixedProduct>(j_scale, density_cd, integral));
      }}
      if (k_scale != 0.0) {{
        const double density_bd = density[{density_offset} + bd];
        if (density_bd != 0.0) {{
          atomicAdd(
              fock + {density_offset} + ac,
              {contribution_name}<MixedProduct>(k_scale, density_bd, integral));
        }}
      }}
    }}
  }}
}}
"""


def emit_direct_force_density_coefficient() -> str:
    """Emit the symmetry-reduced Direct force density contraction.

    The scaled form is the method-neutral primitive used by composed mean-field
    methods. The compatibility wrapper preserves the historical HF convention.
    """

    return """template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient_scaled(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    double coulomb_coefficient, double exchange_coefficient) {
  const std::size_t matrix_size = n * n;
  double coefficient = 0.0;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
    const std::size_t ab = matrix_index(a, b, n);
    const std::size_t ac = matrix_index(a, c, n);
    const std::size_t cd = matrix_index(c, d, n);
    const std::size_t bd = matrix_index(b, d, n);
    if (coulomb_coefficient != 0.0) {
      if constexpr (Unrestricted) {
        const double total_ab =
            density[spin_offset + ab] + density[spin_offset + matrix_size + ab];
        const double total_cd =
            density[spin_offset + cd] + density[spin_offset + matrix_size + cd];
        coefficient += 0.5 * coulomb_coefficient * total_ab * total_cd;
      } else {
        coefficient += 0.5 * coulomb_coefficient *
                       density[physical_offset + ab] * density[physical_offset + cd];
      }
    }
    if (exchange_coefficient != 0.0) {
      if constexpr (Unrestricted) {
        coefficient +=
            0.5 * exchange_coefficient *
            (density[spin_offset + ac] * density[spin_offset + bd] +
             density[spin_offset + matrix_size + ac] *
                 density[spin_offset + matrix_size + bd]);
      } else {
        coefficient += 0.5 * exchange_coefficient *
                       density[physical_offset + ac] * density[physical_offset + bd];
      }
    }
  }
  return coefficient;
}

template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  constexpr double exchange_coefficient = Unrestricted ? -1.0 : -0.5;
  return direct_force_density_coefficient_scaled<Unrestricted>(
      n, physical_offset, spin_offset, density, i, j, k, l, 1.0, exchange_coefficient);
}
"""


def emit_direct_force_component_weight() -> str:
    """Emit normalized Direct-force external component weights."""

    return """__device__ __forceinline__ double direct_force_component_weight(
    const double* ao_coefficients, std::size_t system_ao_begin,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    double density_coefficient) {
  return density_coefficient *
         ao_coefficients[system_ao_begin + i] *
         ao_coefficients[system_ao_begin + j] *
         ao_coefficients[system_ao_begin + k] *
         ao_coefficients[system_ao_begin + l];
}
"""


def emit_generated_shell_fock_accumulation() -> str:
    """Emit the scatter helper embedded in compiler-generated shell kernels."""

    exchange_only = (
        "(task.reversed_shell_pair_mask & kGeneratedDpppExchangeConsumerBit) != 0U"
    )
    coulomb_only = (
        "(task.reversed_shell_pair_mask & kGeneratedDpppCoulombConsumerBit) != 0U"
    )
    return emit_fock_accumulation_cuda(
        function_name="generated_dppp_accumulate_fock",
        parameters="""    const GeneratedDpppShellTask& task,
    const double* density,
    double* fock,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    Integral integral""",
        setup="""  const std::size_t n = static_cast<std::size_t>(task.matrix_order);
  const std::size_t matrix_size = n * n;""",
        permutation_setup="""    std::size_t a = 0, b = 0, c = 0, d = 0;
    generated_dppp_eri_permutation(
        permutation, i, j, k, l, a, b, c, d);
    if (!generated_dppp_unique_permutation(
            permutation, i, j, k, l, a, b, c, d)) continue;""",
        matrix_index="generated_dppp_matrix_index",
        density_offset="task.density_offset",
        spin_offset="task.spin_offset",
        coulomb_scale=f"({exchange_only} ? 0.0 : 1.0)",
        restricted_exchange_scale=(
            f"({coulomb_only} ? 0.0 : ({exchange_only} ? 1.0 : -0.5))"
        ),
        unrestricted_exchange_scale=(
            f"({coulomb_only} ? 0.0 : ({exchange_only} ? 1.0 : -1.0))"
        ),
        description=(
            "Scatter one canonical integral using VIBEQC's shared HF/J-only/K-only convention."
        ),
    ).rstrip("\n")


def emit_direct_fock_accumulation_header() -> str:
    """Emit the native fallback adapter from the same compiler-owned equations."""

    function = emit_fock_accumulation_cuda(
        function_name="accumulate_direct_fock_integral",
        parameters="""    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density, double* fock, std::size_t i, std::size_t j,
    std::size_t k, std::size_t l, Integral integral, bool coulomb_only = false,
    bool exchange_only = false""",
        setup="  const std::size_t matrix_size = n * n;",
        permutation_setup="""    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);""",
        matrix_index="matrix_index",
        density_offset="physical_offset",
        spin_offset="spin_offset",
        coulomb_scale="exchange_only ? 0.0 : 1.0",
        restricted_exchange_scale=("coulomb_only ? 0.0 : (exchange_only ? 1.0 : -0.5)"),
        unrestricted_exchange_scale=(
            "coulomb_only ? 0.0 : (exchange_only ? 1.0 : -1.0)"
        ),
        description=(
            "Scatter one symmetry-canonical ERI into HF, Coulomb-only, or exchange-only matrices."
        ),
        unroll_permutations=False,
    )
    return f"""#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "scf/cuda/direct_eri_symmetry.cuh"
#include "scf/cuda/matrix_index.cuh"

// Generated by the scientific compiler. Native Direct-HF owns only the
// surrounding task/runtime plumbing; the RHF/UHF contraction lives here.

namespace vibeqc::scf::cuda_execution {{

{function}
{emit_direct_force_density_coefficient()}
{emit_direct_force_component_weight()}
}}  // namespace vibeqc::scf::cuda_execution
"""
