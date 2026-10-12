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
    polymorphic_output: bool = False,
) -> str:
    """Emit one canonical-ERI scatter with shared J/K spin semantics.

    Direct response and generated Fock workers may supply a runtime-owned
    accumulation sink. The sink does not own the contraction or spin policy.
    """

    unroll_directive = "#pragma unroll\n" if unroll_permutations else ""
    output_template = ""
    if polymorphic_output:
        parameters = parameters.replace("double* fock", "Output fock")
        output_template = ",\n          typename Output = double*"
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
template <bool Unrestricted, bool MixedProduct = false, typename Integral = double{output_template}>
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
    methods. An optional seed differentiates the same quadratic products,
    preserving the complete canonical orbit without a second ERI traversal.
    The compatibility wrapper preserves the historical HF convention.
    """

    return """/** Quadratic density product, or its directional derivative along seed.
 * Apply scale before multiplication, preserving finite small-coefficient paths.
 */
__device__ __forceinline__ double direct_force_density_product(
    const double* density, const double* seed, std::size_t offset,
    std::size_t first, std::size_t second, double scale) {
  if (seed != nullptr)
    return scale * density[offset + first] * seed[offset + second] +
           scale * seed[offset + first] * density[offset + second];
  return scale * density[offset + first] * density[offset + second];
}

template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient_scaled(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    double coulomb_coefficient, double exchange_coefficient,
    const double* seed = nullptr) {
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
        coefficient += direct_force_density_product(
            density, seed, physical_offset, ab, cd, 0.5 * coulomb_coefficient);
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
        coefficient += direct_force_density_product(
            density, seed, physical_offset, ac, bd, 0.5 * exchange_coefficient);
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
            f"({exchange_only} ? ({coulomb_only} ? -0.5 : 1.0) : ({coulomb_only} ? 0.0 : -0.5))"
        ),
        unrestricted_exchange_scale=(
            f"({exchange_only} ? ({coulomb_only} ? -1.0 : 1.0) : ({coulomb_only} ? 0.0 : -1.0))"
        ),
        description=(
            "Scatter one canonical integral using GENERATIVEQC's shared HF/J-only/K-only convention."
        ),
        polymorphic_output=True,
    ).rstrip("\n")


def emit_direct_bilinear_density_coefficient() -> str:
    """Emit P:G'(D) from the same ordered RHF J - K/2 scatter.

    Each canonical integral contributes through its distinct eightfold orbit.
    Polarize the shared quadratic coefficient, rather than maintaining a second
    Coulomb/exchange formula. The native consumer owns finite audits/traversal.
    """
    return """/** Coefficient of one canonical derivative ERI in P:(J'(D)-K'(D)/2). */
__device__ __forceinline__ double direct_bilinear_density_coefficient(
    std::size_t n, std::size_t offset, const double* density, const double* seed,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  return direct_force_density_coefficient_scaled<false>(
      n, offset, 0, density, i, j, k, l, 1.0, -0.5, seed);
}
"""


def emit_direct_fock_accumulation_header() -> str:
    """Emit the native fallback adapter from the same compiler-owned equations."""

    function = emit_fock_accumulation_cuda(
        function_name="accumulate_direct_fock_integral",
        parameters="""    std::size_t n, std::size_t physical_offset, std::size_t spin_offset,
    const double* density, double* fock, std::size_t i, std::size_t j,
    std::size_t k, std::size_t l, Integral integral, bool coulomb_only = false,
    bool exchange_only = false, bool hf_exchange = false""",
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
        restricted_exchange_scale=(
            "hf_exchange ? -0.5 : (coulomb_only ? 0.0 : (exchange_only ? 1.0 : -0.5))"
        ),
        unrestricted_exchange_scale=(
            "hf_exchange ? -1.0 : (coulomb_only ? 0.0 : (exchange_only ? 1.0 : -1.0))"
        ),
        description=(
            "Scatter one symmetry-canonical ERI into HF, Coulomb-only, or exchange-only matrices."
        ),
        unroll_permutations=False,
        polymorphic_output=True,
    )
    return f"""#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "scf/cuda/direct_eri_symmetry.cuh"
#include "scf/cuda/matrix_index.cuh"

// Generated by the scientific compiler. Native Direct-HF owns only the
// surrounding task/runtime plumbing; the RHF/UHF contraction lives here.

namespace generativeqc::scf::cuda_execution {{

{function}
{emit_direct_force_density_coefficient()}
{emit_direct_force_component_weight()}
{emit_direct_bilinear_density_coefficient()}
}}  // namespace generativeqc::scf::cuda_execution
"""
