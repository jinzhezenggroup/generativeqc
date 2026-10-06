"""Share the retained FP64 forward derivative recurrence across dddd sources.

Dual3 is the existing scalar differentiation algebra. This lowering changes
its lifetime to one primitive-pair product and independent atom, with no new
derivative equations or raw four-index derivative tensor.
"""


def emit_direct_pair_materialized_gradient_support() -> str:
    """Emit a uniform CTA consumer for separate full-range J'/K' dddd forces.

    Six 256-component slots cover the Cartesian shell domain. The caller must
    prove the full-source operator, resident cache and retained recurrence mode.
    Every AO component retains its Schwarz and independent density gates.
    """
    return r"""
#if defined(__CUDACC__)
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_force_scatter.cuh"

namespace generativeqc::scf::cuda_execution {

/** One dddd Dual3 simplex shared by all components and both density channels.
 * The AD scalar carries x/y/z for one atom; repeated shell centers share its
 * seed. Only N-1 distinct atoms are differentiated, preserving translation. */
struct MaterializedDirectPairDerivativeRecurrence {
  using Pair = ShellPairHermiteCoefficients<Dual3, 2, 2>;
  Pair bra[3], ket[3];
  CoulombAuxiliary<Dual3, 8> coulomb;
  Vec3<Dual3> first_product, second_product;
  double p, q, coefficients[4];
};
static_assert(sizeof(MaterializedDirectPairDerivativeRecurrence) <= (32U << 10));

/** Reuse cached primal geometry and seed the incumbent product-center algebra.
 * Individual primitive exponents seed the geometry response; neither combined
 * cache coefficients nor underflowed Gaussian factors are divided back out. */
__device__ inline Vec3<Dual3> prepare_materialized_direct_pair_derivative(
    const PrimitivePairData& cached, double alpha, double beta,
    const Vec3<Dual3>& first, const Vec3<Dual3>& second,
    MaterializedDirectPairDerivativeRecurrence::Pair (&coefficients)[3]) {
  auto product = product_center(alpha, first, beta, second);
  product.x.value = cached.product_center.x;
  product.y.value = cached.product_center.y;
  product.z.value = cached.product_center.z;
  for (unsigned axis = 0; axis < 3; ++axis) {
    const auto a = vec_axis(first, axis), b = vec_axis(second, axis), ab = a - b;
    fill_shell_pair_hermite_geometry<2, 2>(2, 2, vec_axis(product, axis), a, b,
        cached.exponent_sum, qexp(-cached.reduced_exponent * ab * ab), coefficients[axis]);
  }
  return product;
}

/** Uniform CTA entry; all six packets share each atom's prepared pair product.
 * J' and K' use the same component derivative and preserve independent source
 * outputs. Inactive lanes participate in every publish/retire barrier. */
template <bool Unrestricted>
__device__ inline void contract_materialized_direct_pair_full_source_force(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* forces, double coulomb_coefficient, double exchange_coefficient,
    MaterializedDirectPairDerivativeRecurrence& shared,
    MaterializedDirectPairWork* work = nullptr, double* checked_derivatives = nullptr) {
  constexpr unsigned Slots = 6;
  const auto first_pair = task.first_pair, second_pair = task.second_pair;
  const auto system = batch.shell_pair_systems[first_pair];
  if ((active && !active[system]) ||
      (coulomb_coefficient == 0.0 && exchange_coefficient == 0.0)) return;
  const auto si = batch.shell_pair_first[first_pair], sj = batch.shell_pair_second[first_pair];
  const auto sk = batch.shell_pair_first[second_pair], sl = batch.shell_pair_second[second_pair];
  const std::int32_t atoms[4] = {batch.shell_atoms[si], batch.shell_atoms[sj],
                               batch.shell_atoms[sk], batch.shell_atoms[sl]};
  std::int32_t unique_atoms[4];
  const unsigned centers = direct_force_unique_center_atoms(atoms, unique_atoms);
  if (centers <= 1) return;
  const std::size_t n = batch.direct_nbf, physical = std::size_t(system) * n * n;
  const std::size_t spin = 2 * physical, ao_begin = std::size_t(system) * n;
  const auto first_count = shell_ao_pair_count(batch, first_pair);
  const auto second_count = shell_ao_pair_count(batch, second_pair);
  const auto count = first_pair == second_pair ? first_count * (first_count + 1) / 2
                                             : first_count * second_count;
  std::size_t i[Slots]{}, j[Slots]{}, k[Slots]{}, l[Slots]{};
  bool admitted[Slots]{};
  double weights[Slots][2]{}, derivative_sum[Slots][3]{};
  bool any_admitted = false;
  for (unsigned slot = 0; slot < Slots; ++slot) {
    const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
    if (ordinal >= count || !decode_direct_tile_ao_ordinal(batch, task, ordinal,
        first_count, second_count, ao_begin, n, i[slot], j[slot], k[slot], l[slot]) ||
        !direct_ao_quartet_survives_schwarz(schwarz_bounds, physical, n,
            i[slot], j[slot], k[slot], l[slot], screening_tolerance)) continue;
    weights[slot][0] = direct_force_density_coefficient_scaled<Unrestricted>(n, physical, spin,
        density, i[slot], j[slot], k[slot], l[slot], coulomb_coefficient, 0.0);
    weights[slot][1] = direct_force_density_coefficient_scaled<Unrestricted>(n, physical, spin,
        density, i[slot], j[slot], k[slot], l[slot], 0.0, exchange_coefficient);
    admitted[slot] = weights[slot][0] != 0.0 || weights[slot][1] != 0.0;
    any_admitted |= admitted[slot];
  }
  if (!__syncthreads_or(any_admitted)) return;
  const auto bra_begin = batch.shell_pair_primitive_offsets[first_pair];
  const auto ket_begin = batch.shell_pair_primitive_offsets[second_pair];
  const auto nb = batch.shell_primitive_offsets[sj + 1] - batch.shell_primitive_offsets[sj];
  const auto nd = batch.shell_primitive_offsets[sl + 1] - batch.shell_primitive_offsets[sl];
  const std::size_t source_stride = std::size_t(batch.total_atoms) * 3;
  for (unsigned center = 0; center + 1 < centers; ++center) {
    const auto coordinate = std::int64_t(unique_atoms[center]) * 3;
    const auto first = atom_position<Dual3>(batch, atoms[0], coordinate);
    const auto second = atom_position<Dual3>(batch, atoms[1], coordinate);
    const auto third = atom_position<Dual3>(batch, atoms[2], coordinate);
    const auto fourth = atom_position<Dual3>(batch, atoms[3], coordinate);
    Dual3 gradients[Slots]{};
    for (auto bra = bra_begin; bra < batch.shell_pair_primitive_offsets[first_pair + 1]; ++bra) {
      if (threadIdx.x == 0) {
        const auto& cached = batch.shell_primitive_pairs[bra];
        const auto a = batch.shell_primitive_offsets[si] + (bra - bra_begin) / nb;
        const auto b = batch.shell_primitive_offsets[sj] + (bra - bra_begin) % nb;
        shared.p = cached.exponent_sum;
        shared.first_product = prepare_materialized_direct_pair_derivative(cached,
            batch.primitive_exponents[a], batch.primitive_exponents[b], first, second, shared.bra);
        shared.coefficients[0] = batch.primitive_coefficients[a];
        shared.coefficients[1] = batch.primitive_coefficients[b];
        if (work) atomicAdd(&work->bra_preparations, 1ULL);
      }
      __syncthreads();
      for (auto ket = ket_begin; ket < batch.shell_pair_primitive_offsets[second_pair + 1]; ++ket) {
        if (threadIdx.x == 0) {
          const auto& cached = batch.shell_primitive_pairs[ket];
          const auto c = batch.shell_primitive_offsets[sk] + (ket - ket_begin) / nd;
          const auto d = batch.shell_primitive_offsets[sl] + (ket - ket_begin) % nd;
          shared.q = cached.exponent_sum;
          shared.second_product = prepare_materialized_direct_pair_derivative(cached,
              batch.primitive_exponents[c], batch.primitive_exponents[d], third, fourth, shared.ket);
          shared.coefficients[2] = batch.primitive_coefficients[c];
          shared.coefficients[3] = batch.primitive_coefficients[d];
          fill_coulomb<8>(shared.p * shared.q / (shared.p + shared.q),
              shared.first_product, shared.second_product, shared.coulomb);
          if (work) {
            atomicAdd(&work->ket_preparations, 1ULL);
            atomicAdd(&work->coulomb_preparations, 1ULL);
          }
        }
        __syncthreads();
        for (unsigned slot = 0; slot < Slots; ++slot) {
          if (!admitted[slot]) continue;
          const double coefficient = batch.direct_ao_coefficients[ao_begin + i[slot]] *
              batch.direct_ao_coefficients[ao_begin + j[slot]] *
              batch.direct_ao_coefficients[ao_begin + k[slot]] *
              batch.direct_ao_coefficients[ao_begin + l[slot]];
          const double weight = coefficient * shared.coefficients[0] * shared.coefficients[1] *
                                shared.coefficients[2] * shared.coefficients[3];
          gradients[slot] = gradients[slot] + weight * consume_cartesian_coulomb<8>(
              shared.p, shared.q, direct_ao_angular(batch, ao_begin + i[slot]),
              direct_ao_angular(batch, ao_begin + j[slot]),
              direct_ao_angular(batch, ao_begin + k[slot]),
              direct_ao_angular(batch, ao_begin + l[slot]), shared.bra, shared.ket, shared.coulomb);
          if (work) atomicAdd(&work->component_contractions, 1ULL);
        }
        __syncthreads();
      }
    }
    for (unsigned slot = 0; slot < Slots; ++slot) {
      if (!admitted[slot]) continue;
      const double derivative[3] = {gradients[slot].derivative_x, gradients[slot].derivative_y,
                                    gradients[slot].derivative_z};
      const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
      for (unsigned axis = 0; axis < 3; ++axis) {
        derivative_sum[slot][axis] += derivative[axis];
        if (checked_derivatives) checked_derivatives[(ordinal * 4 + center) * 3 + axis] = derivative[axis];
        for (unsigned source = 0; source < 2; ++source) {
          const double value = -weights[slot][source] * derivative[axis];
          if (value != 0.0) atomicAdd(forces + source * source_stride + coordinate + axis, value);
        }
      }
    }
  }
  const auto final_coordinate = std::int64_t(unique_atoms[centers - 1]) * 3;
  for (unsigned slot = 0; slot < Slots; ++slot) {
    if (!admitted[slot]) continue;
    const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (checked_derivatives) checked_derivatives[(ordinal * 4 + centers - 1) * 3 + axis] = -derivative_sum[slot][axis];
      for (unsigned source = 0; source < 2; ++source) {
        const double value = weights[slot][source] * derivative_sum[slot][axis];
        if (value != 0.0) atomicAdd(forces + source * source_stride + final_coordinate + axis, value);
      }
    }
    if (work) atomicAdd(&work->published_components, 1ULL);
  }
}
}  // namespace generativeqc::scf::cuda_execution
#endif
"""
