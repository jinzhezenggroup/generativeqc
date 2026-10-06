"""Share one scalar Coulomb recurrence across dddd components and responses.

The Coulomb IR defines states as Cartesian spatial derivatives, so its first
response is a raised spatial state. Existing Dual3 Hermite coefficients carry
the pair geometry response; the scalar L+1 simplex is prepared once per product.
"""


def emit_direct_pair_materialized_gradient_support() -> str:
    """Emit a uniform CTA consumer for both full-range dddd force layouts.

    Six 256-component slots cover the Cartesian shell domain. The caller must
    prove a full-range operator, resident cache and retained recurrence mode.
    Every AO component retains its Schwarz and independent density gates.
    """
    return r"""
#if defined(__CUDACC__)
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_force_scatter.cuh"
#include "scf/cuda/direct_force_sources.hpp"

namespace generativeqc::scf::cuda_execution {

/** The scalar Coulomb IR's first spatial response is the next Cartesian state.
 * R^n_tuv differentiates (-2*rho)^n F_n(rho |P-Q|^2), as defined by the existing
 * Coulomb derivative algebra. The atom response scales each raised state by
 * d(P-Q)/dR_atom. Full Coulomb and total spatial degree <=8 are the admitted
 * domain; the L+1 scalar simplex supplies every response read. */
struct MaterializedCoulombSpatialResponse {
  const CoulombAuxiliary<double, 9>& scalar_states;
  double product_response;
  __device__ Dual3 at(unsigned n, unsigned t, unsigned u, unsigned v) const {
    return {scalar_states.at(n, t, u, v),
            product_response * scalar_states.at(n, t + 1, u, v),
            product_response * scalar_states.at(n, t, u + 1, v),
            product_response * scalar_states.at(n, t, u, v + 1)};
  }
};

/** Bra Hermite responses for the N-1 independent atoms survive all ket products.
 * One scalar Coulomb simplex serves every atom, component and source response;
 * the ket Hermite response is retired before the next atom replaces it. */
struct MaterializedDirectPairDerivativeRecurrence {
  using Pair = ShellPairHermiteCoefficients<Dual3, 2, 2>;
  Pair bra[3][3], ket[3];
  CoulombAuxiliary<double, 9> coulomb;
  Vec3<Dual3> first_product[3], second_product;
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

/** Uniform CTA entry; all six packets and atoms share each prepared pair product.
 * Separate preserves independent J'/K' outputs; Combined precontracts their
 * signed density weights before the shared component derivative is consumed.
 * Inactive lanes participate in every publish/retire barrier. */
template <bool Unrestricted, DirectForceOutputMode Mode = DirectForceOutputMode::Separate>
__device__ inline void contract_materialized_direct_pair_force(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* forces, double coulomb_coefficient, double exchange_coefficient,
    MaterializedDirectPairDerivativeRecurrence& shared,
    MaterializedDirectPairWork* work = nullptr, double* checked_derivatives = nullptr) {
  constexpr unsigned Slots = 6;
  constexpr unsigned Sources = DirectForceSources<Mode>::count;
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
  double weights[Slots][Sources]{}, derivative_sum[Slots][3]{};
  bool any_admitted = false;
  for (unsigned slot = 0; slot < Slots; ++slot) {
    const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
    if (ordinal >= count || !decode_direct_tile_ao_ordinal(batch, task, ordinal,
        first_count, second_count, ao_begin, n, i[slot], j[slot], k[slot], l[slot]) ||
        !direct_ao_quartet_survives_schwarz(schwarz_bounds, physical, n,
            i[slot], j[slot], k[slot], l[slot], screening_tolerance)) continue;
    // The common source contract carries the caller's signed coefficients;
    // Combined must not introduce an extra exchange or spin factor.
    for (unsigned source = 0; source < Sources; ++source) {
      const auto coefficients = DirectForceSources<Mode>::coefficients(
          source, coulomb_coefficient, exchange_coefficient);
      weights[slot][source] = direct_force_density_coefficient_scaled<Unrestricted>(
          n, physical, spin, density, i[slot], j[slot], k[slot], l[slot],
          coefficients.coulomb, coefficients.exchange);
      admitted[slot] |= weights[slot][source] != 0.0;
    }
    any_admitted |= admitted[slot];
  }
  if (!__syncthreads_or(any_admitted)) return;
  const auto bra_begin = batch.shell_pair_primitive_offsets[first_pair];
  const auto ket_begin = batch.shell_pair_primitive_offsets[second_pair];
  const auto nb = batch.shell_primitive_offsets[sj + 1] - batch.shell_primitive_offsets[sj];
  const auto nd = batch.shell_primitive_offsets[sl + 1] - batch.shell_primitive_offsets[sl];
  const std::size_t source_stride = std::size_t(batch.total_atoms) * 3;
  // Keep a bounded derivative sum per component/independent atom. Primitive
  // traversal and coefficient order remain unchanged within each response.
  double gradients[3][Slots][3]{};
  for (auto bra = bra_begin; bra < batch.shell_pair_primitive_offsets[first_pair + 1]; ++bra) {
    if (threadIdx.x == 0) {
      const auto& cached = batch.shell_primitive_pairs[bra];
      const auto a = batch.shell_primitive_offsets[si] + (bra - bra_begin) / nb;
      const auto b = batch.shell_primitive_offsets[sj] + (bra - bra_begin) % nb;
      shared.p = cached.exponent_sum;
      shared.coefficients[0] = batch.primitive_coefficients[a];
      shared.coefficients[1] = batch.primitive_coefficients[b];
      for (unsigned center = 0; center + 1 < centers; ++center) {
        const auto coordinate = std::int64_t(unique_atoms[center]) * 3;
        shared.first_product[center] = prepare_materialized_direct_pair_derivative(cached,
            batch.primitive_exponents[a], batch.primitive_exponents[b],
            atom_position<Dual3>(batch, atoms[0], coordinate),
            atom_position<Dual3>(batch, atoms[1], coordinate), shared.bra[center]);
        if (work) atomicAdd(&work->bra_preparations, 1ULL);
      }
    }
    __syncthreads();
    for (auto ket = ket_begin; ket < batch.shell_pair_primitive_offsets[second_pair + 1]; ++ket) {
      if (threadIdx.x == 0) {
        const auto& cached = batch.shell_primitive_pairs[ket];
        const auto c = batch.shell_primitive_offsets[sk] + (ket - ket_begin) / nd;
        const auto d = batch.shell_primitive_offsets[sl] + (ket - ket_begin) % nd;
        shared.q = cached.exponent_sum;
        shared.coefficients[2] = batch.primitive_coefficients[c];
        shared.coefficients[3] = batch.primitive_coefficients[d];
        const Vec3<double> first_product{shared.first_product[0].x.value,
            shared.first_product[0].y.value, shared.first_product[0].z.value};
        fill_coulomb<9>(shared.p * shared.q / (shared.p + shared.q),
                       first_product, cached.product_center, shared.coulomb);
        if (work) atomicAdd(&work->coulomb_preparations, 1ULL);
      }
      __syncthreads();
      for (unsigned center = 0; center + 1 < centers; ++center) {
        if (threadIdx.x == 0) {
          const auto& cached = batch.shell_primitive_pairs[ket];
          const auto c = batch.shell_primitive_offsets[sk] + (ket - ket_begin) / nd;
          const auto d = batch.shell_primitive_offsets[sl] + (ket - ket_begin) % nd;
          const auto coordinate = std::int64_t(unique_atoms[center]) * 3;
          shared.second_product = prepare_materialized_direct_pair_derivative(cached,
              batch.primitive_exponents[c], batch.primitive_exponents[d],
              atom_position<Dual3>(batch, atoms[2], coordinate),
              atom_position<Dual3>(batch, atoms[3], coordinate), shared.ket);
          if (work) atomicAdd(&work->ket_preparations, 1ULL);
        }
        __syncthreads();
        const MaterializedCoulombSpatialResponse response{shared.coulomb,
            shared.first_product[center].x.derivative_x - shared.second_product.x.derivative_x};
        for (unsigned slot = 0; slot < Slots; ++slot) {
          if (!admitted[slot]) continue;
          const double coefficient = batch.direct_ao_coefficients[ao_begin + i[slot]] *
              batch.direct_ao_coefficients[ao_begin + j[slot]] *
              batch.direct_ao_coefficients[ao_begin + k[slot]] *
              batch.direct_ao_coefficients[ao_begin + l[slot]];
          const double weight = coefficient * shared.coefficients[0] * shared.coefficients[1] *
                                shared.coefficients[2] * shared.coefficients[3];
          const auto component = weight * consume_cartesian_coulomb_states<8, Dual3>(
              shared.p, shared.q, direct_ao_angular(batch, ao_begin + i[slot]),
              direct_ao_angular(batch, ao_begin + j[slot]),
              direct_ao_angular(batch, ao_begin + k[slot]),
              direct_ao_angular(batch, ao_begin + l[slot]), shared.bra[center], shared.ket, response);
          gradients[center][slot][0] += component.derivative_x;
          gradients[center][slot][1] += component.derivative_y;
          gradients[center][slot][2] += component.derivative_z;
          if (work) atomicAdd(&work->component_contractions, 1ULL);
        }
        __syncthreads();
      }
    }
  }
  for (unsigned center = 0; center + 1 < centers; ++center) {
    const auto coordinate = std::int64_t(unique_atoms[center]) * 3;
    for (unsigned slot = 0; slot < Slots; ++slot) {
      if (!admitted[slot]) continue;
      const auto& derivative = gradients[center][slot];
      const std::size_t ordinal = slot * detail::kDirectQuartetTileSize + threadIdx.x;
      for (unsigned axis = 0; axis < 3; ++axis) {
        derivative_sum[slot][axis] += derivative[axis];
        if (checked_derivatives) checked_derivatives[(ordinal * 4 + center) * 3 + axis] = derivative[axis];
        for (unsigned source = 0; source < Sources; ++source) {
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
      for (unsigned source = 0; source < Sources; ++source) {
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
