"""Emit bounded primitive-pair preparation shared by Cartesian Direct consumers.

This schedule reuses the existing Gaussian cache, Hermite recurrence, Coulomb
simplex and J/K scatter owners. It changes their lifetime, never their formulas.
"""

from __future__ import annotations


def emit_direct_pair_materialized_support() -> str:
    """Prepare once per primitive-pair product and consume all live packet AOs.

    A packet is the existing exact 256-AO-quartet domain. Multiple packets of a
    larger shell quartet remain independently screened and counted; coalescing
    their recurrence is a separate prepared scheduling step, not assumed here.
    """
    return r"""
#if defined(__CUDACC__)
#include "generated_direct_shell_pair_hermite.cuh"
#include "scf/cuda/direct_fock_accumulation.cuh"
#include "scf/cuda/direct_screening.cuh"

namespace generativeqc::scf::cuda_execution {

/** Optional qualification counters. Production passes no allocation. */
struct MaterializedDirectPairWork {
  unsigned long long bra_preparations{}, ket_preparations{}, coulomb_preparations{};
  unsigned long long component_contractions{}, published_components{};
};

/** One CTA-owned recurrence, bounded independently of primitive contraction
 * length. No primitive/AO quartet tensor or geometry-dependent global cache. */
template <unsigned AngularOrder>
struct MaterializedDirectPairRecurrence {
  static_assert(AngularOrder >= 5 && AngularOrder <= kMaximumCoulombOrder);
  using Pair = ShellPairHermiteCoefficients<double, kMaximumAngularMomentum,
                                            kMaximumAngularMomentum>;
  Pair bra[3], ket[3];
  CoulombAuxiliary<double, AngularOrder> coulomb;
  PrimitivePairData first, second;
  double coefficients[4];
};
static_assert(sizeof(MaterializedDirectPairRecurrence<12>) <= (48U << 10));

/** Borrow cached p/mu/P and retain the incumbent axis-Gaussian arithmetic.
 * The cache's combined weighted_coefficient is deliberately not divided to
 * recover raw coefficients: zero/underflow would lose the scientific inputs.
 * Individual coefficient factors retain their original multiplication order. */
template <class Pair>
__device__ inline void prepare_materialized_direct_pair(
    const PrimitivePairData& cached, unsigned first_angular, unsigned second_angular,
    const Vec3<double>& first, const Vec3<double>& second, Pair (&coefficients)[3]) {
  for (unsigned axis = 0; axis < 3; ++axis) {
    const double a = vec_axis(first, axis), b = vec_axis(second, axis), ab = a - b;
    fill_shell_pair_hermite_geometry<kMaximumAngularMomentum, kMaximumAngularMomentum>(
        first_angular, second_angular, vec_axis(cached.product_center, axis), a, b,
        cached.exponent_sum, qexp(-cached.reduced_exponent * ab * ab), coefficients[axis]);
  }
}

/** Uniform CTA entry: admission precedes every barrier. Inactive/tail lanes
 * still participate in publication/retirement but never read an invalid AO.
 * All selected components consume one Coulomb recurrence per pair product.
 * J and K reuse each contracted ERI through the existing symmetry scatter. */
template <bool Unrestricted, unsigned AngularOrder>
__device__ inline void contract_materialized_direct_pair_fock(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* fock, const std::uint64_t* generated_mask,
    MaterializedDirectPairRecurrence<AngularOrder>& shared,
    MaterializedDirectPairWork* work = nullptr, bool coulomb_only = false,
    bool exchange_only = false, double* checked_components = nullptr) {
  if (task.first_pair >= batch.total_shell_pairs || task.second_pair >= batch.total_shell_pairs)
    return;
  const auto first_pair = task.first_pair, second_pair = task.second_pair;
  const auto system = batch.shell_pair_systems[first_pair];
  if (system < 0 || system >= batch.batch_size || batch.shell_pair_systems[second_pair] != system ||
      (active && !active[system])) return;
  const auto si = batch.shell_pair_first[first_pair], sj = batch.shell_pair_second[first_pair];
  const auto sk = batch.shell_pair_first[second_pair], sl = batch.shell_pair_second[second_pair];
  const auto li = batch.shell_angular[si], lj = batch.shell_angular[sj];
  const auto lk = batch.shell_angular[sk], ll = batch.shell_angular[sl];
  const auto shell_class = direct_quartet_shell_class_device(li, lj, lk, ll);
  if (generated_mask && (*generated_mask & (std::uint64_t{1} << shell_class))) return;
  const std::size_t n = batch.direct_nbf, physical = std::size_t(system) * n * n;
  const std::size_t spin = 2 * physical, ao_begin = std::size_t(system) * n;
  const auto first_count = shell_ao_pair_count(batch, first_pair);
  const auto second_count = shell_ao_pair_count(batch, second_pair);
  const auto count = first_pair == second_pair ? first_count * (first_count + 1) / 2
                                               : first_count * second_count;
  const std::size_t ordinal = std::size_t(task.tile) * detail::kDirectQuartetTileSize + threadIdx.x;
  std::size_t i{}, j{}, k{}, l{};
  const bool admitted = ordinal < count &&
      decode_direct_tile_ao_ordinal(batch, task, ordinal, first_count, second_count,
                                    ao_begin, n, i, j, k, l) &&
      direct_ao_quartet_survives_schwarz(schwarz_bounds, physical, n, i, j, k, l,
                                        screening_tolerance);
  if (!__syncthreads_or(admitted)) return;
  const auto first = atom_position<double>(batch, batch.shell_atoms[si], -1);
  const auto second = atom_position<double>(batch, batch.shell_atoms[sj], -1);
  const auto third = atom_position<double>(batch, batch.shell_atoms[sk], -1);
  const auto fourth = atom_position<double>(batch, batch.shell_atoms[sl], -1);
  const auto bra_begin = batch.shell_pair_primitive_offsets[first_pair];
  const auto ket_begin = batch.shell_pair_primitive_offsets[second_pair];
  const auto nb = batch.shell_primitive_offsets[sj + 1] - batch.shell_primitive_offsets[sj];
  const auto nd = batch.shell_primitive_offsets[sl + 1] - batch.shell_primitive_offsets[sl];
  double value = 0.0;
  for (auto bra = bra_begin; bra < batch.shell_pair_primitive_offsets[first_pair + 1]; ++bra) {
    if (threadIdx.x == 0) {
      shared.first = batch.shell_primitive_pairs[bra];
      prepare_materialized_direct_pair(shared.first, li, lj, first, second, shared.bra);
      const auto a = batch.shell_primitive_offsets[si] + (bra - bra_begin) / nb;
      const auto b = batch.shell_primitive_offsets[sj] + (bra - bra_begin) % nb;
      shared.coefficients[0] = batch.primitive_coefficients[a];
      shared.coefficients[1] = batch.primitive_coefficients[b];
      if (work) atomicAdd(&work->bra_preparations, 1ULL);
    }
    __syncthreads();
    for (auto ket = ket_begin; ket < batch.shell_pair_primitive_offsets[second_pair + 1]; ++ket) {
      if (threadIdx.x == 0) {
        shared.second = batch.shell_primitive_pairs[ket];
        prepare_materialized_direct_pair(shared.second, lk, ll, third, fourth, shared.ket);
        const auto c = batch.shell_primitive_offsets[sk] + (ket - ket_begin) / nd;
        const auto d = batch.shell_primitive_offsets[sl] + (ket - ket_begin) % nd;
        shared.coefficients[2] = batch.primitive_coefficients[c];
        shared.coefficients[3] = batch.primitive_coefficients[d];
        const auto p = shared.first.exponent_sum, q = shared.second.exponent_sum;
        fill_coulomb<AngularOrder>(p * q / (p + q), shared.first.product_center,
                                  shared.second.product_center, shared.coulomb);
        if (work) {
          atomicAdd(&work->ket_preparations, 1ULL);
          atomicAdd(&work->coulomb_preparations, 1ULL);
        }
      }
      __syncthreads();
      if (admitted) {
        const double coefficient = batch.direct_ao_coefficients[ao_begin + i] *
            batch.direct_ao_coefficients[ao_begin + j] * batch.direct_ao_coefficients[ao_begin + k] *
            batch.direct_ao_coefficients[ao_begin + l];
        const double weight = coefficient * shared.coefficients[0] * shared.coefficients[1] *
                             shared.coefficients[2] * shared.coefficients[3];
        value += weight * consume_cartesian_coulomb<AngularOrder>(
            shared.first.exponent_sum, shared.second.exponent_sum,
            direct_ao_angular(batch, ao_begin + i), direct_ao_angular(batch, ao_begin + j),
            direct_ao_angular(batch, ao_begin + k), direct_ao_angular(batch, ao_begin + l),
            shared.bra, shared.ket, shared.coulomb);
        if (work) atomicAdd(&work->component_contractions, 1ULL);
      }
      // Retire all readers before the next ket replaces the shared recurrence.
      __syncthreads();
    }
  }
  if (admitted) {
    // Qualification may inspect each final component without affecting the
    // production storage contract, where this borrowed address is null.
    if (checked_components) checked_components[ordinal] = value;
    if (value != 0.0)
      accumulate_direct_fock_integral<Unrestricted>(n, physical, spin, density, fock,
                                                    i, j, k, l, value, coulomb_only, exchange_only);
    if (work) atomicAdd(&work->published_components, 1ULL);
  }
}
}  // namespace generativeqc::scf::cuda_execution
#endif
"""
