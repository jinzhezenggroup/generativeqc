"""Emit bounded primitive-pair preparation shared by Cartesian Direct consumers.

This schedule reuses the existing Gaussian cache, Hermite recurrence, Coulomb
simplex and J/K scatter owners. It changes their lifetime, never their formulas.
"""

from __future__ import annotations


def emit_direct_pair_materialized_support() -> str:
    """Prepare once per primitive-pair product for admitted adjacent packets.

    Each packet is the exact 256-AO-quartet domain with independent component
    screening. Explicit register slots can coalesce a complete shell quartet
    only when its caller proves common shell admission and precision/source
    routing. A single-slot caller preserves the original packet contract.
    """
    return r"""
#if defined(__CUDACC__)
#include "generated_direct_shell_pair_hermite.cuh"
#include "runtime/compensated_atomic.cuh"
#include "scf/cuda/direct_fock_accumulation.cuh"
#include "scf/cuda/direct_screening.cuh"

namespace generativeqc::scf::cuda_execution {

/** Optional qualification counters. Production passes no allocation. */
struct MaterializedDirectPairWork {
  unsigned long long bra_preparations{}, ket_preparations{}, coulomb_preparations{};
  unsigned long long component_contractions{}, published_components{};
};

/** Canonical whole-shell owners use fewer idle lanes in small domains.
 * These compile-time bounds cover every s/p/d/f composition, not just the
 * most frequent shell class. Legacy packet consumers retain 256 lanes. */
__host__ __device__ inline constexpr unsigned canonical_materialized_component_lanes(
    unsigned order) {
  return order <= 5 ? 32 : 256;
}
__host__ __device__ inline constexpr unsigned canonical_materialized_component_slots(
    unsigned order) {
  unsigned maximum = 1;
  for (unsigned shell = 0; shell < 4; ++shell) {
    const unsigned angular = order / 4 + (shell < order % 4);
    maximum *= (angular + 1) * (angular + 2) / 2;
  }
  const unsigned lanes = canonical_materialized_component_lanes(order);
  return (maximum + lanes - 1) / lanes;
}

/** One CTA-owned recurrence, bounded independently of primitive contraction
 * length. No primitive/AO quartet tensor or geometry-dependent global cache. */
template <unsigned AngularOrder, typename Scalar = double>
struct MaterializedDirectPairRecurrence {
  static_assert(AngularOrder <= kMaximumCoulombOrder);
  static constexpr unsigned pair_angular =
      AngularOrder < kMaximumAngularMomentum ? AngularOrder : kMaximumAngularMomentum;
  using Pair = ShellPairHermiteCoefficients<Scalar, pair_angular, pair_angular>;
  Pair bra[3], ket[3];
  CoulombAuxiliary<Scalar, AngularOrder> coulomb;
  PrimitivePairData first, second;
  double coefficients[4];
};
static_assert(sizeof(MaterializedDirectPairRecurrence<12>) <= (48U << 10));

/** Borrow cached p/mu/P and retain the incumbent axis-Gaussian arithmetic.
 * The cache's combined weighted_coefficient is deliberately not divided to
 * recover raw coefficients: zero/underflow would lose the scientific inputs.
 * Individual coefficient factors retain their original multiplication order.
 * A disjoint axis interval lends the same complete Hermite initialization to
 * independent lanes; the default interval preserves serial packet consumers. */
template <class Pair>
__device__ inline void prepare_materialized_direct_pair(
    const PrimitivePairData& cached, unsigned first_angular, unsigned second_angular,
    const Vec3<double>& first, const Vec3<double>& second, Pair (&coefficients)[3],
    unsigned axis_begin = 0U, unsigned axis_end = 3U) {
  for (unsigned axis = axis_begin; axis < axis_end; ++axis) {
    const double a = vec_axis(first, axis), b = vec_axis(second, axis), ab = a - b;
    using Scalar = std::remove_reference_t<decltype(coefficients[axis].at(0, 0, 0))>;
    fill_shell_pair_hermite_geometry<Pair::kIDimension - 1, Pair::kJDimension - 1, Scalar>(
        first_angular, second_angular, vec_axis(cached.product_center, axis), a, b,
        cached.exponent_sum, qexp(-cached.reduced_exponent * ab * ab), coefficients[axis]);
  }
}

/** Descending radial planes depend only on the already-published n+1 plane.
 * Rows within a plane are independent. Reuse the serial owner's Boys roots
 * and axis operation so this changes scheduling, not recurrence arithmetic.
 * Only the first warp prepares the small planes. Other component warps wait
 * at one final CTA publication instead of paying a CTA barrier per plane.
 * All lanes enter, including those without an admitted Cartesian component. */
template <unsigned AngularOrder>
__device__ inline void fill_materialized_coulomb_cooperative(
    double exponent, const Vec3<double>& product, const Vec3<double>& center,
    CoulombAuxiliary<double, AngularOrder>& auxiliary) {
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0)
      fill_coulomb_roots<AngularOrder>(exponent, product, center, auxiliary);
    __syncwarp();
    const Vec3<double> difference{product.x - center.x, product.y - center.y, product.z - center.z};
    for (unsigned plane = 1; plane <= AngularOrder; ++plane) {
      const unsigned radial = AngularOrder - plane;
      const unsigned rows = (plane + 1) * (plane + 2) / 2;
      for (unsigned row = threadIdx.x; row < rows; row += 32) {
        unsigned first_order = 0, second_order = row;
        while (second_order > plane - first_order) {
          second_order -= plane - first_order + 1;
          ++first_order;
        }
        for (unsigned third_order = 0; third_order <= plane - first_order - second_order;
             ++third_order) {
          if (first_order)
            auxiliary.at(radial, first_order, second_order, third_order) =
                raise_coulomb_axis<0>(auxiliary, difference.x, radial, first_order, second_order, third_order);
          else if (second_order)
            auxiliary.at(radial, first_order, second_order, third_order) =
                raise_coulomb_axis<1>(auxiliary, difference.y, radial, first_order, second_order, third_order);
          else if (third_order)
            auxiliary.at(radial, first_order, second_order, third_order) =
                raise_coulomb_axis<2>(auxiliary, difference.z, radial, first_order, second_order, third_order);
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

/** Uniform CTA entry: admission precedes every barrier. Inactive/tail lanes
 * still participate in publication/retirement but never read an invalid AO.
 * All selected components consume one Coulomb recurrence per pair product.
 * J and K reuse each contracted ERI through the existing symmetry scatter. */
template <bool Unrestricted, unsigned AngularOrder, unsigned ComponentSlots = 1,
          unsigned ComponentLanes = detail::kDirectQuartetTileSize, typename Scalar = double,
          bool CooperativePreparation = false, bool ConvolvedContraction = false>
__device__ inline void contract_materialized_direct_pair_fock(
    DeviceBatch batch, const ActiveShellQuartetTile& task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* fock, const std::uint64_t* generated_mask,
    MaterializedDirectPairRecurrence<AngularOrder, Scalar>& shared,
    MaterializedDirectPairWork* work = nullptr, bool coulomb_only = false,
    bool exchange_only = false, double* checked_components = nullptr,
    bool hf_exchange = false,
    generativeqc::integrals::CoulombRange range = generativeqc::integrals::CoulombRange::Full,
    double omega = 0.0, double* separate_coulomb = nullptr, double* separate_exchange = nullptr,
    std::uint64_t* canonical_work = nullptr, std::uint64_t* mixed_work = nullptr) {
  constexpr bool mixed = std::is_same_v<Scalar, MixedPrecisionFloat>;
  static_assert(!CooperativePreparation || std::is_same_v<Scalar, double>);
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
  if constexpr (mixed) {
    if (range != generativeqc::integrals::CoulombRange::Full || !separate_coulomb ||
        separate_exchange || exchange_only || hf_exchange) {
      if (threadIdx.x == 0 && separate_coulomb)
        atomicAdd(separate_coulomb + physical, static_cast<double>(NAN));
      return;
    }
  }
  const auto first_count = shell_ao_pair_count(batch, first_pair);
  const auto second_count = shell_ao_pair_count(batch, second_pair);
  const auto count = first_pair == second_pair ? first_count * (first_count + 1) / 2
                                               : first_count * second_count;
  // Each register slot owns one independently screened packet lane. Callers
  // may coalesce adjacent packets only after admitting the complete shell
  // quartet on the same precision/source route. The dddd stream supplies all
  // six possible packets; the incumbent packet queue supplies one.
  static_assert(ComponentSlots > 0);
  static_assert(ComponentLanes >= 32 && ComponentLanes <= detail::kDirectQuartetTileSize &&
                (ComponentLanes & (ComponentLanes - 1)) == 0);
  std::size_t i[ComponentSlots]{}, j[ComponentSlots]{}, k[ComponentSlots]{}, l[ComponentSlots]{};
  bool admitted[ComponentSlots]{};
  double value[ComponentSlots]{};
  bool any_admitted = false;
  for (unsigned slot = 0; slot < ComponentSlots; ++slot) {
    const std::size_t ordinal = std::size_t(task.tile) * detail::kDirectQuartetTileSize +
                                slot * ComponentLanes + threadIdx.x;
    admitted[slot] = ordinal < count &&
        decode_direct_tile_ao_ordinal(batch, task, ordinal, first_count, second_count,
                                      ao_begin, n, i[slot], j[slot], k[slot], l[slot]) &&
        direct_ao_quartet_survives_schwarz(schwarz_bounds, physical, n,
                                          i[slot], j[slot], k[slot], l[slot],
                                          screening_tolerance);
    if constexpr (mixed) {
      if (admitted[slot])
        admitted[slot] = direct_coulomb_density_active<Unrestricted>(
            n, physical, spin, density, i[slot], j[slot], k[slot], l[slot]);
    }
    any_admitted |= admitted[slot];
  }
  if (!__syncthreads_or(any_admitted)) return;
  const auto first = atom_position<double>(batch, batch.shell_atoms[si], -1);
  const auto second = atom_position<double>(batch, batch.shell_atoms[sj], -1);
  const auto third = atom_position<double>(batch, batch.shell_atoms[sk], -1);
  const auto fourth = atom_position<double>(batch, batch.shell_atoms[sl], -1);
  const auto bra_begin = batch.shell_pair_primitive_offsets[first_pair];
  const auto ket_begin = batch.shell_pair_primitive_offsets[second_pair];
  const auto nb = batch.shell_primitive_offsets[sj + 1] - batch.shell_primitive_offsets[sj];
  const auto nd = batch.shell_primitive_offsets[sl + 1] - batch.shell_primitive_offsets[sl];
  for (auto bra = bra_begin; bra < batch.shell_pair_primitive_offsets[first_pair + 1]; ++bra) {
    // Read immutable cached geometry directly: shared.first is not published
    // until the existing barrier, so other axis owners must not borrow it yet.
    if constexpr (CooperativePreparation || (AngularOrder == 3 && ComponentLanes == 32)) {
      if (threadIdx.x < 3U)
        prepare_materialized_direct_pair(batch.shell_primitive_pairs[bra], li, lj, first, second,
                                         shared.bra, threadIdx.x, threadIdx.x + 1U);
    }
    if (threadIdx.x == 0) {
      shared.first = batch.shell_primitive_pairs[bra];
      if constexpr (!CooperativePreparation && (AngularOrder != 3 || ComponentLanes != 32))
        prepare_materialized_direct_pair(shared.first, li, lj, first, second, shared.bra);
      const auto a = batch.shell_primitive_offsets[si] + (bra - bra_begin) / nb;
      const auto b = batch.shell_primitive_offsets[sj] + (bra - bra_begin) % nb;
      shared.coefficients[0] = batch.primitive_coefficients[a];
      shared.coefficients[1] = batch.primitive_coefficients[b];
      if (work) atomicAdd(&work->bra_preparations, 1ULL);
    }
    __syncthreads();
    for (auto ket = ket_begin; ket < batch.shell_pair_primitive_offsets[second_pair + 1]; ++ket) {
      // All three axes retire with the unchanged Coulomb publication barrier.
      if constexpr (CooperativePreparation || (AngularOrder == 3 && ComponentLanes == 32)) {
        if (threadIdx.x < 3U)
          prepare_materialized_direct_pair(batch.shell_primitive_pairs[ket], lk, ll, third, fourth,
                                           shared.ket, threadIdx.x, threadIdx.x + 1U);
      }
      if (threadIdx.x == 0) {
        shared.second = batch.shell_primitive_pairs[ket];
        if constexpr (!CooperativePreparation && (AngularOrder != 3 || ComponentLanes != 32))
          prepare_materialized_direct_pair(shared.second, lk, ll, third, fourth, shared.ket);
        const auto c = batch.shell_primitive_offsets[sk] + (ket - ket_begin) / nd;
        const auto d = batch.shell_primitive_offsets[sl] + (ket - ket_begin) % nd;
        shared.coefficients[2] = batch.primitive_coefficients[c];
        shared.coefficients[3] = batch.primitive_coefficients[d];
        const auto p = shared.first.exponent_sum, q = shared.second.exponent_sum;
        if constexpr (mixed) {
          // Cached geometry and Gaussian bases remain FP64. Only the shared
          // Hermite/Coulomb recurrence and its component contraction narrow.
          const Vec3<Scalar> bra_center{shared.first.product_center.x,
                                       shared.first.product_center.y,
                                       shared.first.product_center.z};
          const Vec3<Scalar> ket_center{shared.second.product_center.x,
                                       shared.second.product_center.y,
                                       shared.second.product_center.z};
          fill_coulomb<AngularOrder>(EvaluationReal<Scalar>{p * q / (p + q)}, bra_center,
                                    ket_center, shared.coulomb);
        } else {
          if (range == generativeqc::integrals::CoulombRange::Full) {
            if constexpr (!CooperativePreparation)
              fill_coulomb<AngularOrder>(p * q / (p + q), shared.first.product_center,
                                        shared.second.product_center, shared.coulomb);
          }
          else if (!fill_range_coulomb<AngularOrder>(p * q / (p + q), shared.first.product_center,
                      shared.second.product_center, range, omega, shared.coulomb))
            shared.coulomb.at(0, 0, 0, 0) = NAN;
        }
        if (work) {
          atomicAdd(&work->ket_preparations, 1ULL);
          atomicAdd(&work->coulomb_preparations, 1ULL);
        }
      }
      __syncthreads();
      if constexpr (CooperativePreparation) {
        if (range == generativeqc::integrals::CoulombRange::Full) {
          const auto bra_exponent = shared.first.exponent_sum, ket_exponent = shared.second.exponent_sum;
          fill_materialized_coulomb_cooperative<AngularOrder>(
              bra_exponent * ket_exponent / (bra_exponent + ket_exponent),
              shared.first.product_center, shared.second.product_center, shared.coulomb);
        }
      }
      for (unsigned slot = 0; slot < ComponentSlots; ++slot) {
        if (!admitted[slot]) continue;
        const double coefficient = batch.direct_ao_coefficients[ao_begin + i[slot]] *
            batch.direct_ao_coefficients[ao_begin + j[slot]] *
            batch.direct_ao_coefficients[ao_begin + k[slot]] *
            batch.direct_ao_coefficients[ao_begin + l[slot]];
        const double weight = coefficient * shared.coefficients[0] * shared.coefficients[1] *
                             shared.coefficients[2] * shared.coefficients[3];
        // Bounded FP64 owners may reuse the qualified three-axis convolution.
        // Serial/packet consumers retain their six-loop summation by default.
        value[slot] += weight * scalar_value(contract_prepared_cartesian_coulomb<AngularOrder, Scalar>(
            shared.first.exponent_sum, shared.second.exponent_sum,
            direct_ao_angular(batch, ao_begin + i[slot]), direct_ao_angular(batch, ao_begin + j[slot]),
            direct_ao_angular(batch, ao_begin + k[slot]), direct_ao_angular(batch, ao_begin + l[slot]),
            shared.bra, shared.ket, shared.coulomb, ConvolvedContraction));
        if (work) atomicAdd(&work->component_contractions, 1ULL);
      }
      // Retire all readers before the next ket replaces the shared recurrence.
      __syncthreads();
    }
  }
  for (unsigned slot = 0; slot < ComponentSlots; ++slot) {
    if (!admitted[slot]) continue;
    const std::size_t ordinal = std::size_t(task.tile) * detail::kDirectQuartetTileSize +
                                slot * ComponentLanes + threadIdx.x;
    // Qualification may inspect each final component without affecting the
    // production storage contract, where this borrowed address is null.
    if (checked_components) checked_components[ordinal] = value[slot];
    if (value[slot] != 0.0) {
      // A canonical consumer publishes independent positive J/K sources, not
      // HF's weighted Fock matrix. Both retain the authoritative orbit scatter.
      if (separate_coulomb || separate_exchange) {
        if (separate_coulomb)
          accumulate_direct_fock_integral<Unrestricted>(n, physical, spin, density,
              separate_coulomb, i[slot], j[slot], k[slot], l[slot], value[slot], true, false);
        if (separate_exchange)
          accumulate_direct_fock_integral<Unrestricted>(n, physical, spin, density,
              separate_exchange, i[slot], j[slot], k[slot], l[slot], value[slot], false, true);
      } else {
        // Exported-reference HF owns a correction plane; ordinary and
        // independent J/K callers retain their existing accumulation sinks.
        accumulate_direct_fock_integral<Unrestricted>(n, physical, spin, density,
            runtime::CompensatedOutput{fock, batch.reference_fock_correction},
            i[slot], j[slot], k[slot], l[slot], value[slot], coulomb_only, exchange_only, hf_exchange);
      }
    }
    if (canonical_work) {
      atomicAdd(reinterpret_cast<unsigned long long*>(canonical_work), 1ULL);
      atomicAdd(reinterpret_cast<unsigned long long*>(canonical_work + 1), 1ULL);
    }
    if constexpr (mixed) {
      if (mixed_work) atomicAdd(reinterpret_cast<unsigned long long*>(mixed_work), 1ULL);
    }
    if (work) atomicAdd(&work->published_components, 1ULL);
  }
}
}  // namespace generativeqc::scf::cuda_execution
#endif
"""
