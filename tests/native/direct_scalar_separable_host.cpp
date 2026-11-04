/** Host-only execution of the actual scalar task and its pre-publication retry.
 * The test provides a serial CUDA shim; this is never a production CPU route
 * and cannot establish device resources, races, or endpoint performance. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "scf/cuda/direct_force_order5_sources.cuh"

using namespace generativeqc::scf::cuda_execution;
namespace generated_weighted_eri = generativeqc::scf::generated_weighted_eri;
static_assert(!DeviceBatch{}.direct_force_pair_separable);

namespace {
unsigned evaluations{}, fault{};

struct FaultRoots : Order5SourceRoots<kDpppShellClass> {
  static generated_weighted_eri::IndependentGradient evaluate_separable(
      const generated_weighted_eri::Geometry& geometry, const double* bra, const double* ket) {
    auto result = Order5SourceRoots<kDpppShellClass>::evaluate_separable(geometry, bra, ket);
    if (++evaluations == 2U) {
      if (fault == 1U) result.center[0][0] = std::numeric_limits<double>::infinity();
      if (fault == 2U) result.center[0][0] = std::numeric_limits<double>::max() / 8.0;
    }
    return result;
  }
};

template <bool Unrestricted, DirectForceOutputMode Mode, class Roots>
void execute(const DeviceBatch& batch, const double* bounds, const double* density,
             const std::uint8_t* active, double threshold, double coulomb, double exchange,
             double* forces) {
  contract_two_electron_force_low_order_sources_task<Unrestricted, kDpppShellClass, false, Roots,
                                                     Mode>(
      batch, {0U, 1U, 0U}, threshold, bounds, density, active, forces, coulomb, exchange);
}
}  // namespace

extern "C" void scalar_pair_probe(bool unrestricted, bool separate, bool enabled,
                                  unsigned orientation, bool same_ket, unsigned system,
                                  const std::int32_t* atoms, const double* positions,
                                  const double* density, const double* coefficients,
                                  const double* bounds, const double* seed, double threshold,
                                  double coulomb, double exchange, bool mixed,
                                  unsigned injected_fault, double* forces,
                                  unsigned long long* counters) {
  DeviceBatch batch{};
  batch.batch_size = system + 1;
  batch.nbf = batch.direct_nbf = 15;
  batch.total_atoms = batch.total_shells = 4;
  batch.total_shell_pairs = 2;
  batch.positions = positions;
  batch.shell_atoms = atoms;
  const std::uint8_t angular[4] = {2U, 1U, 1U, 1U};
  batch.shell_angular = angular;
  std::int64_t offsets[5] = {0, 6, 9, 12, 15};
  for (auto& offset : offsets) offset += system * 15;
  batch.shell_direct_ao_offsets = offsets;
  batch.direct_ao_coefficients = coefficients;
  std::int32_t shells[4] = {0, 1, 2, same_ket ? 2 : 3};
  if (orientation & 1U) std::swap(shells[0], shells[1]);
  if (orientation & 2U) std::swap(shells[2], shells[3]);
  if (orientation & 4U) {
    std::swap(shells[0], shells[2]);
    std::swap(shells[1], shells[3]);
  }
  const std::int32_t first[2] = {shells[0], shells[2]};
  const std::int32_t second[2] = {shells[1], shells[3]};
  const std::int32_t systems[2] = {std::int32_t(system), std::int32_t(system)};
  batch.shell_pair_first = first;
  batch.shell_pair_second = second;
  batch.shell_pair_systems = systems;
  const std::int64_t primitive_offsets[3] = {0, injected_fault ? 2 : 1, injected_fault ? 3 : 2};
  batch.shell_pair_primitive_offsets = primitive_offsets;
  const double exponents[4] = {0.6, 0.8, 1.1, 0.9};
  PrimitivePairData primitives[3]{};
  for (unsigned pair = 0; pair < 2U; ++pair) {
    for (auto index = primitive_offsets[pair]; index < primitive_offsets[pair + 1U]; ++index) {
      auto& data = primitives[index];
      const double alpha = exponents[first[pair]] + 0.2 * (index - primitive_offsets[pair]);
      const double beta = exponents[second[pair]];
      data.exponent_sum = alpha + beta;
      data.reduced_exponent = alpha * beta / data.exponent_sum;
      data.first_product_scale = alpha / data.exponent_sum;
      data.second_product_scale = beta / data.exponent_sum;
      double squared_distance = 0.0;
      double product[3];
      for (unsigned axis = 0; axis < 3U; ++axis) {
        const auto left = positions[3 * atoms[first[pair]] + axis];
        const auto right = positions[3 * atoms[second[pair]] + axis];
        product[axis] = (alpha * left + beta * right) / data.exponent_sum;
        squared_distance += (left - right) * (left - right);
      }
      data.product_center = {product[0], product[1], product[2]};
      data.weighted_coefficient = std::exp(-data.reduced_exponent * squared_distance);
    }
  }
  batch.shell_primitive_pairs = primitives;
  batch.direct_force_pair_separable = enabled;
  batch.direct_force_relative_response = false;
  batch.direct_force_fp32 = mixed;
  batch.direct_force_seed = seed;
  DirectForceWork work{};
  batch.direct_force_work = &work;
  const std::uint8_t active[2] = {1U, 1U};
  evaluations = 0;
  fault = injected_fault;
  const auto call = [&]<bool Unrestricted, DirectForceOutputMode Mode>() {
    if (fault)
      execute<Unrestricted, Mode, FaultRoots>(batch, bounds, density, active, threshold, coulomb,
                                              exchange, forces);
    else
      execute<Unrestricted, Mode, Order5SourceRoots<kDpppShellClass>>(
          batch, bounds, density, active, threshold, coulomb, exchange, forces);
  };
  if (unrestricted) {
    if (separate)
      call.template operator()<true, DirectForceOutputMode::Separate>();
    else
      call.template operator()<true, DirectForceOutputMode::Combined>();
  } else {
    if (separate)
      call.template operator()<false, DirectForceOutputMode::Separate>();
    else
      call.template operator()<false, DirectForceOutputMode::Combined>();
  }
  const auto& pair = work.classes[kDpppShellClass].scalar_pair;
  const unsigned long long observed[] = {pair.attempted_shells,
                                         pair.accepted_shells,
                                         pair.mask_rejections,
                                         pair.numerical_rejections,
                                         pair.checked_ao_quartets,
                                         pair.bra_density_terms,
                                         pair.ket_density_terms,
                                         pair.primitive_products,
                                         work.classes[kDpppShellClass].scalar.calls,
                                         evaluations,
                                         pair.screening_pair_terms,
                                         pair.screening_pair_proofs,
                                         pair.screening_pair_fallbacks};
  std::copy(std::begin(observed), std::end(observed), counters);
}

/** Exercise the real mask proof with arbitrary pair domains, independently of
 * the DPPP force adapter, including same-shell and same-pair triangles. */
extern "C" bool scalar_pair_full_mask_probe(
    unsigned dimension, unsigned system, const std::int64_t* offsets,
    const std::int32_t* first, const std::int32_t* second, bool same_pair,
    const double* bounds, double threshold, unsigned long long* counters) {
  DeviceBatch batch{};
  batch.direct_nbf = dimension;
  batch.shell_direct_ao_offsets = offsets;
  batch.shell_pair_first = first;
  batch.shell_pair_second = second;
  const std::size_t second_pair = same_pair ? 0U : 1U;
  const auto layout = direct_shell_ao_quartet_layout(batch, 0U, second_pair);
  ScalarSeparableDirectForceWork work{};
  const bool complete = direct_force_complete_schwarz_mask(
      batch, 0U, second_pair, layout, static_cast<std::size_t>(system) * dimension,
      static_cast<std::size_t>(system) * dimension * dimension, threshold, bounds, &work);
  const unsigned long long observed[] = {work.checked_ao_quartets, work.screening_pair_terms,
                                         work.screening_pair_proofs, work.screening_pair_fallbacks};
  std::copy(std::begin(observed), std::end(observed), counters);
  return complete;
}
