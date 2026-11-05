/** Host-only execution of the actual scalar task and its pre-publication retry.
 * The test provides a serial CUDA shim; this is never a production CPU route
 * and cannot establish device resources, races, or endpoint performance. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "scf/cuda/direct_force_order4_sources.cuh"
#include "scf/cuda/direct_force_order5_sources.cuh"

using namespace generativeqc::scf::cuda_execution;
namespace generated_weighted_eri = generativeqc::scf::generated_weighted_eri;
static_assert(!DeviceBatch{}.direct_force_pair_separable);

namespace {
unsigned evaluations{}, fault{};

template <class Roots>
struct FaultRoots : Roots {
  static generated_weighted_eri::IndependentGradient evaluate_separable(
      const generated_weighted_eri::Geometry& geometry, const double* bra, const double* ket) {
    auto result = Roots::evaluate_separable(geometry, bra, ket);
    if (++evaluations == 2U) {
      if (fault == 1U) result.center[0][0] = std::numeric_limits<double>::infinity();
      if (fault == 2U) result.center[0][0] = std::numeric_limits<double>::max() / 8.0;
    }
    return result;
  }
};

template <unsigned ShellClass, bool Unrestricted, DirectForceOutputMode Mode, class Roots>
void execute(const DeviceBatch& batch, const double* bounds, const double* density,
             const std::uint8_t* active, double threshold, double coulomb, double exchange,
             double* forces, bool same_pair) {
  contract_two_electron_force_low_order_sources_task<Unrestricted, ShellClass, false, Roots, Mode>(
      batch, {0U, same_pair ? 0U : 1U, 0U}, threshold, bounds, density, active, forces, coulomb,
      exchange);
}

template <unsigned ShellClass, class Roots>
void run_probe(const std::array<std::uint8_t, 4>& angular, bool unrestricted, bool separate,
               bool enabled, bool relative, unsigned orientation, bool same_ket, bool same_pair,
               unsigned system, const std::int32_t* atoms, const double* positions,
               const double* density, const double* coefficients, const double* bounds,
               const double* seed, double threshold, double coulomb, double exchange, bool mixed,
               unsigned injected_fault, double* forces, unsigned long long* counters) {
  DeviceBatch batch{};
  batch.batch_size = system + 1;
  unsigned dimension = 0U;
  std::int64_t offsets[5]{};
  for (unsigned shell = 0; shell < 4U; ++shell) {
    dimension += (angular[shell] + 1U) * (angular[shell] + 2U) / 2U;
    offsets[shell + 1U] = dimension;
  }
  batch.nbf = batch.direct_nbf = dimension;
  batch.total_atoms = batch.total_shells = 4;
  batch.total_shell_pairs = 2;
  batch.positions = positions;
  batch.shell_atoms = atoms;
  batch.shell_angular = angular.data();
  for (auto& offset : offsets) offset += system * dimension;
  batch.shell_direct_ao_offsets = offsets;
  batch.direct_ao_coefficients = coefficients;
  std::int32_t shells[4] = {0, 1, 2, same_ket ? 2 : 3};
  if (same_pair) {
    shells[2] = 0;
    shells[3] = 1;
  }
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
  batch.direct_force_relative_response = relative;
  batch.direct_force_fp32 = mixed;
  batch.direct_force_seed = seed;
  DirectForceWork work{};
  batch.direct_force_work = &work;
  const std::uint8_t active[2] = {1U, 1U};
  evaluations = 0;
  fault = injected_fault;
  const auto call = [&]<bool Unrestricted, DirectForceOutputMode Mode>() {
    if (fault)
      execute<ShellClass, Unrestricted, Mode, FaultRoots<Roots>>(
          batch, bounds, density, active, threshold, coulomb, exchange, forces, same_pair);
    else
      execute<ShellClass, Unrestricted, Mode, Roots>(batch, bounds, density, active, threshold,
                                                     coulomb, exchange, forces, same_pair);
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
  const auto& pair = work.classes[ShellClass].scalar_pair;
  const unsigned long long observed[] = {pair.attempted_shells,
                                         pair.accepted_shells,
                                         pair.mask_rejections,
                                         pair.numerical_rejections,
                                         pair.checked_ao_quartets,
                                         pair.bra_density_terms,
                                         pair.ket_density_terms,
                                         pair.primitive_products,
                                         work.classes[ShellClass].scalar.calls,
                                         evaluations,
                                         pair.screening_pair_terms,
                                         pair.screening_pair_proofs,
                                         pair.screening_pair_fallbacks};
  std::copy(std::begin(observed), std::end(observed), counters);
}
}  // namespace

/** Exercise each actual roots binding without rewriting production source. */
extern "C" void scalar_pair_probe(unsigned class_index, bool unrestricted, bool separate,
                                  bool enabled, bool relative, unsigned orientation, bool same_ket,
                                  bool same_pair, unsigned system, const std::int32_t* atoms,
                                  const double* positions, const double* density,
                                  const double* coefficients, const double* bounds,
                                  const double* seed, double threshold, double coulomb,
                                  double exchange, bool mixed, unsigned injected_fault,
                                  double* forces, unsigned long long* counters) {
  constexpr std::array<std::uint8_t, 4> angular[] = {{1, 1, 1, 1}, {2, 1, 1, 1}, {2, 0, 1, 1},
                                                     {2, 0, 2, 0}, {2, 1, 1, 0}, {2, 1, 2, 0},
                                                     {2, 2, 1, 0}};
  const auto call = [&]<unsigned ShellClass, class Roots>() {
    run_probe<ShellClass, Roots>(angular[class_index], unrestricted, separate, enabled, relative,
                                 orientation, same_ket, same_pair, system, atoms, positions,
                                 density, coefficients, bounds, seed, threshold, coulomb, exchange,
                                 mixed, injected_fault, forces, counters);
  };
  switch (class_index) {
    case 0:
      call.template operator()<kPpppShellClass, Order4SourceRoots<kPpppShellClass>>();
      break;
    case 1:
      call.template operator()<kDpppShellClass, Order5SourceRoots<kDpppShellClass>>();
      break;
    case 2:
      call.template operator()<kDsppShellClass, Order4SourceRoots<kDsppShellClass>>();
      break;
    case 3:
      call.template operator()<kDsdsShellClass, Order4SourceRoots<kDsdsShellClass>>();
      break;
    case 4:
      call.template operator()<kDppsShellClass, Order4SourceRoots<kDppsShellClass>>();
      break;
    case 5:
      call.template operator()<kDpdsShellClass, Order5SourceRoots<kDpdsShellClass>>();
      break;
    case 6:
      call.template operator()<kDdpsShellClass, Order5SourceRoots<kDdpsShellClass>>();
      break;
    default:
      std::abort();
  }
}

/** Exercise the real mask proof with arbitrary pair domains, independently of
 * the DPPP force adapter, including same-shell and same-pair triangles. */
extern "C" bool scalar_pair_full_mask_probe(unsigned dimension, unsigned system,
                                            const std::int64_t* offsets, const std::int32_t* first,
                                            const std::int32_t* second, bool same_pair,
                                            const double* bounds, double threshold,
                                            unsigned long long* counters) {
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
