#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda/df_md_j.hpp"

namespace generativeqc::scf::cuda_execution {

/** CPU topology/metadata preparation for a single normalized basis. No integral
 * or Hermite coefficient is evaluated here. Angular groups and disjoint E/H
 * intervals are fixed before upload. The separate CUDA owner must authenticate
 * geometry/device/stream identity and retain this storage's device copies.
 * Host peak includes the returned vectors and temporary AO expansion metadata;
 * device bytes describe every array required by this layout, not a whole SCF. */
struct DfMdLayout {
  std::vector<DfMdAo> aos;
  std::vector<DfMdPrimitive> primitives;
  std::array<std::size_t, 8> offsets{};
  std::size_t transform_elements{}, hermite_elements{}, device_bytes{}, host_peak_bytes{};
  bool auxiliary{};
};

namespace df_md_layout_detail {
inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("DF MD metadata addition overflows size_t");
  return a + b;
}
inline std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("DF MD metadata multiplication overflows size_t");
  return a * b;
}
inline unsigned hermites(unsigned angular) {
  return (angular + 1) * (angular + 2) * (angular + 3) / 6;
}
}  // namespace df_md_layout_detail

/** Preflight the entire unscreened optional storage before any large vector
 * allocation. `length_error` reports overflow, launch-domain or budget refusal;
 * `invalid_argument` reports malformed/unsupported metadata. Either is an
 * explicit reason for a future provider to retain its existing bounded path.
 * No default admission is implied. Input coefficients must already carry the
 * canonical radial normalization; this function never normalizes them again. */
namespace df_md_layout_detail {
inline DfMdLayout layout(const core::System& system, bool auxiliary, std::size_t device_limit,
                         std::size_t host_limit, bool materialize) {
  using namespace df_md_layout_detail;
  if (system.shells.empty() || system.atoms.empty() ||
      (system.basis_representation != GENERATIVEQC_BASIS_CARTESIAN &&
       system.basis_representation != GENERATIVEQC_BASIS_SPHERICAL))
    throw std::invalid_argument("DF MD layout requires a nonempty Cartesian/spherical basis");
  // Even one primitive record per shell must fit; reject enormous metadata
  // domains before walking the quadratic unordered-shell-pair census.
  if (multiply(system.shells.size(), sizeof(DfMdPrimitive)) > std::min(device_limit, host_limit))
    throw std::length_error("DF MD shell metadata exceeds the optional budget");
  std::size_t aos = 0, primitive_count = 0;
  DfMdLayout result;
  result.auxiliary = auxiliary;
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > (auxiliary ? 4U : 3U) || shell.primitives.empty() ||
        shell.atom_index >= system.atoms.size())
      throw std::invalid_argument("DF MD supports orbital f / auxiliary g with valid primitives");
    for (double coordinate : system.atoms[shell.atom_index].position)
      if (!std::isfinite(coordinate)) throw std::invalid_argument("DF MD center is nonfinite");
    for (const auto& primitive : shell.primitives)
      if (!(primitive.exponent > 0) || !std::isfinite(primitive.exponent) ||
          !std::isfinite(primitive.coefficient))
        throw std::invalid_argument("DF MD primitive is invalid");
    aos = add(aos, molecule::shell_ao_count(shell.angular_momentum, system.basis_representation));
  }
  for (std::size_t i = 0; i < system.shells.size(); ++i) {
    const auto& a = system.shells[i];
    for (std::size_t j = 0; j < (auxiliary ? 1 : i + 1); ++j) {
      const auto& b = auxiliary ? a : system.shells[j];
      const auto angular = (auxiliary ? 0 : a.angular_momentum) + b.angular_momentum;
      const auto na =
          auxiliary ? 1 : molecule::shell_ao_count(a.angular_momentum, system.basis_representation);
      const auto nb = molecule::shell_ao_count(b.angular_momentum, system.basis_representation);
      const auto count = multiply(auxiliary ? 1 : a.primitives.size(), b.primitives.size());
      primitive_count = add(primitive_count, count);
      result.offsets[angular + 1] = add(result.offsets[angular + 1], count);
      const auto states = multiply(count, hermites(angular));
      result.hermite_elements = add(result.hermite_elements, states);
      result.transform_elements =
          add(result.transform_elements, multiply(states, multiply(na, nb)));
    }
  }
  if (primitive_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF MD primitive domain exceeds a one-dimensional CUDA grid");
  const auto metadata = add(multiply(primitive_count, sizeof(DfMdPrimitive)),
                            multiply(add(aos, auxiliary ? 1 : 0), sizeof(DfMdAo)));
  result.device_bytes =
      add(metadata, multiply(add(result.transform_elements, multiply(2, result.hermite_elements)),
                             sizeof(double)));
  // Public expansion temporaries are bounded by fifteen Cartesian g AOs with
  // six terms each. Include shell offsets, returned object and vector slack
  // below after actual allocation; the preflight deliberately reserves twice
  // the payload capacity without relying on a particular allocator policy.
  const auto payload =
      add(multiply(primitive_count, sizeof(DfMdPrimitive)), multiply(aos, sizeof(DfMdAo)));
  const auto temporary =
      add(multiply(add(system.shells.size(), 1), sizeof(std::size_t)),
          15 * (sizeof(molecule::AoExpansion) + 6 * sizeof(molecule::CartesianExpansionTerm)));
  result.host_peak_bytes = add(sizeof(result), add(multiply(2, payload), multiply(2, temporary)));
  if (result.device_bytes > device_limit || result.host_peak_bytes > host_limit)
    throw std::length_error("DF MD complete layout exceeds the optional budget");

  // Admission queries share this exact census and allocate no topology-sized
  // vectors. Complete layouts continue below only after both caps are proven.
  if (!materialize) return result;

  result.aos.reserve(aos);
  result.primitives.reserve(primitive_count);
  std::vector<std::size_t> shell_offsets;
  shell_offsets.reserve(add(system.shells.size(), 1));
  for (const auto& shell : system.shells) {
    shell_offsets.push_back(result.aos.size());
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      DfMdAo ao;
      if (expansion.empty() || expansion.size() > 6)
        throw std::invalid_argument("DF MD public AO expansion exceeds six terms");
      for (const auto& term : expansion) {
        const auto c = term.component;
        ao.angular[ao.count] = {c[0], c[1], c[2]};
        ao.coefficients[ao.count++] =
            term.coefficient * molecule::cartesian_component_normalization(c);
      }
      result.aos.push_back(ao);
    }
  }
  shell_offsets.push_back(result.aos.size());
  for (std::size_t i = 0; i < system.shells.size(); ++i) {
    const auto& a = system.shells[i];
    const auto ra = system.atoms[a.atom_index].position;
    for (std::size_t j = 0; j < (auxiliary ? 1 : i + 1); ++j) {
      const auto& b = auxiliary ? a : system.shells[j];
      const auto rb = system.atoms[b.atom_index].position;
      for (std::size_t p = 0; p < (auxiliary ? 1 : a.primitives.size()); ++p)
        for (const auto& q : b.primitives) {
          DfMdPrimitive primitive;
          primitive.first_ao = auxiliary ? 0 : shell_offsets[i];
          primitive.second_ao = shell_offsets[auxiliary ? i : j];
          primitive.first_count = auxiliary ? 1 : shell_offsets[i + 1] - primitive.first_ao;
          primitive.second_count = shell_offsets[(auxiliary ? i : j) + 1] - primitive.second_ao;
          primitive.first_angular = auxiliary ? 0 : a.angular_momentum;
          primitive.second_angular = b.angular_momentum;
          primitive.same_shell = !auxiliary && i == j;
          primitive.alpha = auxiliary ? 0 : a.primitives[p].exponent;
          primitive.beta = q.exponent;
          primitive.weight = (auxiliary ? 1 : a.primitives[p].coefficient) * q.coefficient;
          if (!std::isfinite(primitive.weight) || !std::isfinite(primitive.alpha + primitive.beta))
            throw std::invalid_argument("DF MD primitive product is unrepresentable");
          primitive.first = {ra[0], ra[1], ra[2]};
          primitive.second = {rb[0], rb[1], rb[2]};
          result.primitives.push_back(primitive);
        }
    }
  }
  // In-place sort avoids a second primitive-sized temporary. Tie order is
  // deterministic and preserves shell/primitive construction order via an
  // explicit ordinal, temporarily stored in the not-yet-assigned E offset.
  for (std::size_t i = 0; i < result.primitives.size(); ++i)
    result.primitives[i].transform_offset = i;
  std::sort(result.primitives.begin(), result.primitives.end(), [](const auto& a, const auto& b) {
    const auto la = a.first_angular + a.second_angular, lb = b.first_angular + b.second_angular;
    return la != lb ? la < lb : a.transform_offset < b.transform_offset;
  });
  for (std::size_t i = 1; i < result.offsets.size(); ++i)
    result.offsets[i] += result.offsets[i - 1];
  std::size_t transforms = 0, states = 0;
  for (auto& primitive : result.primitives) {
    primitive.transform_offset = transforms;
    primitive.hermite_offset = states;
    const auto count = hermites(primitive.first_angular + primitive.second_angular);
    transforms += primitive.first_count * primitive.second_count * count;
    states += count;
  }
  if (transforms != result.transform_elements || states != result.hermite_elements ||
      result.aos.size() != aos || result.primitives.size() != primitive_count)
    throw std::logic_error("DF MD layout census and generated intervals disagree");
  const auto actual_payload = add(multiply(result.aos.capacity(), sizeof(DfMdAo)),
                                  multiply(result.primitives.capacity(), sizeof(DfMdPrimitive)));
  if (actual_payload > multiply(2, payload) ||
      shell_offsets.capacity() > multiply(2, add(system.shells.size(), 1)))
    throw std::length_error("DF MD host allocator capacity exceeds the reserved envelope");
  return result;
}

}  // namespace df_md_layout_detail

/** Exact optional device footprint with a bounded host-layout preflight.
 * Shares the materializer's census, but constructs no large host vectors and
 * evaluates no integrals. This lets an enclosing planner reserve the owner
 * before allocating its interchangeable raw fallback tiles. */
inline std::size_t df_md_layout_device_bytes(const core::System& system, bool auxiliary,
                                             std::size_t device_limit, std::size_t host_limit) {
  return df_md_layout_detail::layout(system, auxiliary, device_limit, host_limit, false)
      .device_bytes;
}

/** Materialize the checked native metadata and disjoint E/H intervals. */
inline DfMdLayout make_df_md_layout(const core::System& system, bool auxiliary,
                                    std::size_t device_limit, std::size_t host_limit) {
  return df_md_layout_detail::layout(system, auxiliary, device_limit, host_limit, true);
}

}  // namespace generativeqc::scf::cuda_execution
