#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/occupied_projection.hpp"

namespace generativeqc::scf::system_projection {

/** Unsupported model metadata is distinct from failure of an admitted transfer. */
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Controls {
  projection::Policy policy;
  bool unrestricted{};
  std::size_t maximum_host_bytes{};
};

/** Matrix-only capacity; prepared fleets and opaque provider storage are separate. */
struct Capacity {
  std::size_t source_ao{}, target_ao{}, spins{};
  std::array<std::size_t, 2> occupied{};
  std::size_t source_density_count{}, target_density_count{}, numeric_workspace_bytes{};
};

/** Validate metadata without evaluating any overlap or allocating matrix storage.
 * Exact ordered nuclei/geometry and electron/spin identity are required because
 * this operation proposes a new basis seed, not a geometry extrapolation.
 */
inline Capacity capacity(const core::System& target, const core::System& source,
                         const Controls& controls) {
  if (!controls.maximum_host_bytes)
    throw std::invalid_argument("system projection requires a positive host budget");
  if (target.atoms.size() != source.atoms.size() || target.charge != source.charge ||
      target.multiplicity != source.multiplicity || target.electron_count != source.electron_count)
    throw projection::Rejected("source/target ordered nuclei, electrons and spin must match");
  if (!target.ecp_terms.empty() || !source.ecp_terms.empty())
    throw Unsupported("system projection requires all-electron inputs");
  for (std::size_t index = 0; index < target.atoms.size(); ++index) {
    const auto& left = target.atoms[index];
    const auto& right = source.atoms[index];
    if (left.ecp_core || right.ecp_core)
      throw Unsupported("system projection requires all-electron nuclei");
    if (left.atomic_number != right.atomic_number || left.position != right.position)
      throw projection::Rejected("source/target ordered nuclei and geometry must match");
  }
  for (const auto* system : {&target, &source})
    for (const auto& shell : system->shells)
      if (shell.angular_momentum > 4)
        throw Unsupported("system projection supports through-g CPU topology");
  const auto [alpha, beta] = initial_guess::spin_occupations(source);
  if (!controls.unrestricted && alpha != beta)
    throw projection::Rejected("restricted projection requires equal spin occupations");
  const auto ns = molecule::ao_count(source), nt = molecule::ao_count(target);
  const std::size_t spins = controls.unrestricted ? 2 : 1;
  const projection::Input input{
      ns, nt, std::max(alpha, beta), controls.unrestricted ? 1U : 2U, {}, {}, {}, {}};
  const auto inner = projection::capacity(input, controls.policy);
  const auto ss = runtime::size_mul(ns, ns), tt = runtime::size_mul(nt, nt);
  const auto ts = runtime::size_mul(nt, ns);
  const auto source_count = runtime::size_mul(spins, ss);
  const auto target_count = runtime::size_mul(spins, tt);
  // Metrics remain live during projection. Retain both spin results until the
  // last channel passes, so a beta-channel refusal cannot publish alpha alone.
  const auto coefficients =
      runtime::size_mul(nt, controls.unrestricted ? runtime::size_add(alpha, beta) : alpha);
  const auto matrices = runtime::size_add(runtime::size_add(ss, tt), ts);
  const auto detached = runtime::size_add(target_count, coefficients);
  const auto extra = runtime::size_mul(sizeof(double), runtime::size_add(matrices, detached));
  return {ns,
          nt,
          spins,
          {alpha, beta},
          source_count,
          target_count,
          runtime::size_add(inner.numeric_workspace_bytes, extra)};
}

/** Own metrics and every spin projection in one synchronous native scope.
 * Mathematical kernels, thresholds and diagnostics are the existing owners'.
 * All input arrays remain borrowed and immutable; output is detached on success.
 */
inline std::array<projection::Result, 2> evaluate(const core::System& target,
                                                  const core::System& source,
                                                  const Controls& controls,
                                                  std::span<const double> density) {
  const auto required = capacity(target, source, controls);
  if (density.size() != required.source_density_count ||
      !std::all_of(density.begin(), density.end(), [](double x) { return std::isfinite(x); }))
    throw projection::Rejected("system projection requires finite spin-major source density");
  if (required.numeric_workspace_bytes > controls.maximum_host_bytes)
    throw projection::BudgetExceeded{};
  const auto ns = required.source_ao, nt = required.target_ao;
  std::vector<double> ss(ns * ns), tt(nt * nt), ts(nt * ns);
  integrals::cross_overlap(source, source, ss);
  integrals::cross_overlap(target, target, tt);
  integrals::cross_overlap(target, source, ts);
  std::array<projection::Result, 2> result;
  for (std::size_t spin = 0; spin < required.spins; ++spin) {
    const projection::Input input{ns,
                                  nt,
                                  required.occupied[spin],
                                  controls.unrestricted ? 1U : 2U,
                                  ss,
                                  tt,
                                  ts,
                                  density.subspan(spin * ns * ns, ns * ns)};
    result[spin] = projection::project_density(input, controls.policy);
  }
  return result;
}

}  // namespace generativeqc::scf::system_projection
