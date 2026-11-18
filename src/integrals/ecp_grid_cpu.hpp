#pragma once

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "generated_ecp_ao.cuh"
#include "integrals/ecp.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::integrals::ecp_grid_cpu {

struct Component {
  unsigned x, y, z;
  double coefficient;
};

struct AO {
  const core::Shell* shell;
  int primitive_offset{}, primitive_count{}, term_count{};
  double x, y, z;
  Component components[3];
};

struct Jet {
  double v[4];
};

struct SecondJet {
  double v[10];
};

/** Common normalized public-AO view for the raw and directional CPU consumers. */
inline std::vector<AO> orbitals(const core::System& system) {
  std::vector<AO> result;
  for (const auto& shell : system.shells) {
    if (shell.primitives.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("ECP primitive count exceeds the generated index type");
    const auto& position = system.atoms[shell.atom_index].position;
    const auto expansions =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation);
    if (std::any_of(expansions.begin(), expansions.end(),
                    [](const auto& expansion) { return expansion.size() > 3; }))
      throw std::invalid_argument("ECP AO expansion exceeds validated domain");
    for (const auto& expansion : expansions) {
      AO ao{&shell,
            0,
            static_cast<int>(shell.primitives.size()),
            static_cast<int>(expansion.size()),
            position[0],
            position[1],
            position[2],
            {}};
      for (std::size_t term = 0; term < expansion.size(); ++term) {
        const auto& source = expansion[term];
        ao.components[term] = {
            source.component[0], source.component[1], source.component[2],
            generated::ecp_component_coefficient(source.component[0], source.component[1],
                                                 source.component[2], source.coefficient)};
      }
      result.push_back(ao);
    }
  }
  return result;
}

struct Scratch {
  std::vector<Jet> values, projections;
};

struct SecondScratch {
  std::vector<SecondJet> values, projections;
};

/** One shared radial source pass; AO jets/projections are reused by every AO pair. */
template <bool Second = false, class Consumer>
void layer(const core::System& system, std::span<const AO> aos, const EcpRadialPoint& radial,
           std::span<const EcpSpherePoint> sphere, std::size_t center, bool derivatives,
           std::conditional_t<Second, SecondScratch, Scratch>& scratch, Consumer&& consume,
           EcpDirectionWork* work = nullptr) {
  constexpr auto projectors = generated::ecp_projector_count;
  if (work) {
    work->radial_layers = runtime::size_add(work->radial_layers, 1);
    work->contraction_terms = runtime::size_add(work->contraction_terms, system.ecp_terms.size());
  }
  double potentials[generated::ecp_max_projector_angular + 2];
  generated::ecp_potentials(system.ecp_terms.data(), system.ecp_terms.size(), radial, center,
                            potentials);
  if (std::all_of(std::begin(potentials), std::end(potentials),
                  [](double value) { return value == 0; }))
    return;
  const auto count = aos.size(), points = sphere.size();
  if (scratch.values.size() < runtime::size_mul(count, points) ||
      scratch.projections.size() < runtime::size_mul(count, projectors))
    throw std::logic_error("ECP radial source scratch is too small");
  const auto& position = system.atoms[center].position;
  for (std::size_t orbital = 0; orbital < count; ++orbital) {
    const auto& ao = aos[orbital];
    for (std::size_t point = 0; point < points; ++point) {
      if constexpr (Second)
        scratch.values[orbital * points + point] = generated::ecp_evaluate_second_ao<SecondJet>(
            ao, ao.shell->primitives.data(), sphere[point], radial.r, position[0], position[1],
            position[2], true);
      else
        scratch.values[orbital * points + point] = generated::ecp_evaluate_ao<Jet>(
            ao, ao.shell->primitives.data(), sphere[point], radial.r, position[0], position[1],
            position[2], derivatives);
    }
    for (int projector = 0; projector < projectors; ++projector)
      scratch.projections[orbital * projectors + projector] =
          generated::ecp_project(scratch.values.data() + orbital * points, sphere.data(), points,
                                 projector, derivatives, Second ? 10 : 4);
    if (work)
      work->ao_primitive_points = runtime::size_add(
          work->ao_primitive_points,
          runtime::size_mul(points, runtime::size_mul(ao.primitive_count, ao.term_count)));
  }
  for (std::size_t first = 0; first < count; ++first)
    for (std::size_t second = 0; second <= first; ++second) {
      double parts[2][Second ? 43 : 10];
      generated::ecp_contract_potentials(
          potentials, sphere.data(), points, scratch.values.data() + first * points,
          scratch.values.data() + second * points, scratch.projections.data() + first * projectors,
          scratch.projections.data() + second * projectors, derivatives, parts);
      consume(first, second, parts);
    }
  if (work) {
    const auto pairs =
        runtime::size_mul(count % 2 ? count : count / 2, count % 2 ? (count + 1) / 2 : count + 1);
    work->contraction_terms = runtime::size_add(
        work->contraction_terms,
        runtime::size_add(runtime::size_mul(runtime::size_mul(count, points), projectors),
                          runtime::size_mul(pairs, runtime::size_add(points, projectors))));
  }
}

}  // namespace generativeqc::integrals::ecp_grid_cpu
