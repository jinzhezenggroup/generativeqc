#include "integrals/ecp.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "generated_ecp_ao.cuh"
#include "integrals/ecp_gaussian.hpp"
#include "integrals/ecp_grid_cpu.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"

namespace generativeqc::integrals {
namespace {
constexpr int projector_count = generated::ecp_projector_count;
}  // namespace

void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere) {
  generated::ecp_make_grid(radial, angular, radii, sphere);
}

EcpData ecp_integrals(const core::System& system, unsigned radial, unsigned angular,
                      bool derivatives) {
  if (radial < 16 || radial > generated::ecp_cpu_max_radial_points || angular < 8 ||
      angular > generated::ecp_cpu_max_polar_points)
    throw std::invalid_argument(
        "CPU ECP quadrature requires 16..1536 radial and 8..192 polar points");
  const auto n = molecule::ao_count(system);
  const auto size = runtime::size_mul(n, n);
  const auto ncoord = derivatives ? runtime::size_mul(3, system.atoms.size()) : 0;
  const auto derivative_elements = runtime::size_mul(ncoord, size);
  const auto aos = ecp_grid_cpu::orbitals(system);
  EcpData out{n, ncoord, std::vector<double>(size), std::vector<double>(size), {}, {}};
  out.local_derivative.resize(derivative_elements);
  out.nonlocal_derivative.resize(derivative_elements);
  if (system.ecp_terms.empty()) return out;
  std::vector<EcpRadialPoint> radii;
  std::vector<EcpSpherePoint> sphere;
  ecp_quadrature(radial, angular, radii, sphere);
  // A radial layer is still the lifetime boundary. Share its potentials and
  // projected AO jets across every pair, independently of radial grid length.
  ecp_grid_cpu::Scratch scratch{
      std::vector<ecp_grid_cpu::Jet>(runtime::size_mul(n, sphere.size())),
      std::vector<ecp_grid_cpu::Jet>(runtime::size_mul(n, projector_count))};
  runtime::sample_cpu_capacity(runtime::vector_capacities(
      out.local, out.nonlocal, out.local_derivative, out.nonlocal_derivative, aos, radii, sphere,
      scratch.values, scratch.projections));
  for (std::size_t center = 0; center < system.atoms.size(); ++center) {
    if (!system.atoms[center].ecp_core) continue;
    for (const auto& radial_point : radii) {
      ecp_grid_cpu::layer(
          system, aos, radial_point, sphere, center, derivatives, scratch,
          [&](std::size_t a, std::size_t b, const double (&parts)[2][10]) {
            for (unsigned part = 0; part < 2; ++part) {
              const auto& v = parts[part];
              auto& matrix = part == 0 ? out.local : out.nonlocal;
              auto& gradient = part == 0 ? out.local_derivative : out.nonlocal_derivative;
              for (unsigned transpose = 0; transpose < (a == b ? 1U : 2U); ++transpose) {
                const auto item = transpose ? b * n + a : a * n + b;
                matrix[item] += v[0];
                if (derivatives)
                  for (unsigned d = 0; d < 3; ++d) {
                    gradient[(aos[a].shell->atom_index * 3 + d) * size + item] += v[d + 1];
                    gradient[(aos[b].shell->atom_index * 3 + d) * size + item] += v[d + 4];
                    gradient[(center * 3 + d) * size + item] += v[d + 7];
                  }
              }
            }
          });
    }
  }
  return out;
}

EcpData checked_ecp_integrals(const core::System& system, bool derivatives) {
  if (gaussian_ecp_supported(system)) return gaussian_ecp_integrals(system, derivatives);
  const auto coarse = ecp_integrals(system, generated::ecp_coarse_radial_points,
                                    generated::ecp_coarse_polar_points, derivatives);
  runtime::CpuRetainedCapacity retained(runtime::vector_capacities(
      coarse.local, coarse.nonlocal, coarse.local_derivative, coarse.nonlocal_derivative));
  auto fine = ecp_integrals(system, generated::ecp_refined_radial_points,
                            generated::ecp_refined_polar_points, derivatives);
  const auto check = [](const auto& a, const auto& b, bool derivative) {
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!generated::ecp_grid_pair_accepted(a[i], b[i], derivative))
        throw std::runtime_error(
            "ECP quadrature convergence gate failed; this input is outside the validated execution "
            "domain");
  };
  check(coarse.local, fine.local, false);
  check(coarse.nonlocal, fine.nonlocal, false);
  check(coarse.local_derivative, fine.local_derivative, true);
  check(coarse.nonlocal_derivative, fine.nonlocal_derivative, true);
  return fine;
}

std::size_t ecp_workspace_capacity(const core::System& system) {
  if (system.ecp_terms.empty()) return 0;
  const auto ao_count = molecule::ao_count(system);
  const auto polar = generated::ecp_refined_polar_points;
  const auto angular = runtime::size_mul(2 * polar, polar);
  const auto tile = generated::ecp_cuda_radial_tile(ao_count, polar);
  const auto coordinates = runtime::size_add(1, runtime::size_mul(3, system.atoms.size()));
  // Both checked grids can coexist with public value/coordinate output panels.
  auto bytes = runtime::size_mul(
      8 * sizeof(double), runtime::size_mul(runtime::size_mul(ao_count, ao_count), coordinates));
  // Four-component values/projections include simultaneous host/device staging.
  bytes = runtime::size_add(
      bytes,
      runtime::size_mul(
          64, runtime::size_mul(runtime::size_mul(tile, ao_count),
                                runtime::size_add(angular, generated::ecp_projector_count))));
  // Grid/metadata bounds include upload copies and geometric vector growth.
  bytes = runtime::size_add(bytes, runtime::size_mul(4 * sizeof(EcpSpherePoint), angular));
  bytes = runtime::size_add(
      bytes, runtime::size_mul(4 * sizeof(EcpRadialPoint), generated::ecp_refined_radial_points));
  bytes = runtime::size_add(bytes, runtime::size_mul(512, ao_count));
  bytes = runtime::size_add(bytes,
                            runtime::size_mul(4 * sizeof(core::EcpTerm), system.ecp_terms.size()));
  for (const auto& shell : system.shells)
    bytes = runtime::size_add(bytes, runtime::size_mul(64, shell.primitives.size()));
  return gaussian_ecp_supported(system)
             ? std::max(bytes, gaussian_ecp_workspace_capacity(system, true))
             : bytes;
}

void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative) {
  if (hcore.size() != ecp.local.size() || derivative.size() != ecp.local_derivative.size())
    throw std::invalid_argument("ECP and one-electron matrix layouts differ");
  for (std::size_t i = 0; i < hcore.size(); ++i)
    hcore[i] = generated::ecp_add_operator(hcore[i], ecp.local[i], ecp.nonlocal[i]);
  for (std::size_t i = 0; i < derivative.size(); ++i)
    derivative[i] = generated::ecp_add_operator(derivative[i], ecp.local_derivative[i],
                                                ecp.nonlocal_derivative[i]);
}
}  // namespace generativeqc::integrals
