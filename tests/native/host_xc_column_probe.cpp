#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/host_xc_column.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

struct Handle {
  dft::HostGridColumnGeometry& geometry;
  std::span<const double> density;
  std::vector<double> expected_density;
  std::unique_ptr<dft::HostGgaNuclearColumn> source;
  std::size_t mode{}, fail_points{};
  double* active_direction{};
  double* active_density_direction{};

  Handle(dft::HostGridColumnGeometry& geometry, std::size_t spins, const double* density,
         std::size_t size, const char* method, const std::size_t* caps)
      : geometry(geometry), density(density, size) {
    const auto basis = geometry.geometry().basis().nao;
    if (!density || size != spins * basis * basis)
      throw std::invalid_argument("column probe density shape mismatch");
    expected_density.assign(density, density + size);
    auto plan = dft::HostGgaNuclearColumn::plan(geometry, spins);
    if (caps) plan = {caps[0], caps[1], caps[2], caps[3], caps[4], caps[5], caps[6]};
    std::string_view identity = method ? method : "";
    if (!method)
      for (const auto& entry : dft::host_hessian_aot::method_bindings())
        if (entry.spins == spins) identity = entry.identity;
    source = std::make_unique<dft::HostGgaNuclearColumn>(geometry, spins, this->density, identity,
                                                         plan, [this] { validate(); });
  }

  void validate() {
    if (mode == 1 || (mode == 2 && source && source->work().point_visits >= fail_points))
      throw std::runtime_error("injected stale XC column owner");
    if (mode == 3 && source) {
      std::vector<double> right(3 * geometry.geometry().basis().natom),
          output(2 + 2 * right.size());
      source->apply(right, {}, output);
    }
    if ((mode == 4 || mode == 5) && source && source->work().point_visits >= fail_points) {
      if (mode == 4)
        const_cast<double*>(density.data())[0] += .125;
      else if (active_direction)
        active_direction[0] += .125;
      mode = 0;
    }
    if (mode == 6 && active_density_direction && source &&
        source->work().point_visits >= fail_points) {
      active_density_direction[0] += .125;
      mode = 0;
    }
    if (!std::equal(expected_density.begin(), expected_density.end(), density.begin()))
      throw std::runtime_error("changed frozen XC column density");
  }
};
}  // namespace

extern "C" const char* host_xc_column_error() { return error.c_str(); }
extern "C" void* host_xc_column_create(void* geometry, std::size_t spins, const double* density,
                                       std::size_t size, const char* method,
                                       const std::size_t* caps) {
  try {
    return new Handle(*static_cast<generativeqc::dft::HostGridColumnGeometry*>(geometry), spins,
                      density, size, method, caps);
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_xc_column_destroy(Handle* handle) { delete handle; }
extern "C" int host_xc_column_plan(void* geometry, std::size_t spins, std::size_t* values) {
  try {
    const auto plan = generativeqc::dft::HostGgaNuclearColumn::plan(
        *static_cast<generativeqc::dft::HostGridColumnGeometry*>(geometry), spins);
    values[0] = plan.numeric_bytes;
    values[1] = plan.map_terms;
    values[2] = plan.scalar_point_visits;
    values[3] = plan.ao_point_visits;
    values[4] = plan.ao_motion_visits;
    values[5] = plan.grid_pair_visits;
    values[6] = plan.tape_visits;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return 1;
  }
}
extern "C" int host_xc_column_apply(Handle* handle, const double* right, std::size_t right_size,
                                    double* output, std::size_t output_size) {
  try {
    handle->active_direction = const_cast<double*>(right);
    handle->source->apply({right, right_size}, {}, {output, output_size});
    handle->active_direction = nullptr;
    return 0;
  } catch (const std::exception& failure) {
    handle->active_direction = nullptr;
    error = failure.what();
    return 1;
  }
}
extern "C" void host_xc_column_work(Handle* handle, std::size_t* values) {
  const auto work = handle->source->work();
  values[0] = work.columns_started;
  values[1] = work.columns_completed;
  values[2] = work.tiles_completed;
  values[3] = work.point_visits;
  values[4] = work.map_terms;
  values[5] = work.scalar_point_visits;
  values[6] = work.ao_motion_visits;
}
extern "C" int host_xc_column_response(Handle* handle, const double* right, std::size_t right_size,
                                       const double* density_right, std::size_t density_size,
                                       double* output, std::size_t output_size) {
  try {
    handle->active_direction = const_cast<double*>(right);
    handle->active_density_direction = const_cast<double*>(density_right);
    handle->source->apply({right, right_size}, {density_right, density_size},
                          {output, output_size});
    handle->active_direction = nullptr;
    handle->active_density_direction = nullptr;
    return 0;
  } catch (const std::exception& failure) {
    handle->active_direction = nullptr;
    handle->active_density_direction = nullptr;
    error = failure.what();
    return 1;
  }
}
extern "C" void host_xc_column_geometry_work(Handle* handle, std::size_t* values) {
  const auto work = handle->geometry.work();
  values[0] = work.tiles_started;
  values[1] = work.tiles_completed;
  values[2] = work.point_visits;
  values[3] = work.ao_point_visits;
  values[4] = work.pair_visits;
  values[5] = work.forward_nodes;
  values[6] = work.reverse_nodes;
}
extern "C" void host_xc_column_fault(Handle* handle, std::size_t mode, std::size_t points) {
  handle->mode = mode;
  handle->fail_points = points;
}

#if defined(GENERATIVEQC_XC_COLUMN_STANDALONE)
int main() {
  using namespace generativeqc;
  for (const auto representation : {core::kCartesianBasis, core::kSphericalBasis}) {
    core::System system;
    system.atoms = {{1, {0, 0, 0}}, {1, {1.5, .3, -.2}}};
    for (std::uint32_t atom = 0; atom < 2; ++atom)
      for (unsigned degree = 0; degree < 2; ++degree)
        system.shells.push_back({atom, degree, {{.7, 1}}});
    system.basis_representation = representation;
    std::string detail;
    if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS) return 1;
    dft::AoBasis basis(system);
    dft::GridSpec spec;
    spec.radial_points = 4;
    spec.angular_polar = 3;
    spec.angular_azimuth = 5;
    dft::MolecularGrid grid(system, spec);
    const auto atomic_weights = grid.atomic_weights();
    const dft::host_hessian_aot::Program* leaf{};
    for (const auto& entry : dft::host_hessian_aot::programs())
      if (entry.name == "ao_first_motion") leaf = &entry;
    if (!leaf) return 2;
    dft::HostGridGeometry ordinary(basis, grid, 11, dft::HostGridGeometry::plan(basis, grid, 11),
                                   leaf->identity, [] {});
    dft::HostGridColumnGeometry geometry(ordinary, atomic_weights,
                                         dft::HostGridColumnGeometry::plan(ordinary), [] {});
    const std::array<double, 6> right{-.04, .01, .05, .12, -.03, .1};
    const std::array<double, 6> translation{.1, .2, .3, .1, .2, .3};
    for (const auto& binding : dft::host_hessian_aot::method_bindings()) {
      std::vector<double> density(binding.spins * basis.nao * basis.nao);
      for (std::size_t spin = 0; spin < binding.spins; ++spin)
        for (std::size_t orbital = 0; orbital < basis.nao; ++orbital)
          density[(spin * basis.nao + orbital) * basis.nao + orbital] = .2 + .03 * (orbital + spin);
      dft::HostGgaNuclearColumn source(geometry, binding.spins, density, binding.identity,
                                       dft::HostGgaNuclearColumn::plan(geometry, binding.spins),
                                       [] {});
      std::array<double, 14> output;
      source.apply(right, {}, output);
      double curvature{}, projected{};
      for (std::size_t coordinate = 0; coordinate < right.size(); ++coordinate) {
        curvature += output[8 + coordinate] * output[8 + coordinate];
        projected += right[coordinate] * output[2 + coordinate];
      }
      if (!(curvature > 1e-12) || std::abs(projected - output[1]) > 3e-12) return 3;
      const auto frozen = output;
      std::vector<double> density_right(density.size());
      for (std::size_t spin = 0; spin < binding.spins; ++spin)
        for (std::size_t orbital = 0; orbital < basis.nao; ++orbital)
          density_right[(spin * basis.nao + orbital) * basis.nao + orbital] =
              .01 * (1 + spin + orbital);
      source.apply(right, density_right, output);
      double response_norm{};
      for (std::size_t coordinate = 0; coordinate < right.size(); ++coordinate)
        response_norm += std::abs(output[8 + coordinate] - frozen[8 + coordinate]);
      if (!(response_norm > 1e-8)) return 6;
      source.apply(translation, {}, output);
      if (output[1] != 0) return 4;
      for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
        if (output[8 + coordinate] != 0) return 5;
    }
  }
  std::cout << "XC columns: frozen/supplied-density, Cartesian/spherical RKS/UKS\n";
}
#endif
