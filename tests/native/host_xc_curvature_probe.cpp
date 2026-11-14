#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/host_xc_curvature.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

struct Handle {
  dft::HostGridMixedGeometry& geometry;
  std::span<const double> density;
  std::unique_ptr<dft::HostGgaNuclearCurvature> source;
  std::size_t mode{}, fail_points{};

  Handle(dft::HostGridMixedGeometry& geometry, std::size_t spins, const double* density,
         std::size_t size, const char* method, const std::size_t* caps)
      : geometry(geometry), density(density, size) {
    auto plan = dft::HostGgaNuclearCurvature::plan(geometry, spins);
    if (caps) plan = {caps[0], caps[1], caps[2], caps[3]};
    std::string_view identity = method ? method : "";
    if (!method)
      for (const auto& entry : dft::host_hessian_aot::method_bindings())
        if (entry.spins == spins) identity = entry.identity;
    source = std::make_unique<dft::HostGgaNuclearCurvature>(geometry, spins, this->density,
                                                            identity, plan, [this] { validate(); });
  }

  void validate() {
    if (mode == 1 || (mode == 2 && source && source->work().point_visits >= fail_points))
      throw std::runtime_error("injected stale frozen XC owner");
    if (mode == 3 && source) {
      std::vector<double> direction(3 * geometry.geometry().basis().natom);
      std::array<double, 4> output;
      source->apply(direction, direction, output);
    }
  }
};
}  // namespace

extern "C" const char* host_curvature_error() { return error.c_str(); }
extern "C" void* host_curvature_create(void* geometry, std::size_t spins, const double* density,
                                       std::size_t size, const char* method,
                                       const std::size_t* caps) {
  try {
    return new Handle(*static_cast<dft::HostGridMixedGeometry*>(geometry), spins, density, size,
                      method, caps);
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_curvature_destroy(Handle* handle) { delete handle; }
extern "C" int host_curvature_plan(void* geometry, std::size_t spins, std::size_t* values) {
  try {
    const auto plan = dft::HostGgaNuclearCurvature::plan(
        *static_cast<dft::HostGridMixedGeometry*>(geometry), spins);
    values[0] = plan.numeric_bytes;
    values[1] = plan.map_terms;
    values[2] = plan.ao_point_visits;
    values[3] = plan.grid_pair_visits;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return 1;
  }
}
extern "C" int host_curvature_apply(Handle* handle, const double* left, std::size_t left_size,
                                    const double* right, std::size_t right_size, double* output,
                                    std::size_t size) {
  try {
    handle->source->apply({left, left_size}, {right, right_size}, {output, size});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return 1;
  }
}
extern "C" void host_curvature_fault(Handle* handle, std::size_t mode, std::size_t points) {
  handle->mode = mode;
  handle->fail_points = points;
}
extern "C" void host_curvature_work(Handle* handle, std::size_t* values) {
  const auto work = handle->source->work();
  values[0] = work.pairs_started;
  values[1] = work.pairs_completed;
  values[2] = work.tiles_completed;
  values[3] = work.point_visits;
  values[4] = work.map_terms;
}

#if defined(GENERATIVEQC_CURVATURE_STANDALONE)
int main() {
  core::System system;
  system.atoms = {{1, {0, 0, 0}}, {1, {1.5, .3, -.2}}};
  system.shells = {{0, 0, {{.7, 1}}}, {1, 0, {{.7, 1}}}};
  system.basis_representation = core::kSphericalBasis;
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS) return 1;
  dft::AoBasis basis(system);
  dft::GridSpec spec;
  spec.radial_points = 4;
  spec.angular_polar = 3;
  spec.angular_azimuth = 5;
  dft::MolecularGrid grid(system, spec);
  auto atomic_weights = grid.atomic_weights();
  const dft::host_hessian_aot::Program* leaf{};
  for (const auto& entry : dft::host_hessian_aot::programs())
    if (entry.name == "ao_first_motion") leaf = &entry;
  if (!leaf) return 2;
  dft::HostGridGeometry geometry(basis, grid, 11, dft::HostGridGeometry::plan(basis, grid, 11),
                                 leaf->identity, [] {});
  dft::HostGridMixedGeometry mixed(geometry, atomic_weights,
                                   dft::HostGridMixedGeometry::plan(geometry), [] {});
  const std::array<double, 6> left{.1, -.2, .03, -.02, .11, -.08};
  const std::array<double, 6> right{-.04, .01, .05, .12, -.03, .1};
  const std::array<double, 6> translation{.1, .2, .3, .1, .2, .3};
  for (std::size_t spins = 1; spins <= 2; ++spins) {
    std::vector<double> density(4 * spins);
    for (std::size_t spin = 0; spin < spins; ++spin) {
      density[4 * spin] = .8;
      density[4 * spin + 3] = .7;
    }
    Handle owner(mixed, spins, density.data(), density.size(), nullptr, nullptr);
    std::array<double, 4> output;
    owner.source->apply(left, right, output);
    if (!std::all_of(output.begin(), output.end(),
                     [](double value) { return std::isfinite(value); }))
      return 3;
    owner.source->apply(translation, right, output);
    if (std::abs(output[1]) > 1e-13 || std::abs(output[3]) > 1e-13) return 4;
  }
  return 0;
}
#endif
