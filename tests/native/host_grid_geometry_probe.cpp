#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "dft/grid.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;
struct Handle {
  dft::MolecularGrid grid;
};
}  // namespace

extern "C" const char* host_grid_error() { return error.c_str(); }
extern "C" void* host_grid_create(unsigned version, unsigned iterations, std::size_t atoms,
                                  const int* numbers, const double* centers) {
  try {
    core::System system;
    for (std::size_t atom = 0; atom < atoms; ++atom)
      system.atoms.push_back(
          {numbers[atom], {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]}});
    auto spec = version == 4 ? dft::default_grid_spec() : dft::GridSpec{};
    spec.version = version;
    spec.partition_iterations = iterations;
    if (version != 4) {
      spec.radial_points = 5;
      spec.angular_polar = 4;
      spec.angular_azimuth = 7;
      for (const auto& atom : system.atoms)
        spec.element_radii[atom.atomic_number] = .8 + .02 * atom.atomic_number;
    }
    return new Handle{dft::MolecularGrid(system, spec)};
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_grid_destroy(Handle* handle) { delete handle; }
extern "C" std::size_t host_grid_size(Handle* handle) { return handle->grid.point_count(); }
extern "C" std::size_t host_grid_workspace(Handle* handle, std::size_t count) {
  try {
    return handle->grid.weight_direction_workspace_bytes(count);
  } catch (const std::exception& failure) {
    error = failure.what();
    return std::size_t(-1);
  }
}
extern "C" std::size_t host_grid_planned_pairs(Handle* handle, std::size_t count) {
  try {
    return handle->grid.weight_direction_work(count).pair_visits;
  } catch (const std::exception& failure) {
    error = failure.what();
    return std::size_t(-1);
  }
}
extern "C" void host_grid_export(Handle* handle, double* points, double* weights, double* raw,
                                 std::uint32_t* owners) {
  const auto& grid = handle->grid;
  std::copy(grid.points().begin(), grid.points().end(), points);
  std::copy(grid.weights().begin(), grid.weights().end(), weights);
  const auto atomic = grid.atomic_weights();
  std::copy(atomic.begin(), atomic.end(), raw);
  std::copy(grid.owners().begin(), grid.owners().end(), owners);
}
extern "C" double* host_grid_retained_weights(Handle* handle) {
  return const_cast<double*>(handle->grid.weights().data());
}
extern "C" int host_grid_direction(Handle* handle, std::size_t begin, std::size_t count,
                                   const double* direction, std::size_t direction_count,
                                   double* output, std::size_t output_count, double* workspace,
                                   std::size_t workspace_count, std::size_t* work) {
  try {
    const auto result =
        handle->grid.weight_direction(begin, count, {direction, direction_count},
                                      {output, output_count}, {workspace, workspace_count});
    work[0] = result.point_visits;
    work[1] = result.pair_visits;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_grid_adjoint(Handle* handle, const double* sensitivity, double* gradient) {
  try {
    const auto result =
        handle->grid.contract_weight_derivative({sensitivity, handle->grid.point_count()});
    std::copy(result.begin(), result.end(), gradient);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
