#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/host_grid_column_geometry.hpp"
#include "dft/host_grid_mixed_geometry.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

core::System make_system(const double* centers, const int* numbers, std::size_t atoms,
                         unsigned angular, bool cartesian) {
  core::System result;
  result.basis_representation = cartesian ? core::kCartesianBasis : core::kSphericalBasis;
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    result.atoms.push_back(
        {numbers[atom], {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]}});
    for (unsigned degree = 0; degree <= angular; ++degree)
      result.shells.push_back({static_cast<std::uint32_t>(atom),
                               degree,
                               {{.7 + .13 * degree, .7}, {.22 + .04 * degree, .3}}});
  }
  unsigned electrons{};
  for (const auto number : std::span(numbers, atoms)) electrons += number;
  result.multiplicity = electrons % 2 ? 2 : 1;
  std::string detail;
  if (molecule::validate_and_normalize(result, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
  return result;
}

dft::GridSpec prescription(unsigned version, unsigned iterations) {
  auto result = version == 4 ? dft::default_grid_spec() : dft::GridSpec{};
  result.version = version;
  result.partition_iterations = iterations;
  if (version != 4) {
    result.radial_points = 5;
    result.angular_polar = 4;
    result.angular_azimuth = 7;
    for (std::size_t element = 1; element < result.element_radii.size(); ++element)
      result.element_radii[element] = .8 + .02 * element;
  }
  return result;
}

struct Handle {
  core::System molecule;
  dft::AoBasis basis;
  dft::MolecularGrid grid;
  std::vector<double> atomic_weights;
  std::unique_ptr<dft::HostGridGeometry> geometry;
  std::unique_ptr<dft::HostGridMixedGeometry> mixed;
  std::unique_ptr<dft::HostGridColumnGeometry> column;
  std::size_t mode{}, fail_points{}, validations{};
  double* active_column_direction{};

  Handle(const double* centers, const int* numbers, std::size_t atoms, unsigned angular,
         unsigned version, unsigned iterations, std::size_t tile, bool cartesian)
      : molecule(make_system(centers, numbers, atoms, angular, cartesian)),
        basis(molecule),
        grid(molecule, prescription(version, iterations)),
        atomic_weights(grid.atomic_weights()) {
    const dft::host_hessian_aot::Program* leaf{};
    for (const auto& entry : dft::host_hessian_aot::programs())
      if (entry.name == "ao_first_motion") leaf = &entry;
    if (!leaf) throw std::invalid_argument("probe lacks the canonical motion leaf");
    geometry = std::make_unique<dft::HostGridGeometry>(
        basis, grid, tile, dft::HostGridGeometry::plan(basis, grid, tile), leaf->identity,
        [this] { validate(); });
    bind(nullptr);
    bind_column(nullptr);
  }

  void validate() {
    ++validations;
    if (mode == 1 || (mode == 2 && mixed && mixed->work().point_visits >= fail_points))
      throw std::runtime_error("injected stale mixed geometry owner");
    if (mode == 3 && mixed) {
      std::vector<double> direction(3 * basis.natom), ao(4 * basis.nao), moving(12 * basis.nao),
          weight(1), delta(3);
      mixed->read(0, 1, direction, direction, ao, moving, weight, delta);
    }
    if (mode == 4 || (mode == 5 && column && column->work().point_visits >= fail_points))
      throw std::runtime_error("injected stale column geometry owner");
    if (mode == 6 && column) {
      std::vector<double> direction(3 * basis.natom), ao(20 * basis.nao),
          weights(2 + 6 * basis.natom);
      column->read(0, 1, direction, ao, weights);
    }
    if (mode == 7 && active_column_direction && column &&
        column->work().point_visits >= fail_points) {
      active_column_direction[0] += .125;
      mode = 0;
    }
  }

  void bind(const std::size_t* caps) {
    auto limits = dft::HostGridMixedGeometry::plan(*geometry);
    if (caps) limits = {caps[0], caps[1], caps[2], caps[3]};
    auto replacement = std::make_unique<dft::HostGridMixedGeometry>(*geometry, atomic_weights,
                                                                    limits, [this] { validate(); });
    mixed = std::move(replacement);
  }

  void bind_column(const std::size_t* caps) {
    auto limits = dft::HostGridColumnGeometry::plan(*geometry);
    if (caps) limits = {caps[0], caps[1], caps[2], caps[3], caps[4]};
    auto replacement = std::make_unique<dft::HostGridColumnGeometry>(
        *geometry, atomic_weights, limits, [this] { validate(); });
    column = std::move(replacement);
  }
};
}  // namespace

extern "C" const char* host_mixed_error() { return error.c_str(); }
extern "C" void* host_mixed_create(const double* centers, const int* numbers, std::size_t atoms,
                                   unsigned angular, unsigned version, unsigned iterations,
                                   std::size_t tile, bool cartesian) {
  try {
    return new Handle(centers, numbers, atoms, angular, version, iterations, tile, cartesian);
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_mixed_destroy(Handle* handle) { delete handle; }
extern "C" void* host_mixed_reader(Handle* handle) { return handle->mixed.get(); }
extern "C" void* host_grid_column_reader(Handle* handle) { return handle->column.get(); }
extern "C" void host_mixed_shape(Handle* handle, std::size_t* values) {
  values[0] = handle->basis.natom;
  values[1] = handle->basis.nao;
  values[2] = handle->grid.point_count();
  values[3] = handle->geometry->tile_points();
}
extern "C" void host_mixed_plan(Handle* handle, std::size_t* values) {
  const auto plan = dft::HostGridMixedGeometry::plan(*handle->geometry);
  values[0] = plan.numeric_bytes;
  values[1] = plan.point_visits;
  values[2] = plan.ao_point_visits;
  values[3] = plan.pair_visits;
}
extern "C" int host_mixed_bind(Handle* handle, const std::size_t* caps) {
  try {
    handle->bind(caps);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return 1;
  }
}
extern "C" void host_mixed_export(Handle* handle, double* points, double* weights, double* atomic,
                                  std::uint32_t* owners, double* radii) {
  std::copy(handle->grid.points().begin(), handle->grid.points().end(), points);
  std::copy(handle->grid.weights().begin(), handle->grid.weights().end(), weights);
  std::copy(handle->atomic_weights.begin(), handle->atomic_weights.end(), atomic);
  std::copy(handle->grid.owners().begin(), handle->grid.owners().end(), owners);
  for (std::size_t atom = 0; atom < handle->basis.natom; ++atom)
    radii[atom] = handle->grid.spec().element_radii[handle->molecule.atoms[atom].atomic_number];
}
extern "C" int host_mixed_read(Handle* handle, std::size_t begin, std::size_t count,
                               const double* left, std::size_t left_size, const double* right,
                               std::size_t right_size, double* ao, std::size_t ao_size,
                               double* moving, std::size_t moving_size, double* weight,
                               std::size_t weight_size, double* delta, std::size_t delta_size) {
  try {
    handle->mixed->read(begin, count, {left, left_size}, {right, right_size}, {ao, ao_size},
                        {moving, moving_size}, {weight, weight_size}, {delta, delta_size});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return 1;
  }
}
extern "C" void host_mixed_fault(Handle* handle, std::size_t mode, std::size_t points) {
  handle->mode = mode;
  handle->fail_points = points;
}
extern "C" void host_mixed_work(Handle* handle, std::size_t* values) {
  const auto work = handle->mixed->work();
  values[0] = work.tiles_started;
  values[1] = work.tiles_completed;
  values[2] = work.point_visits;
  values[3] = work.ao_point_visits;
  values[4] = work.pair_visits;
}
extern "C" double* host_mixed_borrowed(Handle* handle, unsigned kind) {
  if (kind == 0) return handle->basis.packed.data();
  if (kind == 1) return const_cast<double*>(handle->grid.points().data());
  if (kind == 2) return const_cast<double*>(handle->grid.weights().data());
  return handle->atomic_weights.data();
}

extern "C" void host_grid_column_plan(Handle* handle, std::size_t* values) {
  const auto plan = dft::HostGridColumnGeometry::plan(*handle->geometry);
  values[0] = plan.numeric_bytes;
  values[1] = plan.point_visits;
  values[2] = plan.ao_point_visits;
  values[3] = plan.pair_visits;
  values[4] = plan.tape_visits;
}
extern "C" int host_grid_column_bind(Handle* handle, const std::size_t* caps) {
  try {
    handle->bind_column(caps);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return 1;
  }
}
extern "C" int host_grid_column_read(Handle* handle, std::size_t begin, std::size_t count,
                                     const double* right, std::size_t right_size, double* jets,
                                     std::size_t jets_size, double* weights,
                                     std::size_t weights_size) {
  try {
    handle->active_column_direction = const_cast<double*>(right);
    handle->column->read(begin, count, {right, right_size}, {jets, jets_size},
                         {weights, weights_size});
    handle->active_column_direction = nullptr;
    return 0;
  } catch (const std::exception& failure) {
    handle->active_column_direction = nullptr;
    error = failure.what();
    return 1;
  }
}
extern "C" void host_grid_column_work(Handle* handle, std::size_t* values) {
  const auto work = handle->column->work();
  values[0] = work.tiles_started;
  values[1] = work.tiles_completed;
  values[2] = work.point_visits;
  values[3] = work.ao_point_visits;
  values[4] = work.pair_visits;
  values[5] = work.forward_nodes;
  values[6] = work.reverse_nodes;
}

#if defined(GENERATIVEQC_MIXED_GRID_STANDALONE)
int main() {
  const std::array<double, 6> centers{0, 0, 0, 1.5, .3, -.2};
  const std::array<int, 2> numbers{1, 8};
  Handle handle(centers.data(), numbers.data(), 2, 1, 3, 3, 7, false);
  const std::array<double, 6> left{.1, -.2, .03, -.02, .11, -.08};
  const std::array<double, 6> right{-.04, .01, .05, .12, -.03, .1};
  std::vector<double> ao(4 * 7 * handle.basis.nao), moving(3 * ao.size()), weight(7), delta(21);
  handle.mixed->read(5, 7, left, right, ao, moving, weight, delta);
  if (handle.mixed->work().tiles_completed != 1 ||
      !std::all_of(delta.begin(), delta.end(), [](double value) { return std::isfinite(value); }))
    return 1;
  const std::array<double, 6> translation{.1, .2, .3, .1, .2, .3};
  handle.mixed->read(5, 7, translation, right, ao, moving, weight, delta);
  for (std::size_t point = 0; point < 7; ++point)
    if (std::abs(delta[point]) > 1e-14 || std::abs(delta[14 + point]) > 1e-14) return 2;
  std::vector<double> jets(20 * 7 * handle.basis.nao), column((2 + 12) * 7);
  handle.column->read(5, 7, right, jets, column);
  if (handle.column->work().tiles_completed != 1) return 3;
  handle.column->read(5, 7, translation, jets, column);
  for (std::size_t point = 0; point < 7; ++point) {
    if (column[7 + point] != 0) return 4;
    for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
      if (column[(8 + coordinate) * 7 + point] != 0) return 5;
  }
  return 0;
}
#endif
