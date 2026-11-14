#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/host_xc_nuclear.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

core::System system(const double* centers, unsigned maximum) {
  core::System result;
  result.basis_representation = core::kSphericalBasis;
  for (std::size_t atom = 0; atom < 2; ++atom) {
    result.atoms.push_back({1, {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]}});
    for (unsigned angular = 0; angular <= maximum; ++angular)
      result.shells.push_back({static_cast<std::uint32_t>(atom),
                               angular,
                               {{.7 + .13 * angular, .7}, {.22 + .04 * angular, .3}}});
  }
  std::string detail;
  if (molecule::validate_and_normalize(result, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
  return result;
}

dft::GridSpec prescription(unsigned version) {
  auto result = version == 4 ? dft::default_grid_spec() : dft::GridSpec{};
  result.version = version;
  if (version != 4) {
    result.radial_points = 4;
    result.angular_polar = 3;
    result.angular_azimuth = 5;
    result.element_radii[1] = .9;
  }
  return result;
}

struct Handle {
  core::System molecule;
  dft::AoBasis basis;
  dft::MolecularGrid grid;
  std::span<const double> density;
  std::size_t spins, tile, validations{}, fail_at{}, mode{};
  std::vector<double> replacement;
  std::unique_ptr<dft::HostGridGeometry> geometry;
  std::unique_ptr<dft::HostGgaNuclearXc> xc;

  Handle(const double* centers, unsigned maximum, unsigned version, std::size_t spins,
         std::size_t tile, const double* density)
      : molecule(system(centers, maximum)),
        basis(molecule),
        grid(molecule, prescription(version)),
        density(density, spins * basis.nao * basis.nao),
        spins(spins),
        tile(tile) {}

  void validate() {
    ++validations;
    if (mode && validations >= fail_at) {
      if (mode == 2 && replacement.empty()) {
        replacement = basis.packed;
        basis.packed.swap(replacement);
        return;
      }
      if (mode == 3) {
        std::vector<double> direction(6), output(density.size());
        xc->apply(direction, output);
      }
      throw std::runtime_error("injected stale nuclear owner");
    }
  }

  void bind(const char* motion, const char* method, const std::size_t* geometry_caps,
            const std::size_t* xc_caps) {
    auto limits = dft::HostGridGeometry::plan(basis, grid, tile);
    if (geometry_caps)
      limits = {geometry_caps[0], geometry_caps[1], geometry_caps[2], geometry_caps[3]};
    auto reader = std::make_unique<dft::HostGridGeometry>(basis, grid, tile, limits, motion,
                                                          [this] { validate(); });
    auto resources = dft::HostGgaNuclearXc::plan(*reader, spins);
    if (xc_caps) resources = {xc_caps[0], xc_caps[1], xc_caps[2], xc_caps[3]};
    auto consumer = std::make_unique<dft::HostGgaNuclearXc>(*reader, spins, density, method,
                                                            resources, [this] { validate(); });
    xc.reset();
    geometry = std::move(reader);
    xc = std::move(consumer);
  }
};
}  // namespace

extern "C" const char* host_nuclear_error() { return error.c_str(); }
extern "C" void* host_nuclear_create(const double* centers, unsigned maximum, unsigned version,
                                     std::size_t spins, std::size_t tile, const double* density,
                                     const char* motion, const char* method) {
  try {
    auto result = std::make_unique<Handle>(centers, maximum, version, spins, tile, density);
    result->bind(motion, method, nullptr, nullptr);
    return result.release();
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_nuclear_destroy(Handle* handle) { delete handle; }
extern "C" int host_nuclear_bind(Handle* handle, const char* motion, const char* method,
                                 const std::size_t* geometry_caps, const std::size_t* xc_caps) {
  try {
    handle->bind(motion, method, geometry_caps, xc_caps);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_nuclear_sizes(Handle* handle, std::size_t* output) {
  const auto first = dft::HostGridGeometry::plan(handle->basis, handle->grid, handle->tile);
  const auto second = handle->xc->resources();
  const std::size_t result[]{
      handle->basis.nao,      handle->grid.point_count(), first.numeric_bytes,  first.point_visits,
      first.ao_point_visits,  first.grid_pair_visits,     second.numeric_bytes, second.map_terms,
      second.ao_point_visits, second.grid_pair_visits};
  std::copy(std::begin(result), std::end(result), output);
}
extern "C" void host_nuclear_grid(Handle* handle, double* points, double* weights,
                                  std::uint32_t* owners) {
  std::copy(handle->grid.points().begin(), handle->grid.points().end(), points);
  std::copy(handle->grid.weights().begin(), handle->grid.weights().end(), weights);
  std::copy(handle->grid.owners().begin(), handle->grid.owners().end(), owners);
}
extern "C" double* host_nuclear_borrowed(Handle* handle, unsigned kind) {
  return kind == 0   ? handle->basis.packed.data()
         : kind == 1 ? const_cast<double*>(handle->grid.points().data())
                     : const_cast<double*>(handle->grid.weights().data());
}
extern "C" void host_nuclear_failure(Handle* handle, std::size_t mode, std::size_t at) {
  handle->mode = mode;
  handle->fail_at = at;
  handle->validations = 0;
  if (!handle->replacement.empty()) {
    handle->basis.packed.swap(handle->replacement);
    handle->replacement.clear();
  }
}
extern "C" int host_nuclear_read(Handle* handle, std::size_t begin, std::size_t count,
                                 const double* direction, std::size_t direction_count, double* ao,
                                 double* moving_ao, double* weights, double* moving_weights) {
  try {
    handle->geometry->read(
        begin, count, {direction, direction_count}, {ao, 4 * count * handle->basis.nao},
        {moving_ao, 4 * count * handle->basis.nao}, {weights, count}, {moving_weights, count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_nuclear_apply(Handle* handle, const double* direction,
                                  std::size_t direction_count, double* output,
                                  std::size_t output_count) {
  try {
    handle->xc->apply({direction, direction_count}, {output, output_count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_nuclear_work(Handle* handle, std::size_t* output) {
  const auto first = handle->geometry->work();
  const auto second = handle->xc->work();
  const std::size_t values[]{
      first.tiles_started,         first.tiles_completed,     first.point_visits,
      first.ao_point_visits,       first.grid_pair_visits,    second.directions_started,
      second.directions_completed, second.tiles_completed,    second.point_visits,
      second.map_evaluations,      second.scheduled_map_terms};
  std::copy(std::begin(values), std::end(values), output);
}
