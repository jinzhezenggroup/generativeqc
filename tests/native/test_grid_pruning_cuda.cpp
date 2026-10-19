/** Standalone pruned-grid device qualification; no SCF timing claim.
 * Execute only through a finite Slurm GPU allocation. */
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "dft/grid.hpp"
#include "runtime/resource_ledger.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void close(double actual, double expected, double absolute, double relative, const char* message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
    std::cerr << std::setprecision(17) << message << ": actual=" << actual
              << " expected=" << expected << " error=" << std::abs(actual - expected) << '\n';
    throw std::runtime_error(message);
  }
}

void qualify(const generativeqc::core::System& system, const generativeqc::dft::GridSpec& spec) {
  using namespace generativeqc;
  const dft::MolecularGrid reference(system, spec);
  const auto count = system.atoms.size() * dft::grid_points_per_atom(spec);
  double minimum_separation = std::numeric_limits<double>::infinity();
  for (std::size_t first = 0; first < system.atoms.size(); ++first)
    for (std::size_t second = 0; second < first; ++second) {
      const auto& a = system.atoms[first].position;
      const auto& b = system.atoms[second].position;
      const double separation = std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      if (separation > spec.coincident_tolerance)
        minimum_separation = std::min(minimum_separation, separation);
    }
  const auto capacity = dft::cuda_quadrature_bytes(system.atoms.size(), count);
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->limit = capacity;
  ledger->device = 0;
  runtime::active_device_resource_ledger = ledger;
  {
    const auto actual = dft::MolecularGrid::from_cuda(system, spec, 0);
    require(actual.point_count() == count && actual.owners() == reference.owners(),
            "pruned extent/owners differ");
    const auto raw = reference.atomic_weights();
    const auto view = actual.cuda_view();
    require(bool(view) && view.point_count == count, "missing resident pruned grid");
    require(ledger->peak == capacity && ledger->live == dft::cuda_resident_grid_bytes(count),
            "pruned allocation inventory differs");
    std::vector<double> resident_raw(count), resident_points(3 * count), resident_weights(count);
    require(cudaMemcpy(resident_raw.data(), view.atomic_weights, count * sizeof(double),
                       cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(resident_points.data(), view.points, 3 * count * sizeof(double),
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(resident_weights.data(), view.weights, count * sizeof(double),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
            "resident pruned grid download failed");
    for (std::size_t coordinate = 0; coordinate < actual.points().size(); ++coordinate) {
      close(actual.points()[coordinate], reference.points()[coordinate], 2e-11, 5e-14,
            "pruned coordinates differ");
      close(resident_points[coordinate], actual.points()[coordinate], 0, 0,
            "resident coordinate differs from export");
    }
    std::vector<double> sensitivity(count);
    for (std::size_t point = 0; point < count; ++point) {
      const double* xyz = actual.points().data() + 3 * point;
      double maximum_distance = 0.0;
      for (const auto& atom : system.atoms)
        maximum_distance = std::max(maximum_distance,
                                    std::hypot(xyz[0] - atom.position[0], xyz[1] - atom.position[1],
                                               xyz[2] - atom.position[2]));
      // At order 512 the rational radial tail reaches ~1e5 Bohr. Subtracting
      // independent host/device norms loses absolute digits in Becke mu even
      // for v1/v2, so allow its distance/separation-conditioned roundoff only.
      const double partition_roundoff =
          16 * std::numeric_limits<double>::epsilon() * maximum_distance / minimum_separation;
      close(resident_raw[point], raw[point], 1e-12, 5e-14, "pruned atomic measure differs");
      close(actual.weights()[point] / resident_raw[point], reference.weights()[point] / raw[point],
            3e-12 + partition_roundoff, 5e-11, "pruned Becke partition differs");
      close(resident_weights[point], actual.weights()[point], 0, 0,
            "resident weight differs from export");
      sensitivity[point] = std::exp(-0.2 * (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]));
    }
    const auto actual_response = actual.contract_weight_derivative(sensitivity);
    const auto reference_response = reference.contract_weight_derivative(sensitivity);
    for (std::size_t coordinate = 0; coordinate < actual_response.size(); ++coordinate)
      close(actual_response[coordinate], reference_response[coordinate], 3e-11, 3e-11,
            "pruned grid weight response differs");
  }
  require(ledger->live == 0, "pruned resident grid leaked allocations");
  runtime::active_device_resource_ledger.reset();
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
  try {
    generativeqc::core::System system;
    system.atoms = {{1, {0.1, -0.2, 0.3}}, {8, {1.7, 0.2, -0.4}}};
    generativeqc::dft::GridSpec spec;
    spec.element_radii[1] = 0.8;
    spec.element_radii[8] = 1.4;
    const std::array<std::array<std::size_t, 3>, 5> shapes{
        {{1, 1, 3}, {5, 7, 23}, {48, 16, 32}, {8, 128, 512}, {512, 7, 23}}};
    for (unsigned version : {1U, 2U, 3U}) {
      spec.version = version;
      for (const auto& shape : shapes) {
        std::cout << "version=" << version << " shape=" << shape[0] << ',' << shape[1] << ','
                  << shape[2] << std::endl;
        spec.radial_points = shape[0];
        spec.angular_polar = shape[1];
        spec.angular_azimuth = shape[2];
        qualify(system, spec);
      }
    }
    spec.version = 3;
    spec.radial_points = 24;
    spec.angular_polar = 8;
    spec.angular_azimuth = 16;
    for (double shift : {0.001, 7.125}) {
      for (auto& atom : system.atoms)
        for (double& coordinate : atom.position) coordinate += shift;
      qualify(system, spec);
    }
    system.atoms[1].position = system.atoms[0].position;
    qualify(system, spec);
    std::cout << "Pruned CUDA coordinates, measures, response and allocation lifetime passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
