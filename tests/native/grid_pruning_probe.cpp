#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "dft/grid.hpp"

/** Standalone CPU quadrature and weight-adjoint export, without a KS library.
 * A null output queries the exact extent; invalid prescriptions return zero
 * rather than allowing a C++ exception to cross the test's ctypes boundary. */
extern "C" std::size_t grid_pruning_probe(unsigned version, std::size_t radial, std::size_t polar,
                                          std::size_t azimuth, const double* centers,
                                          double* points, double* weights, double* atomic_weights,
                                          std::uint32_t* owners, double* gradient) {
  try {
    generativeqc::core::System system;
    system.atoms = {{1, {centers[0], centers[1], centers[2]}},
                    {8, {centers[3], centers[4], centers[5]}}};
    generativeqc::dft::GridSpec spec;
    spec.version = version;
    spec.radial_points = radial;
    spec.angular_polar = polar;
    spec.angular_azimuth = azimuth;
    spec.element_radii[1] = 0.8;
    spec.element_radii[8] = 1.4;
    const generativeqc::dft::MolecularGrid grid(system, spec);
    if (points) {
      std::copy(grid.points().begin(), grid.points().end(), points);
      std::copy(grid.weights().begin(), grid.weights().end(), weights);
      std::copy(grid.owners().begin(), grid.owners().end(), owners);
      const auto raw = grid.atomic_weights();
      std::copy(raw.begin(), raw.end(), atomic_weights);
      std::vector<double> sensitivity(grid.point_count());
      for (std::size_t point = 0; point < grid.point_count(); ++point) {
        const double* xyz = points + 3 * point;
        sensitivity[point] = std::exp(-0.2 * (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]));
      }
      const auto response = grid.contract_weight_derivative(sensitivity);
      std::copy(response.begin(), response.end(), gradient);
    }
    return grid.point_count();
  } catch (...) {
    return 0;
  }
}
