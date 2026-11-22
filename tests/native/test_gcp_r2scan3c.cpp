// MB16-43/06 coordinates: grimme-lab/mstore a9070de0..., Apache-2.0.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#ifdef GENERATIVEQC_GCP_CUDA_TEST
#include "dft/dispersion/gcp_cuda.hpp"
#endif

#include "dft/dispersion/gcp_r2scan3c.hpp"

using namespace generativeqc::dft::dispersion;

namespace {
constexpr std::array<std::int32_t, 16> kZ{
    5, 7, 1, 8, 5, 1, 13, 1, 5, 12, 1, 1, 1, 1, 6, 1,
};
constexpr std::array<double, 48> kXyz{
    0.10912945825730,  1.64180252123600,  0.27838149792131,  -2.30085163837888, 0.87765138232225,
    -0.60457694150897, 2.78083551168063,  4.95421363506113,  0.40788634984219,  -5.36229602768251,
    -7.29510945515334, 0.06097106408867,  2.13846114572058,  -0.99012126457352, 0.93647189687052,
    0.09330150731888,  -2.75648066796634, -3.70294675694565, -1.52684105316140, -2.44981814860506,
    -1.02727325811774, -0.45240334635443, 5.86105501765814,  0.30815308772432,  -3.95419048213910,
    -5.52061943693205, -0.31702321028260, 2.68706169520082,  -0.13577304635533, -3.57041492458512,
    -3.79914135008731, 2.06429808651079,  -0.77285245656187, 0.89693752015341,  4.58640300917890,
    3.09718012019731,  2.76317093138142,  -0.62928000132252, 3.08807601371151,  1.00075543259914,
    -3.11885279872042, 1.08659460804098,  0.86969979951508,  4.43363816376984,  1.02355776570620,
    4.05637089597643,  -1.52300699610852, -0.29218485610105,
};

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

GCPStatus evaluate(const std::array<double, 48>& xyz, double* energy, double* gradient) {
#ifdef GENERATIVEQC_GCP_CUDA_TEST
  static GcpCudaPlan plan(0, kZ, GcpCudaPlan::numeric_bytes(kZ.size()));
  plan.execute(xyz, *energy, gradient ? std::span<double>(gradient, 48) : std::span<double>{});
  return GCPStatus::success;
#else
  return evaluate_r2scan3c_gcp(16, kZ.data(), xyz.data(), r2scan3c_gcp_parameters(), energy,
                               gradient);
#endif
}

double energy_of(const std::array<double, 48>& xyz) {
  double energy = 0.0;
  const auto status = evaluate(xyz, &energy, nullptr);
  return status == GCPStatus::success ? energy : NAN;
}
}  // namespace

int main() {
#ifdef GENERATIVEQC_GCP_CUDA_TEST
  if (!std::getenv("SLURM_JOB_ID")) return 77;
#endif
  double energy = -1.0;
  std::array<double, 48> gradient{};
  const auto status = evaluate(kXyz, &energy, gradient.data());
  if (status != GCPStatus::success) return 1;
  if (!near(energy, 0.0113040952, 5.0e-8) || energy_of(kXyz) != energy) return 2;

  for (int axis = 0; axis < 3; ++axis) {
    double total = 0.0;
    for (int atom = 0; atom < 16; ++atom) {
      total += gradient[3 * atom + axis];
    }
    if (std::abs(total) > 2.0e-14) return 3;
  }

  double max_fd_error = 0.0;
  for (double h : {1.0e-4, 3.0e-5}) {
    for (int k = 0; k < 48; ++k) {
      auto plus = kXyz;
      auto minus = kXyz;
      plus[k] += h;
      minus[k] -= h;
      const double fd = (energy_of(plus) - energy_of(minus)) / (2.0 * h);
      max_fd_error = std::max(max_fd_error, std::abs(fd - gradient[k]));
    }
  }
  std::printf("r2SCAN-3c gCP MB16-43/06: E=%.12f max_fd=%.3e\n", energy, max_fd_error);
  if (max_fd_error > 2.0e-8) return 4;

  std::array<std::int32_t, 1> argon{18};
  std::array<double, 3> one_xyz{0.0, 0.0, 0.0};
  std::array<double, 3> one_grad{9.0, 9.0, 9.0};
  energy = 7.0;
  if (evaluate_r2scan3c_gcp(1, argon.data(), one_xyz.data(), r2scan3c_gcp_parameters(), &energy,
                            one_grad.data()) != GCPStatus::success ||
      energy != 0.0 || one_grad != std::array<double, 3>{0.0, 0.0, 0.0}) {
    return 5;
  }

  std::array<std::int32_t, 1> potassium{19};
  energy = 7.0;
  if (evaluate_r2scan3c_gcp(1, potassium.data(), one_xyz.data(), r2scan3c_gcp_parameters(), &energy,
                            one_grad.data()) != GCPStatus::unsupported_element) {
    return 6;
  }

  energy = 7.0;
  if (evaluate_r2scan3c_gcp(16, kZ.data(), kXyz.data(), r2scan3c_gcp_parameters(), &energy,
                            const_cast<double*>(kXyz.data())) != GCPStatus::invalid_argument) {
    return 7;
  }
#ifdef GENERATIVEQC_GCP_CUDA_TEST
  double reference_energy = 0;
  std::array<double, 48> reference_gradient{};
  if (evaluate_r2scan3c_gcp(16, kZ.data(), kXyz.data(), r2scan3c_gcp_parameters(),
                            &reference_energy, reference_gradient.data()) != GCPStatus::success)
    return 8;
  for (std::size_t k = 0; k < gradient.size(); ++k)
    if (!near(gradient[k], reference_gradient[k], 2e-12)) return 9;
  GcpCudaPlan owner(0, kZ, GcpCudaPlan::numeric_bytes(kZ.size()));
  auto coincident = kXyz;
  std::copy_n(coincident.begin(), 3, coincident.begin() + 3);
  energy = 7;
  try {
    owner.execute(coincident, energy, gradient);
    return 10;
  } catch (const std::exception&) {
  }
  if (energy != 7) return 11;
  owner.execute(kXyz, energy, gradient);
  if (!near(energy, reference_energy, 2e-14)) return 12;
  try {
    GcpCudaPlan bounded(0, kZ, GcpCudaPlan::numeric_bytes(kZ.size()) - 1);
    return 13;
  } catch (const std::bad_alloc&) {
  }

  const auto bytes = GcpCudaPlan::numeric_bytes(kZ.size(), true);
  try {
    GcpCudaPlan bounded(0, kZ, bytes - 1, true);
    return 14;
  } catch (const std::bad_alloc&) {
  }
  std::array<double, 48> direction{}, action{};
  for (std::size_t coordinate = 0; coordinate < direction.size(); ++coordinate)
    direction[coordinate] = 0.25 * std::sin(0.9 * coordinate + 0.4);
  action.fill(19);
  try {
    owner.apply_hessian(kXyz, direction, action);
    return 15;
  } catch (const std::invalid_argument&) {
  }
  GcpCudaPlan hessian(0, kZ, bytes, true);
  try {
    hessian.apply_hessian(coincident, direction, action);
    return 16;
  } catch (const std::exception&) {
  }
  if (!std::all_of(action.begin(), action.end(), [](double value) { return value == 19; }))
    return 17;
  hessian.apply_hessian(kXyz, direction, action);

  // Independent simple-dftd3-style Python reference: Richardson-extrapolated
  // directional slopes at displacements 2e-4 and 1e-4 along this direction.
  constexpr double reference_curvature = 1.41315809626e-4;
  double curvature = 0;
  for (std::size_t coordinate = 0; coordinate < action.size(); ++coordinate)
    curvature += action[coordinate] * direction[coordinate];
  if (!near(curvature, reference_curvature, 2e-10)) return 18;

  // Mixed central differences of the independently evaluated host energies.
  // Use two step sizes so truncation/roundoff cannot hide a wrong sign or term.
  const auto host_energy = [](const auto& xyz) {
    double value{};
    const auto status = evaluate_r2scan3c_gcp(16, kZ.data(), xyz.data(), r2scan3c_gcp_parameters(),
                                              &value, nullptr);
    return status == GCPStatus::success ? value : NAN;
  };
  double curvature_error = 0;
  for (const double step : {1.0e-3, 5.0e-4}) {
    for (std::size_t coordinate = 0; coordinate < action.size(); ++coordinate) {
      double expected = 0;
      for (const double motion_sign : {-1.0, 1.0}) {
        for (const double axis_sign : {-1.0, 1.0}) {
          auto displaced = kXyz;
          for (std::size_t index = 0; index < displaced.size(); ++index)
            displaced[index] += step * motion_sign * direction[index];
          displaced[coordinate] += step * axis_sign;
          expected += axis_sign * motion_sign * host_energy(displaced);
        }
      }
      expected /= 4 * step * step;
      if (!near(action[coordinate], expected, 3e-8)) return 20;
      curvature_error = std::max(curvature_error, std::abs(action[coordinate] - expected));
    }
  }
  std::printf("r2SCAN-3c gCP CUDA HVP: max energy-stencil error %.3e\n", curvature_error);
  direction.fill(0.125);
  hessian.apply_hessian(kXyz, direction, action);
  if (!std::all_of(action.begin(), action.end(), [](double value) { return value == 0; }))
    return 19;
#endif
  return 0;
}
