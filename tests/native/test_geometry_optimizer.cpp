#include "generativeqc/geometry_optimizer.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace geo = generativeqc::geometry;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

static void expect_throw(const std::function<void()>& fn, std::string needle) {
  try {
    fn();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(needle) != std::string::npos, "wrong exception text");
    return;
  }
  throw std::runtime_error("expected optimizer failure");
}

int main() {
  // Independent analytic energy/force oracle. The shifted reference model
  // confirms that stage boundaries do not inherit the previous stationary point.
  auto quadratic = [](double center, double stiffness) {
    return [=](std::span<const double> x) -> geo::EnergyForces {
      geo::EnergyForces r;
      r.forces.resize(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
        const double delta = x[i] - center;
        r.energy += 0.5 * stiffness * delta * delta;
        r.forces[i] = -stiffness * delta;
      }
      return r;
    };
  };
  geo::Options limits;
  limits.force_tolerance = 1e-7;
  std::vector<double> start{3, -2, 1};
  auto low = geo::minimize(start, limits, quadratic(0.5, 3));
  require(low.steps > 0 && low.max_force < limits.force_tolerance, "analytic optimizer regression");
  for (double x : low.coordinates) require(std::abs(x - 0.5) < 1e-6, "analytic optimizer regression");
  auto high = geo::minimize(low.coordinates, limits, quadratic(-0.7, 12));
  require(high.steps > 0 && high.max_force < limits.force_tolerance, "analytic optimizer regression");
  for (double x : high.coordinates) require(std::abs(x + 0.7) < 1e-6, "analytic optimizer regression");
  require(high.evaluations >= high.steps + 1, "analytic optimizer regression");

  std::size_t calls = 0;
  auto quartic = [&](std::span<const double> x) -> geo::EnergyForces {
    ++calls;
    double z = x[0];
    return {std::pow(z, 4), {-4 * z * z * z, 0, 0}};
  };
  limits.force_tolerance = 1e-6;
  auto q = geo::minimize(std::vector<double>{1, 0, 0}, limits, quartic);
  require(std::abs(q.coordinates[0]) < 0.01, "analytic optimizer regression");
  require(q.max_force <= limits.force_tolerance, "analytic optimizer regression");
  require(calls >= q.steps + 1, "analytic optimizer regression");

  expect_throw([&] { geo::minimize(std::vector<double>{1, 2}, limits, quadratic(0, 1)); },
               "atom-major");
  expect_throw([&] { geo::minimize(start, limits, [](std::span<const double>) {
                 return geo::EnergyForces{0, {1}};
               }); }, "invalid forces");
  expect_throw([&] { geo::minimize(start, limits, [](std::span<const double> x) {
                 return geo::EnergyForces{std::nan(""), std::vector<double>(x.size(), 0)};
               }); }, "nonfinite");
  geo::Options short_run = limits;
  short_run.max_steps = 1;
  short_run.max_atom_step = 1e-5;
  expect_throw([&] { geo::minimize(start, short_run, quadratic(0, 3)); }, "step budget");
  expect_throw([&] { geo::minimize(start, limits, [](std::span<const double> x) {
                 return geo::EnergyForces{0, std::vector<double>(x.size(), 1)};
               }); }, "line search failed");
  expect_throw([&] { geo::minimize(start, limits, [](std::span<const double>) -> geo::EnergyForces {
                 throw std::runtime_error("electronic method did not converge");
               }); }, "electronic method did not converge");
}
