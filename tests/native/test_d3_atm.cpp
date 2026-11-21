#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/dispersion/d3_response.hpp"

namespace {
using generativeqc::dft::dispersion::d3_atm_workspace_elements;
using generativeqc::dft::dispersion::d3_host_tables;
using generativeqc::dft::dispersion::D3ATMParameters;
using generativeqc::dft::dispersion::D3Status;
using generativeqc::dft::dispersion::evaluate_d3_bj_atm;

constexpr std::array<std::int32_t, 4> kNumbers{6, 8, 7, 1};
constexpr std::array<double, 12> kCoordinates{0.0, 0.0, 0.0, 2.5,  0.1, 0.0,
                                              0.6, 2.7, 0.2, -1.2, 0.8, 2.4};
// Independent oracle: dftd3 1.6.0 RationalDampingParam with
// {s6=1, s8=.7875, a1=.4289, a2=4.4407, alp=14}, evaluated as s9=1 minus s9=0.
constexpr double kOracleEnergy = 3.3280853885481534e-08;
constexpr std::array<double, 12> kOracleGradient{
    3.72472344498586336e-09,  -1.85479558309227088e-08, -3.61045307549533526e-08,
    2.60708265967875448e-08,  -2.63615669102038128e-08, 2.56660793687209677e-09,
    -3.96599503356355378e-09, 5.05725536818215585e-08,  -1.60861908260602298e-08,
    -2.58295550081963018e-08, -5.66303094069503694e-09, 4.96241136441330152e-08};

struct Result {
  D3Status status{};
  double energy{};
  std::array<double, 12> gradient{};
};

Result evaluate(const std::array<double, 12>& xyz, D3ATMParameters parameters, bool gradient) {
  std::vector<double> workspace(d3_atm_workspace_elements(kNumbers.size()));
  Result out{};
  out.status = evaluate_d3_bj_atm(kNumbers.size(), kNumbers.data(), xyz.data(), parameters,
                                  d3_host_tables(), workspace.data(), workspace.size(), &out.energy,
                                  gradient ? out.gradient.data() : nullptr);
  return out;
}

void require_close(double actual, double expected, double tolerance, const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << std::setprecision(17) << message << ": expected " << expected << ", got " << actual
              << '\n';
    throw std::runtime_error(message);
  }
}

void test_external_oracle() {
  const auto result = evaluate(kCoordinates, D3ATMParameters{1.0, 0.0, 0.0, 0.0}, true);
  if (result.status != D3Status::success) throw std::runtime_error("ATM oracle evaluation failed");
  require_close(result.energy, kOracleEnergy, 2.0e-18, "ATM oracle energy mismatch");
  for (std::size_t i = 0; i < result.gradient.size(); ++i)
    require_close(result.gradient[i], kOracleGradient[i], 2.0e-17, "ATM oracle gradient mismatch");
}

void test_finite_difference(D3ATMParameters parameters, double tolerance) {
  const auto analytic = evaluate(kCoordinates, parameters, true);
  if (analytic.status != D3Status::success)
    throw std::runtime_error("analytic ATM evaluation failed");
  constexpr double step = 1.0e-5;
  for (std::size_t q = 0; q < kCoordinates.size(); ++q) {
    auto plus = kCoordinates, minus = kCoordinates;
    plus[q] += step;
    minus[q] -= step;
    const auto ep = evaluate(plus, parameters, false);
    const auto em = evaluate(minus, parameters, false);
    if (ep.status != D3Status::success || em.status != D3Status::success)
      throw std::runtime_error("finite-difference ATM evaluation failed");
    const double numeric = (ep.energy - em.energy) / (2.0 * step);
    require_close(analytic.gradient[q], numeric, tolerance, "ATM finite-difference mismatch");
  }
}

void test_zero_scaling() {
  const auto result = evaluate(kCoordinates, D3ATMParameters{0.0, 0.0, 0.0, 0.0}, true);
  if (result.status != D3Status::success || result.energy != 0.0 ||
      !std::all_of(result.gradient.begin(), result.gradient.end(),
                   [](double x) { return x == 0.0; }))
    throw std::runtime_error("s9=0 must disable the standalone ATM contribution exactly");
}

/** Replay the same staged kernels on the host to exercise finite cutoffs
 * independently of the catalog's uncut entries and CUDA scheduling. */
void test_switched_hessian() {
  using namespace generativeqc::dft::dispersion;
  namespace response = generativeqc::dft::dispersion::d3_response;
  const auto atoms = kNumbers.size();
  const auto tables = d3_host_tables();
  std::array<double, kCoordinates.size()> direction{}, action{};
  for (std::size_t component = 0; component < direction.size(); ++component)
    direction[component] = std::sin(0.53 * component + 0.27);
  D3ModelParameters model{.bj = {1, 0.7875, 0.4289, 4.4407, 0, 3.2, 4.5, 2.0},
                          .zero = {1, 0.722, 1.217, 1, 14, 3.2, 4.5, 2.0},
                          .atm = {1, 3.2, 4.5, 2.0},
                          .atm_enabled = true};
  const auto require_success = [](D3Status status) {
    if (status != D3Status::success) throw std::runtime_error("D3 switched Hessian failed");
  };
  for (const auto damping : {D3Damping::bj, D3Damping::zero}) {
    model.damping = damping;
    if (damping == D3Damping::zero) {
      model.atm_enabled = false;
      model.atm.s9 = 0;
    }
    std::vector<double> scratch(d3_hessian_workspace_elements(atoms));
    const auto work = response::workspace(scratch.data(), atoms);
    for (std::size_t atom = 0; atom < atoms; ++atom)
      require_success(response::weights_for_atom(atoms, atom, kNumbers.data(), kCoordinates.data(),
                                                 direction.data(), 3.2, tables, work));
    for (std::size_t atom = 0; atom < atoms; ++atom)
      require_success(response::local_response(atoms, atom, kNumbers.data(), kCoordinates.data(),
                                               direction.data(), model, tables, work,
                                               action.data() + 3 * atom));
    for (std::size_t atom = 0; atom < atoms; ++atom)
      require_success(response::cn_pullback(atoms, atom, kNumbers.data(), kCoordinates.data(),
                                            direction.data(), 3.2, tables, work,
                                            action.data() + 3 * atom));
    const auto gradient = [&](double displacement) {
      auto geometry = kCoordinates;
      std::array<double, kCoordinates.size()> result{};
      for (std::size_t component = 0; component < geometry.size(); ++component)
        geometry[component] += displacement * direction[component];
      std::vector<double> primal(d3_workspace_elements(atoms));
      double energy{};
      require_success(evaluate_d3_model(atoms, kNumbers.data(), geometry.data(), model, tables,
                                        primal.data(), primal.size(), &energy, result.data()));
      return result;
    };
    const auto stencil = [&](double step) {
      const auto upper = gradient(step), lower = gradient(-step);
      std::array<double, kCoordinates.size()> result{};
      for (std::size_t component = 0; component < result.size(); ++component)
        result[component] = (upper[component] - lower[component]) / (2 * step);
      return result;
    };
    const auto coarse = stencil(1e-4), fine = stencil(5e-5);
    for (std::size_t component = 0; component < action.size(); ++component)
      require_close(action[component], (4 * fine[component] - coarse[component]) / 3, 1e-10,
                    "D3 differentiated cutoff or damping mismatch");
  }
  auto geometry = kCoordinates;
  geometry[4] = 0;
  std::vector<double> scratch(d3_hessian_workspace_elements(2));
  const auto work = response::workspace(scratch.data(), 2);
  if (response::weights_for_atom(2, 0, kNumbers.data(), geometry.data(), direction.data(), 2.5,
                                 tables, work) != D3Status::unsupported)
    throw std::runtime_error("hard CN boundary must not publish a derivative");
}

void test_d3_reference_weight_underflow() {
  using namespace generativeqc::dft::dispersion;
  namespace response = generativeqc::dft::dispersion::d3_response;
  const std::array<d3_data::ElementData, 1> elements{{{0, 2, 1, 1}}};
  const std::array<std::int32_t, 1> numbers{1};
  const std::array<double, 3> stationary{};
  std::vector<double> scratch(d3_hessian_workspace_elements(1));
  const auto work = response::workspace(scratch.data(), 1);
  for (const auto& centers : {std::array{13.4, 13.6}, std::array{14.0, 15.0}}) {
    // The first fixture has a nonzero subnormal norm; the second rounds
    // directly to zero. The primal uses its max reference in either case.
    const D3Tables tables{elements.data(), nullptr, centers.data(), nullptr};
    if (response::weights_for_atom(1, 0, numbers.data(), stationary.data(), stationary.data(), 0,
                                   tables, work) != D3Status::success ||
        work.weights[0] != 0 || work.weights[1] != 1 || work.first_cn[0] != 0 ||
        work.first_cn[1] != 0 || work.second_cn[0] != 0 || work.second_cn[1] != 0)
      throw std::runtime_error("D3 Hessian must preserve the primal reference-weight fallback");
  }
}

}  // namespace

int main() {
  try {
    test_d3_reference_weight_underflow();
    test_external_oracle();
    test_finite_difference(D3ATMParameters{1.0, 0.0, 0.0, 0.0}, 3.0e-12);
    // Exercise all three smooth-cutoff product-rule terms as well as CN response.
    test_finite_difference(D3ATMParameters{1.0, 0.0, 4.5, 2.0}, 3.0e-12);
    test_zero_scaling();
    test_switched_hessian();
    std::cout << "D3(BJ)-ATM oracle, analytic-gradient, smooth-cutoff, and s9=0 tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
