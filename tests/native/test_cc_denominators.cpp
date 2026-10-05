// Independent ordered FP64 arithmetic, boundary admission and provenance gates.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cc/solver.hpp"

namespace {
using namespace generativeqc::cc;
void require(bool value, const char* why) {
  if (!value) throw std::runtime_error(why);
}

void boundary_tests() {
  SolverOptions options;
  auto accepted = [&](double occupied, double shift, bool derived) {
    Problem p;
    p.nocc = p.nvir = 1;
    options.level_shift = shift;
    try {
      initialize_canonical_denominators(p, std::array{occupied, 0.0}, options, derived);
      return true;
    } catch (const std::invalid_argument&) {
      return false;
    }
  };
  const double threshold = options.denominator_threshold;
  for (bool derived : {false, true}) {
    require(!accepted(-threshold, 0.0, derived), "threshold equality admitted");
    require(!accepted(std::nextafter(-threshold, 0.0), 0.0, derived), "below threshold admitted");
    require(accepted(std::nextafter(-threshold, -1.0), 0.0, derived), "above threshold rejected");
    require(!accepted(-threshold, 1.0, derived), "shift hid physical near-zero denominator");
    require(!accepted(0.0, 1.0, derived), "shift hid nonnegative physical denominator");
    require(!accepted(-1.0, -0.1, derived), "negative shift admitted");
    require(!accepted(-std::numeric_limits<double>::infinity(), 0.0, derived), "nonfinite input");
    require(!accepted(-0.75 * std::numeric_limits<double>::max(), 0.0, derived),
            "doubles overflow");
    require(!accepted(-1.0, 0.75 * std::numeric_limits<double>::max(), derived), "shift overflow");
  }
}

void arithmetic_tests() {
  std::uint64_t random = 1904;
  auto unit = [&] {
    random = random * 6364136223846793005ULL + 1442695040888963407ULL;
    return std::ldexp(static_cast<double>(random >> 11), -53);
  };
  bool saw_shifted_sum_difference = false;
  for (unsigned sample = 0; sample < 100; ++sample) {
    Problem derived, explicit_problem;
    derived.nocc = explicit_problem.nocc = 2;
    derived.nvir = explicit_problem.nvir = 3;
    std::array<double, 5> eps{-0.1 - 40.0 * unit(), -0.1 - 40.0 * unit(), 0.1 + unit(),
                              0.1 + unit(), 0.1 + unit()};
    SolverOptions options;
    options.level_shift = unit();
    initialize_canonical_denominators(derived, eps, options, true);
    initialize_canonical_denominators(explicit_problem, eps, options, false);
    require(derived.d2.empty() && derived.d2.capacity() == 0, "derived retained d2");
    require(explicit_problem.canonical_eps.capacity() == 0, "explicit retained spectrum");
    require(denominator_identity(derived) != denominator_identity(explicit_problem),
            "identity alias");
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        for (std::size_t a = 0; a < 3; ++a)
          for (std::size_t b = 0; b < 3; ++b) {
            const auto flat = ((i * 2 + j) * 3 + a) * 3 + b;
            // Volatile intermediates are an independent rounding oracle. This
            // intentionally protects pair-gap grouping, not only tolerance.
            volatile double first = eps[i] - eps[2 + a], second = eps[j] - eps[2 + b];
            volatile double physical = first + second, twice_shift = 2.0 * options.level_shift;
            const double expected = physical - twice_shift;
            require(std::bit_cast<std::uint64_t>(expected) ==
                        std::bit_cast<std::uint64_t>(doubles_denominator_at(derived, flat)),
                    "derived grouping differs from original FP64 arithmetic");
            require(std::bit_cast<std::uint64_t>(expected) ==
                        std::bit_cast<std::uint64_t>(explicit_problem.d2[flat]),
                    "explicit grouping differs from original FP64 arithmetic");
            const double shifted_sum = derived.d1[i * 3 + a] + derived.d1[j * 3 + b];
            saw_shifted_sum_difference |= shifted_sum != expected;
          }
    const auto identity = denominator_identity(derived);
    derived.canonical_eps[0] = std::nextafter(derived.canonical_eps[0], 0.0);
    require(denominator_identity(derived) != identity, "spectrum identity stale");
    derived.canonical_eps[0] = eps[0];
    derived.canonical_denominator_threshold *= 2;
    require(denominator_identity(derived) != identity, "threshold omitted from provenance");
  }
  require(saw_shifted_sum_difference, "rounding regression does not distinguish shifted singles");
}
void representation_tests() {
  Problem p;
  p.nocc = p.nvir = 1;
  for (auto* values : {&p.foo, &p.fov, &p.fvv, &p.ovov, &p.ovvo, &p.oovv, &p.ovvv, &p.ovoo, &p.oooo,
                       &p.vvvv, &p.initial_t1, &p.initial_t2})
    values->assign(1, 0.0);
  const std::array eps{-1.0, 1.0};
  initialize_canonical_denominators(p, eps, SolverOptions{}, true);
  validate_problem(p);
  const auto original = p;
  auto rejected = [&] {
    bool failed = false;
    try {
      validate_problem(p);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    require(failed, "mixed or stale canonical representation admitted");
    p = original;
  };
  p.d2.assign(1, -4.0);
  rejected();
  p.canonical_eps[0] = -2.0;
  rejected();
  p.d1[0] = 0.0;
  rejected();
  p.denominator_representation = DenominatorRepresentation::Explicit;
  rejected();
  p.denominator_representation = static_cast<DenominatorRepresentation>(99);
  rejected();
  initialize_canonical_denominators(p, eps, SolverOptions{}, false);
  // Arbitrary supplied denominators retain their original finite-only contract.
  p.d1[0] = 3.0;
  p.d2[0] = 17.0;
  validate_problem(p);
}
}  // namespace

int main() {
  try {
    boundary_tests();
    arithmetic_tests();
    representation_tests();
    std::cout << "canonical denominator arithmetic, admission and identity gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
