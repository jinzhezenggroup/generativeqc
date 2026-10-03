#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "tensor/cpu_compensated_sum.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double sum(const std::vector<double>& terms) {
  generativeqc::tensor::CpuCompensatedSum result;
  for (double term : terms) result.add(term);
  return result.value();
}
}  // namespace

int main() {
  try {
    require(sum({}) == 0.0, "empty sum is not zero");
    require(sum({1.0e16, 1.0, -1.0e16}) == 1.0, "small cancellation term lost");
    require(sum({1.0, 1.0e16, -1.0e16}) == 1.0, "magnitude swap lost correction");
    require(sum({-1.0e16, -1.0, 1.0e16}) == -1.0, "negative correction lost");
    require(sum({-0.0, 0.0}) == 0.0, "signed-zero input is nonzero");
    const double tiny = std::numeric_limits<double>::denorm_min();
    require(sum({tiny, tiny, -tiny}) == tiny, "subnormal term lost");
    const double infinity = std::numeric_limits<double>::infinity();
    const double maximum = std::numeric_limits<double>::max();
    require(sum({infinity, 1.0}) == infinity, "positive infinity changed");
    require(sum({-infinity, -1.0}) == -infinity, "negative infinity changed");
    require(std::isnan(sum({infinity, -infinity})), "opposing infinities hidden");
    require(std::isnan(sum({std::numeric_limits<double>::quiet_NaN(), 1.0})), "NaN hidden");
    require(sum({maximum, maximum, -maximum}) == infinity, "overflow recovered silently");
    require(sum({-maximum, -maximum, maximum}) == -infinity, "negative overflow recovered");

    const std::vector<double> signed_density{1.0e16, 1.0, -1.0e16};
    const std::vector<double> unit_operator(3, 1.0);
    require(generativeqc::tensor::cpu_compensated_dot(signed_density, unit_operator) == 1.0,
            "signed density dot lost cancellation term");
    require(generativeqc::tensor::cpu_compensated_dot({}, {}) == 0.0, "empty dot is not zero");
    bool shape_rejected = false;
    try {
      (void)generativeqc::tensor::cpu_compensated_dot(signed_density, {});
    } catch (const std::invalid_argument&) {
      shape_rejected = true;
    }
    require(shape_rejected, "dot accepted different extents");
    const std::vector<double> large{maximum}, two{2.0};
    require(std::isinf(generativeqc::tensor::cpu_compensated_dot(large, two)),
            "dot hid a nonfinite FP64 product");

    // Independent wider reduction for a long, non-exact FP64 stream. The unit
    // test remains portable if long double is no wider; the qualification
    // probe separately requires extended precision and a Decimal sum oracle.
    std::vector<double> terms;
    terms.reserve(331776);
    for (std::size_t i = 0; i < 331776; ++i)
      terms.push_back(
          -std::ldexp(1.0 + static_cast<double>(i % 997) / 997.0, -8 - static_cast<int>(i % 50)));
    long double reference = 0.0L;
    long double absolute_sum = 0.0L;
    for (double term : terms) {
      reference += static_cast<long double>(term);
      absolute_sum += std::abs(static_cast<long double>(term));
    }
    if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
      const long double error = std::abs(static_cast<long double>(sum(terms)) - reference);
      require(error <= 2.0L * std::numeric_limits<double>::epsilon() * absolute_sum,
              "compensated reduction differs from long-double oracle");
    }
    // A product that rounds to 1.0 must not be fused with the leading -1.0.
    generativeqc::tensor::CpuCompensatedSum products;
    products.add(-1.0);
    const volatile double a = 1.0 + std::ldexp(1.0, -27);
    const volatile double b = 1.0 - std::ldexp(1.0, -27);
    products.add(a * b);
    require(products.value() == 0.0, "product fused into compensated leading sum");
    std::cout << "CPU compensated FP64 sum controls passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
