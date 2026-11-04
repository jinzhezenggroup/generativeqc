"""Conservative host counterparts for positive CUDA interval arithmetic.

These deliberately widen an RN result by one representable number. They are
outward enclosures, not a claim to simulate CUDA's exact directed-rounding bits.
Zero products/sums remain exact so empty occupied spins are exercised normally.
"""

DIRECTED_ROUNDING_HOST = r"""
#include <cmath>
#include <limits>
inline double __dadd_ru(double left, double right) {
  if (left == 0.0 && right == 0.0) return 0.0;
  return std::nextafter(left + right, std::numeric_limits<double>::infinity());
}
inline double __dmul_ru(double left, double right) {
  if (left == 0.0 || right == 0.0) return left * right;
  return std::nextafter(left * right, std::numeric_limits<double>::infinity());
}
inline double __ddiv_ru(double left, double right) {
  return std::nextafter(left / right, std::numeric_limits<double>::infinity());
}
inline double __dsub_rd(double left, double right) {
  return std::nextafter(left - right, -std::numeric_limits<double>::infinity());
}
inline double __dsqrt_ru(double value) {
  if (value == 0.0) return 0.0;
  return std::nextafter(std::sqrt(value), std::numeric_limits<double>::infinity());
}
"""
