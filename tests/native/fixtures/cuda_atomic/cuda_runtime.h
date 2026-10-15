#pragma once

#include <bit>

#define __device__

namespace cuda_atomic_double {
inline bool interfere = false;
inline unsigned int attempts = 0;
}  // namespace cuda_atomic_double

inline long long __double_as_longlong(double value) { return std::bit_cast<long long>(value); }
inline double __longlong_as_double(long long value) { return std::bit_cast<double>(value); }

inline unsigned long long atomicCAS(unsigned long long* address, unsigned long long compare,
                                    unsigned long long replacement) {
  ++cuda_atomic_double::attempts;
  if (cuda_atomic_double::interfere) {
    cuda_atomic_double::interfere = false;
    *address = __double_as_longlong(7.0);
  }
  const auto previous = *address;
  if (previous == compare) {
    *address = replacement;
  }
  return previous;
}

inline double atomicAdd(double* address, double value) {
  const double previous = *address;
  *address += value;
  return previous;
}
