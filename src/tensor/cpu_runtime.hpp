#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Generated FP64 programs own the sizes. Only publish after every SSA value
// passes its finite check. The budget includes adapter staging and snapshots.
namespace generativeqc_tensor_cpu {
// Byte storage never creates double objects through a cast. memcpy preserves
// representation and avoids alignment, aliasing and object-lifetime assumptions.
template <class T>
T load(const unsigned char* data) noexcept {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}
template <class T>
void store(unsigned char* data, T value) noexcept {
  std::memcpy(data, &value, sizeof(T));
}

// Counts in this separate ABI are bytes, with generated dtype/offset contracts.
// Boolean inputs must contain canonical bytes 0/1; generated evaluation checks
// every live FP64 and Boolean value before publishing any detached output.
template <class Evaluate>
int run_typed(const unsigned char* input, size_t ni, unsigned char* output, size_t no,
              size_t budget, size_t expected_input, size_t expected_output, size_t arena_bytes,
              size_t required_bytes, Evaluate evaluate) noexcept {
  static_assert(CHAR_BIT == 8 && sizeof(double) == 8,
                "typed CPU ABI requires FP64 and 8-bit bytes");
  const auto in = reinterpret_cast<std::uintptr_t>(input);
  const auto out = reinterpret_cast<std::uintptr_t>(output);
  if (ni != expected_input || no != expected_output || budget < required_bytes ||
      arena_bytes < no || (ni && (!input || in % alignof(double))) ||
      (no && (!output || out % alignof(double))) ||
      (ni && no && (in <= out ? out - in < ni : in - out < no)))
    return -1;
  try {
    std::vector<unsigned char> arena(arena_bytes);
    if (!evaluate(input, arena.data())) return -2;
    if (no) std::copy_n(arena.data() + arena_bytes - no, no, output);
    return 0;
  } catch (...) {
    return -3;
  }
}

template <class Evaluate>
int run(const double* input, size_t ni, double* output, size_t no, size_t budget,
        size_t expected_input, size_t expected_output, size_t arena_count, size_t required_bytes,
        Evaluate evaluate) noexcept {
  if (ni != expected_input || no != expected_output || budget < required_bytes || (ni && !input) ||
      (no && !output))
    return -1;
  try {
    for (size_t i = 0; i < ni; ++i)
      if (!std::isfinite(input[i])) return -2;
    std::vector<double> arena(arena_count);
    if (!evaluate(input, arena.data())) return -2;
    // Generated evaluation stores the detached outputs at the arena tail.
    if (no) std::copy_n(arena.data() + arena_count - no, no, output);
    return 0;
  } catch (...) {
    return -3;
  }
}
}  // namespace generativeqc_tensor_cpu
