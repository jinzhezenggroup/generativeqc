#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define GENERATIVEQC_SOURCE_HD __host__ __device__
#else
#define GENERATIVEQC_SOURCE_HD
#endif

namespace generativeqc::scf::cuda_execution {

struct CanonicalDirectSourceSink {
  double* values{};
  const std::int32_t* inverse{};
  const std::uint64_t* prefix{};
  std::size_t dimension{}, first_begin{}, second_begin{};
  bool same_bucket{};

  GENERATIVEQC_SOURCE_HD std::size_t offset(std::size_t first, std::size_t second,
                                            std::size_t third, std::size_t fourth) const {
    const auto bra = first >= second ? first * dimension + second : second * dimension + first;
    const auto ket = third >= fourth ? third * dimension + fourth : fourth * dimension + third;
    auto first_pair = static_cast<std::size_t>(inverse[bra]);
    auto second_pair = static_cast<std::size_t>(inverse[ket]);
    if (same_bucket && first_pair < second_pair) {
      const auto previous = first_pair;
      first_pair = second_pair;
      second_pair = previous;
    }
    const auto row = first_pair - first_begin;
    return (row ? prefix[row - 1] : 0) + second_pair - second_begin;
  }

  GENERATIVEQC_SOURCE_HD void store(std::size_t first, std::size_t second, std::size_t third,
                                    std::size_t fourth, double value) const {
    values[offset(first, second, third, fourth)] = value;
  }
};

}  // namespace generativeqc::scf::cuda_execution

#undef GENERATIVEQC_SOURCE_HD
