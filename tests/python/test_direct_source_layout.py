"""Exercise the compiled packed-source layout independently of CUDA scheduling."""

import subprocess
from pathlib import Path

from conftest import NativeCxx


def test_shell_sources_preserve_screened_sorted_canonical_slots(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """AO swaps, pair swaps, empty rows and shifted bucket ranks keep one owner."""
    root = Path(__file__).resolve().parents[2]
    driver = tmp_path / "source_layout.cpp"
    driver.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include "scf/cuda/direct_source_layout.hpp"

using generativeqc::scf::cuda_execution::CanonicalDirectSourceSink;

int main() {
  std::mt19937 random(2976);
  for (const bool same_bucket : {false, true}) {
    for (unsigned trial = 0; trial < 20; ++trial) {
      constexpr std::size_t dimension = 17;
      constexpr std::size_t first_count = 9;
      const std::size_t second_count = same_bucket ? first_count : 7;
      std::vector<std::pair<std::size_t, std::size_t>> pairs;
      for (std::size_t first = 0; first < dimension; ++first)
        for (std::size_t second = 0; second <= first; ++second)
          pairs.emplace_back(first, second);
      std::shuffle(pairs.begin(), pairs.end(), random);
      const std::size_t first_begin = 113;
      const std::size_t second_begin = same_bucket ? first_begin : 37;
      std::vector<std::int32_t> inverse(dimension * dimension, -1);
      for (std::size_t row = 0; row < first_count; ++row)
        inverse[pairs[row].first * dimension + pairs[row].second] = first_begin + row;
      if (!same_bucket)
        for (std::size_t column = 0; column < second_count; ++column) {
          const auto pair = pairs[first_count + column];
          inverse[pair.first * dimension + pair.second] = second_begin + column;
        }
      std::vector<std::uint64_t> prefix(first_count);
      std::size_t count = 0;
      for (std::size_t row = 0; row < first_count; ++row) {
        const auto limit = same_bucket ? row + 1 : second_count;
        count += row % 3 == 0 ? 0 : random() % (limit + 1);
        prefix[row] = count;
      }
      std::vector<double> values(count, NAN);
      CanonicalDirectSourceSink sink{values.data(), inverse.data(), prefix.data(), dimension,
                                     first_begin, second_begin, same_bucket};
      std::size_t ordinal = 0;
      for (std::size_t row = 0; row < first_count; ++row) {
        const auto columns = prefix[row] - (row ? prefix[row - 1] : 0);
        for (std::size_t column = 0; column < columns; ++column, ++ordinal) {
          const auto bra = pairs[row];
          const auto ket = pairs[same_bucket ? column : first_count + column];
          for (const bool swap_bra : {false, true})
            for (const bool swap_ket : {false, true}) {
              const auto first = swap_bra ? bra.second : bra.first;
              const auto second = swap_bra ? bra.first : bra.second;
              const auto third = swap_ket ? ket.second : ket.first;
              const auto fourth = swap_ket ? ket.first : ket.second;
              assert(sink.offset(first, second, third, fourth) == ordinal);
              if (same_bucket)
                assert(sink.offset(third, fourth, first, second) == ordinal);
              sink.store(first, second, third, fourth, ordinal ? -double(ordinal) : -0.0);
            }
        }
      }
      assert(ordinal == count);
      for (std::size_t index = 0; index < count; ++index) {
        assert(values[index] == -double(index));
        assert(std::signbit(values[index]));
      }
    }
  }
}
"""
    )
    executable = tmp_path / "source_layout"
    native_cxx.build_executable(
        [driver], executable, compile_args=["-std=c++20", f"-I{root / 'src'}"]
    )
    subprocess.run([str(executable)], check=True, timeout=15)
