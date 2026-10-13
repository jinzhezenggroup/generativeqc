// Host execution of actual generated arithmetic and ring views, not CUDA qualification.
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_scf_array_native.hpp"
#include "generated_scf_density_cuda.cuh"
#include "tensor/cuda_ring_gram.cuh"

namespace {
constexpr double sentinel = -987654.0;

bool same(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

void density() {
  using generativeqc::scf::generated::occupied_density_kernel;
  for (std::size_t n : {1U, 3U, 9U}) {
    for (double weight : {0.0, 1.0, 2.0, 0.1, 1.3, -0.3}) {
      for (std::int32_t occupied : {0, 1, static_cast<std::int32_t>(n)}) {
        const std::size_t elements = 2 * n * n;
        std::vector<double> coefficients(elements, std::numeric_limits<double>::quiet_NaN());
        for (std::size_t state = 0; state < 2; ++state)
          for (std::size_t row = 0; row < n; ++row)
            for (std::int32_t orbital = 0; orbital < occupied; ++orbital)
              coefficients[state * n * n + row + orbital * n] =
                  std::sin(0.13 * (row + 3 * orbital + 5 * state + 1));
        const std::int32_t counts[]{occupied, occupied};
        for (bool masked : {false, true}) {
          const std::uint8_t active[]{1, 0};
          for (std::size_t launched : {std::size_t{1}, elements - 1, elements, elements + 3}) {
            std::vector<double> actual(elements + 2, sentinel), expected = actual;
            blockDim.x = 1;
            gridDim.x = static_cast<unsigned>(launched);
            threadIdx.x = 0;
            for (std::size_t element = 0; element < launched; ++element) {
              blockIdx.x = static_cast<unsigned>(element);
              occupied_density_kernel<0>(2, 1, n, counts, coefficients.data(),
                                         masked ? active : nullptr, actual.data() + 1, weight);
            }
            // Independent full-square reference preserves every old DF launch
            // write. Only a complete canonical-weight launch may mirror a pair.
            for (std::size_t element = 0; element < std::min(launched, elements); ++element) {
              const auto state = element / (n * n), row = element % n;
              const auto column = element % (n * n) / n;
              if (masked && !active[state]) continue;
              double value = 0.0;
              long double oracle = 0;
              for (std::int32_t orbital = 0; orbital < occupied; ++orbital) {
                const auto left = coefficients[state * n * n + row + orbital * n];
                const auto right = coefficients[state * n * n + column + orbital * n];
                value += weight * left * right;
                oracle += static_cast<long double>(weight) * left * right;
              }
              expected[element + 1] = value;
              if (std::abs(actual[element + 1] - oracle) > 2e-14L * (1 + std::abs(oracle)))
                throw std::runtime_error("density differs from independent long-double oracle");
            }
            for (std::size_t i = 0; i < actual.size(); ++i)
              if (!same(actual[i], expected[i]))
                throw std::runtime_error("density changed order, mask, sentinel or clipped writes");
            if (launched >= elements && (weight == 1.0 || weight == 2.0)) {
              auto ordinary = std::vector<double>(elements + 2, sentinel);
              for (std::size_t element = 0; element < launched; ++element) {
                blockIdx.x = static_cast<unsigned>(element);
                const auto* mask = masked ? active : nullptr;
                if (weight == 1.0)
                  occupied_density_kernel<1>(2, 1, n, counts, coefficients.data(), mask,
                                             ordinary.data() + 1);
                else
                  occupied_density_kernel<2>(2, 1, n, counts, coefficients.data(), mask,
                                             ordinary.data() + 1);
              }
              if (ordinary != actual) throw std::runtime_error("DF/HF density schedules diverged");
            }
          }
        }
      }
    }
  }
}

void combination() {
  using generativeqc::tensor::RingHistoryElement;
  using generativeqc::tensor::generated::ordered_history_combination;
  for (std::uint32_t capacity : {1U, 2U, 7U, 64U}) {
    for (std::uint32_t first = 0; first < capacity; ++first) {
      for (std::uint32_t count = 0; count <= capacity; ++count) {
        for (std::size_t elements : {1U, 9U, 18U, 37U}) {
          // Poison unused/retired slots and neighboring systems. Rebuild the
          // same ring address for each narrowed live window, without clearing it.
          const auto nan = std::numeric_limits<double>::quiet_NaN();
          std::vector<double> history(3 * capacity * elements, nan), coefficients(capacity, nan);
          std::vector<const double*> chronological;
          auto* system = history.data() + capacity * elements;
          for (std::uint32_t item = 0; item < count; ++item) {
            auto* row = system + ((first + item) % capacity) * elements;
            chronological.push_back(row);
            coefficients[item] = (item % 2 ? -1.0 : 1.0) / (item + 1);
            for (std::size_t element = 0; element < elements; ++element)
              row[element] = std::cos(0.019 * (item + 1) * (element + 2));
          }
          std::vector<double> cpu(elements, sentinel);
          generativeqc::scf::generated::diis_extrapolate(cpu.data(), chronological,
                                                         coefficients.data(), count, elements);
          for (std::size_t element = 0; element < elements; ++element) {
            const auto result = ordered_history_combination(
                coefficients.data(), count,
                RingHistoryElement{system, elements, element, first, capacity});
            double ordered = 0;
            long double oracle = 0;
            for (std::uint32_t item = 0; item < count; ++item) {
              ordered += coefficients[item] * chronological[item][element];
              oracle += static_cast<long double>(coefficients[item]) * chronological[item][element];
            }
            if (!same(result, ordered) || !same(result, cpu[element]) || !std::isfinite(result) ||
                std::abs(result - oracle) > 2e-14L * (1 + std::abs(oracle)))
              throw std::runtime_error("history combination changed order or read a retired slot");
          }
          if (!std::isnan(history.front()) || !std::isnan(history.back()))
            throw std::runtime_error("history input mutated");
        }
      }
    }
  }
}
}  // namespace

int main() {
  density();
  combination();
  std::cout << "retained density and history contracts passed\n";
}
