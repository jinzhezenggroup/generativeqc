// Ring Gram/solve/combine against independently rebuilt chronological histories.
// No CC equations, native solver result, or cached Gram enters the host oracle.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cc/cuda_state.cuh"
#include "solver/diis_ring.hpp"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* p{};
  explicit Device(std::size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~Device() { cudaFree(p); }
  void put(const std::vector<T>& values) {
    check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get(std::size_t n) {
    std::vector<T> out(n);
    check(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
  }
};

double dot(const double* a, const double* b, std::size_t n) {
  // Reproduce the documented FP64 tree with explicit host rounding, and also
  // compare with an independently accumulated extended-precision dot below.
  std::array<double, 256> part{};
  for (std::size_t lane = 0; lane < 256; ++lane)
    for (std::size_t i = lane; i < n; i += 256) {
      volatile double product = a[i] * b[i];
      volatile double sum = part[lane] + product;
      part[lane] = sum;
    }
  for (std::size_t stride = 128; stride; stride /= 2)
    for (std::size_t lane = 0; lane < stride; ++lane) {
      volatile double sum = part[lane] + part[lane + stride];
      part[lane] = sum;
    }
  return part[0];
}

void run(int capacity, std::size_t elements, int mode) {
  const double poison = std::numeric_limits<double>::quiet_NaN();
  generativeqc::solver::DiisRing ring(capacity);
  std::vector<double> rows(capacity * elements, poison), stored(capacity * capacity, poison);
  Device<double> errors(rows.size()), gram(stored.size()), dense(stored.size());
  Device<double> system((capacity + 1) * (capacity + 1)), coefficients(capacity + 1);
  Device<double> reference_coefficients(capacity + 1), result(elements);
  Device<int> status(1), reference_status(1), arithmetic(1);
  errors.put(rows);
  gram.put(stored);
  for (int step = 0; step < 3 * capacity + 3; ++step) {
    const auto slot = ring.push();
    const auto count = static_cast<int>(ring.size()), first = static_cast<int>(ring.first());
    for (std::size_t i = 0; i < elements; ++i)
      rows[slot * elements + i] =
          mode == 1   ? 0.0
          : mode == 2 ? 1.0
                      : std::sin((step + 1.0) * (i + 0.37)) + std::cos(0.173 * (i + step));
    errors.put(rows);
    // Corrupt one old-old entry deliberately. A full Gram rebuild would repair
    // it and fail this sentinel check, even if all ordinary outputs agreed.
    std::size_t canary = stored.size();
    double previous = 0;
    if (count > 1) {
      canary = ring.slot(0) * capacity + ring.slot(0);
      previous = stored[canary];
      stored[canary] = -17.25;
      gram.put(stored);
    }
    generativeqc_tensor::history_gram_row<<<count, 256>>>(errors.p, elements, capacity, first,
                                                          count, slot, gram.p);
    check(cudaGetLastError());
    auto updated = gram.get(stored.size());
    for (int i = 0; i < capacity; ++i)
      for (int j = 0; j < capacity; ++j)
        if (i != static_cast<int>(slot) && j != static_cast<int>(slot) &&
            std::bit_cast<std::uint64_t>(updated[i * capacity + j]) !=
                std::bit_cast<std::uint64_t>(stored[i * capacity + j]))
          throw std::runtime_error("an old-old Gram entry was recomputed or moved");
    if (canary < stored.size()) updated[canary] = previous;
    stored = updated;
    gram.put(stored);
    std::vector<double> expected(count * count);
    for (int i = 0; i < count; ++i)
      for (int j = 0; j < count; ++j) {
        const auto a = ring.slot(i), b = ring.slot(j);
        const double want = dot(rows.data() + a * elements, rows.data() + b * elements, elements);
        const double actual = stored[a * capacity + b];
        if (actual != want) throw std::runtime_error("ordered Gram differs from full rebuild");
        long double independent = 0;
        for (std::size_t k = 0; k < elements; ++k)
          independent += static_cast<long double>(rows[a * elements + k]) * rows[b * elements + k];
        if (std::abs(independent - actual) > 2e-12L * std::max(1.0L, std::abs(independent)))
          throw std::runtime_error("independent Gram oracle failed");
        expected[i * count + j] = want;
      }
    dense.put(expected);
    generativeqc::cc::diis_coefficients<<<1, 1>>>(gram.p, count, system.p, coefficients.p, status.p,
                                                  capacity, first);
    generativeqc::cc::diis_coefficients<<<1, 1>>>(dense.p, count, system.p,
                                                  reference_coefficients.p, reference_status.p);
    check(cudaGetLastError());
    const auto action = status.get(1)[0];
    if (action != reference_status.get(1)[0]) throw std::runtime_error("retirement policy changed");
    if (count > 1 && mode == 1 && action != 2) throw std::runtime_error("zero Gram policy changed");
    if (count > 1 && mode == 2 && action != 1) throw std::runtime_error("singular Gram accepted");
    if (!action && coefficients.get(count) != reference_coefficients.get(count))
      throw std::runtime_error("chronological coefficients changed");
    result.put(std::vector<double>(elements, -987.0));
    arithmetic.put({0});
    generativeqc_tensor::diis_combine_slice<<<generativeqc_tensor::blocks(elements, 256), 256>>>(
        errors.p, coefficients.p, elements, 0, elements, count, status.p, result.p, arithmetic.p,
        capacity, first);
    check(cudaGetLastError());
    const auto values = result.get(elements);
    if (arithmetic.get(1)[0]) throw std::runtime_error("finite combination failed");
    const auto weights = !action ? coefficients.get(count) : std::vector<double>{};
    for (std::size_t i = 0; i < elements; ++i) {
      double want = -987.0;
      if (!action) {
        want = 0;
        for (int row = 0; row < count; ++row) {
          volatile double product = weights[row] * rows[ring.slot(row) * elements + i];
          volatile double sum = want + product;
          want = sum;
        }
      }
      if (values[i] != want) throw std::runtime_error("ordered/guarded combination changed");
    }
    if ((action == 1 || step % 5 == 2) && ring.size() > 1) ring.retire_oldest();
    if (step == capacity + 1) ring.clear();
  }
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    for (int capacity : {1, 2, 3, 8, 20})
      for (std::size_t elements : {1, 17, 256, 259, 4097})
        for (int mode : {0, 1, 2}) run(capacity, elements, mode);
    std::cout << "75 ring cases: exact Gram order, independent dots, no old-old writes, "
                 "wrap/restart/retirement and guarded combination pass\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
