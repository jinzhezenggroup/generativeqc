/** Exercise the actual production kernel against its retained ordered route.
 * Independent long-double dots validate the physical-slot cache, including
 * wraparound, normalized retirement, poisoned inactive systems and uncleared
 * cache storage across a complete count/head reset. */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/cuda/scf_diis_kernels.hpp"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class Value>
struct Device {
  Value* data{};
  std::size_t size{};
  Device(std::size_t count, Value initial) : size(count) {
    std::vector<Value> values(count, initial);
    check(cudaMalloc(&data, size * sizeof(Value)));
    check(cudaMemcpy(data, values.data(), size * sizeof(Value), cudaMemcpyHostToDevice));
  }
  ~Device() { (void)cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::vector<Value> read() const {
    std::vector<Value> result(size);
    check(cudaMemcpy(result.data(), data, size * sizeof(Value), cudaMemcpyDeviceToHost));
    return result;
  }
  void write(const std::vector<Value>& values) {
    if (values.size() != size) throw std::runtime_error("test input shape changed");
    check(cudaMemcpy(data, values.data(), size * sizeof(Value), cudaMemcpyHostToDevice));
  }
};

struct State {
  Device<double> fock_history, residual_history, matrix, coefficients, effective, cache;
  Device<std::uint32_t> count, head;
  Device<generativeqc::tensor::RingGramWork> work;
  State(unsigned history, std::size_t vector_size)
      : fock_history(2 * history * vector_size, std::numeric_limits<double>::quiet_NaN()),
        residual_history(2 * history * vector_size, std::numeric_limits<double>::quiet_NaN()),
        matrix(2 * (history + 1) * (history + 1), -9876),
        coefficients(2 * (history + 1), -9876),
        effective(2 * vector_size, -9876),
        cache(2 * history * history, std::numeric_limits<double>::quiet_NaN()),
        count(2, 0),
        head(2, 0),
        work(2, {}) {}
};

template <class Value>
void identical(const std::vector<Value>& first, const std::vector<Value>& second) {
  if (first.size() != second.size() ||
      std::memcmp(first.data(), second.data(), first.size() * sizeof(Value)))
    throw std::runtime_error("cached Gram changed ordered DIIS state bits");
}

void one_case(unsigned basis, unsigned spins, unsigned history, bool normalized) {
  using namespace generativeqc::scf::cuda_execution;
  const std::size_t vector_size = static_cast<std::size_t>(basis) * basis * spins;
  State legacy(history, vector_size), cached(history, vector_size);
  Device<double> fock(2 * vector_size, 0), residual(2 * vector_size, 0);
  Device<std::uint8_t> active(2, 0);
  active.write({1, 0});
  unsigned long long expected_dots = 0;
  bool saw_retirement = false;
  for (unsigned step = 0; step < 3 * history + 32; ++step) {
    if (step == 13 || step == 23) {
      legacy.count.write({0, 0});
      legacy.head.write({0, 0});
      cached.count.write({0, 0});
      cached.head.write({0, 0});
    }
    std::vector<double> fock_input(2 * vector_size, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> residual_input = fock_input;
    const auto frequency = step % 7 == 1 ? step - 1 : step;
    for (std::size_t element = 0; element < vector_size; ++element) {
      fock_input[element] = std::sin(0.009 * (step + 1) * (element + 1));
      residual_input[element] = std::cos(0.013 * (frequency + 1) * (element + 1));
    }
    fock.write(fock_input);
    residual.write(residual_input);
    const auto before = cached.count.read();
    const auto admitted_count = std::min(before[0] + 1, history);
    if (admitted_count >= 2) expected_dots += admitted_count + (before[0] == 1);
    launch_update_diis_kernel(2, 32, 0, nullptr, 2, basis, spins, history, fock.data, residual.data,
                              active.data, legacy.fock_history.data, legacy.residual_history.data,
                              legacy.matrix.data, legacy.coefficients.data, legacy.count.data,
                              legacy.head.data, legacy.effective.data, normalized);
    launch_update_diis_cached_gram(
        2, 32, 0, nullptr, 2, basis, spins, history, fock.data, residual.data, active.data,
        cached.fock_history.data, cached.residual_history.data, cached.matrix.data,
        cached.coefficients.data, cached.count.data, cached.head.data, cached.effective.data,
        cached.cache.data, normalized, cached.work.data);
    check(cudaGetLastError());
    identical(legacy.count.read(), cached.count.read());
    identical(legacy.head.read(), cached.head.read());
    identical(legacy.coefficients.read(), cached.coefficients.read());
    identical(legacy.effective.read(), cached.effective.read());
    identical(legacy.fock_history.read(), cached.fock_history.read());
    identical(legacy.residual_history.read(), cached.residual_history.read());
    const auto counts = cached.count.read();
    const auto heads = cached.head.read();
    saw_retirement |= counts[0] < admitted_count;
    const auto first = normalized ? (heads[0] + history - counts[0]) % history : 0;
    const auto values = cached.residual_history.read();
    const auto gram = cached.cache.read();
    for (unsigned row = 0; counts[0] >= 2 && row < counts[0]; ++row)
      for (unsigned column = 0; column < counts[0]; ++column) {
        const auto row_slot = (first + row) % history;
        const auto column_slot = (first + column) % history;
        long double expected = 0;
        for (std::size_t element = 0; element < vector_size; ++element)
          expected += static_cast<long double>(values[row_slot * vector_size + element]) *
                      values[column_slot * vector_size + element];
        const double actual = gram[row_slot * history + column_slot];
        if (!std::isfinite(actual) ||
            std::abs(actual - expected) > 3e-12L * (1 + std::abs(expected)))
          throw std::runtime_error("cached Gram differs from independent dot oracle");
      }
    for (std::size_t element = static_cast<std::size_t>(history) * history; element < gram.size();
         ++element)
      if (!std::isnan(gram[element])) throw std::runtime_error("inactive Gram was mutated");
    const auto work = cached.work.read();
    if (work[0].dots != expected_dots || work[0].vector_elements != expected_dots * vector_size ||
        work[1].dots || work[1].vector_elements)
      throw std::runtime_error("incremental vector work is not one live row per insertion");
  }
  if (normalized && history > 2 && !saw_retirement)
    throw std::runtime_error("normalized test never exercised dependent-history retirement");
  const auto work = cached.work.read()[0];
  std::cout << "basis=" << basis << " spins=" << spins << " history=" << history
            << " normalized=" << normalized << " dots=" << work.dots
            << " vector_elements=" << work.vector_elements << '\n';
}
}  // namespace

void invalid_history_state() {
  using namespace generativeqc::scf::cuda_execution;
  constexpr unsigned history = 4;
  constexpr std::size_t vector_size = 9;
  for (bool invalid_head : {false, true}) {
    State state(history, vector_size);
    Device<double> fock(2 * vector_size, 2.5), residual(2 * vector_size, 1.0);
    Device<std::uint8_t> active(2, 1);
    state.count.write({invalid_head ? 0U : history + 1, 0});
    state.head.write({invalid_head ? history : 0U, 0});
    const auto prior_fock = state.fock_history.read();
    const auto prior_residual = state.residual_history.read();
    const auto prior_cache = state.cache.read();
    launch_update_diis_cached_gram(2, 32, 0, nullptr, 2, 3, 1, history, fock.data, residual.data,
                                   active.data, state.fock_history.data,
                                   state.residual_history.data, state.matrix.data,
                                   state.coefficients.data, state.count.data, state.head.data,
                                   state.effective.data, state.cache.data, true, state.work.data);
    check(cudaGetLastError());
    if (state.count.read() != std::vector<std::uint32_t>{0, 1} ||
        state.head.read() != std::vector<std::uint32_t>{0, 1})
      throw std::runtime_error("invalid ordered history did not reset independently");
    identical(state.effective.read(), fock.read());
    const auto after_fock = state.fock_history.read();
    const auto after_residual = state.residual_history.read();
    if (std::memcmp(prior_fock.data(), after_fock.data(), history * vector_size * sizeof(double)) ||
        std::memcmp(prior_residual.data(), after_residual.data(),
                    history * vector_size * sizeof(double)))
      throw std::runtime_error("invalid history was accessed before reset");
    identical(prior_cache, state.cache.read());
    for (const auto& work : state.work.read())
      if (work.dots || work.vector_elements)
        throw std::runtime_error("invalid or singleton history performed dot work");
  }
}

void disabled_history(unsigned history) {
  using namespace generativeqc::scf::cuda_execution;
  State legacy(1, 9), cached(1, 9);
  Device<double> fock(18, 2.5);
  Device<std::uint8_t> active(2, 0);
  active.write({1, 0});
  launch_update_diis_kernel(2, 32, 0, nullptr, 2, 3, 1, history, fock.data, nullptr, active.data,
                            legacy.fock_history.data, legacy.residual_history.data,
                            legacy.matrix.data, legacy.coefficients.data, legacy.count.data,
                            legacy.head.data, legacy.effective.data);
  launch_update_diis_cached_gram(2, 32, 0, nullptr, 2, 3, 1, history, fock.data, nullptr,
                                 active.data, cached.fock_history.data,
                                 cached.residual_history.data, cached.matrix.data,
                                 cached.coefficients.data, cached.count.data, cached.head.data,
                                 cached.effective.data, nullptr, false, cached.work.data);
  check(cudaGetLastError());
  identical(legacy.effective.read(), cached.effective.read());
  identical(legacy.count.read(), cached.count.read());
  identical(legacy.head.read(), cached.head.read());
  for (const auto& work : cached.work.read())
    if (work.dots || work.vector_elements)
      throw std::runtime_error("disabled DIIS performed dot work");
}

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    invalid_history_state();
    disabled_history(0);
    disabled_history(1);
    for (unsigned basis : {3U, 17U, 67U})
      for (unsigned spins : {1U, 2U})
        for (unsigned history : {2U, 3U, 8U, 32U, 33U, 64U})
          for (bool normalized : {false, true}) one_case(basis, spins, history, normalized);
    std::cout << "incremental ordered DIIS Gram contracts passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
