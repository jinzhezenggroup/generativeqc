/** Independent dot oracle for populated, wrapped and retired DIIS histories.
 * Poisoned unused slots catch reads outside the chronological live window;
 * an inactive neighbor must leave its entire output reservation untouched.
 */
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/cuda/scf_diis_kernels.hpp"

namespace {
bool notebook_allocation(std::string& pod, std::string& namespace_name) {
  const char* named_pod = std::getenv("MY_POD_NAME");
  const char* kubernetes = std::getenv("KUBERNETES_SERVICE_HOST");
  const char* jupyter = std::getenv("JUPYTER_SERVER_ROOT");
  if (!named_pod || !*named_pod || !kubernetes || !*kubernetes || !jupyter || !*jupyter)
    return false;
  std::ifstream hostname("/etc/hostname");
  std::ifstream namespace_file("/var/run/secrets/kubernetes.io/serviceaccount/namespace");
  std::string actual_hostname;
  if (!std::getline(hostname, actual_hostname) ||
      !std::getline(namespace_file, namespace_name) || namespace_name.empty() ||
      actual_hostname != named_pod)
    return false;
  pod = named_pod;
  return true;
}

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* data{};
  explicit Device(const std::vector<T>& values) {
    check(cudaMalloc(&data, values.size() * sizeof(T)));
    check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

void check_history(unsigned spins, unsigned old_count, unsigned head) {
  constexpr unsigned n = 67, history = 8, batch = 2;
  const std::size_t size = n * n * spins, parts = (size + 4095) / 4096;
  const double poison = std::numeric_limits<double>::quiet_NaN();
  const auto count = std::min(old_count + 1, history);
  const auto first = (head + 1 + history - count) % history;
  const auto live = [&](unsigned slot) { return (slot + history - first) % history < count; };
  std::vector<double> current(batch * size), stored(batch * history * size, poison);
  for (std::size_t i = 0; i < size; ++i) {
    current[i] = std::sin(0.017 * (i + 1));
    for (unsigned slot = 0; slot < history; ++slot)
      if (live(slot) && slot != head)
        stored[slot * size + i] = std::cos(0.011 * (slot + 1) * (i + 1));
  }
  std::vector<double> result(batch * history * history * parts, poison);
  Device<double> d_current(current), d_stored(stored), d_result(result);
  Device<std::uint8_t> d_active(std::vector<std::uint8_t>{1, 0});
  Device<std::uint32_t> d_count(std::vector<std::uint32_t>{old_count, old_count});
  Device<std::uint32_t> d_head(std::vector<std::uint32_t>{head, head});
  generativeqc::scf::cuda_execution::launch_diis_dot_partials(
      nullptr, batch, n, spins, history, d_current.data, d_stored.data, d_active.data, d_count.data,
      d_head.data, parts, d_result.data);
  check(cudaGetLastError());
  check(cudaMemcpy(result.data(), d_result.data, result.size() * sizeof(double),
                   cudaMemcpyDeviceToHost));
  for (unsigned system = 0; system < batch; ++system)
    for (unsigned row = 0; row < history; ++row)
      for (unsigned column = 0; column < history; ++column) {
        const auto offset = ((system * history + row) * history + column) * parts;
        const bool used = system == 0 && count >= 2 && live(row) && live(column);
        if (!used) {
          for (std::size_t part = 0; part < parts; ++part)
            if (!std::isnan(result[offset + part]))
              throw std::runtime_error("DIIS wrote an inactive/unpopulated partial");
          continue;
        }
        const auto* left = row == head ? current.data() : stored.data() + row * size;
        const auto* right = column == head ? current.data() : stored.data() + column * size;
        // Long-double accumulation is independent of the GPU reduction tree.
        long double expected = 0;
        for (std::size_t i = 0; i < size; ++i)
          expected += static_cast<long double>(left[i]) * right[i];
        double actual = 0;
        for (std::size_t part = 0; part < parts; ++part) actual += result[offset + part];
        if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-11L)
          throw std::runtime_error("DIIS partials differ from the independent ring-history dot");
      }
}

void check_incremental_history(unsigned spins) {
  constexpr unsigned n = 5, history = 4, batch = 3;
  const std::size_t elements = n * n * spins;
  const double poison = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> current(batch * elements), stored(batch * history * elements, poison);
  std::vector<double> gram(batch * history * history, poison);
  std::vector<std::uint8_t> active(batch);
  std::vector<std::uint32_t> count(batch, 0), head(batch, 0);
  Device<double> d_current(current), d_stored(stored), d_gram(gram);
  Device<double> d_fock_history(stored);
  Device<double> d_linear(std::vector<double>(batch * (history + 1) * (history + 1)));
  Device<double> d_coefficients(std::vector<double>(batch * (history + 1)));
  Device<double> d_effective(current);
  Device<std::uint8_t> d_active(active);
  Device<std::uint32_t> d_count(count), d_head(head);
  const auto upload = [](auto& device, const auto& host) {
    check(cudaMemcpy(device.data, host.data(), host.size() * sizeof(host[0]),
                     cudaMemcpyHostToDevice));
  };
  const auto download = [](auto& host, const auto& device) {
    check(cudaMemcpy(host.data(), device.data, host.size() * sizeof(host[0]),
                     cudaMemcpyDeviceToHost));
  };
  const auto same = [](double left, double right) {
    return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
  };
  const auto full_dot = [&](const double* left, const double* right) {
    long double value = 0;
    for (std::size_t i = 0; i < elements; ++i)
      value += static_cast<long double>(left[i]) * right[i];
    return value;
  };
  for (unsigned step = 0; step < 3 * history + 2; ++step) {
    active = {1, static_cast<std::uint8_t>(step % 2 == 0), 0};
    for (unsigned system = 0; system < batch; ++system)
      for (std::size_t i = 0; i < elements; ++i)
        current[system * elements + i] =
            std::sin(0.13 * (step + 1) * (i + 1 + system)) +
            std::cos(0.07 * (step + 3 + system) * (i + 1));
    upload(d_current, current);
    upload(d_active, active);
    const auto previous = gram;
    check(generativeqc::scf::cuda_execution::launch_diis_pending_gram(
        nullptr, batch, n, spins, history, d_current.data, d_stored.data, d_active.data,
        d_count.data, d_head.data, d_gram.data));
    download(gram, d_gram);
    for (unsigned system = 0; system < batch; ++system) {
      const unsigned inserted = head[system];
      const unsigned first = (inserted + history - count[system]) % history;
      for (unsigned row = 0; row < history; ++row)
        for (unsigned column = 0; column < history; ++column) {
          const auto index = (system * history + row) * history + column;
          const bool live_row = row == inserted ||
                                (row + history - first) % history < count[system];
          const bool live_column = column == inserted ||
                                   (column + history - first) % history < count[system];
          const bool new_pair = active[system] && live_row && live_column &&
                                (row == inserted || column == inserted);
          if (!new_pair) {
            if (!same(gram[index], previous[index]))
              throw std::runtime_error("DIIS recomputed an old-old or inactive Gram pair");
            continue;
          }
          const auto* left = row == inserted
                                 ? current.data() + system * elements
                                 : stored.data() + (system * history + row) * elements;
          const auto* right = column == inserted
                                  ? current.data() + system * elements
                                  : stored.data() + (system * history + column) * elements;
          const auto expected = full_dot(left, right);
          if (!std::isfinite(gram[index]) ||
              std::abs(static_cast<long double>(gram[index]) - expected) >
                  2e-12L * std::max(1.0L, std::abs(expected)))
            throw std::runtime_error("incremental DIIS Gram differs from full precision oracle");
        }
    }
    generativeqc::scf::cuda_execution::launch_update_diis_kernel(
        batch, 32, 0, nullptr, batch, n, spins, history, d_current.data, d_current.data,
        d_active.data, d_fock_history.data, d_stored.data, d_linear.data,
        d_coefficients.data, d_count.data, d_head.data, d_effective.data,
        false, false, nullptr, 0, d_gram.data);
    check(cudaGetLastError());
    for (unsigned system = 0; system < batch; ++system) {
      if (!active[system]) continue;
      const auto slot = head[system];
      std::copy_n(current.data() + system * elements, elements,
                  stored.data() + (system * history + slot) * elements);
      count[system] = std::min(count[system] + 1, history);
      head[system] = (slot + 1) % history;
    }
    std::vector<std::uint32_t> actual_count(batch), actual_head(batch);
    download(actual_count, d_count);
    download(actual_head, d_head);
    if (actual_count != count || actual_head != head)
      throw std::runtime_error("incremental DIIS history count or head changed");
    if (step == history + 1) {
      count[0] = head[0] = 0;
      check(cudaMemset(d_count.data, 0, sizeof(std::uint32_t)));
      check(cudaMemset(d_head.data, 0, sizeof(std::uint32_t)));
    }
  }
  active = {1, 0, 0};
  current[0] = poison;
  upload(d_current, current);
  upload(d_active, active);
  check(generativeqc::scf::cuda_execution::launch_diis_pending_gram(
      nullptr, batch, n, spins, history, d_current.data, d_stored.data, d_active.data,
      d_count.data, d_head.data, d_gram.data));
  download(gram, d_gram);
  if (!std::isnan(gram[head[0] * history + head[0]]))
    throw std::runtime_error("nonfinite pending residual was hidden by the Gram reduction");

  current[0] = 17.0;
  upload(d_current, current);
  count[0] = history + 1;
  head[0] = history;
  upload(d_count, count);
  upload(d_head, head);
  const auto before_invalid = gram;
  check(generativeqc::scf::cuda_execution::launch_diis_pending_gram(
      nullptr, batch, n, spins, history, d_current.data, d_stored.data, d_active.data,
      d_count.data, d_head.data, d_gram.data));
  download(gram, d_gram);
  for (std::size_t i = 0; i < gram.size(); ++i)
    if (!same(gram[i], before_invalid[i]))
      throw std::runtime_error("invalid DIIS state wrote a Gram entry");
  generativeqc::scf::cuda_execution::launch_update_diis_kernel(
      batch, 32, 0, nullptr, batch, n, spins, history, d_current.data, d_current.data,
      d_active.data, d_fock_history.data, d_stored.data, d_linear.data,
      d_coefficients.data, d_count.data, d_head.data, d_effective.data,
      false, false, nullptr, 0, d_gram.data);
  check(cudaGetLastError());
  download(count, d_count);
  download(head, d_head);
  if (count[0] != 0 || head[0] != 0)
    throw std::runtime_error("invalid incremental DIIS state was not reset");
  std::vector<double> effective(current.size());
  download(effective, d_effective);
  for (std::size_t i = 0; i < elements; ++i)
    if (effective[i] != current[i])
      throw std::runtime_error("invalid incremental DIIS state did not return current Fock");
}

void check_normalized_retirement(unsigned spins) {
  constexpr unsigned n = 3, history = 4;
  const std::size_t elements = n * n * spins;
  std::vector<double> residual(elements), fock(elements);
  for (std::size_t i = 0; i < elements; ++i) residual[i] = std::sin(0.3 * (i + 1));
  Device<double> d_residual(residual), d_fock(fock);
  Device<std::uint8_t> d_active(std::vector<std::uint8_t>{1});
  Device<double> d_incremental_fock_history(std::vector<double>(history * elements));
  Device<double> d_incremental_residual_history(std::vector<double>(history * elements));
  Device<double> d_incremental_gram(std::vector<double>(history * history));
  Device<double> d_incremental_linear(std::vector<double>((history + 1) * (history + 1)));
  Device<double> d_incremental_coefficients(std::vector<double>(history + 1));
  Device<double> d_incremental_effective(fock);
  Device<std::uint32_t> d_incremental_count(std::vector<std::uint32_t>{0});
  Device<std::uint32_t> d_incremental_head(std::vector<std::uint32_t>{0});
  Device<double> d_default_fock_history(std::vector<double>(history * elements));
  Device<double> d_default_residual_history(std::vector<double>(history * elements));
  Device<double> d_default_linear(std::vector<double>((history + 1) * (history + 1)));
  Device<double> d_default_coefficients(std::vector<double>(history + 1));
  Device<double> d_default_effective(fock);
  Device<std::uint32_t> d_default_count(std::vector<std::uint32_t>{0});
  Device<std::uint32_t> d_default_head(std::vector<std::uint32_t>{0});
  for (unsigned step = 0; step < 2 * history + 1; ++step) {
    for (std::size_t i = 0; i < elements; ++i) fock[i] = step + 0.1 * i;
    check(cudaMemcpy(d_fock.data, fock.data(), elements * sizeof(double), cudaMemcpyHostToDevice));
    check(generativeqc::scf::cuda_execution::launch_diis_pending_gram(
        nullptr, 1, n, spins, history, d_residual.data, d_incremental_residual_history.data,
        d_active.data, d_incremental_count.data, d_incremental_head.data,
        d_incremental_gram.data));
    generativeqc::scf::cuda_execution::launch_update_diis_kernel(
        1, 32, 0, nullptr, 1, n, spins, history, d_fock.data, d_residual.data, d_active.data,
        d_incremental_fock_history.data, d_incremental_residual_history.data,
        d_incremental_linear.data, d_incremental_coefficients.data, d_incremental_count.data,
        d_incremental_head.data, d_incremental_effective.data, true, false, nullptr, 0,
        d_incremental_gram.data);
    check(cudaGetLastError());
    generativeqc::scf::cuda_execution::launch_update_diis_kernel(
        1, 32, 0, nullptr, 1, n, spins, history, d_fock.data, d_residual.data, d_active.data,
        d_default_fock_history.data, d_default_residual_history.data, d_default_linear.data,
        d_default_coefficients.data, d_default_count.data, d_default_head.data,
        d_default_effective.data, true);
    check(cudaGetLastError());
    std::uint32_t incremental_count{}, incremental_head{}, default_count{}, default_head{};
    check(cudaMemcpy(&incremental_count, d_incremental_count.data, sizeof(incremental_count),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&incremental_head, d_incremental_head.data, sizeof(incremental_head),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&default_count, d_default_count.data, sizeof(default_count),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&default_head, d_default_head.data, sizeof(default_head),
                     cudaMemcpyDeviceToHost));
    if (incremental_count != default_count || incremental_head != default_head ||
        incremental_count != (step == 0 ? 1U : 2U))
      throw std::runtime_error("normalized DIIS dependent retirement changed");
    std::vector<double> incremental_effective(elements), default_effective(elements);
    check(cudaMemcpy(incremental_effective.data(), d_incremental_effective.data,
                     elements * sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(default_effective.data(), d_default_effective.data,
                     elements * sizeof(double), cudaMemcpyDeviceToHost));
    if (incremental_effective != default_effective || incremental_effective != fock)
      throw std::runtime_error("dependent DIIS fallback changed effective Fock");
  }
}
}  // namespace

int main() {
  std::string pod, namespace_name;
  const bool notebook = notebook_allocation(pod, namespace_name);
  if (!notebook && !std::getenv("SLURM_JOB_ID")) return 77;
  try {
    check(cudaSetDevice(0));
    if (notebook) {
      cudaDeviceProp device{};
      check(cudaGetDeviceProperties(&device, 0));
      std::cout << "notebook allocation pod=" << pod << " namespace=" << namespace_name
                << " gpu=" << device.name << '\n';
    }
    for (unsigned spins : {1U, 2U})
      for (unsigned count : {0U, 1U, 2U, 6U, 7U, 8U})
        for (unsigned head : {0U, 3U, 7U}) check_history(spins, count, head);
    for (unsigned spins : {1U, 2U}) {
      check_incremental_history(spins);
      check_normalized_retirement(spins);
    }
    std::cout << "validated DIIS partials and incremental circular histories\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
