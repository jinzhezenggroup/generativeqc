/** Real-device lifetime/work test for the pair-materialized Direct schedule.
 * The retained per-component evaluator controls ERIs. A separate host orbit
 * expansion controls RHF/UHF J/K contraction, without calling its scatter.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "generated_direct_pair_cache.cuh"
#include "generated_direct_source_contraction.cuh"
#ifdef GENERATIVEQC_PAIR_MATERIALIZED_DFT_STREAM_TEST
#include "scf/cuda/direct_bounded_dddd.hpp"
#endif

using namespace generativeqc::scf::cuda_execution;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Device {
  T* data{};
  std::size_t count{};
  explicit Device(std::size_t size) : count(size) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    clear();
  }
  explicit Device(const std::vector<T>& values) : Device(values.size()) {
    check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  void clear() { check(cudaMemset(data, 0, count * sizeof(T))); }
  std::vector<T> read() const {
    std::vector<T> values(count);
    check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};

/** Synthetic scientific inputs, with the actual production pair-cache kernel.
 * Two signed primitive contractions and complete Cartesian components per
 * shell expose cancellation, index ordering and all eight high orders. */
struct Fixture {
  DeviceBatch batch{};
  std::vector<std::int64_t> ao_offsets{0}, pair_offsets;
  std::vector<std::uint8_t> ao_angular;
  std::vector<std::int32_t> first_shells, second_shells;
  std::vector<double> host_density, host_schwarz;
  std::vector<void*> owned;
  template <class T>
  T* upload(const std::vector<T>& values) {
    T* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), values.size() * sizeof(T)));
    owned.push_back(pointer);
    check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return pointer;
  }
  explicit Fixture(std::array<unsigned, 4> momenta, bool unrestricted, bool coincident) {
    std::vector<std::uint8_t> angular;
    std::vector<std::int32_t> ao_shells;
    for (const auto l : momenta) {
      const auto shell = static_cast<std::int32_t>(angular.size());
      angular.push_back(l);
      for (int x = l; x >= 0; --x)
        for (int y = int(l) - x; y >= 0; --y) {
          ao_shells.push_back(shell);
          ao_angular.push_back(x);
          ao_angular.push_back(y);
          ao_angular.push_back(l - x - y);
        }
      ao_offsets.push_back(ao_offsets.back() + (l + 1) * (l + 2) / 2);
    }
    const auto n = std::size_t(ao_offsets.back()), matrix = n * n;
    std::vector<double> positions{0.1, -0.2, 0.3, 0.8, 0.4, -0.7, -0.3, 0.9, 0.2, 0.5, -0.6, 1.1};
    std::vector<double> exponents, primitive_coefficients, coefficients(n);
    for (unsigned shell = 0; shell < 4; ++shell)
      for (unsigned p = 0; p < 2; ++p) {
        exponents.push_back(0.5 + 0.13 * shell + 0.2 * p);
        primitive_coefficients.push_back(p ? -0.31 : 0.7);
      }
    for (std::size_t i = 0; i < n; ++i) coefficients[i] = 0.7 + 0.03 * i;
    host_density.resize((unrestricted ? 2 : 1) * matrix);
    host_schwarz.resize(matrix);
    for (std::size_t i = 0; i < matrix; ++i) {
      host_density[i] = std::cos(0.31 * (i / n) + 0.17 * (i % n)) / n;
      if (unrestricted) host_density[matrix + i] = std::sin(0.27 * (i / n) - 0.23 * (i % n)) / n;
      host_schwarz[i] = 0.1 + 0.13 * ((i / n + i % n) % 11);
    }
    for (unsigned first = 0; first < 4; ++first)
      for (unsigned second = 0; second <= first; ++second) {
        first_shells.push_back(first);
        second_shells.push_back(second);
        pair_offsets.push_back(4 * (first_shells.size() - 1));
      }
    pair_offsets.push_back(40);
    batch.batch_size = 1;
    batch.nbf = batch.direct_nbf = n;
    batch.total_atoms = batch.total_shells = 4;
    batch.total_shell_pairs = 10;
    // Shell-density screening reconstructs cross-pair indices from these
    // system prefixes, including on the retained native stream route.
    batch.system_shell_offsets = upload<std::int64_t>({0, 4});
    batch.system_shell_pair_offsets = upload<std::int64_t>({0, 10});
    batch.positions = upload(positions);
    batch.shell_atoms = upload<std::int32_t>(coincident ? std::vector<std::int32_t>{0, 0, 0, 0}
                                                        : std::vector<std::int32_t>{0, 1, 2, 3});
    batch.shell_angular = upload(angular);
    batch.shell_direct_ao_offsets = upload(ao_offsets);
    batch.shell_primitive_offsets = upload<std::int64_t>({0, 2, 4, 6, 8});
    batch.shell_pair_first = upload(first_shells);
    batch.shell_pair_second = upload(second_shells);
    batch.shell_pair_systems = upload<std::int32_t>(std::vector<std::int32_t>(10, 0));
    batch.shell_pair_primitive_offsets = upload(pair_offsets);
    batch.direct_ao_angular = upload(ao_angular);
    // The retained production dispatcher looks up shell owners through this
    // map; the explicit-shell ERI oracle intentionally does not need it.
    batch.direct_ao_shells = upload(ao_shells);
    batch.direct_ao_coefficients = upload(coefficients);
    batch.primitive_exponents = upload(exponents);
    batch.primitive_coefficients = upload(primitive_coefficients);
    auto* pairs = upload(std::vector<PrimitivePairData>(40));
    batch.shell_primitive_pairs = pairs;
    build_shell_primitive_pair_cache_kernel<<<10, 32>>>(batch, pairs);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
  }
  ~Fixture() {
    for (auto* pointer : owned) cudaFree(pointer);
  }
};

template <unsigned A, unsigned B, unsigned C, unsigned D>
__global__ void retained_components(DeviceBatch batch, const ActiveShellQuartetTile* tasks,
                                    const double* schwarz, double threshold, double* values) {
  const auto task = tasks[blockIdx.x];
  const auto first_count = shell_ao_pair_count(batch, task.first_pair);
  const auto second_count = shell_ao_pair_count(batch, task.second_pair);
  const auto count = task.first_pair == task.second_pair ? first_count * (first_count + 1) / 2
                                                         : first_count * second_count;
  const auto ordinal = std::size_t(task.tile) * 256 + threadIdx.x;
  std::size_t i{}, j{}, k{}, l{};
  if (ordinal >= count ||
      !decode_direct_tile_ao_ordinal(batch, task, ordinal, first_count, second_count, 0,
                                     batch.direct_nbf, i, j, k, l) ||
      !direct_ao_quartet_survives_schwarz(schwarz, 0, batch.direct_nbf, i, j, k, l, threshold))
    return;
  values[ordinal] = contracted_eri_cartesian_source_shell_class<A, B, C, D, double>(
      batch, i, j, k, l, batch.shell_pair_first[task.first_pair],
      batch.shell_pair_second[task.first_pair], batch.shell_pair_first[task.second_pair],
      batch.shell_pair_second[task.second_pair], -1);
}

template <bool Unrestricted, unsigned Order, bool WholeShell = false>
__global__ void shared_components(DeviceBatch batch, const ActiveShellQuartetTile* tasks,
                                  const double* schwarz, double threshold, const double* density,
                                  const std::uint8_t* active, double* fock, unsigned channel,
                                  MaterializedDirectPairWork* work, double* values) {
  __shared__ MaterializedDirectPairRecurrence<Order> shared;
  // Forty register slots cover the largest through-f shell quartet. The DFT
  // dddd production stream needs only six, retaining the same admission math.
  contract_materialized_direct_pair_fock<Unrestricted, Order, WholeShell ? 40 : 1>(
      batch, tasks[blockIdx.x], threshold, schwarz, density, active, fock, nullptr, shared, work,
      channel == 1, channel == 2, values);
}

std::size_t triangle_row(std::size_t ordinal) {
  std::size_t row = 0;
  while ((row + 1) * (row + 2) / 2 <= ordinal) ++row;
  return row;
}
std::array<std::size_t, 2> host_pair(const Fixture& fixture, unsigned pair, std::size_t ordinal) {
  const auto first = fixture.first_shells[pair], second = fixture.second_shells[pair];
  if (first == second) {
    const auto row = triangle_row(ordinal);
    return {std::size_t(fixture.ao_offsets[first]) + row,
            std::size_t(fixture.ao_offsets[second]) + ordinal - row * (row + 1) / 2};
  }
  const auto second_count = fixture.ao_offsets[second + 1] - fixture.ao_offsets[second];
  return {std::size_t(fixture.ao_offsets[first]) + ordinal / second_count,
          std::size_t(fixture.ao_offsets[second]) + ordinal % second_count};
}
void close(double actual, double expected, const char* message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 3e-11 + 3e-11 * std::abs(expected))
    throw std::runtime_error(message);
}

template <unsigned A, unsigned B, unsigned C, unsigned D, bool Unrestricted,
          bool WholeShell = false>
void qualify(bool same_pair, bool coincident, double threshold) {
  constexpr auto order = A + B + C + D;
  Fixture fixture({D, C, B, A}, Unrestricted, coincident);
  const auto n = std::size_t(fixture.batch.direct_nbf), matrix = n * n;
  const unsigned first_pair = 8, second_pair = same_pair ? 8 : 1;
  const auto first_count = ((A + 1) * (A + 2) / 2) * ((B + 1) * (B + 2) / 2);
  const auto second_count = ((C + 1) * (C + 2) / 2) * ((D + 1) * (D + 2) / 2);
  const std::size_t count =
      same_pair ? first_count * (first_count + 1) / 2 : first_count * second_count;
  std::vector<ActiveShellQuartetTile> tasks;
  for (unsigned tile = 0; tile * 256 < count; ++tile)
    tasks.push_back({first_pair, second_pair, tile});
  Device<ActiveShellQuartetTile> dtasks(tasks);
  Device<double> density(fixture.host_density), schwarz(fixture.host_schwarz);
  Device<double> retained(count), materialized(count), fock((Unrestricted ? 2 : 1) * matrix);
  Device<std::uint8_t> active(std::vector<std::uint8_t>{1});
  Device<MaterializedDirectPairWork> work(1);
  retained_components<A, B, C, D>
      <<<tasks.size(), 256>>>(fixture.batch, dtasks.data, schwarz.data, threshold, retained.data);
  check(cudaGetLastError());
  const auto expected_values = retained.read();
  std::size_t admitted_count = 0, live_packets = 0;
  for (unsigned channel = 0; channel < 3; ++channel) {
    work.clear();
    materialized.clear();
    fock.clear();
    shared_components<Unrestricted, order, WholeShell><<<WholeShell ? 1 : tasks.size(), 256>>>(
        fixture.batch, dtasks.data, schwarz.data, threshold, density.data, active.data, fock.data,
        channel, work.data, materialized.data);
    check(cudaGetLastError());
    const auto values = materialized.read(), actual_fock = fock.read();
    std::vector<double> expected_fock(actual_fock.size(), 0.0);
    std::set<std::size_t> packets;
    admitted_count = 0;
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      // A different host decoding implementation controls admission and orbit.
      const auto p = same_pair ? triangle_row(ordinal) : ordinal / second_count;
      const auto q = same_pair ? ordinal - p * (p + 1) / 2 : ordinal % second_count;
      const auto ij = host_pair(fixture, first_pair, p), kl = host_pair(fixture, second_pair, q);
      const auto i = ij[0], j = ij[1], k = kl[0], l = kl[1];
      const bool admitted =
          fixture.host_schwarz[i + n * j] * fixture.host_schwarz[k + n * l] >= threshold;
      if (admitted) {
        ++admitted_count;
        packets.insert(ordinal / 256);
      }
      close(values[ordinal], expected_values[ordinal],
            "cached component differs from raw evaluator");
      const std::set<std::array<std::size_t, 4>> orbit{{i, j, k, l}, {j, i, k, l}, {i, j, l, k},
                                                       {j, i, l, k}, {k, l, i, j}, {l, k, i, j},
                                                       {k, l, j, i}, {l, k, j, i}};
      for (const auto& abcd : orbit) {
        const auto a = abcd[0], b = abcd[1], c = abcd[2], d = abcd[3];
        // CUDA dense views are column-major; do not reuse the device helper
        // in the independent host contraction.
        const auto total = fixture.host_density[c + n * d] +
                           (Unrestricted ? fixture.host_density[matrix + c + n * d] : 0.0);
        for (unsigned spin = 0; spin < (Unrestricted ? 2U : 1U); ++spin) {
          if (channel != 2)
            expected_fock[spin * matrix + a + n * b] += total * expected_values[ordinal];
          if (channel != 1)
            // A K-only consumer publishes positive K; the combined HF
            // consumer applies its restricted/unrestricted exchange factor.
            expected_fock[spin * matrix + a + n * c] +=
                (channel == 2 ? 1.0 : (Unrestricted ? -1.0 : -0.5)) *
                fixture.host_density[spin * matrix + b + n * d] * expected_values[ordinal];
        }
      }
    }
    live_packets = packets.size();
    for (std::size_t i = 0; i < actual_fock.size(); ++i)
      close(actual_fock[i], expected_fock[i],
            "shared J/K differs from independent orbit contraction");
    const auto counters = work.read()[0];
    const auto live_sources = WholeShell ? std::size_t(admitted_count != 0) : live_packets;
    if (counters.bra_preparations != 4 * live_sources ||
        counters.ket_preparations != 16 * live_sources ||
        counters.coulomb_preparations != 16 * live_sources ||
        counters.component_contractions != 16 * admitted_count ||
        counters.published_components != admitted_count)
      throw std::runtime_error("pair-materialized executed recurrence/work mismatch");
  }
  active.clear();
  work.clear();
  fock.clear();
  shared_components<Unrestricted, order, WholeShell><<<WholeShell ? 1 : tasks.size(), 256>>>(
      fixture.batch, dtasks.data, schwarz.data, threshold, density.data, active.data, fock.data, 0,
      work.data, materialized.data);
  check(cudaGetLastError());
  if (work.read()[0].coulomb_preparations)
    throw std::runtime_error("inactive system consumed recurrence");
  std::cout << "order=" << order << " uhf=" << Unrestricted << " same=" << same_pair
            << " whole_shell=" << WholeShell << " live_packets=" << live_packets
            << " components=" << admitted_count
            << " recurrence=" << 16 * (WholeShell ? std::size_t(admitted_count != 0) : live_packets)
            << '\n';
}
#ifdef GENERATIVEQC_PAIR_MATERIALIZED_DFT_STREAM_TEST
/** Exercise the real native launcher used by DFT J/K, including its fallbacks.
 * The ERIs and host symmetry orbit remain independent of stream scheduling. */
template <bool Unrestricted>
void qualify_dft_stream() {
  Fixture fixture({2, 2, 2, 2}, Unrestricted, false);
  const auto n = std::size_t(fixture.batch.direct_nbf), matrix = n * n;
  constexpr unsigned pair = 8, count = 36 * 37 / 2;
  std::vector<ActiveShellQuartetTile> tasks;
  for (unsigned tile = 0; tile * 256 < count; ++tile) tasks.push_back({pair, pair, tile});
  Device<ActiveShellQuartetTile> dtasks(tasks);
  Device<double> density(fixture.host_density), schwarz(fixture.host_schwarz), retained(count);
  Device<double> fock((Unrestricted ? 2 : 1) * matrix);
  Device<std::uint8_t> active(std::vector<std::uint8_t>{1});
  Device<std::uint32_t> cursor(1);
  Device<unsigned long long> census(1);
  Device<MaterializedDirectPairWork> work(1);
  retained_components<2, 2, 2, 2>
      <<<tasks.size(), 256>>>(fixture.batch, dtasks.data, schwarz.data, 0.0, retained.data);
  check(cudaGetLastError());
  const auto values = retained.read();
  Device<std::uint32_t> order(std::vector<std::uint32_t>{pair});
  std::vector<std::uint32_t> offsets(20, 0);
  offsets[11] = 1;  // One dd pair in system zero; other classes are unused.
  Device<std::uint32_t> class_offsets(offsets);
  Device<double> shell_bounds(std::vector<double>(10, 1.0));
  Device<generativeqc::scf::detail::GeneratedShellPairDensityBounds> density_bounds(
      std::vector<generativeqc::scf::detail::GeneratedShellPairDensityBounds>(10, {1.0, 1.0, 1.0}));
  GeneratedShellPairStream topology{};
  topology.batch_size = 1;
  topology.pair_order = order.data;
  topology.pair_class_offsets = class_offsets.data;
  topology.shell_pair_bounds = shell_bounds.data;
  topology.shell_pair_density_bounds = density_bounds.data;
  for (unsigned channel = 0; channel < 3; ++channel) {
    topology.fock_consumer = static_cast<generativeqc::scf::detail::GeneratedFockConsumer>(channel);
    Device<GeneratedShellPairStream> stream(std::vector<GeneratedShellPairStream>{topology});
    std::vector<double> expected((Unrestricted ? 2 : 1) * matrix, 0.0);
    for (unsigned ordinal = 0; ordinal < count; ++ordinal) {
      const auto p = triangle_row(ordinal), q = ordinal - p * (p + 1) / 2;
      const auto ij = host_pair(fixture, pair, p), kl = host_pair(fixture, pair, q);
      const auto i = ij[0], j = ij[1], k = kl[0], l = kl[1];
      const std::set<std::array<std::size_t, 4>> orbit{{i, j, k, l}, {j, i, k, l}, {i, j, l, k},
                                                       {j, i, l, k}, {k, l, i, j}, {l, k, i, j},
                                                       {k, l, j, i}, {l, k, j, i}};
      for (const auto& abcd : orbit) {
        const auto a = abcd[0], b = abcd[1], c = abcd[2], d = abcd[3];
        const auto total = fixture.host_density[c + n * d] +
                           (Unrestricted ? fixture.host_density[matrix + c + n * d] : 0.0);
        for (unsigned spin = 0; spin < (Unrestricted ? 2U : 1U); ++spin) {
          if (channel != 2) expected[spin * matrix + a + n * b] += total * values[ordinal];
          if (channel != 1)
            expected[spin * matrix + a + n * c] +=
                (channel == 2 ? 1.0 : (Unrestricted ? -1.0 : -0.5)) *
                fixture.host_density[spin * matrix + b + n * d] * values[ordinal];
        }
      }
    }
    for (unsigned route = 0; route < 6; ++route) {
      auto batch = fixture.batch;
      batch.direct_pair_materialized_values = route != 0;
      if (route == 2) batch.shell_primitive_pairs = nullptr;
      if (route == 3) batch.direct_coulomb_reachable = 1;
      if (route == 4) batch.direct_hermite_convolution = 1;
      if (route == 5) batch.shell_pair_primitive_offsets = nullptr;
      cursor.clear();
      census.clear();
      work.clear();
      fock.clear();
      std::cout << "DFT stream uhf=" << Unrestricted << " channel=" << channel << " route=" << route
                << '\n'
                << std::flush;
      launch_bounded_direct_dddd_streaming_kernel_scaled(
          Unrestricted, DirectScreeningPurpose::Fock, false, 3, 32, 0, nullptr, batch, stream.data,
          0.0, schwarz.data, density.data, active.data, fock.data, cursor.data, nullptr,
          census.data, 1.0, Unrestricted ? -1.0 : -0.5, work.data);
      check(cudaGetLastError());
      const auto actual = fock.read();
      for (std::size_t item = 0; item < actual.size(); ++item)
        close(actual[item], expected[item],
              "DFT stream differs from independent orbit contraction");
      if (census.read()[0] != 1) throw std::runtime_error("DFT stream shell admission changed");
      const auto counters = work.read()[0];
      if (counters.coulomb_preparations != (route == 1 ? 16U : 0U) ||
          counters.published_components != (route == 1 ? count : 0U))
        throw std::runtime_error("DFT stream did not execute its selected source");
    }
    for (unsigned empty = 0; empty < 2; ++empty) {
      auto batch = fixture.batch;
      batch.direct_pair_materialized_values = true;
      active.clear();
      if (empty) {
        const std::uint8_t enabled = 1;
        check(cudaMemcpy(active.data, &enabled, sizeof(enabled), cudaMemcpyHostToDevice));
      }
      cursor.clear();
      census.clear();
      work.clear();
      fock.clear();
      launch_bounded_direct_dddd_streaming_kernel_scaled(
          Unrestricted, DirectScreeningPurpose::Fock, false, 3, 32, 0, nullptr, batch, stream.data,
          empty ? 2.0 : 0.0, schwarz.data, density.data, active.data, fock.data, cursor.data,
          nullptr, census.data, 1.0, Unrestricted ? -1.0 : -0.5, work.data);
      check(cudaGetLastError());
      if (census.read()[0] || work.read()[0].coulomb_preparations)
        throw std::runtime_error("empty DFT stream consumed a source");
      for (const auto value : fock.read()) close(value, 0.0, "empty DFT stream published a matrix");
    }
    const std::uint8_t enabled = 1;
    check(cudaMemcpy(active.data, &enabled, sizeof(enabled), cudaMemcpyHostToDevice));
  }
  std::cout << "DFT native dddd stream, RHF/UHF J/K and bounded fallbacks PASS\n";
}
#endif

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::strcmp(argv[1], "--lifetime") == 0) {
      // Full-order numerical/memcheck/initcheck gates run independently.
      // Racecheck targets shared publication, retirement, multiple packets,
      // inactive lanes and empty claims without instrumenting the expensive
      // ffff scalar oracle's private recurrence at every component.
      qualify<2, 2, 1, 1, false, true>(false, false, 0.0);
      qualify<2, 2, 2, 2, true, true>(true, true, 0.8);
      qualify<2, 2, 2, 2, false, true>(false, false, 2.0);
      std::cout << "pair-materialized shared lifecycle PASS\n";
      return 0;
    }
    qualify<2, 1, 1, 1, false>(false, false, 0.0);
    qualify<2, 1, 1, 1, true>(false, true, 0.8);
    qualify<2, 2, 1, 1, false>(false, false, 0.0);
    qualify<2, 2, 2, 1, true>(false, false, 0.8);
    qualify<2, 2, 2, 2, false>(true, true, 0.0);
    qualify<3, 2, 2, 2, true>(false, false, 0.8);
    qualify<3, 3, 2, 2, false>(false, false, 0.0);
    qualify<3, 3, 3, 2, true>(false, false, 0.8);
    qualify<3, 3, 3, 3, false>(false, false, 0.0);
    std::cout << "pair-materialized full/tail, order5..12, RHF/UHF J/K and exact work PASS\n";
    qualify<2, 2, 2, 2, false, true>(false, false, 0.0);
    qualify<2, 2, 2, 2, true, true>(true, true, 0.8);
    qualify<2, 2, 2, 2, false, true>(false, false, 2.0);
    qualify<3, 3, 3, 3, true, true>(false, false, 0.8);
    std::cout << "pair-materialized whole-shell admission and exact work PASS\n";
#ifdef GENERATIVEQC_PAIR_MATERIALIZED_DFT_STREAM_TEST
    qualify_dft_stream<false>();
    qualify_dft_stream<true>();
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
