/** Standalone phase-A LR source qualification; no SCF, native lease or selector.
 * Input/orbit references are prepared independently with pinned Cartesian Libcint.
 * This executable must be compiled and run only in a separately authorized job.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated_direct_pair_cache.cuh"
#include "generated_direct_source_contraction.cuh"

using namespace generativeqc::scf::cuda_execution;
using Range = generativeqc::integrals::CoulombRange;
constexpr double poison = -9.25e245;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class T>
struct Device {
  T* data{};
  std::size_t count;
  explicit Device(std::size_t n) : count(n) {
    check(cudaMalloc(reinterpret_cast<void**>(&data), std::max(n, std::size_t{1}) * sizeof(T)));
  }
  explicit Device(const std::vector<T>& values) : Device(values.size()) {
    if (count) check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data); }
  Device(const Device&) = delete;
  std::vector<T> read() const {
    std::vector<T> values(count);
    if (count) check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};
struct Task {
  ActiveShellQuartetTile tile;
  std::size_t components;
};
struct Case {
  std::string name;
  std::array<unsigned, 4> angular;
  double omega, threshold;
  std::size_t values, task_count;
  std::vector<std::int64_t> offsets, primitive_offsets;
  std::vector<double> positions, coefficients, exponents, weights, density, expected, raw_k;
  std::vector<unsigned> powers, visits;
  std::vector<double> bounds;
  std::vector<std::int32_t> canonical;
  std::vector<Task> tasks;
  std::vector<std::uint16_t> maps;
};
template <class T>
std::vector<T> read_vector(std::istream& input) {
  std::size_t count{};
  input >> count;
  require(count <= 4096, "fixture vector capacity refused");
  std::vector<T> values(count);
  for (auto& value : values) input >> value;
  return values;
}
unsigned pair_id(unsigned first, unsigned second) {
  if (first < second) std::swap(first, second);
  return first * (first + 1) / 2 + second;
}
Case read_case(std::istream& input) {
  Case c;
  input >> c.name;
  for (auto& value : c.angular) input >> value;
  input >> c.omega >> c.threshold >> c.values >> c.task_count;
  require(c.values <= 192 && c.task_count <= 3, "fixture extent refused");
  c.offsets = read_vector<std::int64_t>(input);
  c.primitive_offsets = read_vector<std::int64_t>(input);
  c.positions = read_vector<double>(input);
  c.powers = read_vector<unsigned>(input);
  c.coefficients = read_vector<double>(input);
  c.exponents = read_vector<double>(input);
  c.weights = read_vector<double>(input);
  c.density = read_vector<double>(input);
  c.expected = read_vector<double>(input);
  c.visits = read_vector<unsigned>(input);
  c.raw_k = read_vector<double>(input);
  require(c.offsets.size() == 5 && c.primitive_offsets.size() == 5 && c.positions.size() == 12,
          "fixture shell metadata malformed");
  const auto n = std::size_t(c.offsets.back());
  require(n && n <= 24 && c.density.size() == n * n && c.raw_k.size() == n * n &&
              c.coefficients.size() == n && c.powers.size() == 3 * n &&
              c.exponents.size() == c.weights.size() &&
              c.weights.size() == std::size_t(c.primitive_offsets.back()) &&
              c.expected.size() == c.values && c.visits.size() == c.values,
          "fixture buffers malformed");
  c.bounds.assign(n * n, 0.0);
  std::size_t count{};
  input >> count;
  require(count <= 192, "bound metadata capacity refused");
  for (std::size_t entry = 0; entry < count; ++entry) {
    unsigned i{}, j{};
    double value{};
    input >> i >> j >> value;
    require(i < n && j < n && std::isfinite(value) && value >= 0, "invalid Schwarz fixture");
    c.bounds[i * n + j] = c.bounds[j * n + i] = value;
  }
  input >> count;
  require(count == c.values, "canonical inventory mismatch");
  c.canonical.resize(4 * count);
  for (auto& value : c.canonical) {
    input >> value;
    require(value >= 0 && value < n, "invalid AO");
  }
  const auto order = c.angular[0] + c.angular[1] + c.angular[2] + c.angular[3];
  require(order >= 3 && order <= 5, "unsupported angular order");
  for (auto value : c.angular) require(value <= 2, "unsupported shell angular momentum");
  const auto capacity = 32 * canonical_materialized_component_slots(order);
  for (std::size_t task = 0; task < c.task_count; ++task) {
    unsigned a{}, b{}, d{}, e{};
    std::size_t components{}, map_count{};
    input >> a >> b >> d >> e >> components >> map_count;
    require(a < 4 && b <= a && d < 4 && e <= d && components <= capacity && map_count == capacity,
            "invalid bounded task metadata");
    c.tasks.push_back({{pair_id(a, b), pair_id(d, e), 0}, components});
    for (std::size_t component = 0; component < capacity; ++component) {
      unsigned value{};
      input >> value;
      require(value == MaterializedDirectPairValueSink::rejected || value < c.values,
              "invalid canonical map target");
      c.maps.push_back(static_cast<std::uint16_t>(value));
    }
  }
  require(bool(input), "truncated standalone input");
  return c;
}
struct Inputs {
  DeviceBatch batch{};
  std::vector<void*> owned;
  template <class T>
  T* upload(const std::vector<T>& values) {
    T* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer),
                     std::max(values.size(), std::size_t{1}) * sizeof(T)));
    owned.push_back(pointer);
    if (!values.empty())
      check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return pointer;
  }
  explicit Inputs(const Case& c, cudaStream_t stream) {
    const auto n = std::size_t(c.offsets.back());
    batch.batch_size = 1;
    batch.nbf = batch.direct_nbf = n;
    batch.total_atoms = batch.total_shells = 4;
    batch.total_shell_pairs = 10;
    batch.system_shell_offsets = upload<std::int64_t>({0, 4});
    batch.system_shell_pair_offsets = upload<std::int64_t>({0, 10});
    batch.positions = upload(c.positions);
    batch.shell_atoms = upload<std::int32_t>({0, 1, 2, 3});
    batch.shell_angular = upload(std::vector<std::uint8_t>(c.angular.begin(), c.angular.end()));
    batch.shell_direct_ao_offsets = upload(c.offsets);
    batch.shell_primitive_offsets = upload(c.primitive_offsets);
    batch.direct_ao_angular = upload(std::vector<std::uint8_t>(c.powers.begin(), c.powers.end()));
    std::vector<std::int32_t> ao_shells(n), first, second;
    for (unsigned shell = 0; shell < 4; ++shell)
      for (auto ao = c.offsets[shell]; ao < c.offsets[shell + 1]; ++ao) ao_shells[ao] = shell;
    batch.direct_ao_shells = upload(ao_shells);
    batch.direct_ao_coefficients = upload(c.coefficients);
    batch.primitive_exponents = upload(c.exponents);
    batch.primitive_coefficients = upload(c.weights);
    std::vector<std::int64_t> pair_offsets{0};
    for (unsigned a = 0; a < 4; ++a)
      for (unsigned b = 0; b <= a; ++b) {
        first.push_back(a);
        second.push_back(b);
        pair_offsets.push_back(pair_offsets.back() +
                               (c.primitive_offsets[a + 1] - c.primitive_offsets[a]) *
                                   (c.primitive_offsets[b + 1] - c.primitive_offsets[b]));
      }
    batch.shell_pair_first = upload(first);
    batch.shell_pair_second = upload(second);
    batch.shell_pair_systems = upload<std::int32_t>(std::vector<std::int32_t>(10, 0));
    batch.shell_pair_primitive_offsets = upload(pair_offsets);
    auto* cache = upload(std::vector<PrimitivePairData>(pair_offsets.back()));
    batch.shell_primitive_pairs = cache;
    build_shell_primitive_pair_cache_kernel<<<10, 32, 0, stream>>>(batch, cache);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
  }
  ~Inputs() {
    for (auto pointer : owned) cudaFree(pointer);
  }
};
__global__ void reset_outputs(double* values, double* scalar, unsigned* visits, std::size_t count,
                              double* exchange, std::size_t matrix,
                              MaterializedDirectPairWork* work, unsigned* refusal) {
  for (auto i = threadIdx.x; i < count + 2; i += blockDim.x) {
    values[i] = scalar[i] = poison;
    visits[i] = 0;
  }
  for (auto i = threadIdx.x; i < matrix; i += blockDim.x) exchange[i] = 0;
  if (!threadIdx.x) {
    *work = {};
    *refusal = 0;
  }
}
template <unsigned Order>
__global__ void candidate_values(DeviceBatch batch, const Task* tasks, const std::uint16_t* maps,
                                 std::size_t count, const double* bounds, double threshold,
                                 double omega, double* values, unsigned* visits, unsigned* refusal,
                                 MaterializedDirectPairWork* work, Range range = Range::Long,
                                 unsigned malformed = 0) {
  __shared__ MaterializedDirectPairRecurrence<Order, double> shared;
  constexpr auto capacity = 32 * canonical_materialized_component_slots(Order);
  const auto task = tasks[blockIdx.x];
  MaterializedDirectPairValueSink sink{maps + blockIdx.x * capacity,
                                       capacity,
                                       task.components,
                                       count,
                                       values + 1,
                                       visits + 1,
                                       refusal};
  if (malformed == 1) --sink.map_count;
  if (malformed == 2) ++sink.component_count;
  if (malformed == 3) sink.value_count = 193;
  // The wrapper has no density/Fock arguments, and internally binds both null.
  produce_materialized_direct_pair_values<Order>(batch, task.tile, threshold, bounds, nullptr,
                                                 shared, sink, omega, work, range);
}
template <unsigned Order>
__global__ void incumbent_values(DeviceBatch batch, const std::int32_t* indices,
                                 const unsigned* expected_visits, std::size_t count, double omega,
                                 double* values) {
  const auto slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= count || !expected_visits[slot]) return;
  const auto* q = indices + 4 * slot;
  const auto shell_class =
      direct_quartet_shell_class_device(batch.shell_angular[batch.direct_ao_shells[q[0]]],
                                        batch.shell_angular[batch.direct_ao_shells[q[1]]],
                                        batch.shell_angular[batch.direct_ao_shells[q[2]]],
                                        batch.shell_angular[batch.direct_ao_shells[q[3]]]);
  values[slot + 1] = dispatch_contracted_eri_cartesian_source_shell_class<Order, double>(
      shell_class, batch, 0, q[0], q[1], q[2], q[3], -1, Range::Long, omega);
}
template <unsigned Order>
__global__ void incumbent_fock(DeviceBatch batch, const Task* tasks, const double* bounds,
                               double threshold, double omega, const double* density,
                               double* exchange) {
  __shared__ MaterializedDirectPairRecurrence<Order, double> shared;
  contract_materialized_direct_pair_fock<false, Order,
                                         canonical_materialized_component_slots(Order), 32>(
      batch, tasks[blockIdx.x].tile, threshold, bounds, density, nullptr, nullptr, nullptr, shared,
      nullptr, false, false, nullptr, false, Range::Long, omega, nullptr, exchange);
}
void close(double actual, double expected, double atol, double rtol, const char* message) {
  require(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <= atol + rtol * std::abs(expected),
          message);
}
std::vector<double> orbit_exchange(const Case& c, const std::vector<double>& values) {
  const auto n = std::size_t(c.offsets.back());
  std::vector<double> result(n * n);
  for (std::size_t slot = 0; slot < c.values; ++slot) {
    if (!c.visits[slot]) continue;
    const auto* q = c.canonical.data() + 4 * slot;
    const auto i = q[0], j = q[1], k = q[2], l = q[3];
    const std::set<std::array<int, 4>> orbit{{i, j, k, l}, {j, i, k, l}, {i, j, l, k},
                                             {j, i, l, k}, {k, l, i, j}, {l, k, i, j},
                                             {k, l, j, i}, {l, k, j, i}};
    for (const auto& p : orbit)
      result[p[0] * n + p[2]] += values[slot + 1] * c.density[p[1] * n + p[3]];
  }
  return result;
}
template <unsigned Order>
void resources() {
  cudaFuncAttributes candidate{}, scalar{}, fock{};
  check(cudaFuncGetAttributes(&candidate, candidate_values<Order>));
  check(cudaFuncGetAttributes(&scalar, incumbent_values<Order>));
  check(cudaFuncGetAttributes(&fock, incumbent_fock<Order>));
  require(candidate.sharedSizeBytes <= (16U << 10), "candidate static shared limit exceeded");
  for (const auto& item :
       {std::pair{"candidate", candidate}, std::pair{"scalar", scalar}, std::pair{"fock", fock}})
    std::cout << "resource order=" << Order << " symbol=" << item.first
              << " registers=" << item.second.numRegs
              << " local_bytes=" << item.second.localSizeBytes
              << " shared_bytes=" << item.second.sharedSizeBytes
              << " max_threads=" << item.second.maxThreadsPerBlock << '\n';
}
template <unsigned Order>
void qualify(const Case& c, cudaStream_t stream, bool raw_diagnostics) {
  Inputs input(c, stream);
  Device<Task> tasks(c.tasks);
  Device<std::uint16_t> maps(c.maps);
  Device<double> bounds(c.bounds), density(c.density);
  Device<std::int32_t> indices(c.canonical);
  Device<unsigned> expected_visits(c.visits);
  Device<double> values(c.values + 2), scalar(c.values + 2), exchange(c.raw_k.size());
  Device<unsigned> visits(c.values + 2), refusal(1);
  Device<MaterializedDirectPairWork> work(1);
  const auto launch = [&] {
    reset_outputs<<<1, 256, 0, stream>>>(values.data, scalar.data, visits.data, c.values,
                                         exchange.data, c.raw_k.size(), work.data, refusal.data);
    candidate_values<Order><<<c.task_count, 32, 0, stream>>>(
        input.batch, tasks.data, maps.data, c.values, bounds.data, c.threshold, c.omega,
        values.data, visits.data, refusal.data, work.data);
    if (c.values)
      incumbent_values<Order><<<(c.values + 255) / 256, 256, 0, stream>>>(
          input.batch, indices.data, expected_visits.data, c.values, c.omega, scalar.data);
    incumbent_fock<Order><<<c.task_count, 32, 0, stream>>>(
        input.batch, tasks.data, bounds.data, c.threshold, c.omega, density.data, exchange.data);
  };
  unsigned execution = 0;
  const auto validate = [&] {
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    const auto actual = values.read(), retained = scalar.read();
    const auto hits = visits.read();
    const auto from_candidate = orbit_exchange(c, actual),
               from_scalar = orbit_exchange(c, retained), fock = exchange.read();
    const auto observed = work.read()[0];
    const auto dump = [&] {
      std::cout << std::setprecision(17) << "raw_case={\"name\":\"" << c.name
                << "\",\"execution\":" << execution;
      const auto vector = [&](const char* name, const auto& data) {
        std::cout << ",\"" << name << "\":[";
        for (std::size_t index = 0; index < data.size(); ++index) {
          if (index) std::cout << ',';
          if (std::isfinite(static_cast<double>(data[index])))
            std::cout << data[index];
          else
            std::cout << "\"nonfinite\"";
        }
        std::cout << ']';
      };
      vector("candidate_with_guards", actual);
      vector("scalar_with_guards", retained);
      vector("reference", c.expected);
      vector("visits_with_guards", hits);
      vector("expected_visits", c.visits);
      vector("K_reference", c.raw_k);
      vector("K_candidate", from_candidate);
      vector("K_scalar", from_scalar);
      vector("K_fock", fock);
      std::cout << ",\"actual_work\":[" << observed.bra_preparations << ','
                << observed.ket_preparations << ',' << observed.coulomb_preparations << ','
                << observed.component_contractions << ',' << observed.published_components << "]}"
                << std::endl;
    };
    if (raw_diagnostics && execution == 0) dump();
    try {
      require(!refusal.read()[0], "candidate refused admitted fixture");
      require(actual.front() == poison && actual.back() == poison && retained.front() == poison &&
                  retained.back() == poison && !hits.front() && !hits.back(),
              "output guard modified");
      for (std::size_t i = 0; i < c.values; ++i) {
        require(hits[i + 1] == c.visits[i], "one-write/holes inventory failed");
        if (!c.visits[i]) {
          require(actual[i + 1] == poison && retained[i + 1] == poison, "screened hole modified");
          continue;
        }
        for (auto value : {actual[i + 1], retained[i + 1]}) {
          close(value, c.expected[i], 2e-12, 2e-10, "Libcint emitted-value gate failed");
          close(value, c.expected[i], 3e-11, 3e-11, "Libcint complete-component gate failed");
        }
        close(actual[i + 1], retained[i + 1], 3e-11, 3e-11, "candidate/scalar parity failed");
      }
      for (std::size_t i = 0; i < c.raw_k.size(); ++i) {
        close(from_candidate[i], c.raw_k[i], 3e-11, 3e-11, "candidate full raw K failed");
        close(from_scalar[i], c.raw_k[i], 3e-11, 3e-11, "scalar full raw K failed");
        close(fock[i], c.raw_k[i], 3e-11, 3e-11, "ordinary Fock sink full raw K failed");
      }
      MaterializedDirectPairWork expected{};
      constexpr auto capacity = 32 * canonical_materialized_component_slots(Order);
      for (std::size_t t = 0; t < c.tasks.size(); ++t) {
        const auto live = std::count_if(
            c.maps.begin() + t * capacity, c.maps.begin() + (t + 1) * capacity,
            [](auto target) { return target != MaterializedDirectPairValueSink::rejected; });
        if (!live) continue;
        const auto& tile = c.tasks[t].tile;
        auto primitive_pairs = [&](unsigned p) {
          unsigned a = 0;
          while ((a + 1) * (a + 2) / 2 <= p) ++a;
          const auto b = p - a * (a + 1) / 2;
          return (c.primitive_offsets[a + 1] - c.primitive_offsets[a]) *
                 (c.primitive_offsets[b + 1] - c.primitive_offsets[b]);
        };
        const auto bra = primitive_pairs(tile.first_pair), ket = primitive_pairs(tile.second_pair);
        expected.bra_preparations += bra;
        expected.ket_preparations += bra * ket;
        expected.coulomb_preparations += bra * ket;
        expected.component_contractions += live * bra * ket;
        expected.published_components += live;
      }
      require(observed.bra_preparations == expected.bra_preparations &&
                  observed.ket_preparations == expected.ket_preparations &&
                  observed.coulomb_preparations == expected.coulomb_preparations &&
                  observed.component_contractions == expected.component_contractions &&
                  observed.published_components == expected.published_components,
              "actual recurrence-work census failed");
      double max_value_error = 0, max_emitted_scaled = 0, max_complete_scaled = 0, max_k_error = 0;
      for (std::size_t i = 0; i < c.values; ++i)
        if (c.visits[i]) {
          for (auto value : {actual[i + 1], retained[i + 1]}) {
            const auto error = std::abs(value - c.expected[i]);
            max_value_error = std::max(max_value_error, error);
            max_emitted_scaled =
                std::max(max_emitted_scaled, error / (2e-12 + 2e-10 * std::abs(c.expected[i])));
            max_complete_scaled =
                std::max(max_complete_scaled, error / (3e-11 + 3e-11 * std::abs(c.expected[i])));
          }
        }
      for (std::size_t i = 0; i < c.raw_k.size(); ++i)
        for (auto value : {from_candidate[i], from_scalar[i], fock[i]})
          max_k_error = std::max(max_k_error, std::abs(value - c.raw_k[i]));
      std::cout << std::setprecision(17) << "execution name=" << c.name << " index=" << execution
                << " value_max_abs=" << max_value_error
                << " emitted_gate_max_fraction=" << max_emitted_scaled
                << " complete_gate_max_fraction=" << max_complete_scaled
                << " rawK_max_abs=" << max_k_error << " bra=" << observed.bra_preparations
                << " ket=" << observed.ket_preparations
                << " radial=" << observed.coulomb_preparations
                << " contractions=" << observed.component_contractions
                << " publications=" << observed.published_components << " PASS" << std::endl;
    } catch (...) {
      if (!raw_diagnostics || execution != 0) dump();
      throw;
    }
    ++execution;
  };
  launch();
  validate();
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch();
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (unsigned replay = 0; replay < 2; ++replay) {
    check(cudaGraphLaunch(executable, stream));
    validate();
  }
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
}
template <unsigned Order>
void qualify_refusals(const Case& c, cudaStream_t stream) {
  Inputs input(c, stream);
  Device<Task> tasks(c.tasks);
  Device<double> bounds(c.bounds), values(c.values + 2), scalar(c.values + 2),
      exchange(c.raw_k.size());
  Device<unsigned> visits(c.values + 2), refusal(1);
  Device<MaterializedDirectPairWork> work(1);
  constexpr auto capacity = 32 * canonical_materialized_component_slots(Order);
  for (unsigned fault = 0; fault < 8; ++fault) {
    auto map = c.maps;
    if (fault == 0) map[1] = map[0];                                // alias
    if (fault == 1) map[0] = static_cast<std::uint16_t>(c.values);  // out of bounds
    if (fault == 2) {
      require(c.tasks[0].components < capacity, "refusal fixture needs a tail");
      map[c.tasks[0].components] = 0;  // missing tail sentinel
    }
    Device<std::uint16_t> maps(map);
    reset_outputs<<<1, 256, 0, stream>>>(values.data, scalar.data, visits.data, c.values,
                                         exchange.data, c.raw_k.size(), work.data, refusal.data);
    candidate_values<Order><<<1, 32, 0, stream>>>(
        input.batch, tasks.data, maps.data, c.values, bounds.data, c.threshold,
        fault == 7 ? -1.0 : c.omega, values.data, visits.data, refusal.data, work.data,
        fault == 6 ? Range::Full : Range::Long, fault >= 3 && fault <= 5 ? fault - 2 : 0);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    require(refusal.read()[0] != 0, "malformed source request did not refuse");
    const auto after = values.read();
    const auto hits = visits.read();
    require(std::all_of(after.begin(), after.end(), [](auto v) { return v == poison; }) &&
                std::all_of(hits.begin(), hits.end(), [](auto v) { return v == 0; }),
            "refusal published a partial value");
    const auto observed = work.read()[0];
    require(!observed.bra_preparations && !observed.ket_preparations &&
                !observed.coulomb_preparations && !observed.component_contractions &&
                !observed.published_components,
            "refusal performed source work");
  }
  std::cout << "source metadata/envelope refusal before publication PASS\n";
}
int main(int argc, char** argv) {
  try {
    require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--summary-only"),
            "usage: materialized_lr_values <fixtures.txt> [--summary-only]");
    const bool raw_diagnostics = argc == 2;
    std::ifstream input(argv[1]);
    std::string magic;
    std::size_t count{};
    input >> magic >> count;
    require(magic == "GQCLR1" && count && count <= 10000, "invalid fixture header");
    resources<3>();
    resources<4>();
    resources<5>();
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (std::size_t i = 0; i < count; ++i) {
      const auto c = read_case(input);
      const auto order = c.angular[0] + c.angular[1] + c.angular[2] + c.angular[3];
      if (i == 0) qualify_refusals<3>(c, stream);
      if (order == 3)
        qualify<3>(c, stream, raw_diagnostics);
      else if (order == 4)
        qualify<4>(c, stream, raw_diagnostics);
      else
        qualify<5>(c, stream, raw_diagnostics);
      std::cout << "case=" << i << " name=" << c.name
                << " PASS ordinary+graph2 Libcint emitted+complete rawK work\n";
    }
    check(cudaStreamDestroy(stream));
    std::cout << "bounded LR canonical values PASS cases=" << count
              << "; fixture only, no native integration/performance claim\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
