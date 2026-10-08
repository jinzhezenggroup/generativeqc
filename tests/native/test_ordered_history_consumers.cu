// Real CUDA SCC-mixer API fixture. Independent expected arithmetic lives in
// test_ordered_history_consumers.py, not in this fixture or generated helpers.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_scc_mixer.cuh"

namespace {
using namespace generativeqc::xtb::detail::cuda;
constexpr std::size_t kTotal = 62;
constexpr std::uint64_t kToken = 0x713;
constexpr std::array<std::int64_t, 4> kVectorOffsets{0, 10, 31, 62};
constexpr std::array<std::array<std::int64_t, 4>, 3> kFieldOffsets{
    {{0, 1, 4, 8}, {0, 3, 9, 18}, {0, 6, 18, 36}}};
constexpr double kCanary = 123456.75;
void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <class T>
struct Device {
  T* p = nullptr;
  std::size_t count;
  explicit Device(std::size_t n) : count(n) {
    checked(cudaMallocManaged(reinterpret_cast<void**>(&p), n * sizeof(T)));
    std::fill_n(p, n, T{});
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() {
    if (p) cudaFree(p);
  }
};
struct Fixture {
  int memory;
  bool graph_mode;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Device<std::int64_t> shell_offsets{4}, atom_offsets{4};
  Device<double> q{8}, dipoles{18}, quadrupoles{36};
  Device<double> current{kTotal}, previous{kTotal}, previous_residual{kTotal};
  Device<double> df, u, weights;
  Device<double> rms{3}, maximum{3};
  Device<std::uint64_t> iterations{3}, restarts{3};
  Device<generativeqc_xtb_status_t> statuses{3};
  Device<std::uint8_t> initialized{3}, converged{3}, active{3};
  Device<double> residual{kTotal}, mixed{kTotal}, delta_f{kTotal}, new_u{kTotal};
  Device<double> beta, coefficients;
  Device<std::uint32_t> stage_sequence{1}, canonical_sequence{1}, device_error{1};
  Gfn2SccDeviceBatch batch;
  Gfn2SccDeviceConstMultipoles raw;
  Gfn2SccDeviceMultipoles output;
  Gfn2SccMixerDevicePolicy policy;
  Gfn2SccMixerDeviceState state;
  Gfn2SccMixerDeviceWorkspace workspace;
  Gfn2SccIterationDeviceActivity activity;
  std::string error;

  Fixture(int m, bool capture)
      : memory(m),
        graph_mode(capture),
        df(kTotal * m),
        u(kTotal * m),
        weights(3 * m),
        beta(3 * m * m),
        coefficients(3 * m) {
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::copy(kFieldOffsets[0].begin(), kFieldOffsets[0].end(), shell_offsets.p);
    const std::int64_t atoms[]{0, 1, 3, 6};
    std::copy_n(atoms, 4, atom_offsets.p);
    batch = {3, 8, 6, 4, 4, kToken, shell_offsets.p, atom_offsets.p};
    raw = {q.p, 8, dipoles.p, 18, quadrupoles.p, 36, kToken};
    output = {q.p, 8, dipoles.p, 18, quadrupoles.p, 36, kToken};
    policy = {memory, .3, .04, .08, kToken};
    state.current_inputs = current.p;
    state.previous_inputs = previous.p;
    state.previous_residuals = previous_residual.p;
    state.df_history = df.p;
    state.u_history = u.p;
    state.omega = weights.p;
    state.residual_rms = rms.p;
    state.residual_maximum = maximum.p;
    state.iterations = iterations.p;
    state.restart_counts = restarts.p;
    state.system_statuses = statuses.p;
    state.initialized = initialized.p;
    state.residual_converged = converged.p;
    state.total_vector_elements = kTotal;
    state.history_elements = kTotal * m;
    state.omega_elements = 3 * m;
    state.batch_elements = 3;
    state.plan_token = kToken;
    workspace.residual = residual.p;
    workspace.mixed = mixed.p;
    workspace.delta_f = delta_f.p;
    workspace.new_u = new_u.p;
    workspace.beta = beta.p;
    workspace.coefficients = coefficients.p;
    workspace.sequence_active = stage_sequence.p;
    workspace.vector_elements = kTotal;
    workspace.beta_elements = 3 * m * m;
    workspace.coefficient_elements = 3 * m;
    workspace.sequence_elements = 1;
    workspace.plan_token = kToken;
    std::fill_n(active.p, 3, std::uint8_t{1});
    canonical_sequence.p[0] = 1;
    activity = {active.p, canonical_sequence.p, 3, 1, kToken};
    std::array<double, kTotal> initial{};
    for (std::size_t s = 0; s < 3; ++s)
      for (auto c = kVectorOffsets[s]; c < kVectorOffsets[s + 1]; ++c)
        initial[c] = .01 * (s + 1) + .001 * (c - kVectorOffsets[s] + 1);
    put_raw(initial.data());
    checked(initialize_gfn2_scc_mixer_cuda(batch, policy, raw, state, workspace, device_error.p,
                                           stream));
    checked(cudaStreamSynchronize(stream));
    if (device_error.p[0] != 0)
      throw std::runtime_error("CUDA initialization reported device error");
    if (graph_mode) {
      checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      checked(mix_gfn2_scc_broyden_cuda(batch, policy, activity, raw, output, state, workspace,
                                        device_error.p, stream));
      checked(cudaStreamEndCapture(stream, &graph));
      checked(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
  }
  ~Fixture() {
    if (stream) cudaStreamSynchronize(stream);
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    if (stream) cudaStreamDestroy(stream);
  }
  void put_raw(const double* packed) {
    const std::array<double*, 3> fields{q.p, dipoles.p, quadrupoles.p};
    for (std::size_t s = 0; s < 3; ++s) {
      auto c = kVectorOffsets[s];
      for (std::size_t f = 0; f < 3; ++f)
        for (auto i = kFieldOffsets[f][s]; i < kFieldOffsets[f][s + 1]; ++i)
          fields[f][i] = packed[c++];
    }
  }
  void get_raw(double* packed) const {
    const std::array<const double*, 3> fields{q.p, dipoles.p, quadrupoles.p};
    for (std::size_t s = 0; s < 3; ++s) {
      auto c = kVectorOffsets[s];
      for (std::size_t f = 0; f < 3; ++f)
        for (auto i = kFieldOffsets[f][s]; i < kFieldOffsets[f][s + 1]; ++i)
          packed[c++] = fields[f][i];
    }
  }
  void launch() {
    if (graph_mode)
      checked(cudaGraphLaunch(executable, stream));
    else
      checked(mix_gfn2_scc_broyden_cuda(batch, policy, activity, raw, output, state, workspace,
                                        device_error.p, stream));
    checked(cudaStreamSynchronize(stream));
  }
  void poison_history(std::size_t s) {
    const auto old = iterations.p[s];
    const auto dimension = kVectorOffsets[s + 1] - kVectorOffsets[s];
    const auto newest = old ? (old - 1) % memory : 0;
    for (int slot = 0; slot < memory; ++slot) {
      if (old == 0 || static_cast<std::uint64_t>(slot) >= old ||
          static_cast<std::uint64_t>(slot) == newest) {
        const auto index = memory * kVectorOffsets[s] + slot * dimension;
        std::fill_n(df.p + index, dimension, std::numeric_limits<double>::quiet_NaN());
        std::fill_n(u.p + index, dimension, std::numeric_limits<double>::quiet_NaN());
        weights.p[s * memory + slot] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  int step(const double* packed, bool poison) {
    put_raw(packed);
    std::fill_n(active.p, 3, std::uint8_t{1});
    std::array<std::uint64_t, 3> old{};
    for (std::size_t s = 0; s < 3; ++s) {
      old[s] = iterations.p[s];
      if (poison) poison_history(s);
    }
    std::fill_n(beta.p, beta.count, kCanary);
    std::fill_n(coefficients.p, coefficients.count, kCanary);
    const auto sticky_before = device_error.p[0];
    launch();
    if (device_error.p[0] != sticky_before)
      throw std::runtime_error("CUDA transition reported device error " +
                               std::to_string(device_error.p[0]));
    for (std::size_t s = 0; s < 3; ++s) {
      if (statuses.p[s] != GENERATIVEQC_XTB_STATUS_SUCCESS || iterations.p[s] != old[s] + 1)
        throw std::runtime_error("CUDA transition did not commit every healthy member");
      const auto count = std::min<std::uint64_t>(memory, old[s]);
      // CUDA keeps capacity leading dimension even for partial histories.
      for (int row = 0; row < memory; ++row)
        for (int column = 0; column < memory; ++column)
          if ((static_cast<std::uint64_t>(row) >= count ||
               static_cast<std::uint64_t>(column) >= count) &&
              beta.p[s * memory * memory + row * memory + column] != kCanary)
            throw std::runtime_error("CUDA padded beta canary changed");
      for (auto i = count; i < static_cast<std::uint64_t>(memory); ++i)
        if (coefficients.p[s * memory + i] != kCanary)
          throw std::runtime_error("CUDA inactive coefficient tail changed");
    }
    return 0;
  }
  template <class Function>
  void each_persistent(Function&& fn) {
    fn(current);
    fn(previous);
    fn(previous_residual);
    fn(df);
    fn(u);
    fn(weights);
    fn(rms);
    fn(maximum);
    fn(iterations);
    fn(restarts);
    fn(statuses);
    fn(initialized);
    fn(converged);
  }
  std::vector<unsigned char> snapshot() {
    std::vector<unsigned char> result;
    each_persistent([&](auto& array) {
      const auto* bytes = reinterpret_cast<const unsigned char*>(array.p);
      result.insert(result.end(), bytes, bytes + array.count * sizeof(*array.p));
    });
    return result;
  }
  void restore(const std::vector<unsigned char>& saved) {
    std::size_t cursor = 0;
    each_persistent([&](auto& array) {
      const auto bytes = array.count * sizeof(*array.p);
      std::memcpy(array.p, saved.data() + cursor, bytes);
      cursor += bytes;
    });
  }
  int fault(int kind) {
    if ((kind == 1 || kind == 2 || kind == 5) && memory < 2)
      throw std::runtime_error("fault needs retained history");
    if (kind == 6 && memory < 4) throw std::runtime_error("singular fault needs two retained rows");
    const auto original = snapshot();
    std::array<double, kTotal> input{}, raw_after{};
    for (std::size_t s = 0; s < 3; ++s)
      for (auto c = kVectorOffsets[s]; c < kVectorOffsets[s + 1]; ++c)
        input[c] = current.p[c] + .03 * (c - kVectorOffsets[s] + 1);
    const auto newest = (iterations.p[0] - 1) % memory;
    const auto retained = (newest + 1) % memory;
    if (kind == 0)
      input[0] = std::numeric_limits<double>::quiet_NaN();
    else if (kind == 1)
      df.p[retained * 10] = std::numeric_limits<double>::quiet_NaN();
    else if (kind == 2)
      u.p[retained * 10] = std::numeric_limits<double>::infinity();
    else if (kind == 3)
      iterations.p[0] = std::numeric_limits<std::uint64_t>::max();
    else if (kind == 4)
      previous_residual.p[0] = 1e308;
    else if (kind == 5)
      weights.p[retained] = 1e308;
    else if (kind == 6) {
      for (const auto slot : {retained, (retained + 1) % memory}) {
        std::fill_n(df.p + slot * 10, 10, 0.0);
        df.p[slot * 10] = 1.0;
        weights.p[slot] = std::ldexp(1.0, 500);
      }
    } else
      throw std::runtime_error("unknown fault");
    put_raw(input.data());
    active.p[0] = 1;
    active.p[1] = active.p[2] = 0;
    const auto expected = snapshot();
    const auto previous_status = statuses.p[0];
    const auto sticky_before = device_error.p[0];
    launch();
    const bool failed = statuses.p[0] == GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR &&
                        device_error.p[0] != 0 &&
                        (sticky_before == 0 || device_error.p[0] == sticky_before);
    statuses.p[0] = previous_status;
    get_raw(raw_after.data());
    const bool retained_bytes =
        snapshot() == expected && std::memcmp(input.data(), raw_after.data(), sizeof(input)) == 0;
    restore(original);
    std::fill_n(active.p, 3, std::uint8_t{1});
    if (!failed || !retained_bytes)
      throw std::runtime_error(
          "CUDA failure changed persistent history, raw values, peers or diagnostics");
    error = "CUDA semantic failure retained; sticky code " + std::to_string(device_error.p[0]);
    return 0;
  }
};
}  // namespace

namespace {
thread_local std::string construction_error;
}
extern "C" int consumer_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
  cudaDeviceProp properties{};
  if (cudaSetDevice(0) != cudaSuccess || cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
    return 0;
  return properties.managedMemory ? 1 : 0;
}
extern "C" void* consumer_create(int memory, int graph) {
  try {
    return new Fixture(memory, graph != 0);
  } catch (const std::exception& e) {
    construction_error = e.what();
    return nullptr;
  }
}
extern "C" void consumer_destroy(void* handle) { delete static_cast<Fixture*>(handle); }
extern "C" const char* consumer_error(void* handle) {
  return handle ? static_cast<Fixture*>(handle)->error.c_str() : construction_error.c_str();
}
extern "C" int consumer_step(void* handle, const double* raw, int poison) {
  auto& f = *static_cast<Fixture*>(handle);
  try {
    return f.step(raw, poison != 0);
  } catch (const std::exception& e) {
    f.error = e.what();
    return 1;
  }
}
extern "C" int consumer_fault(void* handle, int kind) {
  auto& f = *static_cast<Fixture*>(handle);
  try {
    return f.fault(kind);
  } catch (const std::exception& e) {
    f.error = e.what();
    return 1;
  }
}
extern "C" int consumer_copy(void* handle, double* out, std::uint64_t* metadata) {
  const auto& f = *static_cast<Fixture*>(handle);
  f.get_raw(out);
  out += kTotal;
  auto copy = [&](const double* values, std::size_t count) {
    std::copy_n(values, count, out);
    out += count;
  };
  copy(f.current.p, kTotal);
  copy(f.previous.p, kTotal);
  copy(f.previous_residual.p, kTotal);
  copy(f.df.p, kTotal * f.memory);
  copy(f.u.p, kTotal * f.memory);
  copy(f.weights.p, 3 * f.memory);
  copy(f.rms.p, 3);
  copy(f.maximum.p, 3);
  for (int s = 0; s < 3; ++s) {
    metadata[s] = f.iterations.p[s];
    metadata[3 + s] = f.restarts.p[s];
    metadata[6 + s] = f.statuses.p[s];
    metadata[9 + s] = f.initialized.p[s];
    metadata[12 + s] = f.converged.p[s];
  }
  return 0;
}
