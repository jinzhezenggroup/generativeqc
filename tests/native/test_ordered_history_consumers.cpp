// Independent public/internal API probe for the method-independent production CPU Johnson mixer.
// The Python test supplies its own chronological Johnson oracle.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "solver/cpu/johnson_broyden.hpp"

namespace {
using namespace generativeqc::solver::cpu;
using generativeqc::solver::BroydenPolicy;
using generativeqc::solver::BroydenResult;
using generativeqc::solver::BroydenStatusEncoding;
constexpr std::size_t kTotal = 62;
constexpr std::array<std::int64_t, 4> kVectorOffsets{0, 10, 31, 62};
constexpr std::array<std::array<std::int64_t, 4>, 3> kFieldOffsets{
    {{0, 1, 4, 8}, {0, 3, 9, 18}, {0, 6, 18, 36}}};
constexpr double kCanary = 123456.75;
struct Aligned {
  void* p;
  explicit Aligned(std::size_t n) : p(nullptr) {
    // Preserve the 64-byte alignment and match POSIX allocation with free.
    if (posix_memalign(&p, 64, n ? n : 64) != 0) throw std::bad_alloc{};
    std::memset(p, 0, n);
  }
  ~Aligned() { std::free(p); }
  Aligned(const Aligned&) = delete;
  Aligned& operator=(const Aligned&) = delete;
};
struct Fixture {
  int memory;
  BroydenPlan plan;
  Aligned raw{576};
  std::unique_ptr<Aligned> state_memory, scratch_memory;
  BroydenVectorView vector;
  BroydenState state;
  BroydenWorkspace scratch;
  std::string error;

  explicit Fixture(int m) : memory(m) {
    BroydenVectorLayoutView layout;
    layout.batch_size = 3;
    layout.workspace_size_bytes = 576;
    layout.workspace_alignment = 64;
    layout.field_count = 3;
    const std::size_t starts[]{0, 64, 256};
    for (std::size_t f = 0; f < 3; ++f) {
      const auto count = kFieldOffsets[f].back();
      layout.fields[f] = {starts[f], static_cast<std::size_t>(count) * sizeof(double), count,
                          kFieldOffsets[f].data(), 4};
      vector.fields[f] = reinterpret_cast<double*>(static_cast<char*>(raw.p) + starts[f]);
    }
    vector.workspace_base = raw.p;
    vector.workspace_size_bytes = 576;
    vector.field_count = 3;
    check(make_broyden_plan(layout, BroydenPolicy{memory, .3, .04, .08},
                            BroydenStatusEncoding{0, 1, 6}, plan, error));
    state_memory = std::make_unique<Aligned>(plan.state_size_bytes());
    scratch_memory = std::make_unique<Aligned>(plan.workspace_size_bytes());
    check(bind_broyden_state(plan, state_memory->p, plan.state_size_bytes(), state, error));
    check(bind_broyden_workspace(plan, scratch_memory->p, plan.workspace_size_bytes(), scratch,
                                 error));
    std::array<double, kTotal> initial{};
    for (std::size_t s = 0; s < 3; ++s) {
      for (auto c = kVectorOffsets[s]; c < kVectorOffsets[s + 1]; ++c) {
        initial[c] = .01 * (s + 1) + .001 * (c - kVectorOffsets[s] + 1);
      }
    }
    put_raw(initial.data());
    check(initialize_broyden_state(plan, vector, state, error));
  }
  void check(BroydenResult result) {
    if (result != BroydenResult::success) throw std::runtime_error(error);
  }
  void put_raw(const double* packed) {
    for (std::size_t s = 0; s < 3; ++s) {
      std::size_t c = kVectorOffsets[s];
      for (std::size_t f = 0; f < 3; ++f) {
        for (auto i = kFieldOffsets[f][s]; i < kFieldOffsets[f][s + 1]; ++i)
          vector.fields[f][i] = packed[c++];
      }
    }
  }
  void get_raw(double* packed) const {
    for (std::size_t s = 0; s < 3; ++s) {
      std::size_t c = kVectorOffsets[s];
      for (std::size_t f = 0; f < 3; ++f) {
        for (auto i = kFieldOffsets[f][s]; i < kFieldOffsets[f][s + 1]; ++i)
          packed[c++] = vector.fields[f][i];
      }
    }
  }
  void poison_history(std::size_t s) {
    const auto old = state.iterations[s];
    const auto dimension = kVectorOffsets[s + 1] - kVectorOffsets[s];
    const auto newest = old ? (old - 1) % memory : 0;
    for (int slot = 0; slot < memory; ++slot) {
      if (old == 0 || static_cast<std::uint64_t>(slot) >= old ||
          static_cast<std::uint64_t>(slot) == newest) {
        const auto index = memory * kVectorOffsets[s] + slot * dimension;
        std::fill_n(state.df_history + index, dimension, std::numeric_limits<double>::quiet_NaN());
        std::fill_n(state.u_history + index, dimension, std::numeric_limits<double>::quiet_NaN());
        state.omega[s * memory + slot] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  int step(const double* packed, bool poison) {
    put_raw(packed);
    for (std::size_t s = 0; s < 3; ++s) {
      if (poison) poison_history(s);
      const auto old = state.iterations[s];
      const auto count = std::min<std::uint64_t>(memory, old);
      std::fill_n(scratch.beta, memory * memory, kCanary);
      std::fill_n(scratch.coefficients, memory, kCanary);
      check(mix_broyden_system(plan, s, vector, state, scratch, error));
      // CPU beta is compact in active history_count, not capacity-strided.
      for (std::size_t i = count * count; i < static_cast<std::size_t>(memory * memory); ++i)
        if (scratch.beta[i] != kCanary) throw std::runtime_error("CPU compact beta tail changed");
      for (std::size_t i = count; i < static_cast<std::size_t>(memory); ++i)
        if (scratch.coefficients[i] != kCanary)
          throw std::runtime_error("CPU coefficient tail changed");
      for (std::size_t i = 0; i < count; ++i) {
        const auto chronological_slot = (old - count + i) % memory;
        if (scratch.history_slots[i] != static_cast<std::int64_t>(chronological_slot))
          throw std::runtime_error("CPU chronological slot sequence changed");
      }
    }
    return 0;
  }
  int fault(int kind) {
    if ((kind == 1 || kind == 2 || kind == 5) && memory < 2)
      throw std::runtime_error("fault needs retained history");
    if (kind == 6 && memory < 4) throw std::runtime_error("singular fault needs two retained rows");
    const auto bytes = plan.state_size_bytes();
    std::vector<unsigned char> original(bytes), expected(bytes), raw_before(576);
    std::memcpy(original.data(), state.workspace_base, bytes);
    std::array<double, kTotal> input{};
    for (std::size_t s = 0; s < 3; ++s)
      for (auto c = kVectorOffsets[s]; c < kVectorOffsets[s + 1]; ++c)
        input[c] = state.current_inputs[c] + .03 * (c - kVectorOffsets[s] + 1);
    const auto newest = (state.iterations[0] - 1) % memory;
    const auto retained = (newest + 1) % memory;
    if (kind == 0)
      input[0] = std::numeric_limits<double>::quiet_NaN();
    else if (kind == 1)
      state.df_history[retained * 10] = std::numeric_limits<double>::quiet_NaN();
    else if (kind == 2)
      state.u_history[retained * 10] = std::numeric_limits<double>::infinity();
    else if (kind == 3)
      state.iterations[0] = std::numeric_limits<std::uint64_t>::max();
    else if (kind == 4)
      state.previous_residuals[0] = 1e308;
    else if (kind == 5)
      state.omega[retained] = 1e308;
    else if (kind == 6) {
      for (const auto slot : {retained, (retained + 1) % memory}) {
        std::fill_n(state.df_history + slot * 10, 10, 0.0);
        state.df_history[slot * 10] = 1.0;
        state.omega[slot] = std::ldexp(1.0, 500);
      }
    } else
      throw std::runtime_error("unknown fault");
    put_raw(input.data());
    std::memcpy(expected.data(), state.workspace_base, bytes);
    std::memcpy(raw_before.data(), raw.p, 576);
    const auto previous_status = state.system_statuses[0];
    const auto result = mix_broyden_system(plan, 0, vector, state, scratch, error);
    const auto failure_text = error;
    const char* expected_error[] = {"residual contains",    "coefficient is not finite",
                                    "result is not finite", "iteration counter",
                                    "residual difference",  "matrix overflowed",
                                    "system is not usable"};
    const bool failed = failure_text.find(expected_error[kind]) != std::string::npos &&
                        result == BroydenResult::numerical_failure &&
                        state.system_statuses[0] == 6 && !failure_text.empty();
    state.system_statuses[0] = previous_status;
    const bool retained_bytes = std::memcmp(expected.data(), state.workspace_base, bytes) == 0 &&
                                std::memcmp(raw_before.data(), raw.p, 576) == 0;
    std::memcpy(state.workspace_base, original.data(), bytes);
    if (!failed || !retained_bytes)
      throw std::runtime_error(
          "CPU failure changed persistent history, raw values, peers or diagnostics: " +
          failure_text);
    error = failure_text;
    return 0;
  }
};
}  // namespace

namespace {
thread_local std::string construction_error;
}
extern "C" int consumer_available() { return 1; }
extern "C" void* consumer_create(int memory, int /* graph */) {
  try {
    return new Fixture(memory);
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
  copy(f.state.current_inputs, kTotal);
  copy(f.state.previous_inputs, kTotal);
  copy(f.state.previous_residuals, kTotal);
  copy(f.state.df_history, kTotal * f.memory);
  copy(f.state.u_history, kTotal * f.memory);
  copy(f.state.omega, 3 * f.memory);
  copy(f.state.residual_rms, 3);
  copy(f.state.residual_maximum, 3);
  for (int s = 0; s < 3; ++s) {
    metadata[s] = f.state.iterations[s];
    metadata[3 + s] = f.state.restart_counts[s];
    metadata[6 + s] = f.state.system_statuses[s];
    metadata[9 + s] = f.state.initialized[s];
    metadata[12 + s] = f.state.converged[s];
  }
  return 0;
}
