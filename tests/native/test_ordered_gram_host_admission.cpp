/** Exercise the real public launcher through a process-local CUDA API recorder.
 * No kernel executes: positive submission checks make a missing ELF interposer
 * fail closed. The Python driver supplies independent integer byte oracles. */
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "scf/cuda/scf_diis_kernels.hpp"

// Model the HF consumer's import using only test-owned names.
namespace generativeqc::scf::detail {
struct AdmissionHeaderClientTag {};
}  // namespace generativeqc::scf::detail
namespace generativeqc::scf {
namespace {
using namespace cuda_execution;
using HeaderClientTag = detail::AdmissionHeaderClientTag;
}  // namespace
}  // namespace generativeqc::scf

namespace {
struct Invocation {
  dim3 grid, block;
  std::size_t shared;
  cudaStream_t source_stream;
};

const Invocation* pending{};
unsigned submissions{};
const char* submission_error{};
constexpr std::size_t shared_bytes = 37;
cudaStream_t stream() { return reinterpret_cast<cudaStream_t>(std::uintptr_t{24576}); }
generativeqc::tensor::RingGramWork* work() {
  return reinterpret_cast<generativeqc::tensor::RingGramWork*>(std::uintptr_t{28672});
}

bool same(dim3 left, dim3 right) {
  return left.x == right.x && left.y == right.y && left.z == right.z;
}

void record(const void* kernel, dim3 grid, dim3 block, std::size_t shared,
            cudaStream_t actual_stream) {
  ++submissions;
  if (!pending || !kernel) {
    submission_error = "unexpected CUDA submission";
    return;
  }
  const auto& expected = *pending;
  // The private kernel argument packet is intentionally opaque. Consistent
  // producer parameter reordering must not change this public admission gate.
  if (!same(grid, expected.grid) || !same(block, expected.block) || shared != expected.shared ||
      actual_stream != expected.source_stream)
    submission_error = "public CUDA launch configuration changed";
}

void submit(dim3 grid, dim3 block, std::int32_t batch, std::int32_t nbf, std::int32_t spins,
            std::uint32_t history, const double* fock, const double* residual,
            const std::uint8_t* active, double* fock_history, double* residual_history,
            double* solve, double* coefficients, std::uint32_t* count, std::uint32_t* head,
            double* effective, double* cache, bool normalize = false,
            generativeqc::tensor::RingGramWork* work_record = nullptr, std::size_t shared = 0,
            cudaStream_t source_stream = nullptr) {
  const Invocation request{grid, block, shared, source_stream};
  const auto before = submissions;
  pending = &request;
  submission_error = nullptr;
  try {
    if (!normalize && !work_record)
      generativeqc::scf::cuda_execution::launch_update_diis_cached_gram(
          grid, block, shared, source_stream, batch, nbf, spins, history, fock, residual, active,
          fock_history, residual_history, solve, coefficients, count, head, effective, cache);
    else
      generativeqc::scf::cuda_execution::launch_update_diis_cached_gram(
          grid, block, shared, source_stream, batch, nbf, spins, history, fock, residual, active,
          fock_history, residual_history, solve, coefficients, count, head, effective, cache,
          normalize, work_record);
  } catch (...) {
    pending = nullptr;
    if (submissions != before) throw std::runtime_error("rejected binding submitted CUDA work");
    throw;
  }
  pending = nullptr;
  if (submissions != before + 1)
    throw std::runtime_error("admitted public launcher missed CUDA API interception");
  if (submission_error) throw std::runtime_error(submission_error);
}

int probe() {
  double fock{}, residual{}, fock_history{}, residual_history{}, solve{}, rhs{}, effective{},
      cache{};
  std::uint8_t active = 1;
  std::uint32_t count{}, head{};
  auto call = [&](dim3 grid, dim3 block, int batch, int n, int spins, unsigned history, bool alias,
                  bool missing_cache, bool missing_history) {
    submit(grid, block, batch, n, spins, history, &fock, &residual, &active,
           missing_history ? nullptr : &fock_history, &residual_history, &solve, &rhs, &count,
           &head, &effective,
           missing_cache ? nullptr
           : alias       ? &solve
                         : &cache);
  };
  unsigned refused = 0;
  auto reject = [&](auto action) {
    const auto before = submissions;
    bool rejected = false;
    try {
      action();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected || submissions != before)
      throw std::runtime_error("missing pre-submission invalid_argument rejection");
    ++refused;
  };
  // Preserve the original thirteen rejection scenarios and three admissions.
  reject([&] { call(1, 32, 1, 3, 1, 4, true, false, false); });
  reject([&] { call(1, 32, 1, 3, 1, 4, false, true, false); });
  reject([&] { call(1, 32, 1, 3, 1, 4, false, false, true); });
  reject([&] { call(dim3(1, 2), 32, 1, 3, 1, 4, false, false, false); });
  reject([&] { call(dim3(1, 1, 2), 32, 1, 3, 1, 4, false, false, false); });
  reject([&] { call(0, 32, 1, 3, 1, 4, false, false, false); });
  reject([&] { call(1, 16, 1, 3, 1, 4, false, false, false); });
  reject([&] { call(1, dim3(32, 2), 1, 3, 1, 4, false, false, false); });
  reject([&] { call(1, 32, 0, 3, 1, 4, false, false, false); });
  reject([&] { call(1, 32, 1, 0, 1, 4, false, false, false); });
  reject([&] { call(1, 32, 1, 3, 3, 4, false, false, false); });
  reject([&] { call(1, 32, 1, 3, 1, 65, false, false, false); });
  reject([&] { call(INT_MAX, 32, INT_MAX, INT_MAX, 2, 64, false, false, false); });
  if (refused != 13 || submissions) throw std::runtime_error("missing original rejection");
  for (unsigned history : {0U, 1U})
    submit(1, 32, 1, 3, 1, history, &fock, nullptr, &active, nullptr, nullptr, nullptr, nullptr,
           nullptr, nullptr, &effective, nullptr);
  call(1, 32, 1, 3, 1, 64, false, false, false);
  const auto initial_admissions = submissions;
  if (initial_admissions != 3) throw std::runtime_error("missing original admission");

  auto binding = [&](unsigned history, unsigned missing) {
    submit(1, 32, 1, 3, 1, history, missing == 0 ? nullptr : &fock,
           missing == 1 ? nullptr : &residual, missing == 2 ? nullptr : &active,
           missing == 3 ? nullptr : &fock_history, missing == 4 ? nullptr : &residual_history,
           missing == 5 ? nullptr : &solve, missing == 6 ? nullptr : &rhs,
           missing == 7 ? nullptr : &count, missing == 8 ? nullptr : &head,
           missing == 9 ? nullptr : &effective, missing == 10 ? nullptr : &cache);
  };
  for (unsigned missing = 0; missing < 11; ++missing) {
    reject([&] { binding(2, missing); });
    for (unsigned history : {0U, 1U}) {
      if (missing == 0 || missing == 2 || missing == 9)
        reject([&] { binding(history, missing); });
      else
        binding(history, missing);
    }
  }
  reject([&] { call(1, dim3(32, 1, 2), 1, 3, 1, 4, false, false, false); });
  reject([&] { call(1, 32, -1, 3, 1, 4, false, false, false); });
  reject([&] { call(1, 32, 1, -1, 1, 4, false, false, false); });
  for (int spins : {-1, 0}) reject([&] { call(1, 32, 1, 3, spins, 4, false, false, false); });
  reject([&] { call(1, 32, 2, 3, 1, 4, false, false, false); });
  reject([&] { call(1, 32, 1, 3, 1, UINT_MAX, false, false, false); });
  if (refused != 37) throw std::runtime_error("missing extended rejection");

  fock = residual = fock_history = residual_history = solve = rhs = effective = cache =
      std::numeric_limits<double>::quiet_NaN();
  active = 0;
  count = head = UINT_MAX;
  call(2, 32, 1, 3, 2, 2, false, false, false);
  if (!std::isnan(fock) || !std::isnan(residual) || !std::isnan(fock_history) ||
      !std::isnan(residual_history) || !std::isnan(solve) || !std::isnan(rhs) ||
      !std::isnan(effective) || !std::isnan(cache) || active || count != UINT_MAX ||
      head != UINT_MAX)
    throw std::runtime_error("host admission touched borrowed state");
  auto* device = reinterpret_cast<double*>(std::uintptr_t{4096});
  submit(1, 32, 1, 3, 1, 2, device, device, reinterpret_cast<std::uint8_t*>(std::uintptr_t{8192}),
         device, device, device, device, reinterpret_cast<std::uint32_t*>(std::uintptr_t{12288}),
         reinterpret_cast<std::uint32_t*>(std::uintptr_t{16384}), device,
         reinterpret_cast<double*>(std::uintptr_t{20480}));
  for (unsigned history : {0U, 1U}) call(1, 32, 1, 3, 1, history, true, false, false);
  for (bool normalize : {false, true})
    submit(1, 32, 1, 3, 1, 2, &fock, &residual, &active, &fock_history, &residual_history, &solve,
           &rhs, &count, &head, &effective, &cache, normalize, work(), shared_bytes, stream());

  unsigned checked = 0, history;
  int batch, n, spins, expected;
  while (std::cin >> batch >> n >> spins >> history >> expected) {
    bool admitted = true;
    try {
      call(static_cast<unsigned>(batch), 32, batch, n, spins, history, false, false, false);
    } catch (const std::invalid_argument&) {
      admitted = false;
    }
    if (admitted != bool(expected)) throw std::runtime_error("independent byte oracle mismatch");
    ++checked;
  }
  std::cout << "{\"maximum_bytes\":" << std::numeric_limits<std::size_t>::max()
            << ",\"scalar_bytes\":" << sizeof(double) << ",\"fixed_rejections\":" << refused
            << ",\"initial_admissions\":" << initial_admissions
            << ",\"checked_boundaries\":" << checked << "}\n";
  return 0;
}
}  // namespace

// This symbol is exported by the test executable, so calls originating in the
// linked production shared library are intercepted as well as test objects.
extern "C" __attribute__((visibility("default"))) cudaError_t CUDARTAPI
cudaLaunchKernel(const void* kernel, dim3 grid, dim3 block, void**, std::size_t shared,
                 cudaStream_t actual_stream) {
  record(kernel, grid, block, shared, actual_stream);
  return cudaSuccess;
}

int main() {
  try {
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visible || *visible)
      throw std::runtime_error("host admission recorder requires CUDA_VISIBLE_DEVICES=\"\"");
    return probe();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
