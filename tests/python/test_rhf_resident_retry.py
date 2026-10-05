"""Bounded host tests for resident-ERI resource failure and exact retry.

The resource inventory/preparation, Owner allocation sequence, automatic policy,
and public retry controller are extracted from production. CUDA allocation,
recurrence, stream and later derivative work are host doubles. These tests do
not execute physical integrals, initialize CUDA, or qualify GPU numerics.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, start: str) -> str:
    begin = source.index(start)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_resident_refusal_preserves_complete_exact_frame(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    direct = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    owner = (ROOT / "src/hf/rhf_frame_response.cu").read_text()
    header = (ROOT / "src/hf/rhf_frame_response.hpp").read_text()
    declarations = [
        "struct DirectJkFailure",
        "void direct_jk_check(",
        "void direct_jk_require(",
        "std::size_t direct_jk_product(",
        "std::size_t direct_jk_sum(",
        "std::size_t canonical_bucket_values(",
        "struct DirectJkDownloadFence",
        "template <class Function>\ngenerativeqc_status direct_jk_guard(",
        "std::size_t cuda_direct_jk_resident_value_bytes(",
        "generativeqc_status prepare_cuda_direct_jk_resident_values(",
    ]
    definitions = [
        _definition(direct, item) + (";" if item.startswith("struct") else "")
        for item in declarations
    ]
    begin = owner.index("    if (resident_budget) {")
    end = owner.index("    state.o = o;", begin)
    allocation = owner[begin:end]
    policy = _definition(header, "std::size_t resident_jk_allowance(")
    wrapper = _definition(owner, "RHFFrameResponseResult rhf_frame_response_cuda(\n")
    probe = tmp_path / "resident_retry.cpp"
    probe.write_text(
        PREFIX
        + "\n\n".join(definitions)
        + ATTEMPT_PREFIX.replace("RESOURCE_POLICY", policy)
        + allocation
        + ATTEMPT_SUFFIX
        + wrapper
        + MAIN
    )
    executable = tmp_path / "resident_retry"
    compile_owner(compiler, tmp_path, [probe], executable)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "10 retry-controller checks passed" in result.stdout


PREFIX = r"""

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorLaunchFailure };
enum generativeqc_status {
  GENERATIVEQC_STATUS_SUCCESS,
  GENERATIVEQC_STATUS_OUT_OF_MEMORY,
  GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
  GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
  GENERATIVEQC_STATUS_CUDA_ERROR,
  GENERATIVEQC_STATUS_INVALID_ARGUMENT
};
constexpr int cudaMemcpyDeviceToHost = 0;
enum class DirectCoulombRange { Full };
struct State {
  cudaError_t allocation{}, pending{}, sync{}, copy{};
  int current{};
  bool nan{}, publication_fail{};
  int allocations{}, frees{}, launches{}, syncs{};
  size_t limit{SIZE_MAX}, live{};
} state;
std::unordered_map<void*, size_t> sizes;
const char* cudaGetErrorString(cudaError_t e) {
  return e == cudaErrorMemoryAllocation ? "allocation" : "execution";
}
generativeqc_status source_cuda_status(cudaError_t e) {
  return e == cudaSuccess                 ? GENERATIVEQC_STATUS_SUCCESS
         : e == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                          : GENERATIVEQC_STATUS_CUDA_ERROR;
}
cudaError_t cudaGetDevice(int* d) {
  *d = state.current;
  return cudaSuccess;
}
cudaError_t cudaGetLastError() {
  auto e = state.pending;
  state.pending = cudaSuccess;
  return e;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++state.syncs;
  return state.sync;
}
cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) {
  std::memset(p, v, n);
  return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t) {
  if (state.copy == cudaSuccess) std::memcpy(d, s, n);
  return state.copy;
}
namespace generativeqc {
namespace runtime {
bool checked_multiply(size_t a, size_t b, size_t& o) {
  if (b && a > SIZE_MAX / b) return false;
  o = a * b;
  return true;
}
bool checked_add(size_t a, size_t b, size_t& o) {
  if (a > SIZE_MAX - b) return false;
  o = a + b;
  return true;
}
cudaError_t resource_cuda_malloc(void** o, size_t n) {
  if (state.allocation != cudaSuccess) return state.allocation;
  if (n > state.limit - state.live) return cudaErrorMemoryAllocation;
  *o = ::operator new(n);
  sizes[*o] = n;
  state.live += n;
  ++state.allocations;
  return cudaSuccess;
}
cudaError_t resource_cuda_free(void* p) {
  state.live -= sizes.at(p);
  sizes.erase(p);
  ::operator delete(p);
  ++state.frees;
  return cudaSuccess;
}
}  // namespace runtime
}  // namespace generativeqc
namespace runtime = generativeqc::runtime;
struct Slots : std::vector<void*> {
  void push_back(void* p) {
    if (state.publication_fail) throw std::bad_alloc();
    std::vector<void*>::push_back(p);
  }
};
struct Diagnostic {
  size_t batch_size{1}, device_bytes{16}, resident_value_count{}, resident_value_bytes{},
      host_bytes{}, host_preparation_bytes{};
};
struct CudaDirectJkPlan {
  const int* canonical_pairs{reinterpret_cast<const int*>(1)};
  double screening_tolerance{};
  std::vector<std::array<size_t, 8>> canonical_pair_offsets{{{0, 2, 2, 2, 2, 2, 2, 2}}};
  int device_id{};
  double* resident_values{};
  size_t resident_value_count{}, device_bytes{16};
  Diagnostic diagnostic;
  cudaStream_t stream{reinterpret_cast<void*>(1)};
  int numerical_value{};
  int* numerical_failure{&numerical_value};
  int canonical_batch{}, canonical_cartesian{};
  double* canonical_bounds{};
  Slots allocations;
  ~CudaDirectJkPlan() {
    for (auto p : allocations) runtime::resource_cuda_free(p);
  }
};
int direct_jk_pair_rows(CudaDirectJkPlan*, unsigned, size_t) { return 0; }
void launch_canonical_jk_kernel(cudaStream_t, int, int, int, unsigned, const int*, int, size_t,
                                size_t n, size_t, size_t m, bool same, bool, bool, bool,
                                DirectCoulombRange, double, double, const double*, const double*,
                                double*, double*, std::uint64_t*, double* target) {
  size_t count = same ? n * (n + 1) / 2 : n * m;
  for (size_t i = 0; i < count; ++i)
    target[i] = state.nan ? std::numeric_limits<double>::quiet_NaN() : double(i + 1);
  if (count) ++state.launches;
}
void launch_independent_jk_finite_kernel(cudaStream_t, const double* v, size_t n, int* e) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(v[i])) *e = 1;
}
"""

ATTEMPT_PREFIX = r"""

#include <chrono>
#include <memory>
#include <optional>
#include <span>
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
void status(generativeqc_status c, const std::string& d) {
  if (c == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (c != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(d);
}
void cuda_resource_check(cudaError_t e) {
  if (e == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (e != cudaSuccess) throw std::runtime_error("CUDA execution fault");
}
size_t checked_add(size_t a, size_t b) {
  size_t c;
  if (!runtime::checked_add(a, b, c)) throw std::bad_alloc();
  return c;
}
size_t checked_mul(size_t a, size_t b) {
  size_t c;
  if (!runtime::checked_multiply(a, b, c)) throw std::bad_alloc();
  return c;
}
size_t bytes(size_t n) { return checked_mul(n, sizeof(double)); }
namespace scf {
using ::prepare_cuda_direct_jk_resident_values;
Diagnostic cuda_direct_jk_plan_diagnostic(CudaDirectJkPlan* p) { return p->diagnostic; }
bool cuda_direct_jk_linear_available(CudaDirectJkPlan* p) { return p->canonical_pairs; }
}
template <class T>
struct Buffer {
  T* p{};
  size_t n{};
  void allocate(int, size_t count, cudaStream_t) {
    if (runtime::resource_cuda_malloc((void**)&p, count * sizeof(T)) != cudaSuccess)
      throw std::bad_alloc();
    n = count;
  }
  size_t size() { return n; }
  ~Buffer() {
    if (p) runtime::resource_cuda_free(p);
  }
};
struct Event {
  void create(int) {}
};
struct RHFFrameResponseOptions {
  std::optional<size_t> resident_jk_maximum_bytes;
  double orbital_screening_tolerance{};
  bool relax_orbitals{true};
  RESOURCE_POLICY
};
extern int failure_mode;
struct Reason {
  void operator=(const std::string&) {
    if (failure_mode == 8) throw std::bad_alloc();
  }
};
struct Stats {
  double resident_jk_setup_seconds{}, applied_screening{}, requested_screening{};
  bool linear_screening_available{}, jk_timing_measured{};
  Reason resident_jk_reason;
  size_t resident_jk_bytes{}, resident_jk_values{}, direct_device_bytes{},
      numeric_capacity_bytes{1ULL << 30}, owned_device_bytes{};
};
namespace core {
struct System {};
}
struct PhysicalReference {};
int destroyed_preconditioners{}, attempts{}, exact_attempts{}, failure_mode{};
struct RHFFrameDFPreconditioner {
  ~RHFFrameDFPreconditioner() { ++destroyed_preconditioners; }
};
struct RHFFrameResponseResult {
  size_t numeric_capacity_bytes{};
  bool resident_jk_discarded_attempt{};
  double resident_jk_retry_seconds{};
  std::string resident_jk_reason;
  bool diagonal{};
};
constexpr size_t resident = 2080ULL * 2081 / 2 * 8, mandatory = 458768;
RHFFrameResponseResult rhf_frame_response_cuda_attempt(
    const core::System&, const PhysicalReference&, std::span<const double>, std::span<const double>,
    int, const RHFFrameResponseOptions& options,
    std::unique_ptr<RHFFrameDFPreconditioner> preconditioner, bool& resident_values_prepared,
    size_t& attempted_capacity) {
  ++attempts;
  const auto resident_budget = options.resident_jk_allowance(64, 512ULL << 20);
  attempted_capacity = resident_budget + mandatory;
  if (!resident_budget) ++exact_attempts;
  if (failure_mode == 6 && !resident_budget) throw std::bad_alloc();
  if (failure_mode == 7) throw std::bad_alloc();
  auto direct = std::make_unique<CudaDirectJkPlan>();
  direct->canonical_pair_offsets = {{{0, 2080, 2080, 2080, 2080, 2080, 2080, 2080}}};
  Stats stats;
  std::string detail;
  int device = 0;
  bool profile = false;
  auto stream = direct->stream;
  size_t nn = 64 * 64, arena_elements = 0;
  Buffer<double> storage;
  Buffer<int> error;
  Buffer<std::uint64_t> census;
  Event jk_start, jk_stop;
"""

ATTEMPT_SUFFIX = r"""

if (failure_mode == 2 || failure_mode == 6) {
  Buffer<double> later_derivative;
  later_derivative.allocate(0, 256, stream);
}
if (failure_mode == 3) throw std::runtime_error("numerical failure");
if (failure_mode == 4) throw std::runtime_error("CUDA execution failure");
if (failure_mode == 5) {
  state.pending = cudaErrorLaunchFailure;
  throw std::bad_alloc();
}
RHFFrameResponseResult result;
result.numeric_capacity_bytes = mandatory;
result.diagonal = !preconditioner;
return result;
}
"""

MAIN = r"""

void check(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
int main() {
  int passed = 0;
  auto reset = [&] {
    state = {};
    check(sizes.empty(), "live registry from prior test");
    destroyed_preconditioners = attempts = exact_attempts = failure_mode = 0;
  };
  auto call = [&](bool with_preconditioner = true) {
    return rhf_frame_response_cuda(
        core::System{}, PhysicalReference{}, {}, {}, 0, RHFFrameResponseOptions{},
        with_preconditioner ? std::make_unique<RHFFrameDFPreconditioner>() : nullptr);
  };
  auto test = [&](const char* name, auto f) {
    reset();
    f();
    check(!state.live && sizes.empty(), "leaked buffer");
    ++passed;
    std::cout << "PASS " << name << "\n";
  };
  test("cache fits but constructor mandatory owner does not", [&] {
    state.limit = resident + 4096;
    auto r = call();
    check(attempts == 2 && exact_attempts == 1 && r.diagonal && r.resident_jk_retry_seconds > 0 &&
              r.resident_jk_discarded_attempt &&
              r.numeric_capacity_bytes == (512ULL << 20) + mandatory &&
              destroyed_preconditioners == 1,
          "missing exact retry");
  });
  test("later derivative temporary crowds out after owner success", [&] {
    state.limit = resident + mandatory + 1024;
    failure_mode = 2;
    auto r = call();
    check(attempts == 2 && exact_attempts == 1 && r.diagonal && r.resident_jk_retry_seconds > 0,
          "later phase not retried");
  });
  test("post-publication diagnostic allocation failure retries", [&] {
    failure_mode = 8;
    auto r = call();
    check(attempts == 2 && exact_attempts == 1 && r.diagonal && r.resident_jk_discarded_attempt,
          "post-publication failure not recognized");
  });
  test("successful resident attempt preserved", [&] {
    auto r = call();
    check(attempts == 1 && exact_attempts == 0 && !r.diagonal && r.resident_jk_retry_seconds == 0,
          "unneeded fallback");
  });
  test("cache allocation refusal remains ordinary exact single attempt", [&] {
    state.limit = mandatory + 4096;
    auto r = call();
    check(attempts == 1 && exact_attempts == 0 && !r.diagonal && r.resident_jk_retry_seconds == 0,
          "refusal retried");
  });
  test("allocation failure before publication propagates", [&] {
    failure_mode = 7;
    bool caught = false;
    try {
      call();
    } catch (const std::bad_alloc&) {
      caught = true;
    }
    check(caught && attempts == 1, "early failure repeated");
  });
  for (const auto mode : {3, 4})
    test(mode == 3 ? "numerical failure never retries" : "execution failure never retries", [&] {
      failure_mode = mode;
      bool caught = false;
      try {
        call();
      } catch (const std::runtime_error&) {
        caught = true;
      }
      check(caught && attempts == 1, "scientific/CUDA fault hidden");
    });
  test("OOM with pending nonallocation CUDA failure propagates", [&] {
    failure_mode = 5;
    bool caught = false;
    try {
      call();
    } catch (const std::runtime_error&) {
      caught = true;
    }
    check(caught && attempts == 1 && state.pending == cudaSuccess, "pending CUDA failure hidden");
  });
  test("second exact OOM terminates without further retry", [&] {
    state.limit = resident + mandatory + 1024;
    failure_mode = 6;
    bool caught = false;
    try {
      call();
    } catch (const std::bad_alloc&) {
      caught = true;
    }
    check(caught && attempts == 2 && exact_attempts == 1, "unbounded retry");
  });
  std::cout << passed
            << " retry-controller checks passed; exact production wrapper and Owner allocation "
               "prefix, host allocator emulation only\n";
}
"""
