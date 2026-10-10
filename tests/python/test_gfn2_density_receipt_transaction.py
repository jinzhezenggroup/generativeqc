"""Exercise production density receipt ownership with host CUDA stubs.

This verifies failed-candidate settlement/readback control flow without treating
stubbed execution as CUDA or scientific evidence.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native/src"


def test_density_diagnostic_define_covers_parent_layout_consumer() -> None:
    """Keep the opt-in workspace ABI identical across the combined library."""
    cmake = (ROOT / "cmake/GenerativeQCGfn2Runtime.cmake").read_text()
    diagnostic = cmake.split("if(GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS)", 1)[
        1
    ].split("endif()", 1)[0]
    define = "GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=1"
    assert "target_compile_definitions(generativeqc_gfn2_cuda PRIVATE" in diagnostic
    assert "target_compile_definitions(${target} PRIVATE" in diagnostic
    assert diagnostic.count(define) == 2


def _block(source: str, marker: str, start: int = 0) -> str:
    begin = source.index(marker, start)
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def _probe_source(runtime: str, runtime_header: str, density_header: str) -> str:
    entry = runtime.index(
        "generativeqc_xtb_status_t execute_restricted_gfn2_cuda_impl("
    )
    failure = runtime.index("const auto fail_working_transaction =", entry)
    inference = runtime.index("generativeqc_xtb_status_t execute_inference_locked(")
    construction = runtime.index(
        "if (density_diagnostics_enabled) {", runtime.index("report factory rejected")
    )
    members_start = runtime.index("  bool density_diagnostics_enabled = false;")
    members_end = runtime.index(
        "  bool failed_density_snapshot_valid = false;", members_start
    )
    members_end += len("  bool failed_density_snapshot_valid = false;")
    fragments = {
        "STATUS": _block(density_header, "enum Gfn2DensityReceiptStatus"),
        "RECEIPT": _block(density_header, "struct Gfn2DensityDeviceReceipt"),
        "SNAPSHOT": _block(runtime_header, "struct Gfn2CudaDensityDiagnosticSnapshot"),
        "ARENA": _block(runtime, "class DeviceArena"),
        "MEMBERS": runtime[members_start:members_end],
        "COPY": _block(runtime, "bool copy_density_diagnostics_locked("),
        "ALLOCATE": _block(runtime, "if (density_diagnostics_enabled) {", construction),
        "RESET": _block(runtime, "if (density_diagnostics_enabled) {", inference),
        "GRAPH": _block(
            runtime,
            "if (density_diagnostics_enabled) {",
            runtime.index("const Gfn2SccLoopLaunchResult loop ="),
        ),
        "BEGIN": _block(
            runtime, "if (implementation.density_diagnostics_enabled) {", entry
        ),
        "FAILED": _block(
            runtime,
            "if (candidate != nullptr && implementation.density_diagnostics_enabled &&",
            failure,
        ),
        "FINISH": _block(
            runtime,
            "if (implementation.density_diagnostics_enabled) {",
            runtime.index(
                "const auto final_status = finish(transaction_status);", entry
            ),
        ),
        "ENABLE": _block(
            runtime, "bool Gfn2CudaExecutionCache::enable_density_diagnostics()"
        ),
        "READ": _block(
            runtime, "bool Gfn2CudaExecutionCache::read_density_diagnostics("
        ),
    }
    source = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS 1
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess = 0;
constexpr int cudaMemcpyDeviceToHost = 1;
using generativeqc_xtb_status_t = int;
constexpr int GENERATIVEQC_XTB_STATUS_SUCCESS = 0;
constexpr int GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR = 1;
constexpr int GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT = 2;
constexpr int GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED = 3;
bool fail_allocation = false, fail_reset = false, fail_sync = false;
bool fail_count_copy = false, fail_receipt_copy = false;
bool fail_host_allocation = false, fail_next_new = false;
void* operator new(std::size_t bytes) {
  if (fail_next_new) {
    fail_next_new = false;
    throw std::bad_alloc();
  }
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
std::unordered_map<void*, std::size_t> allocations;
int cudaMalloc(void** result, std::size_t bytes) {
  if (fail_allocation) return 1;
  *result = std::malloc(bytes);
  if (!*result) return 1;
  allocations[*result] = bytes;
  return cudaSuccess;
}
int cudaFree(void* pointer) {
  if (allocations.erase(pointer) != 1) return 1;
  std::free(pointer);
  return cudaSuccess;
}
bool live(const void* pointer, std::size_t bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  for (const auto& [base_pointer, size] : allocations) {
    const auto base = reinterpret_cast<std::uintptr_t>(base_pointer);
    if (address >= base && address - base <= size && bytes <= size - (address - base))
      return true;
  }
  return false;
}
int cudaMemsetAsync(void* pointer, int value, std::size_t bytes, cudaStream_t) {
  if (fail_reset || !live(pointer, bytes)) return 1;
  std::memset(pointer, value, bytes);
  return cudaSuccess;
}
int cudaStreamSynchronize(cudaStream_t) { return fail_sync ? 1 : cudaSuccess; }
int cudaMemcpy(void* destination, const void* source, std::size_t bytes, int) {
  if ((bytes == sizeof(std::uint64_t) && fail_count_copy) ||
      (bytes != sizeof(std::uint64_t) && fail_receipt_copy) || !live(source, bytes)) return 1;
  std::memcpy(destination, source, bytes);
  if (bytes == sizeof(std::uint64_t) && fail_host_allocation) {
    fail_host_allocation = false;
    fail_next_new = true;
  }
  return cudaSuccess;
}
std::string cuda_error_message(const char* action, int) { return action; }
struct ScopedCudaDevice {
  ScopedCudaDevice(int, std::string&) {}
  bool ok() const { return true; }
  int restore(std::string&) { return 0; }
};
void require(bool condition, const char* text) {
  if (!condition) throw std::runtime_error(text);
}
namespace generativeqc::xtb::generated {
unsigned gfn2_electronic_matrix_tiles(std::int64_t, std::int64_t) { return 2u; }
}
namespace cuda {
@STATUS@;
@RECEIPT@;
}
using cuda::Gfn2DensityDeviceReceipt;
@SNAPSHOT@;
@ARENA@;
struct Sink {
  std::uint64_t* diagnostic_receipt_count = nullptr;
  Gfn2DensityDeviceReceipt* diagnostic_receipts = nullptr;
  std::int64_t diagnostic_receipt_capacity = 0;
};
struct Prepared {
  struct Host {
    std::uint64_t plan_token = 0;
    struct { std::int64_t batch_size = 1; } basis;
    struct { std::int32_t maximum_iterations = 2; } key;
  } host;
  struct { struct { std::int64_t total_matrix_elements = 9; } density_batch; } plan_seed;
  struct { struct { Sink density_workspace; } workspace; } scc_binding;
  DeviceArena density_receipt_arena;
  std::int64_t density_receipt_capacity = 0;
  std::uint32_t density_grid_tiles = 0;
  bool submitted = false;
};
struct Loop { std::uint32_t execution_mode; std::uint64_t submitted_graphs; };
class Gfn2CudaExecutionCache {
 public:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Gfn2CudaExecutionCache();
  bool enable_density_diagnostics() noexcept;
  bool read_density_diagnostics(Gfn2CudaDensityDiagnosticSnapshot&, std::string&) const;
};
struct Gfn2CudaExecutionCache::Impl {
  std::int32_t device_id = 0;
  cudaStream_t stream = nullptr;
  mutable std::mutex mutex;
  @MEMBERS@
  std::unique_ptr<Prepared> prepared;
  @COPY@
  int allocate(Prepared& current, std::string& error) {
    auto* candidate = &current;
    int cuda_status = cudaSuccess;
    @ALLOCATE@
    return GENERATIVEQC_XTB_STATUS_SUCCESS;
  }
  int reset(Prepared& current, std::string& error) {
    int cuda_status = cudaSuccess;
    @RESET@
    return GENERATIVEQC_XTB_STATUS_SUCCESS;
  }
  void graph(Loop loop) {
    const bool capture_bounded_scc = false;
    @GRAPH@
  }
};
Gfn2CudaExecutionCache::Gfn2CudaExecutionCache() : impl_(std::make_unique<Impl>()) {}
@ENABLE@
@READ@
void call(Gfn2CudaExecutionCache& cache, std::uint64_t token, std::uint64_t attempted,
          bool fail, bool reject_before_reset = false, unsigned mode = 1,
          std::uint64_t submitted_graphs = 1) {
  auto& implementation = *cache.impl_;
  @BEGIN@
  if (reject_before_reset) return;
  std::string error;
  Prepared* working = implementation.prepared.get();
  std::unique_ptr<Prepared> candidate;
  if (working == nullptr || working->host.plan_token != token) {
    candidate = std::make_unique<Prepared>();
    candidate->host.plan_token = token;
    const auto allocation_status = implementation.allocate(*candidate, error);
    if (allocation_status != GENERATIVEQC_XTB_STATUS_SUCCESS) {
      require(allocation_status == GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED,
              "allocation failure changed its status");
      return;
    }
    working = candidate.get();
  }
  if (implementation.reset(*working, error) != GENERATIVEQC_XTB_STATUS_SUCCESS) return;
  auto& sink = working->scc_binding.workspace.density_workspace;
  require(*sink.diagnostic_receipt_count == 0, "receipt reset retained prior count");
  *sink.diagnostic_receipt_count = attempted;
  for (std::uint64_t index = 0;
       index < std::min<std::uint64_t>(attempted, sink.diagnostic_receipt_capacity); ++index) {
    sink.diagnostic_receipts[index] = {};
    sink.diagnostic_receipts[index].slot = index;
    sink.diagnostic_receipts[index].system = token;
    sink.diagnostic_receipts[index].plain_visits = 10 + index;
  }
  implementation.graph({mode, submitted_graphs});
  if (fail) {
    error = "original numerical failure";
    @FAILED@
    require(error == "original numerical failure", "diagnostic changed endpoint error");
  } else if (candidate != nullptr) {
    implementation.prepared = std::move(candidate);
  }
  const auto transaction_status = fail ? GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR
                                       : GENERATIVEQC_XTB_STATUS_SUCCESS;
  const auto final_status = transaction_status;
  @FINISH@
}
Gfn2CudaDensityDiagnosticSnapshot read(Gfn2CudaExecutionCache& cache) {
  Gfn2CudaDensityDiagnosticSnapshot snapshot;
  std::string error;
  require(cache.read_density_diagnostics(snapshot, error), error.c_str());
  return snapshot;
}
void scenario(const std::string& name) {
  Gfn2CudaExecutionCache cache;
  require(cache.enable_density_diagnostics(), "density diagnostics unavailable");
  if (name == "capacity_reject") {
    Prepared oversized;
    oversized.host.basis.batch_size = 1000000;
    std::string error;
    require(cache.impl_->allocate(oversized, error) ==
                GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT &&
                !error.empty() && allocations.empty(),
            "receipt budget admission allocated an oversized arena");
    return;
  }
  if (name == "allocation_failure") {
    fail_allocation = true;
    call(cache, 31, 0, true);
    require(cache.impl_->prepared == nullptr && allocations.empty(),
            "allocation failure retained a candidate");
    return;
  }
  if (name == "reset_failure") {
    fail_reset = true;
    call(cache, 31, 0, true);
    Gfn2CudaDensityDiagnosticSnapshot snapshot;
    std::string error;
    require(!cache.read_density_diagnostics(snapshot, error) && allocations.empty(),
            "reset failure exposed stale receipts");
    return;
  }
  if (name == "first_failure" || name == "first_failure_empty") {
    const auto count = name == "first_failure" ? 3u : 0u;
    call(cache, 31, count, true);
    require(cache.impl_->prepared == nullptr && allocations.empty(),
            "failed first candidate retained a device arena");
    const auto snapshot = read(cache);
    require(snapshot.call_id == 1 && snapshot.plan_token == 31 &&
                snapshot.attempted_receipts == count && snapshot.receipts.size() == count &&
                !snapshot.endpoint_completed && snapshot.graph_submitted,
            "failed first call lost its partial or empty receipt identity");
    return;
  }
  if (name == "replacement_failure") {
    call(cache, 31, 2, false);
    auto* original = cache.impl_->prepared.get();
    call(cache, 47, 1, true);
    require(cache.impl_->prepared.get() == original && allocations.size() == 1,
            "failed replacement published topology or retained its arena");
    const auto snapshot = read(cache);
    require(snapshot.call_id == 2 && snapshot.plan_token == 47 &&
                snapshot.receipts.size() == 1 && snapshot.receipts[0].system == 47,
            "replacement returned stale committed-topology receipts");
    return;
  }
  if (name == "replay_reset") {
    call(cache, 31, 2, false);
    const auto first = read(cache);
    call(cache, 31, 1, false);
    const auto second = read(cache);
    require(first.call_id == 1 && second.call_id == 2 &&
                first.plan_token == second.plan_token && second.attempted_receipts == 1 &&
                second.receipts.size() == 1 && second.endpoint_completed,
            "same-topology replay retained old receipts");
    return;
  }
  if (name == "rejected_before_reset") {
    call(cache, 31, 1, true);
    call(cache, 47, 0, true, true);
    Gfn2CudaDensityDiagnosticSnapshot snapshot;
    std::string error;
    require(!cache.read_density_diagnostics(snapshot, error),
            "pre-reset rejection returned previous failed snapshot");
    return;
  }
  if (name == "overflow" || name == "bounded_fallback") {
    call(cache, 31, name == "overflow" ? 19 : 1, true, false,
         name == "overflow" ? 1 : 0, name == "overflow" ? 1 : 0);
    const auto snapshot = read(cache);
    require(snapshot.receipt_capacity == 16 &&
                snapshot.attempted_receipts == (name == "overflow" ? 19u : 1u) &&
                snapshot.receipts.size() == (name == "overflow" ? 16u : 1u) &&
                snapshot.graph_submitted == (name == "overflow"),
            "overflow or bounded-fallback status was hidden");
    return;
  }
  if (name == "failed_count_copy" || name == "failed_receipt_copy" ||
      name == "failed_sync" || name == "failed_host_allocation") {
    fail_count_copy = name == "failed_count_copy";
    fail_receipt_copy = name == "failed_receipt_copy";
    fail_sync = name == "failed_sync";
    fail_host_allocation = name == "failed_host_allocation";
    call(cache, 31, 1, true);
    fail_count_copy = fail_receipt_copy = fail_sync = fail_host_allocation = false;
    Gfn2CudaDensityDiagnosticSnapshot snapshot;
    std::string error;
    require(cache.impl_->prepared == nullptr && allocations.empty() &&
                !cache.read_density_diagnostics(snapshot, error),
            "failed optional readback exposed an incomplete snapshot");
    return;
  }
  throw std::runtime_error("unknown scenario");
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected scenario");
    scenario(argv[1]);
    require(allocations.empty(), "device arena leaked");
    std::cout << argv[1] << ": passed (host CUDA stubs)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""
    for name, fragment in fragments.items():
        source = source.replace(f"@{name}@", fragment)
    return source


@pytest.fixture(scope="module")
def density_transaction_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("gfn2-density-receipt-transaction")
    source = directory / "probe.cpp"
    source.write_text(
        _probe_source(
            (NATIVE / "runtime/gfn2_cuda_execution.cu").read_text(),
            (NATIVE / "runtime/gfn2_cuda_execution.hpp").read_text(),
            (NATIVE / "backends/cuda/gfn2_density.cuh").read_text(),
        )
    )
    return required_native_cxx.build_executable(
        [source], directory / "probe", compile_args=["-std=c++17", "-O2"]
    )


@pytest.mark.parametrize(
    "scenario",
    (
        "allocation_failure",
        "capacity_reject",
        "reset_failure",
        "first_failure",
        "first_failure_empty",
        "replacement_failure",
        "replay_reset",
        "rejected_before_reset",
        "overflow",
        "bounded_fallback",
        "failed_count_copy",
        "failed_receipt_copy",
        "failed_sync",
        "failed_host_allocation",
    ),
)
def test_density_receipt_lifecycle(
    density_transaction_probe: Path, scenario: str
) -> None:
    result = subprocess.run(
        [str(density_transaction_probe), scenario],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
