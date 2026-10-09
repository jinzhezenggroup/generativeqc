"""Host execution of production receipt ownership and readback control flow.

The CUDA allocation/copy/stream and provider boundaries are explicit stubs.
The arena, diagnostic methods, reset, metadata, and failure-settlement lambda
are extracted from the runtime under test. This is not CUDA execution evidence.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native/src"


def _block(source: str, marker: str, start: int = 0) -> str:
    """Retain a complete production definition or statement, including its body."""
    begin = source.index(marker, start)
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def _probe_source(runtime: str, header: str, mixer: str) -> str:
    entry = runtime.index(
        "generativeqc_xtb_status_t execute_restricted_gfn2_cuda_impl("
    )
    inference = runtime.index("generativeqc_xtb_status_t execute_inference_locked(")
    construction = runtime.index(
        "if (mixer_diagnostics_enabled) {", runtime.index("report factory rejected")
    )
    receipt_members = re.search(
        r"    DeviceArena mixer_receipt_arena;\n    std::int64_t mixer_receipt_capacity = 0;",
        runtime,
    )
    assert receipt_members is not None
    begin = runtime.index("  bool mixer_diagnostics_enabled = false;")
    end = runtime.index("  mutable std::mutex mutex;", begin)
    members = runtime[begin:end]
    helper_marker = "bool copy_mixer_diagnostics_locked("
    helper = _block(runtime, helper_marker) if helper_marker in runtime else ""
    fragments = {
        "RECEIPT": _block(mixer, "struct Gfn2SccMixerDeviceReceipt"),
        "STAGES": _block(mixer, "enum Gfn2SccMixerReceiptStage"),
        "SNAPSHOT": _block(header, "struct Gfn2CudaMixerDiagnosticSnapshot"),
        "ARENA": _block(runtime, "class DeviceArena"),
        "RECEIPT_MEMBERS": receipt_members.group(),
        "MEMBERS": members,
        "DESTRUCTOR": _block(runtime, "~Impl()"),
        "HELPER": helper,
        "ALLOCATE": _block(runtime, "if (mixer_diagnostics_enabled) {", construction),
        "RESET": _block(runtime, "if (mixer_diagnostics_enabled) {", inference),
        "GRAPH": _block(
            runtime, "if (mixer_diagnostics_enabled && !capture_bounded_scc) {"
        ),
        "BEGIN": _block(
            runtime, "if (implementation.mixer_diagnostics_enabled) {", entry
        ),
        "FINISH": _block(
            runtime,
            "if (implementation.mixer_diagnostics_enabled) {",
            runtime.index("  }();", entry),
        ),
        "FAIL": _block(runtime, "const auto fail_working_transaction =", entry) + ";",
        "ENABLE": _block(
            runtime, "bool Gfn2CudaExecutionCache::enable_mixer_diagnostics()"
        ),
        "READ": _block(runtime, "bool Gfn2CudaExecutionCache::read_mixer_diagnostics("),
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

using generativeqc_xtb_status_t = int;
constexpr int GENERATIVEQC_XTB_STATUS_SUCCESS = 0;
constexpr int GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR = 1;
constexpr int GENERATIVEQC_XTB_STATUS_INVALID_ARGUMENT = 2;
constexpr int GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED = 3;
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess = 0;
constexpr int cudaMemcpyDeviceToHost = 1;
int selected_device = 0;
int wrong_device_frees = 0;
bool reject_copy = false;
bool reject_receipt_copy = false;
bool fail_diagnostic_sync = false;
bool reject_sync = false;
bool fail_snapshot_allocation = false;
bool fail_next_allocation = false;
int injected_faults = 0;
void* operator new(std::size_t bytes) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    ++injected_faults;
    throw std::bad_alloc();
  }
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
struct Allocation { std::size_t bytes; int device; };
std::unordered_map<void*, Allocation> allocations;
bool live_range(const void* pointer, std::size_t bytes) {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  for (const auto& item : allocations) {
    const auto base = reinterpret_cast<std::uintptr_t>(item.first);
    if (begin >= base && begin - base <= item.second.bytes &&
        bytes <= item.second.bytes - (begin - base)) return true;
  }
  return false;
}
int cudaMalloc(void** output, std::size_t bytes) {
  *output = std::malloc(bytes);
  if (!*output) return 1;
  allocations[*output] = {bytes, selected_device};
  return cudaSuccess;
}
int cudaFree(void* pointer) {
  const auto found = allocations.find(pointer);
  if (found == allocations.end()) return 1;
  if (found->second.device != selected_device) ++wrong_device_frees;
  std::free(pointer);
  allocations.erase(found);
  return cudaSuccess;
}
int cudaGetDevice(int* device) { *device = selected_device; return cudaSuccess; }
int cudaSetDevice(int device) { selected_device = device; return cudaSuccess; }
int cudaStreamSynchronize(cudaStream_t) {
  if (reject_sync) { ++injected_faults; return 1; }
  return cudaSuccess;
}
int cudaMemsetAsync(void* destination, int value, std::size_t bytes, cudaStream_t) {
  if (!live_range(destination, bytes)) return 1;
  std::memset(destination, value, bytes);
  return cudaSuccess;
}
int cudaMemcpy(void* destination, const void* source, std::size_t bytes, int) {
  if (reject_copy || (reject_receipt_copy && bytes != sizeof(std::uint64_t))) {
    ++injected_faults;
    return 1;
  }
  if (!live_range(source, bytes)) return 1;
  std::memcpy(destination, source, bytes);
  if (fail_snapshot_allocation && bytes == sizeof(std::uint64_t)) {
    fail_snapshot_allocation = false;
    fail_next_allocation = true;
  }
  return cudaSuccess;
}
int cublasDestroy(void*) { return 0; }
std::string cuda_error_message(const char* action, int) { return action; }
struct ScopedCudaDevice {
  int previous;
  ScopedCudaDevice(int device, std::string&) : previous(selected_device) {
    cudaSetDevice(device);
  }
  ~ScopedCudaDevice() { cudaSetDevice(previous); }
  bool ok() const { return true; }
  int restore(std::string&) { return cudaSetDevice(previous); }
};
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
namespace cuda {
@RECEIPT@;
@STAGES@;
}
using cuda::Gfn2SccMixerDeviceReceipt;
@SNAPSHOT@;
@ARENA@;
struct Sink {
  Gfn2SccMixerDeviceReceipt* receipts = nullptr;
  std::uint64_t* receipt_count = nullptr;
  std::int64_t receipt_capacity = 0;
};
struct Prepared {
  struct Host {
    std::uint64_t plan_token = 0;
    struct { std::int64_t batch_size = 1; } basis;
    struct { std::int64_t maximum_iterations = 2; } key;
  } host;
  struct { struct { Sink mixer_workspace; } workspace; } scc_binding;
  bool submitted = false;
  @RECEIPT_MEMBERS@
};
struct Loop { unsigned execution_mode; std::uint64_t submitted_graphs; };
class Gfn2CudaExecutionCache {
 public:
  struct Impl {
    int device_id = 0;
    cudaStream_t stream = nullptr;
    bool handles_created = false;
    void* blas = nullptr;
    struct { void reset() {} } eigen_handles;
    mutable std::mutex mutex;
    @MEMBERS@
    @DESTRUCTOR@
    @HELPER@
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
    int settle_public_submissions_locked(Prepared& current, int status, std::string&) {
      cudaStreamSynchronize(stream);
      current.submitted = false;
      if (fail_diagnostic_sync) reject_sync = true;
      return status;
    }
  };
  std::unique_ptr<Impl> impl_ = std::make_unique<Impl>();
  bool enable_mixer_diagnostics() noexcept;
  bool read_mixer_diagnostics(Gfn2CudaMixerDiagnosticSnapshot&, std::string&) const;
};
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
    require(implementation.allocate(*candidate, error) == 0, "receipt allocation failed");
    working = candidate.get();
  }
  require(implementation.reset(*working, error) == 0, "receipt reset failed");
  auto& sink = working->scc_binding.workspace.mixer_workspace;
  require(*sink.receipt_count == 0, "new endpoint retained its old attempted count");
  *sink.receipt_count = attempted;
  for (std::uint64_t index = 0; index < std::min<std::uint64_t>(attempted, sink.receipt_capacity); ++index) {
    sink.receipts[index] = {};
    sink.receipts[index].invocation = index;
    sink.receipts[index].system = token;
    sink.receipts[index].iteration_before = index;
    sink.receipts[index].coefficient_dot_elements = 11 * (index + 1);
    sink.receipts[index].completed_stages = cuda::kMixerResidual;
  }
  implementation.graph({mode, submitted_graphs});
  const auto abort_topology_candidate = []() {};
  @FAIL@
  int transaction_status = GENERATIVEQC_XTB_STATUS_SUCCESS;
  if (fail) {
    error = "original numerical endpoint failure";
    transaction_status = fail_working_transaction(GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR);
    require(transaction_status == GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR,
            "diagnostic retention replaced the original endpoint status");
    require(error == "original numerical endpoint failure",
            "diagnostic retention replaced the original endpoint error");
  } else if (candidate != nullptr) {
    implementation.prepared = std::move(candidate);
  }
  @FINISH@
}

Gfn2CudaMixerDiagnosticSnapshot read(Gfn2CudaExecutionCache& cache) {
  Gfn2CudaMixerDiagnosticSnapshot snapshot;
  std::string error;
  const bool available = cache.read_mixer_diagnostics(snapshot, error);
  require(available, error.c_str());
  require(error.empty(), "successful diagnostic read retained an error");
  return snapshot;
}

void scenario(const std::string& name) {
  Gfn2CudaExecutionCache cache;
  require(cache.enable_mixer_diagnostics(), "diagnostics could not be enabled");
  if (name == "enabled_state") {
    Gfn2CudaMixerDiagnosticSnapshot snapshot;
    std::string error;
    require(!cache.read_mixer_diagnostics(snapshot, error) && !error.empty(),
            "read before the first call returned an uninitialized receipt");
    require(cache.enable_mixer_diagnostics() && allocations.empty(),
            "repeated enable before topology allocated or changed diagnostic state");
    call(cache, 31, 1, false);
    require(!cache.enable_mixer_diagnostics(), "late enable changed a prepared topology");
    snapshot = read(cache);
    require(snapshot.call_id == 1 && snapshot.receipts.size() == 1,
            "late enable rejection invalidated the completed diagnostic call");
  } else if (name == "first_failure_partial" || name == "first_failure_empty") {
    const unsigned count = name == "first_failure_partial" ? 1 : 0;
    call(cache, 31, count, true);
    require(cache.impl_->prepared == nullptr && allocations.empty(),
            "failed candidate retained a prepared topology or device arena");
    const auto snapshot = read(cache);
    require(snapshot.call_id == 1 && snapshot.plan_token == 31 &&
                snapshot.attempted_receipts == count && snapshot.receipts.size() == count &&
                !snapshot.endpoint_completed && snapshot.graph_submitted,
            "first-call failure lost its call identity or partial receipts");
    if (count) require(snapshot.receipts[0].coefficient_dot_elements == 11 &&
                           snapshot.receipts[0].status == GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR &&
                           snapshot.receipts[0].completed_stages == cuda::kMixerResidual,
                       "partial failed-stage receipt changed during retention");
  } else if (name == "replacement_failure") {
    call(cache, 31, 2, false);
    auto* original = cache.impl_->prepared.get();
    call(cache, 47, 1, true);
    require(cache.impl_->prepared.get() == original && allocations.size() == 1,
            "failed replacement changed committed topology or retained another device arena");
    const auto snapshot = read(cache);
    require(snapshot.call_id == 2 && snapshot.plan_token == 47 &&
                snapshot.receipts.size() == 1 && snapshot.receipts[0].system == 47 &&
                !snapshot.endpoint_completed,
            "replacement failure returned stale committed-topology diagnostics");
  } else if (name == "recovery_replay") {
    call(cache, 31, 2, true);
    call(cache, 47, 1, false);
    auto snapshot = read(cache);
    require(snapshot.call_id == 2 && snapshot.plan_token == 47 && snapshot.endpoint_completed &&
                snapshot.attempted_receipts == 1 && snapshot.receipts.size() == 1 &&
                snapshot.receipts[0].system == 47,
            "recovery retained failed-call receipts or metadata");
    call(cache, 47, 0, false);
    snapshot = read(cache);
    require(snapshot.call_id == 3 && snapshot.plan_token == 47 && snapshot.endpoint_completed &&
                snapshot.attempted_receipts == 0 && snapshot.receipts.empty(),
            "same-topology replay retained stale receipt count or data");
  } else if (name == "rejected_before_reset") {
    call(cache, 31, 1, true);
    call(cache, 47, 0, true, true);
    Gfn2CudaMixerDiagnosticSnapshot snapshot;
    snapshot.call_id = 123;
    std::string error;
    require(!cache.read_mixer_diagnostics(snapshot, error) && !error.empty() &&
                snapshot.call_id == 123,
            "rejected-before-reset call exposed a previous failed-call snapshot");
  } else if (name == "overflow") {
    call(cache, 31, 5, true);
    const auto snapshot = read(cache);
    require(snapshot.attempted_receipts == 5 && snapshot.receipts.size() == 2 &&
                snapshot.receipts[1].invocation == 1,
            "bounded retention hid overflow or copied beyond receipt capacity");
  } else if (name == "bounded_fallback") {
    call(cache, 31, 1, false, false, 0, 0);
    const auto snapshot = read(cache);
    require(snapshot.endpoint_completed && snapshot.graph_family == 0 &&
                !snapshot.graph_submitted && snapshot.receipts.size() == 1,
            "bounded direct fallback falsely reported a submitted graph");
  } else if (name == "failed_count_copy" || name == "failed_receipt_copy" ||
             name == "failed_diagnostic_sync" || name == "failed_snapshot_allocation") {
    reject_copy = name == "failed_count_copy";
    reject_receipt_copy = name == "failed_receipt_copy";
    fail_diagnostic_sync = name == "failed_diagnostic_sync";
    fail_snapshot_allocation = name == "failed_snapshot_allocation";
    call(cache, 31, 1, true);
    require(injected_faults != 0, "diagnostic fault injection was not exercised");
    reject_copy = false;
    reject_receipt_copy = false;
    fail_diagnostic_sync = false;
    reject_sync = false;
    fail_snapshot_allocation = false;
    fail_next_allocation = false;
    require(allocations.empty(), "failed diagnostic copy retained candidate device storage");
    Gfn2CudaMixerDiagnosticSnapshot snapshot;
    std::string error;
    require(!cache.read_mixer_diagnostics(snapshot, error),
            "failed diagnostic copy exposed an incomplete snapshot");
  } else {
    throw std::runtime_error("unknown scenario");
  }
  selected_device = 7;
  cache.impl_.reset();
  require(selected_device == 7 && allocations.empty() && wrong_device_frees == 0,
          "cache teardown leaked storage or freed it on the wrong CUDA device");
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected scenario");
    scenario(argv[1]);
    std::cout << argv[1] << ": passed (host CUDA/provider stubs)\n";
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
def receipt_transaction_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("gfn2-mixer-receipt-transaction")
    source = directory / "probe.cpp"
    source.write_text(
        _probe_source(
            (NATIVE / "runtime/gfn2_cuda_execution.cu").read_text(),
            (NATIVE / "runtime/gfn2_cuda_execution.hpp").read_text(),
            (NATIVE / "backends/cuda/gfn2_scc_mixer.cuh").read_text(),
        )
    )
    return required_native_cxx.build_executable(
        [source], directory / "probe", compile_args=["-std=c++17", "-O2"]
    )


@pytest.mark.parametrize(
    "scenario",
    (
        "enabled_state",
        "first_failure_partial",
        "first_failure_empty",
        "replacement_failure",
        "recovery_replay",
        "rejected_before_reset",
        "overflow",
        "bounded_fallback",
        "failed_count_copy",
        "failed_receipt_copy",
        "failed_diagnostic_sync",
        "failed_snapshot_allocation",
    ),
)
def test_call_owned_receipts_survive_transaction_outcomes(
    receipt_transaction_probe: Path, scenario: str
) -> None:
    result = subprocess.run(
        [str(receipt_transaction_probe), scenario],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
