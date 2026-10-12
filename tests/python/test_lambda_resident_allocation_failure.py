"""Exercise the real optional-arena failure boundary with CUDA error doubles.

This is a host failure-classification test, not a CUDA numerical qualification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_optional_resident_arena_oom_clears_only_the_allocation_error(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    source = (ROOT / "src/cc/lambda_response_cuda.cu").read_text()
    resident = source[source.index("class CudaLambdaResidentBackend final") :]
    start = resident.index(
        "} catch (const generativeqc_tensor::DeviceAllocationError&) {"
    )
    body = resident[start : resident.index("} catch (...) {", start)]
    probe = tmp_path / "resident_allocation_failure.cpp"
    probe.write_text(
        r"""
#include <new>
#include <stdexcept>
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorLaunchFailure };
cudaError_t pending = cudaSuccess, synchronization = cudaSuccess;
cudaError_t cudaGetLastError() { auto result = pending; pending = cudaSuccess; return result; }
cudaError_t cudaStreamSynchronize(int) { return synchronization; }
namespace generativeqc_tensor {
struct DeviceAllocationError : std::runtime_error { using std::runtime_error::runtime_error; };
struct DeviceRuntimeError : std::runtime_error { using std::runtime_error::runtime_error; };
}
void cuda_check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw generativeqc_tensor::DeviceAllocationError("oom");
  if (status != cudaSuccess) throw generativeqc_tensor::DeviceRuntimeError("execution");
}
struct Owner { int stream_ = 7, synchronizations_ = 0; };
struct Probe {
  Owner owner_;
  int cleanups = 0;
  void cleanup() noexcept { ++cleanups; }
  void fail() {
    try {
      throw generativeqc_tensor::DeviceAllocationError("optional arena");
"""
        + body
        + r"""
    }
  }
};
int main() {
  for (auto previous : {cudaSuccess, cudaErrorMemoryAllocation, cudaErrorLaunchFailure}) {
    for (auto stream_error : {cudaSuccess, cudaErrorLaunchFailure}) {
      pending = previous;
      synchronization = stream_error;
      Probe probe;
      bool refused = false, execution_failed = false;
      try { probe.fail(); }
      catch (const std::bad_alloc&) { refused = true; }
      catch (const generativeqc_tensor::DeviceRuntimeError&) { execution_failed = true; }
      const bool must_propagate = previous == cudaErrorLaunchFailure ||
                                  stream_error == cudaErrorLaunchFailure;
      if (probe.cleanups != 1 || probe.owner_.synchronizations_ != 1 || pending != cudaSuccess ||
          refused == must_propagate || execution_failed != must_propagate)
        return 1;
      // A fresh generated action reads the CUDA last-error slot after launch.
      // The recoverable OOM must not poison that original host-vector fallback.
      if (!must_propagate) cuda_check(cudaGetLastError());
    }
  }
}
"""
    )
    executable = tmp_path / "resident_allocation_failure"
    required_native_cxx.build_executable(
        [probe], executable, compile_args=("-std=c++20", "-Wall", "-Wextra", "-Werror")
    )
    subprocess.run([str(executable)], check=True, timeout=15)
