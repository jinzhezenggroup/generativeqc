"""Execute census ownership/error paths with fault-injected CUDA operations."""

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_census_capture_lifetime_errors_and_unavailable_counts(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """No captured borrower or failed copy may publish valid executed work."""
    mock = tmp_path / "runtime/resource_cuda.cuh"
    mock.parent.mkdir()
    mock.write_text(MOCK)
    source = tmp_path / "diagnostic.cpp"
    source.write_text(DRIVER)
    binary = native_cxx.build_executable(
        [source],
        tmp_path / "diagnostic",
        compile_args=["-std=c++20", f"-I{tmp_path}", f"-I{ROOT / 'src'}"],
    )
    output = tmp_path / "census.jsonl"
    subprocess.run([binary, output], check=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["valid"] for row in rows] == [False, False, True, True]
    assert [row["complete_bounded_spd_domain"] for row in rows] == [
        False,
        False,
        True,
        False,
    ]
    complete = rows[2]["classes"]
    assert complete[0]["executed_shells"] == 7
    assert complete[0]["executed_components"] is None
    assert complete[0]["coulomb_preparations"] is None
    assert all(
        row["executed_shells"] is None for row in complete if row["angular_order"] == 6
    )
    assert all(row["executed_shells"] is None for row in rows[3]["classes"])


MOCK = r"""
#pragma once
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>
using cudaStream_t = int;
// Match compatibility headers that omit CUDA's capture-specific enumerators.
enum cudaError_t : int { cudaSuccess, cudaErrorUnknown, cudaErrorMemoryAllocation };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive };
constexpr int cudaMemcpyDeviceToHost = 0;
inline cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
inline cudaError_t allocation = cudaSuccess, zeroing = cudaSuccess, copying = cudaSuccess;
inline std::vector<int> calls;
inline bool pending{};
inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* state) {
  assert(stream == 17); calls.push_back(1); *state = capturing; return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* pointer, int value, std::size_t bytes, cudaStream_t stream) {
  assert(stream == 17); calls.push_back(3); pending = true;
  std::memset(pointer, value, bytes); return zeroing;
}
inline cudaError_t cudaMemcpyAsync(void* target, const void* source, std::size_t bytes,
                                   int, cudaStream_t stream) {
  assert(stream == 17); calls.push_back(4); pending = true;
  if (copying == cudaSuccess) std::memcpy(target, source, bytes);
  return copying;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  assert(stream == 17); calls.push_back(5); pending = false; return cudaSuccess;
}
namespace generativeqc::runtime {
inline cudaError_t resource_cuda_malloc(void** target, std::size_t bytes) {
  assert(capturing == cudaStreamCaptureStatusNone); calls.push_back(2);
  if (allocation == cudaSuccess) *target = std::malloc(bytes);
  return allocation;
}
inline cudaError_t resource_cuda_free(void* pointer) {
  assert(!pending); calls.push_back(6); std::free(pointer); return cudaSuccess;
}
}
"""

DRIVER = r"""
#include "scf/cuda/direct_force_diagnostic.hpp"
using namespace generativeqc::scf::cuda_execution;
int main(int argc, char** argv) {
  assert(argc == 2);
  unsetenv("GENERATIVEQC_DIRECT_FORCE_CENSUS");
  { DirectForceDiagnostic disabled(17);
    assert(disabled.prepare() == cudaSuccess && disabled.device() == nullptr);
    assert(disabled.finish(cudaErrorUnknown) == cudaErrorUnknown); }
  assert(calls.empty());
  setenv("GENERATIVEQC_DIRECT_FORCE_CENSUS", argv[1], 1);
  capturing = cudaStreamCaptureStatusActive;
  { DirectForceDiagnostic captured(17);
    assert(static_cast<int>(captured.prepare()) == 900); }
  assert(calls == std::vector<int>{1}); calls.clear();
  capturing = cudaStreamCaptureStatusNone; allocation = cudaErrorMemoryAllocation;
  { DirectForceDiagnostic failed(17);
    assert(failed.prepare() == cudaErrorMemoryAllocation); }
  assert((calls == std::vector<int>{1,2})); calls.clear();
  allocation = cudaSuccess; zeroing = cudaErrorUnknown;
  { DirectForceDiagnostic failed(17); assert(failed.prepare() == cudaErrorUnknown); }
  assert((calls == std::vector<int>{1,2,3,5,6})); calls.clear(); zeroing = cudaSuccess;
  // A failed submission still drains all previous borrowers before release.
  { DirectForceDiagnostic failed(17); assert(failed.prepare() == cudaSuccess);
    assert(failed.finish(cudaErrorUnknown) == cudaErrorUnknown); }
  assert((calls == std::vector<int>{1,2,3,4,5,6})); calls.clear();
  copying = cudaErrorUnknown;
  { DirectForceDiagnostic failed(17); assert(failed.prepare() == cudaSuccess);
    assert(failed.finish(cudaSuccess) == cudaErrorUnknown); }
  assert((calls == std::vector<int>{1,2,3,4,5,6})); calls.clear(); copying = cudaSuccess;
  { DirectForceDiagnostic good(17); assert(good.prepare() == cudaSuccess);
    good.set_complete_domain(true); good.device()->classes[0].executed_shells = 7;
    assert(good.finish(cudaSuccess) == cudaSuccess); }
  assert((calls == std::vector<int>{1,2,3,4,5,6})); calls.clear();
  { DirectForceDiagnostic partial(17); assert(partial.prepare() == cudaSuccess);
    partial.set_complete_domain(true); partial.device()->classes[0].generated_owned = 1;
    assert(partial.finish(cudaSuccess) == cudaSuccess); }
  assert((calls == std::vector<int>{1,2,3,4,5,6}));
}
"""
