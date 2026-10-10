"""Exercise the real host graph owner with a deterministic CUDA lifecycle stub."""

import subprocess
from pathlib import Path

from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_capture_exception_abandons_stream_and_allows_retry(
    native_cxx: NativeCxx, tmp_path: Path
) -> None:
    """Compile real graph ownership against the shared observation adapter's CUDA ABI."""
    (tmp_path / "cuda_runtime_api.h").write_text(
        """#pragma once
#include <cstddef>
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaErrorInvalidResourceHandle = 400;
constexpr int cudaStreamCaptureModeThreadLocal = 1;
constexpr unsigned long long cudaGraphInstantiateFlagDeviceLaunch = 4;
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3 };
cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, int);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
"""
    )
    harness = tmp_path / "capture.cpp"
    harness.write_text(
        r"""#include "scf/cuda/rhf_graph.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
bool capturing = false;
int graphs = 0, executables = 0;
std::vector<std::array<std::uint64_t, 10>> events;
bool observer_throws = false;
bool observer_recurses = false;
void observer(const std::uint64_t* values, std::size_t count, void*) {
  if (count == 14 && values[0] == 2) return;
  if (count != 10 || values[0] != 1 || !values[2] || !values[5] || !values[6]) std::terminate();
  std::array<std::uint64_t, 10> event{};
  std::copy_n(values, count, event.begin());
  events.push_back(event);
  if (observer_recurses) {
    using namespace generativeqc::runtime;
    observe_residency_graph(ResidencyGraphEvent::definition, 99, ResidencyGraphRole::hf_iteration,
                            0, nullptr, nullptr, nullptr);
  }
  if (observer_throws) throw std::runtime_error("observer only");
}
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t p) {
  delete static_cast<int*>(p); --executables; return cudaSuccess;
}
cudaError_t cudaGraphDestroy(cudaGraph_t p) {
  delete static_cast<int*>(p); --graphs; return cudaSuccess;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamBeginCapture(cudaStream_t, int) {
  if (capturing) return cudaErrorInvalidResourceHandle;
  capturing = true; return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) {
  if (!capturing) return cudaErrorInvalidResourceHandle;
  capturing = false; *graph = new int(1); ++graphs; return cudaSuccess;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* out, cudaGraph_t, unsigned long long) {
  *out = new int(1); ++executables; return cudaSuccess;
}
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t p, cudaStream_t) {
  return p ? cudaSuccess : cudaErrorInvalidResourceHandle;
}
int main() {
  using namespace generativeqc::runtime;
  std::uint64_t errors = 99;
  if (bind_residency_observer(nullptr, nullptr) != 1) return 10;
  if (bind_residency_observer(observer, nullptr) != 0) return 11;
  if (bind_residency_observer(observer, nullptr) != 1) return 12;
  if (unbind_residency_observer(observer, &errors, &errors) != 1) return 13;
  std::thread foreign([] {
    observe_residency_graph(ResidencyGraphEvent::definition, 1, ResidencyGraphRole::hf_iteration,
                            0, nullptr, nullptr, nullptr);
  });
  foreign.join();
  if (!events.empty()) return 14;
  for (bool post : {false, true}) {
    events.clear();
    {
      generativeqc::scf::cuda_execution::RhfIterationGraphs owner;
      auto capture = [&](const std::function<generativeqc_status()>& body) {
        return post ? owner.capture_post_eigensolver(0, nullptr, body)
                    : owner.capture_iteration(0, nullptr, false, body);
      };
      try {
        capture([]() -> generativeqc_status { throw std::logic_error("original"); });
        return 2;
      } catch (const std::logic_error& error) {
        if (std::string(error.what()) != "original") return 3;
      }
      if (capturing || graphs || executables) {
        std::cerr << "capture remained active after exception\n";
        return 4;
      }
      auto rejected = capture([] { return GENERATIVEQC_STATUS_NOT_IMPLEMENTED; });
      if (rejected.ok() || rejected.body_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED ||
          capturing || graphs || executables) return 5;
      if (!capture([] { return GENERATIVEQC_STATUS_SUCCESS; }).ok()) return 6;
      if (capturing || graphs != 1 || executables != 1) return 7;
      if (events.size() != 1 || events[0][1] != 1 || events[0][3] != (post ? 2 : 1)) return 15;
      auto replay = post ? owner.launch_post_eigensolver(nullptr)
                         : owner.launch_iteration(nullptr);
      if (replay != cudaSuccess) return 8;
      if (events.size() != 3 || events[1][1] != 2 || events[2][1] != 3 ||
          events[2][8] != cudaSuccess || events[0][2] != events[2][2]) return 16;
    }
    if (capturing || graphs || executables) return 9;
    if (events.size() != 4 || events.back()[1] != 4 || events.back()[2] != events[0][2]) return 17;
  }
  {
    generativeqc::scf::cuda_execution::RhfIterationGraphs owner;
    events.clear();
    observer_throws = true;
    observer_recurses = true;
    if (!owner.capture_iteration(0, nullptr, true, [] { return GENERATIVEQC_STATUS_SUCCESS; }).ok()) return 18;
    const auto original_generation = events.front()[2];
    if (events.front()[4] != cudaGraphInstantiateFlagDeviceLaunch) return 19;
    if (owner.launch_iteration(nullptr) != cudaSuccess) return 20;
    if (!owner.capture_iteration(0, nullptr, false, [] { return GENERATIVEQC_STATUS_SUCCESS; }).ok()) return 21;
    if (events[3][1] != 4 || events[3][2] != original_generation ||
        events[4][1] != 1 || events[4][2] <= original_generation) return 22;
  }
  if (graphs || executables) return 23;
  if (unbind_residency_observer(observer, nullptr, &errors) != 0 || errors != 12) return 24;
  if (unbind_residency_observer(observer, nullptr, &errors) != 1) return 25;
  residency_graph_generation = std::numeric_limits<std::uint64_t>::max();
  if (next_residency_graph_generation() != 0 || next_residency_graph_generation() != 0) return 26;
  std::cout << "graph capture lifecycle passed\n";
}
"""
    )
    executable = tmp_path / "capture"
    native_cxx.build_executable(
        [ROOT / "src/scf/cuda/rhf_graph.cpp", harness],
        executable,
        compile_args=(
            "-std=c++20",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        link_args=("-pthread",),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
