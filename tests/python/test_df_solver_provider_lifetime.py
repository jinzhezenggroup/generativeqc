"""CC solver releases must not hide another owner's measured provider growth.

Compile the live cleanup and optional-arena fallback with host CUDA doubles.
This tests resource ordering and fallback selection, not GPU execution.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def provider_lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a host C++ compiler")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/cc/cuda_solver.cu").read_text()
    begin = source.index("  void cleanup() noexcept {")
    cleanup = source[begin : source.index("\n  template <class Output>", begin)]
    begin = source.index("      auto allocate_numeric = [&]() {")
    end = source.index("      cuda_check(allocation);", begin)
    fallback = source[begin : end + len("      cuda_check(allocation);")]
    directory = tmp_path_factory.mktemp("df-solver-provider-lifetime")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(
        PREFIX + cleanup + "\nvoid retry() {\n" + fallback + "\n}\n};\n}\n" + MAIN
    )
    build = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-pthread",
            "-I" + str(ROOT / "src"),
            str(unit),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    return executable


@pytest.mark.parametrize(
    "operation",
    [
        "cleanup",
        "fallback",
        "history-fallback",
        "conventional-fallback",
        "conventional-history-fallback",
        "workspace-fallback",
        "workspace-history-fallback",
    ],
)
def test_provider_release_waits_for_other_owner_measurement(
    provider_lifetime_probe: Path, operation: str
) -> None:
    result = subprocess.run(
        [str(provider_lifetime_probe), operation],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <chrono>
#include <algorithm>
#include <cstddef>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include "runtime/allocation_measurement.hpp"
using namespace std::chrono_literals;
constexpr int cudaErrorMemoryAllocation=2;
static std::promise<void> released;
static std::promise<void> partial_freed;
static bool fail_history=false;
static bool workspace_enabled=false;
static int workspace_releases=0;
static int destroys=0, frees=0, allocations=0, cleared=0;
int cudaStreamSynchronize(void*) { return 0; }
int cublasDestroy(void*) { ++destroys; released.set_value(); return 0; }
int cudaEventDestroy(void*) { return 0; }
int cudaFree(void*) { if(++frees==1) partial_freed.set_value(); return 0; }
int cudaStreamDestroy(void*) { return 0; }
int cudaGetLastError() { ++cleared; return 0; }
int cudaMalloc(void** pointer,std::size_t) {
  if (++allocations==(fail_history ? 2 : 1)) return cudaErrorMemoryAllocation;
  *pointer=reinterpret_cast<void*>(4); return 0;
}
void cuda_check(int code) { if(code) throw std::runtime_error("CUDA failure"); }
void blas_check(int code) { if(code) throw std::runtime_error("BLAS failure"); }
namespace generativeqc::cc {
constexpr std::size_t kContractionProviderAllowance=96ULL<<20;
std::size_t checked_add(std::size_t a,std::size_t b) { return a+b; }
struct Contractions {
  void *handle=reinterpret_cast<void*>(2), *stream=reinterpret_cast<void*>(1);
  std::size_t workspace_bytes() const { return workspace_enabled ? 4ULL<<20 : 0; }
  bool release_workspace() {
    std::lock_guard<std::mutex> lock(generativeqc::runtime::allocation_measurement_mutex);
    workspace_enabled=false; ++workspace_releases; return true;
  }
  void release_locked() {
    if (handle) {
      if (stream) cuda_check(cudaStreamSynchronize(stream));
      blas_check(cublasDestroy(handle));
    }
    handle=nullptr;
    stream=nullptr;
  }
  void reset_locked() noexcept {
    if (handle) {
      if (stream) (void)cudaStreamSynchronize(stream);
      (void)cublasDestroy(handle);
    }
    handle=nullptr;
    stream=nullptr;
  }
};
struct Owner {
  void *stream=reinterpret_cast<void*>(1);
  Contractions contractions;
  void *trial_begin{}, *trial_end{};
  unsigned char *base=reinterpret_cast<unsigned char*>(3);
  // Histories are a second numeric owner; their release shares the lock.
  unsigned char *history_base=reinterpret_cast<unsigned char*>(5);
  struct Plan { bool matrix_gemm=true; std::size_t auxiliary_batch_size=1; } plan;
  struct { std::size_t nocc=1,nvir=1; } p;
  struct { bool df_auxiliary_reduction=true; } options;
  std::size_t naux=1,combined=0;
  Plan df_iteration_plan(std::size_t,std::size_t,std::size_t,bool,bool,bool) { return {}; }
  std::size_t build_layout() { return layout.total; }
  bool conventional_prepared=false;
  bool replay_matrix=false;
  std::size_t replay_batch=0;
  bool pairs_enabled=false;
  struct { std::size_t total=1024,history_bytes=0; } layout;
  struct { unsigned synchronizations=0; std::size_t owned_device_bytes=0,numeric_capacity_bytes=0;
           bool df_pair_resource_refused=false; } diagnostic;
  int replans=0;
  void scalar_plan() {
    conventional_prepared=false; plan.matrix_gemm=false; layout.total=512; ++replans;
  }
"""

MAIN = r"""
int main(int argc,char** argv) {
  if(argc!=2) return 99;
  const auto operation=std::string(argv[1]);
  fail_history=operation=="history-fallback" || operation=="conventional-history-fallback" ||
               operation=="workspace-history-fallback";
  workspace_enabled=operation=="workspace-fallback" || operation=="workspace-history-fallback";
  const bool optional_workspace=workspace_enabled;
  const bool fallback=operation!="cleanup";
  generativeqc::cc::Owner owner;
  if(optional_workspace) {
    owner.pairs_enabled=owner.replay_matrix=true; owner.plan.auxiliary_batch_size=32;
  }
  if (operation=="conventional-fallback" || operation=="conventional-history-fallback") {
    owner.conventional_prepared=true;
    owner.plan.matrix_gemm=false;
    owner.naux=0;
  }
  if(fail_history) {
    owner.base=owner.history_base=nullptr;owner.layout.history_bytes=256;owner.combined=2048;
  }
  auto release=released.get_future();
  auto partial_release=partial_freed.get_future();
  std::promise<void> started;
  auto ready=started.get_future();
  std::unique_lock<std::mutex> measurement(
      generativeqc::runtime::allocation_measurement_mutex);
  std::thread worker([&] {
    started.set_value();
    if(fallback) owner.retry(); else owner.cleanup();
  });
  ready.wait();
  const bool released_during_measurement=release.wait_for(250ms)==std::future_status::ready;
  const bool freed_during_measurement=partial_release.wait_for(0ms)==std::future_status::ready;
  const bool workspace_released_during_measurement=workspace_releases!=0;
  measurement.unlock();
  worker.join();
  if(released_during_measurement || freed_during_measurement || workspace_released_during_measurement) {
    std::cerr << "Provider released during another owner's allocation measurement: "
                 "a 96 MiB release can hide 160 MiB growth as 64 MiB.\n";
    return 1;
  }
  if(optional_workspace) {
    if(destroys || !owner.contractions.handle || workspace_releases!=1 || workspace_enabled ||
       owner.replans || !owner.pairs_enabled || !owner.replay_matrix ||
       owner.plan.auxiliary_batch_size!=32 || owner.layout.total!=1024 ||
       allocations!=(fail_history ? 4 : 2) || cleared!=(fail_history ? 2 : 1)) return 5;
    owner.cleanup();
    return owner.base || owner.history_base || owner.stream || destroys!=1 ||
           frees!=(fail_history ? 3 : 2);
  }
  if(destroys!=1 || owner.contractions.handle) return 2;
  if(fallback) {
    if(owner.conventional_prepared || owner.plan.matrix_gemm || owner.replans!=1 || allocations!=(fail_history ? 4 : 2) ||
       cleared!=(fail_history ? 2 : 1) ||
       owner.layout.total!=512 || owner.base!=reinterpret_cast<unsigned char*>(4)) return 3;
    owner.cleanup();
  }
  if(owner.base || owner.history_base || owner.stream || frees!=(fail_history ? 3 : 2)) return 4;
}
"""
