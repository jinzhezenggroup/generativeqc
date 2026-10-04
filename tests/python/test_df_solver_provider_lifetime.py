"""DF solver releases must not hide another owner's measured provider growth.

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
    begin = source.index("      auto allocation = cudaMalloc(")
    end = source.index("      cuda_check(allocation);", begin)
    fallback = source[begin : end + len("      cuda_check(allocation);")]
    directory = tmp_path_factory.mktemp("df-solver-provider-lifetime")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(
        PREFIX + cleanup + "\nvoid retry() {\n" + fallback + "\n}\n};\n}\n" + MAIN
    )
    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("operation", ["cleanup", "fallback"])
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
static int destroys=0, frees=0, allocations=0, cleared=0;
int cudaStreamSynchronize(void*) { return 0; }
int cublasDestroy(void*) { ++destroys; released.set_value(); return 0; }
int cudaEventDestroy(void*) { return 0; }
int cudaFree(void*) { ++frees; return 0; }
int cudaStreamDestroy(void*) { return 0; }
int cudaGetLastError() { ++cleared; return 0; }
int cudaMalloc(void** pointer,std::size_t) {
  if (++allocations==1) return cudaErrorMemoryAllocation;
  *pointer=reinterpret_cast<void*>(4); return 0;
}
void cuda_check(int code) { if(code) throw std::runtime_error("CUDA failure"); }
void blas_check(int code) { if(code) throw std::runtime_error("BLAS failure"); }
namespace generativeqc::cc {
struct Owner {
  void *stream=reinterpret_cast<void*>(1), *blas=reinterpret_cast<void*>(2);
  void *trial_begin{}, *trial_end{};
  unsigned char *base=reinterpret_cast<unsigned char*>(3);
  struct { bool matrix_gemm=true; } plan;
  struct { std::size_t total=1024; } layout;
  int replans=0;
  void scalar_plan() { plan.matrix_gemm=false; layout.total=512; ++replans; }
"""

MAIN = r"""
int main(int argc,char** argv) {
  if(argc!=2) return 99;
  const bool fallback=std::string(argv[1])=="fallback";
  generativeqc::cc::Owner owner;
  auto release=released.get_future();
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
  measurement.unlock();
  worker.join();
  if(released_during_measurement) {
    std::cerr << "Provider released during another owner's allocation measurement: "
                 "a 96 MiB release can hide 160 MiB growth as 64 MiB.\n";
    return 1;
  }
  if(destroys!=1 || owner.blas) return 2;
  if(fallback) {
    if(owner.plan.matrix_gemm || owner.replans!=1 || allocations!=2 || cleared!=1 ||
       owner.layout.total!=512 || owner.base!=reinterpret_cast<unsigned char*>(4)) return 3;
    owner.cleanup();
  }
  if(owner.base || owner.stream || frees!=1) return 4;
}
"""
