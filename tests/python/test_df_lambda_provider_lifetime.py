"""Provider release must not corrupt another owner's measured allocation delta."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include "runtime/allocation_measurement.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
namespace runtime = generativeqc::runtime;
using cudaStream_t = void*;
using cublasHandle_t = void*;
constexpr int cudaErrorMemoryAllocation = 2;
std::promise<void> released;
std::atomic<int> handles{0}, allocations{0}, release_calls{0};
int malloc_calls = 0;
bool fail_first_allocation = false;
int cudaStreamSynchronize(cudaStream_t) { return 0; }
int cudaStreamDestroy(cudaStream_t) { return 0; }
int cudaGetLastError() { return 0; }
int cublasDestroy(cublasHandle_t p) {
  delete static_cast<int*>(p);
  --handles;
  if (++release_calls == 1) released.set_value();
  return 0;
}
int cudaMalloc(void** p, std::size_t bytes) {
  if (++malloc_calls == 1 && fail_first_allocation) return cudaErrorMemoryAllocation;
  *p = new unsigned char[bytes];
  ++allocations;
  return 0;
}
int cudaFree(void* p) {
  delete[] static_cast<unsigned char*>(p);
  --allocations;
  return 0;
}
void cuda_check(int code) { if (code) throw std::runtime_error("CUDA failure"); }
void blas_check(int code) { if (code) throw std::runtime_error("BLAS failure"); }
"""

DRIVER = r"""
int main(int argc, char** argv) {
  if (argc != 2) return 10;
  const bool fallback = std::string(argv[1]) == "fallback";
  std::promise<void> started;
  auto entered = started.get_future();
  auto release = released.get_future();
  // A different owner's before/after cudaMemGetInfo interval is active. A
  // released old handle (or its arena) would hide some new provider storage.
  std::unique_lock<std::mutex> measurement(runtime::allocation_measurement_mutex);
  std::thread worker([&] {
    started.set_value();
    if (fallback) {
      allocation_fallback();
    } else {
      Storage storage;
      storage.blas = new int(1);
      ++handles;
    }
  });
  entered.wait();
  const bool released_during_measurement =
      release.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
  measurement.unlock();
  worker.join();
  if (released_during_measurement) {
    std::cerr << "provider release escaped allocation measurement lock\n";
    return 1;
  }
  if (release_calls != 1 || handles || allocations) return 2;
  if (fallback && malloc_calls != 2) return 3;
}
"""


@pytest.fixture(scope="module")
def provider_lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/cc/df_lambda_cuda.cu").read_text()
    storage = source[source.index("struct Storage {") : source.index("\n__global__")]
    start = source.index("    auto allocation = cudaMalloc(")
    end = source.index("    cuda_check(allocation);", start) + len(
        "    cuda_check(allocation);"
    )
    # Compile the actual optional matrix/staged allocation fallback, including
    # provider destruction, rather than a model of its ownership decisions.
    fallback = (
        r"""
void allocation_fallback() {
  Storage storage;
  storage.blas = new int(1);
  ++handles;
  struct { bool df_matrix_gemm = true, df_auxiliary_reduction = true; } metrics;
  std::size_t cursor = 16, fallback_cursor = 4;
  auto capacity = [] {};
  auto scalar_plan = [&] { metrics.df_matrix_gemm = false; cursor = 8; };
  fail_first_allocation = true;
"""
        + source[start:end]
        + r"""
  if (metrics.df_matrix_gemm || !metrics.df_auxiliary_reduction || cursor != 8)
    throw std::runtime_error("incorrect scalar fallback");
}
"""
    )
    folder = tmp_path_factory.mktemp("df-lambda-provider-lifetime")
    unit, obj, binary = folder / "probe.cpp", folder / "probe.o", folder / "probe"
    unit.write_text(PREFIX + storage + fallback + DRIVER)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-pthread",
            "-I" + str(ROOT / "src"),
            "-c",
            str(unit),
            "-o",
            str(obj),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, "-pthread", str(obj), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("path", ["destructor", "fallback"])
def test_provider_release_serializes_with_measurement(
    provider_lifetime_probe: Path, path: str
) -> None:
    result = subprocess.run(
        [str(provider_lifetime_probe), path],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
