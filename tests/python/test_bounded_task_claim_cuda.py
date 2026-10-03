"""Protect the generated-task compactor's shared claim across skipped work.

Adapted from the claim-protocol regression authored by njzjz-bot in
0b99c6ce298f2726373f1909ad10a37b5acffc43 and its nonpaged adaptation in PR #1776.
The optional CUDA probe exercises the production claim prefix and the compactor's
strided candidate traversal, not integral math or complete-endpoint performance.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _claim_prefix() -> str:
    source = (ROOT / "src/scf/cuda/direct_bounded_tasks.cu").read_text()
    start = source.index("  while (true) {")
    stop = source.index("    if (block_quartet >= total) return;", start)
    return source[start:stop]


def _check_claim_barriers(claim: str) -> None:
    statements = "\n".join(
        line for line in claim.splitlines() if not line.strip().startswith("//")
    )
    assert statements.split("{", 1)[1].lstrip().startswith("__syncthreads();")
    assert statements.count("__syncthreads();") == 2
    assert statements.index("__syncthreads();") < statements.index("atomicAdd(")
    assert statements.rindex("__syncthreads();") > statements.index("atomicAdd(")


def test_task_claim_consumption_barrier_precedes_every_leader_overwrite() -> None:
    """Inactive/screened continues must encounter a CTA barrier before writing."""
    _check_claim_barriers(_claim_prefix())


@pytest.mark.parametrize("replacement", ["", "__syncwarp();"])
def test_task_claim_contract_rejects_missing_or_warp_only_barrier(
    replacement: str,
) -> None:
    claim = _claim_prefix().replace("__syncthreads();", replacement, 1)
    with pytest.raises(AssertionError):
        _check_claim_barriers(claim)


@pytest.fixture(scope="module")
def task_claim_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.environ.get("GENERATIVEQC_CLAIM_CUDA_TEST") != "1":
        pytest.skip("opt-in finite Slurm CUDA claim qualification")
    assert os.environ.get("SLURM_JOB_ID"), "CUDA qualification requires finite Slurm"
    cache = shutil.which("ccache")
    compiler = Path(os.environ["CUDA_PATH"]) / "bin/nvcc"
    assert cache and compiler.is_file()
    subprocess.run([cache, "--version"], check=True, timeout=10)
    directory = tmp_path_factory.mktemp("bounded-task-claim")
    source = directory / "probe.cu"
    source.write_text(
        r"""
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
constexpr unsigned threads = 128;
constexpr unsigned workers = 4;
constexpr unsigned products_count = 8;
constexpr unsigned counts[products_count] = {528, 1024, 1, 17, 256, 63, 0, 129};
__global__ void probe(const unsigned* candidate_counts, unsigned long long* global_cursor,
                      unsigned* visits, unsigned skip_mode) {
  __shared__ unsigned long long block_quartet;
  const std::size_t total = products_count;
"""
        + _claim_prefix()
        + r"""
    // Delay a nonleader warp before its first read, not merely a later reload.
    if (threadIdx.x / 32 == 1) {
      const auto began = clock64();
      while (clock64() - began < 100000ULL) {}
    }
    if (block_quartet >= total) return;
    const auto observed = block_quartet;
    if ((skip_mode & 1U) && observed % 3 == 0) continue;  // inactive system
    if ((skip_mode & 2U) && observed % 3 == 1) continue;  // block screening
    if (skip_mode & 4U) continue;  // all claims rejected
    const std::size_t candidate_count = candidate_counts[observed];
    for (std::size_t candidate = threadIdx.x; candidate < candidate_count;
         candidate += blockDim.x) {
      atomicAdd(visits + observed * threads + threadIdx.x, 1U);
    }
    // The production compactor has this barrier, bypassed by outer continues.
    __syncthreads();
  }
}
void checked(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::abort();
  }
}
int main() {
  unsigned *device_counts{}, *visits{};
  unsigned long long* cursor{};
  checked(cudaMalloc(&device_counts, sizeof(counts)));
  checked(cudaMalloc(&visits, products_count * threads * sizeof(unsigned)));
  checked(cudaMalloc(&cursor, sizeof(*cursor)));
  checked(cudaMemcpy(device_counts, counts, sizeof(counts), cudaMemcpyHostToDevice));
  for (unsigned replay = 0; replay < 16; ++replay) {
    for (unsigned skip_mode = 0; skip_mode < 5; ++skip_mode) {
      for (unsigned long long initial : {0ULL, 1ULL, 8ULL, 11ULL}) {
        checked(cudaMemset(visits, 0, products_count * threads * sizeof(unsigned)));
        checked(cudaMemcpy(cursor, &initial, sizeof(initial), cudaMemcpyHostToDevice));
        probe<<<workers, threads>>>(device_counts, cursor, visits, skip_mode);
        checked(cudaGetLastError());
        checked(cudaDeviceSynchronize());
        std::vector<unsigned> actual(products_count * threads);
        checked(cudaMemcpy(actual.data(), visits, actual.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        unsigned long long claimed{};
        checked(cudaMemcpy(&claimed, cursor, sizeof(claimed), cudaMemcpyDeviceToHost));
        assert(claimed == std::max(initial, static_cast<unsigned long long>(products_count)) + workers);
        for (unsigned ordinal = 0; ordinal < products_count; ++ordinal) {
          const bool skipped = ((skip_mode & 1U) && ordinal % 3 == 0) ||
                               ((skip_mode & 2U) && ordinal % 3 == 1) || (skip_mode & 4U);
          for (unsigned lane = 0; lane < threads; ++lane) {
            unsigned expected = 0;
            if (ordinal >= initial && !skipped && lane < counts[ordinal])
              expected = 1 + (counts[ordinal] - 1 - lane) / threads;
            assert(actual[ordinal * threads + lane] == expected);
          }
        }
      }
    }
  }
  checked(cudaFree(cursor));
  checked(cudaFree(visits)); checked(cudaFree(device_counts));
  std::puts("inactive and screened task claims preserve every reader");
}
"""
    )
    executable = directory / "probe"
    subprocess.run(
        [
            cache,
            str(compiler),
            "-std=c++20",
            "-arch=sm_120",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    return executable


@pytest.mark.parametrize("sanitizer", [None, "synccheck", "racecheck"])
def test_cuda_task_claim_readers_survive_skipped_products(
    task_claim_probe: Path,
    sanitizer: str | None,
) -> None:
    """Stress the source claim protocol; independent endpoint gates are separate."""
    command = [str(task_claim_probe)]
    if sanitizer:
        tool = shutil.which("compute-sanitizer")
        assert tool, "requested synchronization qualification needs compute-sanitizer"
        command = [tool, "--tool", sanitizer, "--error-exitcode", "91", *command]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=180
    )
    assert "preserve every reader" in result.stdout
    if sanitizer == "racecheck":
        assert "RACECHECK SUMMARY: 0 hazards" in result.stdout + result.stderr
    elif sanitizer:
        assert "ERROR SUMMARY: 0 errors" in result.stdout + result.stderr
