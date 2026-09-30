"""The asynchronous scalar upload must borrow its caller's fenced storage."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _seed_helper(text: str) -> str:
    start = text.index("  void stage_reference_seed(")
    return text[start : text.index("\n  }", start) + len("\n  }")]


def _probe(helper: str) -> str:
    return (
        r"""
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
const void* queued_source = nullptr;
std::size_t copies = 0;
constexpr int cudaMemcpyHostToDevice = 1;
int cudaMemcpyAsync(void*, const void* source, std::size_t count, int, void*) {
  if (count != sizeof(double)) throw std::runtime_error("wrong scalar copy size");
  queued_source = source;
  ++copies;
  return 0;
}
void cuda_check(int status) { if (status) throw std::runtime_error("copy failed"); }
std::size_t checked_add(std::size_t a, std::size_t b) { return a + b; }
struct Probe {
  double device_seed = 0.0;
  struct State { double* bar_reference_electronic_energy; } state{&device_seed};
  void* stream = nullptr;
  std::size_t h2d = 0;
"""
        + helper
        + r"""
};
int main() {
  Probe owner;
  for (double seed : {0.0, 1.0, -0.25}) {
    queued_source = nullptr;
    owner.stage_reference_seed(seed);
    // Compare addresses before any deferred read: the pre-fix version fails
    // deterministically here without dereferencing an expired stack pointer.
    if (queued_source != &seed) {
      std::cerr << "queued scalar does not borrow the caller's live storage\n";
      return 1;
    }
    std::memcpy(&owner.device_seed, queued_source, sizeof(double));
    if (owner.device_seed != seed) return 2;
  }
  for (double seed : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    const auto before = copies;
    bool rejected = false;
    try { owner.stage_reference_seed(seed); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || copies != before) return 3;
  }
  if (owner.h2d != 3 * sizeof(double)) return 4;
  std::cout << "caller-owned deferred scalar and nonfinite guards passed\n";
}
"""
    )


def test_reference_seed_upload_borrows_fenced_caller_storage(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    text = (ROOT / "src/cc/lambda_response_cuda.cu").read_text()
    source = tmp_path / "seed.cpp"
    source.write_text(_probe(_seed_helper(text)))
    executable = tmp_path / "seed"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
