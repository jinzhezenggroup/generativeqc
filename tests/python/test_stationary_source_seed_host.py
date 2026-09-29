"""Compile the production source-seed ABI against a deterministic CUDA queue stub."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_source_seed_rejects_replacement_and_drains_failed_uploads(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    begin = source.index("int stationary_seed_sources_v1(")
    end = source.index("int stationary_tasks(", begin)
    body = source[begin:end]
    harness = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
using std::size_t;
using std::uint64_t;
constexpr int cudaSuccess = 0;
constexpr size_t stationary_source_count = 7;
const double* pending_input = nullptr;
double* pending_output = nullptr;
size_t pending_count = 0;
bool throw_upload = false, fail_sync = false;
unsigned sync_attempts = 0;
int cudaStreamSynchronize(int) {
  ++sync_attempts;
  if (fail_sync) { fail_sync = false; return 1; }
  if (pending_input) std::copy_n(pending_input, pending_count, pending_output);
  pending_input = nullptr;
  return 0;
}
void cuda_check(int value) { if (value) throw std::runtime_error("CUDA error"); }
namespace generativeqc_stationary_cuda {
struct Owner {
  struct Context { int stream = 1; } context;
  size_t atoms = 2;
  bool failed = false, source_seed_ready = true, geometry_pending = false;
  uint64_t launches = 3, source_seed_launch_epoch = 3, synchronizations = 0;
  uint64_t uploads = 0, h2d_calls = 0;
  double* sources = nullptr;
};
void check(Owner& p) { if (p.failed) throw std::runtime_error("failed owner"); }
template<class F> int guarded(Owner* p, char*, size_t, F function) {
  try { function(); return 0; }
  catch (...) { if (p) p->failed = true; return 1; }
}
void upload(Owner& p, double* output, const double* input, size_t count, int) {
  pending_input = input; pending_output = output; pending_count = count;
  p.uploads += count * sizeof(double); ++p.h2d_calls;
  if (throw_upload) throw std::runtime_error("partial submission");
}
}
'''
    main = r'''
int main() {
  using namespace generativeqc_stationary_cuda;
  std::array<double, 42> input{}, output{};
  for (size_t i = 0; i < input.size(); ++i) input[i] = double(i);
  Owner p; p.sources = output.data(); char error[128]{};
  auto invoke = [&] { return stationary_seed_sources_v1(&p, input.data(), input.size(), error, sizeof(error)); };
  auto require = [](bool ok) { if (!ok) throw std::runtime_error("seed contract failed"); };
  require(invoke() == 0 && input == output && !p.source_seed_ready);
  require(p.h2d_calls == 1 && p.uploads == sizeof(input) && p.synchronizations == 1);
  require(invoke() != 0 && p.h2d_calls == 1);
  p = Owner{}; p.sources = output.data(); p.launches += 1;
  require(invoke() != 0 && p.h2d_calls == 0);
  p = Owner{}; p.sources = output.data(); p.geometry_pending = true;
  require(invoke() != 0 && p.h2d_calls == 0);
  p = Owner{}; p.sources = output.data(); input[0] = NAN;
  require(invoke() != 0 && p.h2d_calls == 0);
  input[0] = 0;
  p = Owner{}; p.sources = output.data();
  require(stationary_seed_sources_v1(&p, input.data(), input.size()-1, error, sizeof(error)) != 0);
  require(p.h2d_calls == 0);
  p = Owner{}; p.sources = output.data(); throw_upload = true;
  auto before = sync_attempts;
  require(invoke() != 0 && p.failed && !pending_input && sync_attempts == before+1);
  throw_upload = false;
  p = Owner{}; p.sources = output.data(); fail_sync = true;
  before = sync_attempts;
  require(invoke() != 0 && p.failed && !pending_input && sync_attempts == before+2);
}
'''
    cpp = tmp_path / "seed.cpp"
    cpp.write_text(harness + body + main)
    binary = tmp_path / "seed"
    subprocess.run(  # noqa: S603 -- compile checked-out source with fixed flags
        [compiler, "-std=c++20", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(  # noqa: S603 -- run only the just-compiled test harness
        [str(binary)], check=True, capture_output=True, text=True
    )
