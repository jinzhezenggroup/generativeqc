"""Host-execute grid publication control flow with instrumented CUDA stand-ins."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
using I = std::int64_t;
int fault = 0, downloads = 0, fences = 0;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
int cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, int kind, int) {
  if (kind == cudaMemcpyDeviceToHost) {
    ++downloads;
    if (fault == 2) throw std::runtime_error("injected output submission failure");
  }
  if (bytes) std::memcpy(dst, src, bytes);
  return 0;
}
int cudaMemsetAsync(void* dst, int value, std::size_t bytes, int) {
  if (bytes) std::memset(dst, value, bytes);
  return 0;
}
void cuda_check(int value) { if (value) throw std::runtime_error("CUDA stand-in failure"); }
int cudaGetLastError() { return fault == 6 ? 1 : 0; }
template<class F> int guarded(char*, std::size_t, F operation) {
  try { operation(); return 0; } catch (...) { return 1; }
}
struct Metrics { double input_ms{},kernel_ms{},packing_ms{},library_ms{},output_ms{}; };
struct Context {
  std::mutex mutex;
  Metrics metrics;
  int error_value{}, *error = &error_value, stream = 1;
  void check_device() {}
  template<class F> void section(bool profile, double& metric, F operation) {
    operation();
    if (profile) {
      ++fences;
      if (&metric == &metrics.output_ms && fault == 3)
        throw std::runtime_error("injected output synchronization failure");
    }
  }
};
struct GridPlan {
  Context context;
  bool view_ready=true,features_ready=true,density_jets_ready=true;
  bool local=true,density_ready=true,use_orbitals=false,orbital_ready=false;
  std::size_t generation=7,nao=2,active_capacity=2,capacity=2,jets=4;
  std::size_t last_points{},last_active{},natom=1,nprimitive=2;
  std::size_t orbital_count[2]{},orbital_tile=1;
  unsigned feature_mask=15;
  double basis[64]{},points[6]{},features[26]{},ao[64]{},density[8]{},local_density[8]{};
  double work[128]{},psi[64]{},factor_panel[64]{}, *factors[2]{};
  std::size_t ao_ids[2]{};
};
void ao_kernel(const double*, std::size_t, std::size_t, std::size_t, const double*,
               std::size_t,std::size_t,double*,int* error,const std::size_t*) {
  if (fault == 1) *error = 7;
}
template<class... A> void gather_factor(A&&...) {}
template<class... A> void gemm(A&&...) {}
template<class... A> void orbital_feature_kernel(A&&...) {}
template<class... A> void finish_orbital_sigma(A&&...) {}
template<class... A> void gather_density(A&&...) {}
template<class... A> void feature_kernel(A&&...) {}
"""

MAIN = r"""
int main(int argc,char** argv) {
  if (argc != 2) return 1;
  const int mode = std::atoi(argv[1]);
  GridPlan p;
  const double points[6]{};
  const std::size_t ids[2]{0,1};
  double output[26]{};
  const bool deferred = mode == 4 || mode == 5;
  fault = mode == 4 ? 1 : mode;
  const auto npoint = mode == 7 ? 0u : 2u;
  const int status = grid_cuda_run_selected_impl(&p,points,npoint,1,ids,2,
      mode == 5 ? output : nullptr,nullptr,deferred,nullptr,0);
  const bool expected_success = mode == 0 || mode == 4 || mode == 7;
  if ((status == 0) != expected_success) return 2;
  if (p.view_ready != expected_success) return 3;
  if (!expected_success && p.density_jets_ready) return 4;
  if (mode == 4 && (downloads || fences || !*p.context.error || !p.density_jets_ready)) return 5;
  if (mode == 0 && (downloads != 1 || fences != 1 || !p.density_jets_ready)) return 6;
  if (mode == 7 && (p.density_jets_ready || downloads || fences != 1)) return 7;
  // A successful later generation can recover; no stale publication is required.
  fault = 0;
  if (grid_cuda_run_selected_impl(&p,points,2,1,ids,2,nullptr,nullptr,0,nullptr,0)) return 8;
  if (!p.view_ready || !p.density_jets_ready || p.generation != 9) return 9;
}
"""


@pytest.fixture(scope="module")
def publication_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/cuda_grid.cu").read_text()
    begin = source.index("static int grid_cuda_run_selected_impl(")
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    body = re.sub(r"<<<.*?>>>", "", source[begin:end], flags=re.DOTALL)
    directory = tmp_path_factory.mktemp("grid-publication")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text("\n".join((PREFIX, body, MAIN)))
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("mode", range(8))
def test_grid_publication_requires_the_selected_error_gate(
    publication_probe: Path, mode: int
) -> None:
    process = subprocess.run(
        [str(publication_probe), str(mode)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 0, (mode, process.returncode, process.stderr)
