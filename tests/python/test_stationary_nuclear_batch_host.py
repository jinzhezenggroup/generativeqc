"""Execute emitted nuclear pair/batch code on the host, without a CUDA claim."""

from __future__ import annotations

import ast
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
#include <limits>
#include <vector>
using std::isfinite;
#define __device__
#define __global__
#define __noinline__
struct { size_t x{}; } blockIdx, threadIdx;
constexpr size_t stationary_nuclear_source = 6;
int atomicExch(int* p, int v) { int old=*p; *p=v; return old; }
double finite(double value, int* error, double fallback) {
  if (isfinite(value)) return value;
  atomicExch(error, 1); return fallback;
}
"""

MAIN = r"""
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  const size_t na = std::strtoul(argv[1], nullptr, 10);
  const int fault = std::atoi(argv[2]);
  std::vector<double> charges(na), centers(3*na), initial(24*na, 0.125);
  for (size_t a=0; a<na; ++a) {
    charges[a]=1+a%8;
    centers[3*a]=0.173*a;
    centers[3*a+1]=std::sin(double(a)+0.4);
    centers[3*a+2]=std::cos(double(a)+0.7);
  }
  if (na==2) centers={-0.5,0,0,0.5,0,0};
  unsigned kind = fault==1 ? 99 : 0;
  if (fault==2) charges[0]=0;
  if (fault==3) charges[0]=std::numeric_limits<double>::quiet_NaN();
  if (fault==4) centers[0]=std::numeric_limits<double>::infinity();
  if (fault==5) std::copy_n(centers.data(),3,centers.data()+3);
  auto single=initial, batch=initial;
  int single_error=0, batch_error=0;
  for (size_t a=0; a<na && !single_error; ++a)
    for (size_t b=0; b<a && !single_error; ++b)
      nuclear_kernel(kind,a,b,charges[a],charges[b],centers.data(),na,
                     single.data(),&single_error);
  nuclear_all_kernel(kind,charges.data(),centers.data(),na,batch.data(),&batch_error);
  if (single_error!=batch_error || single!=batch) return 2;
  if (bool(batch_error)!=bool(fault)) return 3;
  for (size_t i=0; i<initial.size(); ++i)
    if ((i<18*na || i>=21*na) && batch[i]!=initial[i]) return 4;
  if (na==2 && !fault) {
    if (batch[18*na]!=2.125 || batch[18*na+3]!=-1.875) return 5;
  }
  // Only the one scheduled worker may touch the source arena.
  for (int inactive=0; inactive<2; ++inactive) {
    blockIdx.x=inactive; threadIdx.x=1-inactive;
    batch=initial; batch_error=0;
    nuclear_all_kernel(0,charges.data(),centers.data(),na,batch.data(),&batch_error);
    if (batch!=initial || batch_error) return 6;
  }
}
"""


@pytest.fixture(scope="module")
def nuclear_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from generativeqc_compiler.integral.first_derivative_native import (
        emit_first_derivative_cuda,
    )

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    path = ROOT / "python/generativeqc_compiler/method/stationary_cuda.py"
    tree = ast.parse(path.read_text())
    value = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_STATIONARY_SCIENTIFIC_KERNELS"
            for target in node.targets
        )
    )
    kernels = ast.literal_eval(value)
    begin = kernels.index("__device__ bool nuclear_pair(")
    end = kernels.index("__global__ void validate_centers(", begin)
    # Strip includes/qualifiers only. The nuclear primitive and request dispatch
    # are the actual compiler-emitted CUDA arithmetic, not a hand-written oracle.
    primitive = emit_first_derivative_cuda((("nuclear", ()),))
    primitive = "\n".join(
        line for line in primitive.splitlines() if not line.startswith("#include")
    )
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    stride = re.search(r"\brecord_stride\s*=\s*(\d+)", header)
    assert stride is not None
    directory = tmp_path_factory.mktemp("stationary-nuclear")
    source, executable = directory / "probe.cpp", directory / "probe"
    source.write_text(
        PREFIX
        + f"\nconstexpr size_t record_stride = {stride[1]};\n"
        + primitive
        + kernels[begin:end]
        + MAIN
    )
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize(
    ("atoms", "fault"),
    [(n, 0) for n in (1, 2, 7, 18, 36, 128)] + [(7, f) for f in range(1, 6)],
)
def test_emitted_nuclear_batch_matches_ordered_pair_execution(
    nuclear_probe: Path, atoms: int, fault: int
) -> None:
    result = subprocess.run(
        [str(nuclear_probe), str(atoms), str(fault)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (atoms, fault, result.returncode, result.stderr)
