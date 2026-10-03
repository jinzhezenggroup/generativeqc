"""Execute generated ERI orbit materialization with an independent value oracle."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)

ROOT = Path(__file__).resolve().parents[2]


def test_dense_orbit_coverage_and_unique_contracted_work(tmp_path: Path) -> None:
    """Check every dense slot, padded launch tail, repeated indices and batches."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache are required")
    header = emit_direct_cartesian_contraction_headers()[
        "generated_direct_eri_materialization.cuh"
    ]
    body = header[
        header.index("__device__ inline void materialize_eri_system_orbit(") :
    ]
    body = body.split("\n}  // namespace generativeqc::scf::cuda_execution", 1)[0]
    source, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    source.write_text(PREFIX + body + MAIN)
    subprocess.run(
        [cache, compiler, "-std=c++20", "-O2", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "56 9834496 1274406" in result.stdout
    assert "28 614656 82621" in result.stdout
    native = (ROOT / "src/scf/cuda/direct_cached_tensor_kernels.cu").read_text()
    kernel = native.split("__global__ void build_eri_kernel", 1)[1].split(
        "__global__ void build_fock_kernel", 1
    )[0]
    assert "materialize_eri_orbit(batch, element, eri)" in kernel
    assert "contracted_eri<" not in kernel


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated CUDA device",
)
@pytest.mark.parametrize("case", ("rhf", "uhf", "spherical-d", "spherical-f"))
def test_energy_only_cuda_batches_preserve_physical_oracle(case: str) -> None:
    """Cover ordinary RHF/UHF resident consumers, including real d/f AOs."""
    import numpy as np
    from generativeqc import Calculator, Primitive, Shell

    assert os.environ.get("SLURM_JOB_ID")
    if case.startswith("spherical"):
        angular = 2 if case.endswith("d") else 3
        basis = (
            Shell(0, 0, (Primitive(1.5, 1.0),)),
            Shell(0, angular, (Primitive(0.8, 1.0),)),
            Shell(1, 0, (Primitive(1.2, 1.0),)),
        )
        atoms = [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
        method, charge, multiplicity = "rhf", 1, 1
    else:
        basis = "sto-3g"
        atoms = [("O", (0.0, 0.0, 0.0)), ("H", (1.43233673, 0.0, 1.10715266))]
        if case == "rhf":
            atoms.append(("H", (-1.43233673, 0.0, 1.10715266)))
        method, charge, multiplicity = case, 0, 1 if case == "rhf" else 2
    shifted = [
        (z, (x, y, zc + (0.015 if i == 1 else 0.0)))
        for i, (z, (x, y, zc)) in enumerate(atoms)
    ]
    common = {
        "method": method,
        "basis": basis,
        "basis_representation": "spherical",
        "max_iterations": 160,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-15,
    }
    cpu = Calculator(device="cpu", **common)
    gpu = Calculator(device="cuda", **common)
    args = {
        "charges": [charge, charge],
        "multiplicities": [multiplicity, multiplicity],
    }
    with cpu.prepare_batch([atoms, shifted], **args) as reference:
        expected = reference.execute(properties=("energy",), strict=True)
    with gpu.prepare_batch([atoms, shifted], warm_start=True, **args) as batch:
        for coordinates in (
            None,
            None,
            [[xyz for _, xyz in shifted], [xyz for _, xyz in atoms]],
        ):
            actual = batch.execute(coordinates, properties=("energy",), strict=True)
            assert all(
                row.executed_backend == "cuda" and row.converged for row in actual.items
            )
            target = (
                expected.energies if coordinates is None else expected.energies[::-1]
            )
            np.testing.assert_allclose(actual.energies, target, atol=3e-9, rtol=0.0)


PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#define __device__
struct DeviceBatch { int nbf, batch_size; };
std::size_t calls=0;
// Independent triangular pair-of-pairs IDs identify the eightfold equivalence
// classes; the generated producer uses a full-grid lexicographic predicate.
double expected(int system,int i,int j,int k,int l) {
  const auto first=std::minmax(i,j),second=std::minmax(k,l);
  const auto p=first.second*(first.second+1)/2+first.first;
  const auto q=second.second*(second.second+1)/2+second.first;
  const auto a=std::max(p,q),b=std::min(p,q);
  return 1.0 + system*10000000.0 + a*(a+1)/2+b;
}
template<class Scalar>
Scalar contracted_eri(DeviceBatch,int system,int i,int j,int k,int l,int derivative) {
  if(derivative!=-1) throw std::runtime_error("unexpected derivative materialization");
  ++calls;
  return expected(system,i,j,k,l);
}
"""

MAIN = r"""
int main() {
  for(int n : {1,2,3,7,14,28,56}) {
    const int batches=n<28?3:1;
    const std::size_t count=static_cast<std::size_t>(n)*n*n*n;
    std::vector<double> eri(batches*count,std::numeric_limits<double>::quiet_NaN());
    calls=0;
    for(std::size_t element=0;element<eri.size()+255;++element)
      materialize_eri_orbit({n,batches},element,eri.data());
    const std::size_t pairs=n*(n+1)/2,unique=pairs*(pairs+1)/2;
    if(calls!=batches*unique) return 1;
    for(int system=0;system<batches;++system)
      for(int i=0;i<n;++i) for(int j=0;j<n;++j)
        for(int k=0;k<n;++k) for(int l=0;l<n;++l) {
          const auto index=system*count+((static_cast<std::size_t>(i)*n+j)*n+k)*n+l;
          if(eri[index]!=expected(system,i,j,k,l)) return 2;
        }
    // Selecting item 2 must read that item's metadata, but write only the
    // single-system destination rather than a batch-offset output slot.
    std::vector<double> selected(count+32,-123.0);
    calls=0;
    for(std::size_t element=0;element<count+255;++element)
      materialize_eri_system_orbit({n,3},2,element,selected.data()+16);
    if(calls!=unique) return 3;
    for(std::size_t i=0;i<16;++i)
      if(selected[i]!=-123.0 || selected[count+16+i]!=-123.0) return 4;
    for(int i=0;i<n;++i) for(int j=0;j<n;++j)
      for(int k=0;k<n;++k) for(int l=0;l<n;++l) {
        const auto index=((static_cast<std::size_t>(i)*n+j)*n+k)*n+l;
        if(selected[16+index]!=expected(2,i,j,k,l)) return 5;
      }
    std::cout<<n<<' '<<count<<' '<<unique<<'\n';
  }
}
"""
