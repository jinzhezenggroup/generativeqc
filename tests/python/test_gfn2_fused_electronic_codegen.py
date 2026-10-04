"""Compiled generated electronic helpers preserve the native FMA contract."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.tensor.scf_cuda import (
    density_template_hash,
    weighted_density_template_hash,
)


def test_native_electronic_fma_cancellation_and_publication(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    header = tmp_path / "generated_gfn2_electronic_native.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_gfn2_electronic_native.py"),
            "--output",
            str(header),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    generated = header.read_text()
    assert density_template_hash() in generated
    assert weighted_density_template_hash() in generated

    source = tmp_path / "fused.cpp"
    source.write_text(r"""
#include <cmath>
#include <limits>
#include <initializer_list>
#include "generated_gfn2_electronic_native.hpp"
int main() {
  using namespace generativeqc::xtb::generated;
  double out=123.;
  if(!gfn2_population_update_tensor(1e308,2.,1e308,out) || out!=std::fma(-1e308,2.,1e308)) return 1;
  if(!gfn2_core_energy_update_tensor(1e308,2.,-1e308,out) || out!=std::fma(1e308,2.,-1e308)) return 2;
  double old=0.;
  for(double potential : {1e308,1e308,-1e308,-1e308}) old=std::fma(-0.25,potential,old);
  if(!gfn2_scalar_hamiltonian_update_tensor(0.5,1e308,1e308,-1e308,-1e308,0.,out) || out!=old) return 3;
  for(double x : {0.,0.1,-0.3,1e-200,1e200}) {
    const double expected=std::fma(-x,0.2,0.17);
    if(!gfn2_population_update_tensor(x,0.2,0.17,out) || out!=expected) return 4;
    const double expected_h=std::fma(-0.5*x,0.3,std::fma(-0.5*x,-0.2,0.13));
    if(!gfn2_multipole_hamiltonian_update_tensor(x,x,0.3,-0.2,0.13,out) || out!=expected_h) return 5;
  }
  out=123.;
  if(gfn2_core_energy_update_tensor(1e308,2.,1e308,out) || out!=123.) return 6;
  if(gfn2_population_update_tensor(std::numeric_limits<double>::quiet_NaN(),1.,1.,out) || out!=123.) return 7;
  double ew=0.;
  if(!gfn2_energy_weight_tensor(0.25,-2.,ew) || ew!=-0.5) return 8;
  double wc=0.;
  if(!gfn2_weighted_coefficient_tensor(3.,0.25,wc) || wc!=0.75) return 9;
  double contribution=0.;
  if(!gfn2_density_contribution_tensor(wc,2.,contribution) || contribution!=1.5) return 10;
  out=0.1;
  if(!gfn2_density_update_tensor(wc,2.,out,out) || out!=std::fma(wc,2.,0.1)) return 11;
  double charge=0.,mag=0.;
  if(!gfn2_restricted_population_publish_tensor(-0.3,1.0,charge) || charge!=0.7) return 11;
  if(!gfn2_spin_population_publish_tensor(-0.3,-0.2,1.0,charge,mag) ||
     charge!=0.5 || mag!=-0.1) return 12;
}
""")
    binary = tmp_path / "fused"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


def test_gfn2_density_consumers_share_generated_scalar_science() -> None:
    root = Path(__file__).resolve().parents[2]
    cuda_generator = (root / "tools/generate_gfn2_electronic_cuda.py").read_text()
    assert "density_template_hash()" in cuda_generator
    assert "weighted_density_template_hash()" in cuda_generator
    cpu = (root / "src/xtb/native/src/model/gfn2/eigensolver.cpp").read_text()
    cuda = (root / "src/xtb/native/src/backends/cuda/gfn2_density.cu").read_text()
    assert "gfn2_weighted_coefficient_tensor(" in cpu
    assert "gfn2_energy_weight_tensor(" in cpu
    assert "gfn2_weighted_coefficient_cuda_tensor(" in cuda
    assert "gfn2_density_contribution_cuda_tensor(" in cuda
    assert "gfn2_density_update_cuda_tensor(" in cuda
    assert "gfn2_energy_weight_cuda_tensor(" in cuda
    assert "fma(density_left, second, density)" not in cuda
    assert "fma(weighted_left, second, weighted_density)" not in cuda


def test_gfn2_mulliken_publication_consumers_use_generated_transforms() -> None:
    root = Path(__file__).resolve().parents[2]
    cpu = (root / "src/xtb/native/src/model/gfn2/mulliken.cpp").read_text()
    cuda = (root / "src/xtb/native/src/backends/cuda/gfn2_mulliken.cu").read_text()
    for source in (cpu, cuda):
        assert "spin_population_publish" in source
        assert "restricted_population_publish" in source
        assert "const double charge = alpha + beta" not in source
        assert "const double magnetization = alpha - beta" not in source
