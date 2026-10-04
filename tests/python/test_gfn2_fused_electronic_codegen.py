"""Compiled generated electronic helpers preserve the native FMA contract."""

import os
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
  const double alpha=-0.3, beta=-0.2;
  if(!gfn2_spin_population_publish_tensor(alpha,beta,1.0,charge,mag) ||
     charge!=alpha+beta+1.0 || mag!=alpha-beta) return 12;
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


def test_gfn2_cpu_generated_density_failures_mark_staging_status() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/xtb/native/src/model/gfn2/eigensolver.cpp").read_text()
    begin = source.index("  double band_energy = 0.0;")
    end = source.index("  const std::size_t spin_matrix_count", begin)
    publication = source[begin:end]
    failure = "return NumericalResult::kDataFailure;"
    status = (
        "thermodynamics.system_statuses[system] = "
        "GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED;"
    )
    failure_lines = [
        index for index, line in enumerate(publication.splitlines()) if failure in line
    ]
    assert failure_lines
    lines = publication.splitlines()
    for index in failure_lines:
        assert status in "\n".join(lines[max(0, index - 2) : index])


def test_gfn2_mulliken_publication_consumers_use_generated_transforms() -> None:
    root = Path(__file__).resolve().parents[2]
    cpu = (root / "src/xtb/native/src/model/gfn2/mulliken.cpp").read_text()
    cuda = (root / "src/xtb/native/src/backends/cuda/gfn2_mulliken.cu").read_text()
    for source in (cpu, cuda):
        assert "spin_population_publish" in source
        assert "restricted_population_publish" in source
        assert "const double charge = alpha + beta" not in source
        assert "const double magnetization = alpha - beta" not in source


def test_density_helper_failure_cannot_publish_previous_eigensolution(
    tmp_path: Path,
) -> None:
    """Run the real solver and batch commit with a deterministic LAPACK provider."""
    compiler, ccache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or ccache is None or sys.platform != "linux":
        pytest.skip(
            "host C++ compiler, ccache, and ELF section garbage collection required"
        )
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=15)
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_gfn2_electronic_native.py"),
            "--output",
            str(tmp_path / "generated_gfn2_electronic_native.hpp"),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    source = tmp_path / "failure_publication.cpp"
    source.write_text(r"""
#include "model/gfn2/eigensolver.cpp"
#include <iostream>
using namespace generativeqc::xtb::detail::gfn2;
static double injected_eigenvalue = -0.5, injected_coefficient = 1.0;
static LapackInt test_syevd(LapackInt, char, char, LapackInt n, double* a,
                           LapackInt, double* w, double*, LapackInt,
                           LapackInt*, LapackInt) {
  if (n != 1) std::abort();
  a[0] = injected_coefficient;
  w[0] = injected_eigenvalue;
  return 0;
}
static void test_trsm(int, int, int, int, int, LapackInt, LapackInt, double,
                     const double*, LapackInt, double*, LapackInt) {}
static void test_gemm(int, int, int, LapackInt m, LapackInt n, LapackInt k,
                     double alpha, const double* a, LapackInt, const double* b,
                     LapackInt, double, double* out, LapackInt) {
  if (m != 1 || n != 1 || k != 1) std::abort();
  out[0] = alpha * a[0] * b[0];
}
int check(int spins, double coefficient, double eigenvalue) {
  // Exercise actual native arithmetic/publication; only LAPACK/BLAS dispatch
  // is injected. No chemistry or external-provider qualification is claimed.
  auto backend = CpuLinearAlgebraAccess::make(
      {}, nullptr, nullptr, test_syevd, test_trsm, test_gemm, nullptr);
  EigensolverPlanData data;
  data.batch_size = 1;
  data.orbital_offsets = {0, 1};
  data.matrix_offsets = {0, 1};
  data.spin_channels = {spins};
  data.alpha_electron_counts = {1.0};
  data.beta_electron_counts = {1.0};
  for (auto& field : data.wavefunction_fields) field.system_offsets = {0, spins};
  data.wavefunction_fields[2].system_offsets = {0, 2};
  double factor = 1.0;
  std::uint64_t generation = 1;
  generativeqc_xtb_status_t overlap_status = GENERATIVEQC_XTB_STATUS_SUCCESS;
  EigensolverOverlapCache overlap;
  overlap.cholesky_factors = &factor;
  overlap.geometry_generations = &generation;
  overlap.system_statuses = &overlap_status;
  double scratch[20]{}, staged[10]{}, staged_thermo[5]{};
  generativeqc_xtb_status_t staged_status = GENERATIVEQC_XTB_STATUS_SUCCESS;
  EigensolverWorkspace workspace;
  workspace.coefficients = scratch;
  workspace.eigenvalues = scratch + 2;
  workspace.occupations = scratch + 4;
  workspace.densities = scratch + 6;
  workspace.energy_weighted_densities = scratch + 8;
  workspace.lapack_work = scratch + 10;
  workspace.batch_coefficients = staged;
  workspace.batch_eigenvalues = staged + 2;
  workspace.batch_occupations = staged + 4;
  workspace.batch_densities = staged + 6;
  workspace.batch_energy_weighted_densities = staged + 8;
  workspace.batch_system_statuses = &staged_status;
  workspace.batch_chemical_potentials = staged_thermo;
  workspace.batch_entropies = staged_thermo + 2;
  workspace.batch_band_energies = staged_thermo + 3;
  workspace.batch_free_energies = staged_thermo + 4;
  const auto staging_wavefunction = make_batch_staging_wavefunction(workspace);
  const auto staging_thermodynamics = make_batch_staging_thermodynamics(data, workspace);
  double hamiltonian[2]{1.0, 1.0};
  injected_coefficient = 1.0;
  injected_eigenvalue = -0.5;
  if (solve_system_unchecked(data, 0, overlap, 1, hamiltonian, 0.0, backend,
          workspace, staging_wavefunction, staging_thermodynamics) !=
      NumericalResult::kSuccess) return 1;
  if (staged_status != GENERATIVEQC_XTB_STATUS_SUCCESS) return 2;

  injected_coefficient = coefficient;
  injected_eigenvalue = eigenvalue;
  if (solve_system_unchecked(data, 0, overlap, 1, hamiltonian, 0.0, backend,
          workspace, staging_wavefunction, staging_thermodynamics) !=
      NumericalResult::kDataFailure) return 3;
  double published[10], published_thermo[5];
  std::fill_n(published, 10, 123.25);
  std::fill_n(published_thermo, 5, 123.25);
  generativeqc_xtb_status_t published_status = GENERATIVEQC_XTB_STATUS_SUCCESS;
  EigensolverWavefunctionView output;
  output.coefficients = published;
  output.eigenvalues = published + 2;
  output.occupations = published + 4;
  output.density = published + 6;
  output.energy_weighted_density = published + 8;
  EigensolverThermodynamicsView thermo{
      &published_status, 1, published_thermo, 2, published_thermo + 2, 1,
      published_thermo + 3, 1, published_thermo + 4, 1};
  // The batch caller intentionally commits per-system results after data
  // failure, relying on the recorded status to suppress numerical publication.
  commit_batch_solve_results(data, workspace, output, thermo);
  if (published_status != GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED) return 4;
  for (double value : published) if (value != 123.25) return 5;
  for (double value : published_thermo) if (value != 123.25) return 6;
  return 0;
}
int main() {
  struct Case { int spins; double coefficient, eigenvalue; };
  const Case cases[] = {
      {1, 1.0, 1e308},    // restricted orbital energy weight overflows
      {1, 1e308, 0.0},    // restricted coefficient weighting overflows
      {1, 1e100, 1e250},  // restricted energy-weighted coefficient overflows
      {2, 1e100, 1e250},  // unrestricted energy-weighted coefficient overflows
  };
  for (const auto& test : cases) {
    if (int result = check(test.spins, test.coefficient, test.eigenvalue)) {
      std::cerr << "spins=" << test.spins << " coefficient=" << test.coefficient
                << " eigenvalue=" << test.eigenvalue << " failure=" << result << '\n';
      return result;
    }
  }
}
""")
    obj = tmp_path / "failure_publication.o"
    subprocess.run(
        [
            ccache,
            compiler,
            "-std=c++20",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            "-I",
            str(root / "src/xtb/native/src"),
            "-I",
            str(root / "src/xtb/native"),
            "-I",
            str(root / "include"),
            "-I",
            str(tmp_path),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
        timeout=60,
    )
    binary = tmp_path / "failure_publication"
    subprocess.run(
        [compiler, "-Wl,--gc-sections", str(obj), "-ldl", "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
