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
    compiler, ccache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or ccache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=15)
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
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_weighted_gram_native.py"),
            "--output",
            str(tmp_path / "generated_weighted_gram_native.hpp"),
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
#include "generated_weighted_gram_native.hpp"
int main() {
  using namespace generativeqc::xtb::generated;
  namespace gram = generativeqc::tensor::weighted_gram::generated;
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
  if(!gram::energy_weight(0.25,-2.,ew) || ew!=-0.5) return 8;
  double wc=0.;
  if(!gram::weighted_coefficient(3.,0.25,wc) || wc!=0.75) return 9;
  double contribution=0.;
  if(!gram::density_contribution(wc,2.,contribution) || contribution!=1.5) return 10;
  out=0.1;
  if(!gram::density_update(wc,2.,out,out) || out!=std::fma(wc,2.,0.1)) return 11;
  double charge=0.,mag=0.;
  if(!gfn2_restricted_population_publish_tensor(-0.3,1.0,charge) || charge!=0.7) return 11;
  const double alpha=-0.3, beta=-0.2;
  if(!gfn2_spin_population_publish_tensor(alpha,beta,1.0,charge,mag) ||
     charge!=alpha+beta+1.0 || mag!=alpha-beta) return 12;
}
""")
    binary = tmp_path / "fused"
    obj = tmp_path / "fused.o"
    subprocess.run(
        [
            ccache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=30,
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
    common = (root / "src/tensor/weighted_gram.hpp").read_text()
    from generativeqc_compiler.method.gfn2_density_lowering import (
        emit_gfn2_density_contract,
    )

    assert cuda.count('#include "generated_gfn2_density_contract.inc"') == 2
    assert "gfn2_density_update_cuda_tensor(" not in cuda
    cuda = cuda.replace(
        '#include "generated_gfn2_density_contract.inc"', emit_gfn2_density_contract()
    )
    assert '#include "tensor/weighted_gram.hpp"' in cpu
    assert "weighted_gram::execute_column_major(" in cpu
    assert "weighted_gram::energy_weights_inplace(" in cpu
    assert "weighted_gram::generated::energy_weight(" in cpu
    assert "generated::weighted_coefficient(" in common
    assert "form_density_column_major(" not in cpu
    assert "gfn2_weighted_coefficient_tensor(" not in cpu
    assert "gfn2_energy_weight_tensor(" not in cpu
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
            str(root / "tools/generate_weighted_gram_native.py"),
            "--output",
            str(tmp_path / "generated_weighted_gram_native.hpp"),
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
static double injected_eigenvalues[2], injected_coefficients[2];
static int injected_spin, expected_spins, gemm_calls;
static LapackInt expected_n = 1;
static const EigensolverWorkspace* traced_workspace;
static bool forbid_allocation = false;
void* operator new(std::size_t size) {
  if (forbid_allocation) std::abort();
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
static LapackInt test_syevd(LapackInt, char, char, LapackInt n, double* a,
                           LapackInt, double* w, double*, LapackInt,
                           LapackInt*, LapackInt) {
  if (n != expected_n || injected_spin >= expected_spins) std::abort();
  if (n == 1) {
    a[0] = injected_coefficients[injected_spin];
    w[0] = injected_eigenvalues[injected_spin];
  } else {
    for (LapackInt orbital = 0; orbital < n; ++orbital) {
      w[orbital] = orbital - 1.25 + 0.375 * injected_spin;
      for (LapackInt row = 0; row < n; ++row)
        a[row + orbital * n] = (row % 2 ? -1. : 1.) *
                               (0.125 + (row + 1.) / (orbital + 2. + injected_spin));
    }
  }
  ++injected_spin;
  return 0;
}
static void test_trsm(int, int, int, int, int, LapackInt, LapackInt, double,
                     const double*, LapackInt, double*, LapackInt) {}
static void test_gemm(int layout, int ta, int tb, LapackInt m, LapackInt n, LapackInt k,
                     double alpha, const double* a, LapackInt lda, const double* b,
                     LapackInt ldb, double beta, double* out, LapackInt ldc) {
  const int spin = gemm_calls / 2;
  const bool energy_weighted = gemm_calls % 2 != 0;
  const auto& workspace = *traced_workspace;
  const std::size_t matrix_count = static_cast<std::size_t>(n) * n;
  if (layout != 102 || ta != 111 || tb != 112 || m != expected_n || n != expected_n ||
      k != expected_n || lda != n || ldb != n || ldc != n || alpha != 1.0 || beta != 0.0 ||
      spin >= expected_spins || a != workspace.lapack_work ||
      b != workspace.coefficients + spin * matrix_count ||
      out != (energy_weighted ? workspace.energy_weighted_densities : workspace.densities) +
             spin * matrix_count) std::abort();
  for (LapackInt orbital = 0; orbital < n; ++orbital) {
    const double occupation = expected_spins == 1
                                  ? workspace.occupations[orbital] + workspace.occupations[n + orbital]
                                  : workspace.occupations[spin * n + orbital];
    const double energy = workspace.eigenvalues[spin * n + orbital];
    const double weight = energy_weighted ? occupation * energy : occupation;
    for (LapackInt row = 0; row < n; ++row)
      if (a[row + orbital * n] != b[row + orbital * n] * weight) std::abort();
    // Check the method's borrowed vector still occupies lapack_work+n*n.
    const double expected_vector = expected_spins == 1 ? weight : occupation * energy;
    if (workspace.lapack_work[matrix_count + orbital] != expected_vector) std::abort();
  }
  ++gemm_calls;
  for (LapackInt column = 0; column < n; ++column) {
    for (LapackInt row = 0; row < n; ++row) {
      double sum = 0.0;
      for (LapackInt orbital = 0; orbital < n; ++orbital)
        sum += a[row + orbital * n] * b[column + orbital * n];
      out[row + column * n] = sum;
    }
  }
}
int check(int spins, double coefficient, double eigenvalue, int expected_calls, bool late_beta) {
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
  constexpr double guard = 987.125;
  std::array<double, 22> scratch_arena;
  std::array<double, 12> staging_arena;
  std::array<double, 7> thermo_arena;
  scratch_arena.fill(guard);
  staging_arena.fill(guard);
  thermo_arena.fill(guard);
  double* scratch = scratch_arena.data() + 1;
  double* staged = staging_arena.data() + 1;
  double* staged_thermo = thermo_arena.data() + 1;
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
  traced_workspace = &workspace;
  expected_spins = spins;
  expected_n = 1;
  injected_spin = gemm_calls = 0;
  std::fill_n(injected_coefficients, 2, 1.0);
  std::fill_n(injected_eigenvalues, 2, -0.5);
  forbid_allocation = true;
  const auto success = solve_system_unchecked(data, 0, overlap, 1, hamiltonian, 0.0, backend,
          workspace, staging_wavefunction, staging_thermodynamics);
  forbid_allocation = false;
  if (success != NumericalResult::kSuccess || gemm_calls != 2 * spins) return 1;
  if (staged_status != GENERATIVEQC_XTB_STATUS_SUCCESS) return 2;
  const auto previous_staging = staging_arena;
  const auto previous_thermo = thermo_arena;

  for (int spin = late_beta ? 1 : 0; spin < spins; ++spin) {
    injected_coefficients[spin] = coefficient;
    injected_eigenvalues[spin] = eigenvalue;
  }
  injected_spin = gemm_calls = 0;
  forbid_allocation = true;
  const auto failure = solve_system_unchecked(data, 0, overlap, 1, hamiltonian, 0.0, backend,
          workspace, staging_wavefunction, staging_thermodynamics);
  forbid_allocation = false;
  if (failure != NumericalResult::kDataFailure || gemm_calls != expected_calls) return 3;
  if (staging_arena != previous_staging || thermo_arena != previous_thermo) return 7;
  if (scratch_arena.front() != guard || scratch_arena.back() != guard ||
      factor != 1.0 || generation != 1 || overlap_status != GENERATIVEQC_XTB_STATUS_SUCCESS)
    return 8;
  for (std::size_t i = 13; i < scratch_arena.size(); ++i)
    if (scratch_arena[i] != guard) return 9;
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
int check_success(LapackInt n, int spins) {
  const std::size_t matrix_count = static_cast<std::size_t>(n) * n;
  auto backend = CpuLinearAlgebraAccess::make(
      {}, nullptr, nullptr, test_syevd, test_trsm, test_gemm, nullptr);
  EigensolverPlanData data;
  data.batch_size = 1;
  data.orbital_offsets = {0, n};
  data.matrix_offsets = {0, static_cast<std::int64_t>(matrix_count)};
  data.spin_channels = {spins};
  data.alpha_electron_counts = {n - 0.25};
  data.beta_electron_counts = {n - 0.75};
  data.wavefunction_fields[0].system_offsets = {0, spins * static_cast<std::int64_t>(matrix_count)};
  data.wavefunction_fields[1].system_offsets = {0, spins * n};
  data.wavefunction_fields[2].system_offsets = {0, 2 * n};
  data.wavefunction_fields[3].system_offsets = data.wavefunction_fields[0].system_offsets;
  data.wavefunction_fields[4].system_offsets = data.wavefunction_fields[0].system_offsets;
  std::array<double, 256> scratch, staged, published;
  std::array<double, 7> staged_thermo, published_thermo;
  constexpr double guard = 987.125;
  scratch.fill(guard);
  staged.fill(guard);
  published.fill(guard);
  staged_thermo.fill(guard);
  published_thermo.fill(guard);
  EigensolverWorkspace workspace;
  workspace.coefficients = scratch.data() + 1;
  workspace.eigenvalues = workspace.coefficients + 2 * matrix_count;
  workspace.occupations = workspace.eigenvalues + 2 * n;
  workspace.densities = workspace.occupations + 2 * n;
  workspace.energy_weighted_densities = workspace.densities + 2 * matrix_count;
  workspace.lapack_work = workspace.energy_weighted_densities + 2 * matrix_count;
  workspace.batch_coefficients = staged.data() + 1;
  workspace.batch_eigenvalues = workspace.batch_coefficients + 2 * matrix_count;
  workspace.batch_occupations = workspace.batch_eigenvalues + 2 * n;
  workspace.batch_densities = workspace.batch_occupations + 2 * n;
  workspace.batch_energy_weighted_densities = workspace.batch_densities + 2 * matrix_count;
  generativeqc_xtb_status_t staged_status = GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED;
  workspace.batch_system_statuses = &staged_status;
  workspace.batch_chemical_potentials = staged_thermo.data() + 1;
  workspace.batch_entropies = staged_thermo.data() + 3;
  workspace.batch_band_energies = staged_thermo.data() + 4;
  workspace.batch_free_energies = staged_thermo.data() + 5;
  const auto staging_wavefunction = make_batch_staging_wavefunction(workspace);
  const auto staging_thermodynamics = make_batch_staging_thermodynamics(data, workspace);
  std::array<double, 25> factor{};
  for (LapackInt i = 0; i < n; ++i) factor[i + i * n] = 1.0;
  const auto original_factor = factor;
  std::uint64_t generation = 1;
  generativeqc_xtb_status_t overlap_status = GENERATIVEQC_XTB_STATUS_SUCCESS;
  EigensolverOverlapCache overlap;
  overlap.cholesky_factors = factor.data();
  overlap.geometry_generations = &generation;
  overlap.system_statuses = &overlap_status;
  std::array<double, 50> hamiltonians{};
  traced_workspace = &workspace;
  expected_spins = spins;
  expected_n = n;
  injected_spin = gemm_calls = 0;
  forbid_allocation = true;
  const auto result = solve_system_unchecked(data, 0, overlap, 1, hamiltonians.data(), 0.0, backend,
                                             workspace, staging_wavefunction, staging_thermodynamics);
  forbid_allocation = false;
  if (result != NumericalResult::kSuccess || gemm_calls != 2 * spins ||
      staged_status != GENERATIVEQC_XTB_STATUS_SUCCESS) return 10;
  // Independently known zero-temperature fractional occupations and the method's
  // original restricted versus alpha-then-beta band accumulation order.
  double expected_band = 0.0;
  for (int spin = 0; spin < 2; ++spin) {
    for (LapackInt orbital = 0; orbital < n; ++orbital) {
      const double expected_occupation = orbital + 1 < n ? 1.0 : spin == 0 ? 0.75 : 0.25;
      if (workspace.batch_occupations[spin * n + orbital] != expected_occupation) return 11;
    }
  }
  for (int spin = 0; spin < spins; ++spin) {
    for (LapackInt orbital = 0; orbital < n; ++orbital) {
      const double occupation = spins == 1
                                    ? workspace.batch_occupations[orbital] + workspace.batch_occupations[n + orbital]
                                    : workspace.batch_occupations[spin * n + orbital];
      const double energy = orbital - 1.25 + 0.375 * spin;
      expected_band += occupation * energy;
    }
    for (LapackInt row = 0; row < n; ++row) {
      for (LapackInt column = 0; column < n; ++column) {
        long double density = 0.0L, weighted_density = 0.0L;
        for (LapackInt orbital = 0; orbital < n; ++orbital) {
          const double occupation = spins == 1
                                        ? workspace.batch_occupations[orbital] + workspace.batch_occupations[n + orbital]
                                        : workspace.batch_occupations[spin * n + orbital];
          const double* c = workspace.batch_coefficients + spin * matrix_count;
          const long double term = static_cast<long double>(c[row * n + orbital]) *
                                   occupation * c[column * n + orbital];
          density += term;
          weighted_density += term * (orbital - 1.25 + 0.375 * spin);
        }
        const std::size_t index = spin * matrix_count + row * n + column;
        const auto close = [](double actual, long double expected) {
          return std::abs(static_cast<long double>(actual) - expected) <=
                 2e-13L * std::max(1.0L, std::abs(expected));
        };
        if (!close(workspace.batch_densities[index], density) ||
            !close(workspace.batch_energy_weighted_densities[index], weighted_density)) return 12;
      }
    }
  }
  if (*workspace.batch_band_energies != expected_band ||
      *workspace.batch_free_energies != expected_band) return 13;
  EigensolverWavefunctionView output;
  output.coefficients = published.data() + 1;
  output.eigenvalues = output.coefficients + 2 * matrix_count;
  output.occupations = output.eigenvalues + 2 * n;
  output.density = output.occupations + 2 * n;
  output.energy_weighted_density = output.density + 2 * matrix_count;
  generativeqc_xtb_status_t published_status = GENERATIVEQC_XTB_STATUS_EIGENSOLVER_FAILED;
  EigensolverThermodynamicsView thermo{
      &published_status, 1, published_thermo.data() + 1, 2, published_thermo.data() + 3, 1,
      published_thermo.data() + 4, 1, published_thermo.data() + 5, 1};
  forbid_allocation = true;
  commit_batch_solve_results(data, workspace, output, thermo);
  forbid_allocation = false;
  if (published != staged || published_thermo != staged_thermo ||
      published_status != GENERATIVEQC_XTB_STATUS_SUCCESS) return 17;
  if (scratch.front() != guard || staged.front() != guard ||
      staged_thermo.front() != guard || staged_thermo.back() != guard ||
      factor != original_factor || generation != 1 || overlap_status != GENERATIVEQC_XTB_STATUS_SUCCESS)
    return 14;
  for (std::size_t i = 1 + 7 * matrix_count + 5 * n; i < scratch.size(); ++i)
    if (scratch[i] != guard) return 15;
  for (std::size_t i = 1 + 6 * matrix_count + 4 * n; i < staged.size(); ++i)
    if (staged[i] != guard) return 16;
  return 0;
}

int main() {
  for (LapackInt n : {2, 3, 5}) {
    for (int spins : {1, 2}) {
      if (int result = check_success(n, spins)) {
        std::cerr << "n=" << n << " spins=" << spins << " failure=" << result << '\n';
        return result;
      }
    }
  }
  struct Case { int spins; double coefficient, eigenvalue; int calls; bool late_beta; };
  const Case cases[] = {
      {1, 1.0, 1e308, 0, false},    // restricted orbital energy weight overflows
      {1, 1e308, 0.0, 0, false},    // restricted coefficient weighting overflows
      {1, 1e100, 1e250, 1, false},  // restricted energy-weighted coefficient overflows
      {2, 1e100, 1e250, 1, false},  // unrestricted alpha weighted coefficient overflows
      {2, 1e100, 1e250, 3, true},   // late beta W fails after alpha P/W and beta P
  };
  for (const auto& test : cases) {
    if (int result = check(test.spins, test.coefficient, test.eigenvalue,
                           test.calls, test.late_beta)) {
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
            str(root / "src"),
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
