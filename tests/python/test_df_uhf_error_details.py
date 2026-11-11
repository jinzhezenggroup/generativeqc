"""Preserve source-indexed UHF provider failures without requiring a real GPU."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from _cc_owner_test_support import compile_owner
from _df_scf_test_support import df_convergence_owner
from test_huckel_df_error_boundaries import _SHIM, ROOT

if TYPE_CHECKING:
    from pathlib import Path

_UHF_SHIM = r"""
bool fail_compact;
void reject_cuda_df_preliminary_guess(const ScfOptions&) {}
std::pair<std::size_t, std::size_t> spin_occupations(const core::System&) { return {1, 0}; }
initial_guess::EigenOperation df_setup_eigen(CudaDensityFittingJkPlan* plan, std::size_t slot) {
  if (plan->sources.at(slot) == 1) throw std::bad_alloc();
  return [](const Matrix&, const Matrix*, const Matrix*, std::size_t) {
    return EigenResult{{1}, {1}};
  };
}
namespace initial_guess {
std::pair<Matrix, Matrix> normalized_warm_uhf_density(
    const integrals::IntegralData&, std::size_t, std::size_t, const Matrix&) {
  return {{1}, {0}};
}
}
std::pair<Matrix, Matrix> prepare_initial_uhf_density(
    const integrals::IntegralData&, const Matrix&, std::size_t, std::size_t,
    const Matrix*, std::optional<EigenResult>&, std::optional<EigenResult>&,
    int, const initial_guess::EigenOperation&) { return {{1}, {0}}; }
generativeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan*, const Matrix&, const Matrix&, const Matrix&, const Matrix&,
    const std::vector<std::int32_t>&, const std::vector<std::int32_t>&, const Matrix&,
    unsigned, double, double, Matrix&, Matrix&,
    std::vector<CudaDensityFittingDeviceScfItem>&, std::string& detail, const Matrix&, unsigned) {
  detail = "injected device SCF allocation failure";
  return fail_compact ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}
generativeqc_status execute_cuda_density_fitting_uhf_jk(
    CudaDensityFittingJkPlan*, const Matrix&, const Matrix&, Matrix&, Matrix&, Matrix&,
    std::string& detail) {
  detail = "injected device JK failure";
  return injected_status;
}
template<class... Arguments> void finalize_density_fitting_uhf(Arguments&&...) {}
template<class... Arguments> double uhf_electronic_energy(Arguments&&...) { return 0; }
Matrix concatenate(const Matrix& alpha, const Matrix& beta) {
  Matrix result = alpha;
  result.insert(result.end(), beta.begin(), beta.end());
  return result;
}
std::pair<Matrix, Matrix> split_spin_matrices(const Matrix&, std::size_t) {
  return {{1}, {0}};
}
EigenResult iteration_df_eigen(const Matrix&, const Matrix&, const Matrix&, std::size_t,
                              CudaDensityFittingJkPlan*, std::size_t) { return {{1}, {1}}; }
Matrix density_from_orbitals(const Matrix&, std::size_t, std::size_t, double) { return {1}; }
"""

_UHF_MAIN = r"""
}
int main(int argc, char** argv) {
  using namespace generativeqc;
  using namespace generativeqc::scf;
  assert(argc == 3);
  fail_compact = std::string(argv[1]) == "compact";
  const std::string failure(argv[2]);
  injected_status = failure == "CUDA_ERROR" ? GENERATIVEQC_STATUS_CUDA_ERROR
      : failure == "OUT_OF_MEMORY" ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                  : GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  ScfOptions options;
  options.max_iterations = 2;
  std::vector<core::System> systems(3);
  for (std::size_t source = 0; source < systems.size(); ++source)
    systems[source].atoms.push_back({static_cast<int>(source + 1), {0, 0, 0}});
  const auto results = run_uhf_density_fitting_cuda_bucket_impl(
      systems, std::nullopt, options, std::vector<const Matrix*>(3), 0, nullptr,
      nullptr, nullptr, nullptr);
  assert(results.size() == 3 && live_arenas == 0);
  assert(results[1].status == GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  assert(results[1].detail.empty());
  for (const auto source : {0U, 2U}) {
    assert(results[source].status == (fail_compact ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                 : injected_status));
    assert(results[source].detail == (fail_compact ? "injected device SCF allocation failure"
                                                 : "injected device JK failure"));
    assert(!results[source].scf.converged);
  }
}
"""


@pytest.fixture(scope="module")
def uhf_error_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the complete production bucket with injected numerical providers."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("requires host C++")
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    begin = source.index(
        "std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_impl("
    )
    end = source.index("\n#endif  // GENERATIVEQC_HAS_CUDA", begin)
    directory = tmp_path_factory.mktemp("uhf_error_details")
    probe = directory / "probe.cpp"
    probe.write_text(
        df_convergence_owner() + _SHIM + _UHF_SHIM + source[begin:end] + _UHF_MAIN
    )
    binary = directory / "probe"
    compile_owner(compiler, directory, [probe], binary)
    return binary


@pytest.mark.parametrize(
    "failure", ("CUDA_ERROR", "OUT_OF_MEMORY", "NUMERICAL_FAILURE")
)
def test_device_jk_failure_retains_details(uhf_error_probe: Path, failure: str) -> None:
    """A failed middle source must not shift the surviving source diagnostics."""
    subprocess.run([str(uhf_error_probe), "jk", failure], check=True, timeout=10)


def test_device_scf_allocation_failure_retains_details(uhf_error_probe: Path) -> None:
    """Return the compact SCF failure message without attempting a host fallback."""
    subprocess.run(
        [str(uhf_error_probe), "compact", "OUT_OF_MEMORY"], check=True, timeout=10
    )
