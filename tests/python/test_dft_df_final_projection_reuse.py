"""Source-contract tests for the PBE0-DF final occupied-projection force reuse."""

import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_public_cuda_hit_gate_requires_projection_arithmetic() -> None:
    check = runpy.run_path(
        str(ROOT / "tests/python/test_dft_df_final_projection_cuda.py")
    )["_assert_projection_reuse"]
    admitted = "response_reused_final_fitted_projection"
    completed = "response_final_fitted_projection_reused"
    for partial in ({admitted: 1}, {completed: 1}):
        with pytest.raises(AssertionError):
            check([{"counters": partial}], True, 9)
        # Completing a different ordinary-response row cannot qualify reuse.
        with pytest.raises(AssertionError):
            check(
                [{"counters": partial}, {"counters": {"atom_coordinates": 9}}], True, 9
            )
    check([{"counters": {completed: 1, "atom_coordinates": 9}}], True, 9)
    check([{"counters": {admitted: 1, completed: 1, "atom_coordinates": 9}}], True, 9)
    check([{"counters": {}}], False, 9)
    for counter in (admitted, completed):
        with pytest.raises(AssertionError):
            check([{"counters": {counter: 1}}], False, 9)


def test_dft_force_consumes_ks_projection_before_generic_response() -> None:
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()

    assert "resident_final_fitted_projection(expected, lease" in method
    assert "energy_derivative_components_with_fitted_projection" in method
    # The final U aliases provider scratch: exchange must consume it before J
    # can run an ordinary response that revokes/reuses the same allocation.
    body = prepared[
        prepared.index(
            "energy_derivative_components_with_fitted_projection"
        ) : prepared.index("PreparedFockPlan::retained_energy_derivative")
    ]
    assert body.index("provider.derivative(exchange") < body.index(
        "provider.derivative(coulomb"
    )
    assert "borrowed.projection != plan->auxiliary_tile_values" in lower
    assert 'trace_counter("response_reused_final_fitted_projection", 1)' in lower


def test_dft_coulomb_response_borrows_final_rks_density() -> None:
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    provider = (ROOT / "src/scf/cuda_fock_provider.cpp").read_text()
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    bridge = (ROOT / "src/scf/cuda/df_gradient_bridge.cu").read_text()

    assert "resident_final_density(expected, lease, lease_detail)" in method
    assert "CudaDfBorrowedResponseDensity" in method
    assert "energy_derivative_components_with_cuda_df_state" in method

    body = prepared[
        prepared.index("energy_derivative_components_with_cuda_df_state") :
        prepared.index("PreparedFockPlan::retained_energy_derivative")
    ]
    assert "provider.derivative(exchange, density, beta, projection)" in body
    assert "provider.derivative(coulomb, density, beta, nullptr, response_density)" in body
    assert provider.count("borrowed_response_density") >= 2
    assert "select_borrowed_density" in lower
    assert "terms[0].exchange_coefficient != 0.0" in lower
    assert "select_borrowed_density ? borrowed_response_density : nullptr" in lower
    assert 'trace_counter("response_borrowed_density", 1)' in bridge
    assert "if (borrowed_density)" in bridge
    assert "densities = borrowed_density->density" in bridge


def test_borrowed_coulomb_density_admission_is_j_only(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    begin = lower.index("  const bool select_borrowed_density =")
    end = lower.index('  const char* storage_control', begin)
    admission = lower[begin:end]
    source = tmp_path / "density_admission.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <string>
#include <vector>
#include "scf/cuda_density_fitting.hpp"
using namespace generativeqc::scf;
struct Source {
  int device_id=0;
  void* stream=reinterpret_cast<void*>(1);
  std::size_t matrix_elements=4,batch_size=1;
};
struct Term {
  std::vector<double> density=std::vector<double>(4);
  double coulomb_coefficient=1.0,exchange_coefficient=0.0;
};
int admit(int kind,bool& selected) {
  Source source, *plan=&source;
  std::size_t system=0;
  std::vector<Term> terms(1);
  bool host_weights=false;
  double values[4]{};
  CudaDfBorrowedResponseDensity lease{0,values,4,plan->stream};
  const CudaDfBorrowedResponseDensity* borrowed_response_density=&lease;
  std::string detail;
  if(kind==1) lease.device_id=1;
  if(kind==2) lease.stream=reinterpret_cast<void*>(2);
  if(kind==3) lease.matrix_elements=3;
  if(kind==4) plan->batch_size=2;
  if(kind==5) terms[0].density.resize(3);
  if(kind==6) terms[0].coulomb_coefficient=0;
  if(kind==7) terms[0].exchange_coefficient=.25;
  if(kind==8) terms.emplace_back();
  if(kind==9) host_weights=true;
  if(kind==10) borrowed_response_density=nullptr;
"""
        + admission
        + r"""
  selected=select_borrowed_density;
  return GENERATIVEQC_STATUS_SUCCESS;
}
int main() {
  for(int kind=0;kind<11;++kind) {
    bool selected=false;
    const auto status=admit(kind,selected);
    if(kind>=1 && kind<=8) assert(status==GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    else assert(status==GENERATIVEQC_STATUS_SUCCESS);
    assert(selected==(kind==0));
  }
}
"""
    )
    executable = tmp_path / "density_admission"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_dft_projection_reuse_keeps_explicit_fallback_controls() -> None:
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    assert 'projection != "off"' in lower
    assert 'occupied_source != "raw"' in lower
    assert 'space != "dense"' in lower
    assert 'storage != "jk-scratch"' in lower


def test_optional_projection_admission_preserves_bounded_fallback() -> None:
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    selection = lower[
        lower.index("const bool select_borrowed_projection") : lower.index(
            "bool borrowed_fitted_occupied"
        )
    ]
    assert (
        "plan->value_storage.pairs == DfPairStorage::SymmetricLowerSingle" in selection
    )
    assert lower.index("const std::string_view projection =") < lower.index(
        "const bool select_borrowed_projection"
    )
    retry = lower[
        lower.index(
            "if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && borrowed_fitted_occupied"
        ) : lower.index(
            "if (status == GENERATIVEQC_STATUS_SUCCESS && borrow && matching_source"
        )
    ]
    assert 'space == "auto" && occupied_source == "auto"' in retry
    assert "final_state, nullptr" in retry


def test_borrowed_projection_executes_storage_and_provider_admission(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    admission = lower[
        lower.index("  const bool select_borrowed_projection =") : lower.index(
            "  const bool fitted_occupied_requested ="
        )
    ]
    source = tmp_path / "admission.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <string_view>
#include "scf/cuda_density_fitting.hpp"
using namespace generativeqc::scf;
struct Source {
  int device_id=0;
  void* stream=reinterpret_cast<void*>(1);
  std::size_t nbf=24,naux=24,batch_size=1,completed_occupied_projection_rank=5;
  bool streamed=false,integral_source=true;
  std::vector<bool> metric_full_rank{true},metric_response_valid{true};
  double values[1]{};
  const double* auxiliary_tile_values=values;
  DfValueStorageOptions value_storage{DfPairStorage::SymmetricLowerSingle,5};
};
generativeqc_status admit(Source* plan, const CudaDfBorrowedFittedProjection* borrowed_fitted_projection,
    std::string_view projection, std::string_view occupied_source, std::string_view space,
    std::string_view storage, bool host_weights, bool& selected) {
  std::size_t system=0;
  std::vector<int> terms{1};
  std::string detail;
"""
        + admission
        + r"""
  selected=borrowed_fitted_occupied;
  return GENERATIVEQC_STATUS_SUCCESS;
}
int main() {
  for (int kind=0; kind<16; ++kind) {
    Source plan;
    CudaDfBorrowedFittedProjection lease{0,plan.values,plan.values,24,24,5,plan.stream};
    std::string_view projection="auto",occupied_source="auto",space="auto",storage="auto";
    bool host_weights=false,selected=false;
    const auto* borrowed=&lease;
    if (kind==1) plan.value_storage.pairs=DfPairStorage::Dense;
    if (kind==2) plan.value_storage.pairs=DfPairStorage::SymmetricLower;
    if (kind==3) projection="off";
    if (kind==4) occupied_source="raw";
    if (kind==5) space="dense";
    if (kind==6) storage="jk-scratch";
    if (kind==7) host_weights=true;
    if (kind==8) borrowed=nullptr;
    if (kind==9) plan.value_storage.rank_capacity=0;
    if (kind==10) plan.completed_occupied_projection_rank=0;
    if (kind==11) plan.metric_full_rank[0]=false;
    if (kind==12) plan.streamed=true;
    if (kind==13) lease.stream=reinterpret_cast<void*>(2);
    if (kind==14) lease.projection=nullptr;
    if (kind==15) plan.metric_response_valid[0]=false;
    const auto status=admit(&plan,borrowed,projection,occupied_source,space,storage,host_weights,selected);
    assert(status==(kind>=9 ? GENERATIVEQC_STATUS_INVALID_ARGUMENT : GENERATIVEQC_STATUS_SUCCESS));
    assert(selected==(kind==0));
  }
}
"""
    )
    executable = tmp_path / "admission"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
