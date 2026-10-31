"""Execute resident J/K ownership and failure handling without a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_ks_fitted_projection_lifetime import _definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_controller_rejects_foreign_projection_leases(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Equal dimensions cannot authorize another owner, epoch or generation."""
    controller = (ROOT / "src/dft/cuda_stationary_controller.cpp").read_text()
    begin = controller.index("    if (fitted_projection &&")
    end = controller.index("    const bool composite", begin)
    source = tmp_path / "projection_identity.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <stdexcept>
#include "dft/cuda_ks.hpp"
using namespace generativeqc::dft;
int main() {
  for (int mutation=0; mutation<11; ++mutation) {
    double values[4]{};
    const int device_=0;
    const std::size_t aos_=2;
    bool fitted=true;
    struct { struct { bool present=true; } exchange; } spec;
    struct { void* stream=reinterpret_cast<void*>(1); } provider;
    CudaKsResidentDensityBinding density{0,values,nullptr,4,1,provider.stream,7,8,9};
    CudaKsResidentFittedProjectionBinding lease{0,values,values,2,3,1,provider.stream,7,8,9};
    auto* fitted_projection=&lease;
    if (mutation==1) ++lease.owner;
    if (mutation==2) ++lease.solve_epoch;
    if (mutation==3) ++lease.generation;
    if (mutation==4) ++lease.device_id;
    if (mutation==5) ++lease.nbf;
    if (mutation==6) lease.stream=reinterpret_cast<void*>(2);
    if (mutation==7) lease.occupied_coefficients=nullptr;
    if (mutation==8) density.spins=2;
    if (mutation==9) fitted=false;
    if (mutation==10) spec.exchange.present=false;
    bool rejected=false;
    try {
"""
        + controller[begin:end]
        + r"""
    } catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected == (mutation!=0));
  }
}
"""
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "projection_identity",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_resident_force_consumes_projection_before_scratch_reuse(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Exercise the real adapter with a bridge that revokes U on every call.

    The fake bridge checks the detached scientific coefficients, device density
    identity and callback channel. Failure injection ensures a failed K never
    continues into J; an absent projection still executes both bounded sources.
    A total-only restricted consumer receives one correctly weighted J+K
    source, while unrestricted consumers retain the independent components.
    """
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    body = _definition(
        prepared,
        "generativeqc_status PreparedFockPlan::consume_fitted_energy_derivatives(",
    )
    source = tmp_path / "handoff.cpp"
    source.write_text(PREFIX + body + DRIVER)
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "handoff",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=30)


PREFIX = r"""
#include <cassert>
#include <memory>
#include "scf/cuda_density_fitting.hpp"
#include "scf/df_response_weights.hpp"
namespace generativeqc::scf {
struct CudaDensityFittingJkPlan {};
struct Data {
  core::System system;
  const core::System *df_gradient_orbital=&system, *df_gradient_auxiliary=&system;
  struct { std::size_t nbf=2; std::vector<double> three_center, metric; } raw;
  unsigned df_gradient_mapping=0;
  std::size_t df_gradient_budget=4096;
};
struct State {
  std::unique_ptr<CudaDensityFittingJkPlan> cuda_df=std::make_unique<CudaDensityFittingJkPlan>();
  std::unique_ptr<Data> fitted=std::make_unique<Data>();
  bool retained_fitted_derivative_order=true;
};
struct PreparedFockPlan {
  State* impl_;
  ResolvedFockBuild model;
  const ResolvedFockBuild& strategy() const { return model; }
  generativeqc_status consume_fitted_energy_derivatives(int, void*, const double*, const double*,
      const double*, const CudaDerivativeSourceConsumer&, std::string&, bool,
      const CudaDfBorrowedFittedProjection*, bool=false) const;
};
struct CudaFockProviderView {
  CudaFockProviderView(CudaDensityFittingJkPlan*, const Data&) {}
  void validate(const ResolvedFockBuild& model) { assert(model.spec.derivative_order == 1); }
};
const double *expected_alpha, *expected_beta, *expected_total;
const CudaDfBorrowedFittedProjection* expected_projection;
bool scratch_live, fail_exchange, unrestricted, range;
std::vector<CudaDerivativeSourceKind> publications;
generativeqc_status execute_cuda_density_fitting_generated_force_response(
    CudaDensityFittingJkPlan*, std::size_t, const core::System&, const core::System&,
    std::span<const double>, const std::vector<double>&,
    std::span<const DensityFittingDensityResponse> terms, unsigned, std::size_t budget,
    std::size_t, std::vector<double>&, std::string&, DfGradientResources*,
    const CudaDfFinalStateToken*, const CudaDfBorrowedFittedProjection* projection,
    const CudaDfBorrowedResponseDensity*, const CudaDfGradientPrecisionRequest*,
    const CudaDfResidentForceRequest* request) {
  assert(request && request->device_id == 0 && request->stream == reinterpret_cast<void*>(1));
  assert(request->matrix_elements == 4 && budget == 4096);
  const bool exchange = request->kind != CudaDerivativeSourceKind::Coulomb;
  const bool combined = request->kind == CudaDerivativeSourceKind::FullRangeJkSum;
  assert(terms.size() == (exchange && unrestricted ? 2U : 1U));
  for (const auto& term : terms) {
    assert(term.density.empty());
    assert(term.coulomb_coefficient == (exchange && !combined ? 0.0 : 1.0));
    assert(term.exchange_coefficient == (exchange ? .125 : 0.0));
  }
  assert(request->densities[0] == (exchange ? expected_alpha : expected_total));
  if (exchange && unrestricted) assert(request->densities[1] == expected_beta);
  assert(projection == (exchange ? expected_projection : nullptr));
  if (projection) assert(scratch_live);
  scratch_live = false;
  if (exchange && fail_exchange) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  if (exchange && !combined)
    assert(request->kind == (range ? CudaDerivativeSourceKind::RangeCorrection
                                  : CudaDerivativeSourceKind::Exchange));
  request->consume({request->kind, CudaDerivativeConvention::EnergyGradient, 0,
                    request->stream, expected_alpha, 3, 1});
  return GENERATIVEQC_STATUS_SUCCESS;
}
"""

DRIVER = r"""
}  // namespace generativeqc::scf
int main() {
  using namespace generativeqc::scf;
  double alpha[4]{}, beta[4]{}, total[4]{}, coefficients[2]{}, projected[4]{};
  expected_alpha=alpha; expected_beta=beta; expected_total=total;
  const auto stream=reinterpret_cast<void*>(1);
  const CudaDerivativeSourceConsumer consume=[](const auto& view) {
    publications.push_back(view.kind);
  };
  for (int mode=0; mode<14; ++mode) {
    State state;
    PreparedFockPlan plan{&state, {}};
    plan.model.spec.spin=FockSpin::Restricted;
    plan.model.spec.coulomb.present=true;
    plan.model.spec.coulomb.coefficient=1;
    plan.model.spec.exchange.present=true;
    plan.model.spec.exchange.coefficient=-.25;
    CudaDfBorrowedFittedProjection lease{0,coefficients,projected,2,2,1,stream};
    expected_projection=mode==1 || mode==4 || mode==5 || mode==6 || mode==11 || mode==13
                            ? nullptr : &lease;
    scratch_live=true; fail_exchange=mode==2 || mode==12;
    unrestricted=mode==4 || mode==13; range=mode==5 || mode==8;
    publications.clear();
    if (mode==3) lease.projection=nullptr;
    if (mode==4 || mode==7 || mode==13) plan.model.spec.spin=FockSpin::Unrestricted;
    if (mode==5) plan.model.spec.coulomb.present=false;
    if (mode==6 || mode==9) plan.model.spec.exchange.present=false;
    std::string detail;
    const auto status=plan.consume_fitted_energy_derivatives(
        0,stream,alpha,beta,total,consume,detail,range,expected_projection,mode>=10);
    if (mode==3 || (mode>=7 && mode<=9)) {
      assert(status==GENERATIVEQC_STATUS_INVALID_ARGUMENT && publications.empty());
    } else if (mode==2 || mode==12) {
      assert(status==GENERATIVEQC_STATUS_NUMERICAL_FAILURE && publications.empty());
    } else {
      assert(status==GENERATIVEQC_STATUS_SUCCESS);
      if (mode==10 || mode==11) {
        assert(publications.size()==1);
        assert(publications[0]==CudaDerivativeSourceKind::FullRangeJkSum);
      }
      else if (mode==5 || mode==6) assert(publications.size()==1);
      else {
        assert(publications.size()==2);
        assert(publications[0]==CudaDerivativeSourceKind::Exchange);
        assert(publications[1]==CudaDerivativeSourceKind::Coulomb);
      }
    }
  }
}
"""
