"""Exercise the composed controller with real GMRES and synthetic device storage.

This is host integration coverage, not CUDA numerical qualification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from test_resident_gmres_failures import SOURCE

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_finished_warmup_proposal_reaches_the_strict_physical_solver(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    source = (ROOT / "src/cc/lambda_response_cuda.cu").read_text()
    start = source.index('  Trace gmres_trace("cc_lambda_gmres");')
    controller = source[
        source.index("  const auto diagonal =", start) : source.index(
            "  gmres_trace.finish();", start
        )
    ]
    # Reuse the existing deterministic vector-storage backend. The actual host
    # and resident GMRES implementations still perform every physical residual.
    backend = SOURCE[: SOURCE.index("int main() {")].replace(
        "struct Backend final :", "struct Backend :"
    )
    unit = tmp_path / "composition.cpp"
    unit.write_text(
        backend
        + r"""
#include <memory>
#include <tuple>
#include "cc/lambda_response.hpp"
using namespace generativeqc;
namespace tensor { struct CudaKrylovVectors { static constexpr std::size_t kProviderAllowance=128; }; }
std::size_t checked_add(std::size_t a,std::size_t b) { return a+b; }
struct Owner {
  bool mixed{}, active{}, finished{}, nonfinite{};
  int refusal{};
  std::size_t warm_actions{},strict_actions{};
  bool begin_mixed_precision(){ active=mixed; return active; }
  void finish_mixed_precision(){ require(active,"no active warmup");active=false;finished=true; }
  std::size_t h2d_bytes() const{return 0;}
  std::size_t d2h_bytes() const{return 0;}
  std::size_t synchronizations() const{return 0;}
  std::size_t owned_device_bytes() const{return 64;}
  std::size_t numeric_capacity_bytes() const{return 128;}
  std::size_t device_limit_bytes() const{return refusal==4?64:8192;}
  void apply(std::span<const double> x,std::span<double> y) {
    if(active)++warm_actions;else ++strict_actions;
    y[0]=(active?2.2:2.0)*x[0]; y[1]=(active?4.7:4.0)*x[1];
    if(active && nonfinite)y[0]=std::numeric_limits<double>::quiet_NaN();
  }
};
struct Layout { std::size_t dimension() const{return 2;} };
struct CudaLambdaResidentBackend : Backend {
  Owner& owner;
  CudaLambdaResidentBackend(Owner& value,const ResponsePlan& plan,std::span<const double> diagonal)
      :Backend(resident_gmres_workspace(plan).vector_slots,plan),owner(value) {
    require(!owner.active,"resident allocation before strict epoch");
    if(owner.refusal==1)throw std::bad_alloc{};
    if(owner.refusal==2)throw std::runtime_error("injected stream failure");
    right_preconditioned=!diagonal.empty();
  }
  static std::size_t required_bytes(const ResponsePlan& plan,const Layout&,bool) {
    return resident_gmres_workspace(plan).vector_slots*2*sizeof(double);
  }
  std::size_t provider_observed_bytes() const{return 32;}
  int provider_version() const{return 1;}
  int runtime_version() const{return 1;}
  void apply(std::size_t dst,std::size_t src) override {
    require(!owner.active,"resident execution before strict epoch");
    owner.apply(data.at(src),data.at(dst));
  }
};
int main() {
 for(bool mixed:{false,true}) for(bool selected:{false,true})
 for(bool preconditioned:{false,true}) for(int refusal:{0,1,2,3,4})
 for(bool invalid:{false,true}) {
  cc::LambdaOptions options;
  options.cuda_resident_krylov=selected;
  options.max_bytes=refusal==3?64:8192;
  options.gmres.absolute_tolerance=1e-12;options.gmres.relative_tolerance=0;
  options.gmres.max_iterations=16;options.gmres.restart=2;
  // One warmup iteration deliberately permits finite exhaustion. It must still
  // become only a proposal to a separately prepared strict controller.
  options.df_mixed_precision_max_iterations=1;
  options.df_mixed_precision_warmup_tolerance=1e-12;
  Owner owner;owner.mixed=mixed;owner.nonfinite=invalid;owner.refusal=refusal;
  Layout layout;
  struct {std::size_t naux=5,reference_retained_bytes=16;} p;
  const std::array<double,2> rhs{1,2},independent{2,4};
  const LinearResponseProblem response_problem(2,[&](auto x,auto y){owner.apply(x,y);});
  const auto response_plan=prepare_response(response_problem,options.gmres);
  auto work_snapshot=[] { return 0; };auto trace_work=[](int){};int phase_work=0;
  const bool expected_failure=(mixed && invalid) || (selected && refusal==2);
  bool failed=false;
  try {
"""
        + controller
        + r"""
   require(!owner.active,"approximate epoch escaped to publication");
   require(owner.finished==mixed,"warmup finish lost");
   require(solved.converged() && solved.residual_norm<1e-12,"strict physical residual failed");
   require(std::abs(solved.solution[0]-0.5)<1e-12 && std::abs(solved.solution[1]-0.5)<1e-12,
           "approximate solution published");
   require(owner.strict_actions>0,"strict physical action missing");
   require((warmup_actions>0)==mixed && (owner.warm_actions>0)==mixed,"warmup lost");
   require(resident_selected==(selected && refusal==0),"optional refusal changed route");
   require(warmup_solution.empty()!=mixed,"finite proposal not retained");
   if(mixed)require(std::abs(warmup_solution[0]-solved.solution[0])>1e-4,
                    "warmup proposal accidentally exact");
   (void)std::tie(warmup_iterations,warmup_preconditioner_actions,resident_bytes,
       resident_provider_bytes,resident_device_admission,resident_extra_admission,
       resident_provider_version,resident_runtime_version,krylov_h2d,krylov_d2h,krylov_fences);
  } catch(const std::runtime_error&) { failed=true; }
  require(failed==expected_failure,"numerical/stream failure swallowed or unexpected failure");
  if(mixed && invalid)require(owner.strict_actions==0,"nonfinite warmup retried strict");
 }
}
"""
    )
    executable = tmp_path / "composition"
    required_native_cxx.build_executable(
        [
            unit,
            ROOT / "src/response/native_gmres.cpp",
            ROOT / "src/response/resident_gmres.cpp",
        ],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)
