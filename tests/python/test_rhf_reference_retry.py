"""Host-execute physical-reference publication and its bounded CUDA retry.

The exact owner block and download helper run with deferred transport and
fault-injected numerical actions. This tests control flow/ownership, not device
numerics, physical convergence, or complete correlated-endpoint performance.
Admission snapshots must record only the final validated publication capacities.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _publication(text: str) -> str:
    begin = text.index(
        "  if (options.export_physical_reference) {",
        text.index("// Energy and every force term consume this same P/F(P)."),
    )
    end = text.index("\n    return outputs;\n  }", begin)
    return text[begin : end + len("\n    return outputs;\n  }")]


_PREFIX = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "posthf/capacity.hpp"
#include "scf/types.hpp"
using cudaStream_t = int;
constexpr int cudaSuccess=0, cudaMemcpyDeviceToHost=2;
static int mode=0, validations=0, copies=0, builds=0, solves=0, products=0, energies=0;
static int plan_capacity_samples=0, candidate_capacity_samples=0;
struct MockPlan {
  std::size_t reference_admitted_plan_host_bytes=101;
  std::size_t reference_admitted_candidate_host_bytes=202;
};
static MockPlan plan;
static std::vector<std::function<void()>> pending;
static std::vector<std::string> actions;
void require(bool ok, const char* message) {if(!ok)throw std::logic_error(message);}
int cudaMemcpyAsync(void* to,const void* from,std::size_t bytes,int,cudaStream_t) {
  if((mode==7 && ++copies==5) || (mode==15 && validations==1 && builds==1))return 1;
  pending.emplace_back([=]{std::memcpy(to,from,bytes);});return 0;
}
int cudaStreamSynchronize(cudaStream_t) {
  auto work=std::move(pending);pending.clear();for(auto& fn:work)fn();return 0;
}
int cudaMemGetInfo(std::size_t* a,std::size_t* b){*a=*b=4096;return 0;}
int cudaPeekAtLastError(){return mode==16 && builds==1 ? 1 : 0;}
namespace generativeqc_tensor {
void cuda_check(int status){if(status)throw std::runtime_error("transport");}
}
namespace generativeqc::scf {
void validate_physical_reference(PhysicalReference& ref) {
  ++validations;actions.push_back("validate");
  require(pending.empty(),"validation precedes transport completion");
  require(plan.reference_admitted_plan_host_bytes==101 &&
              plan.reference_admitted_candidate_host_bytes==202 &&
              plan_capacity_samples==0 && candidate_capacity_samples==0,
          "admission snapshots changed before final validation");
  if(mode==5)throw std::invalid_argument("shape");
  if(mode==6)throw std::bad_alloc();
  if(mode==2 || ((mode==1 || mode==3 || mode==4 || (mode>=9 && mode<=12) ||
                   mode==15 || mode==16) && validations==1))
    throw std::runtime_error("canonicality");
  require(ref.nbf==2 && ref.nocc==1,"shape transport");
  require(ref.overlap==std::vector<double>({1,3,2,4}),"column-major conversion");
  require(ref.density[0]==(builds ? 22 : 11),"wrong density generation");
  require(ref.fock[0]==(builds ? 44 : 33),"wrong Fock generation");
}
}
"""

_ACTIONS = r"""
using namespace generativeqc::scf;
struct MockHost {std::vector<std::size_t> occupied{1};};
std::size_t hf_cuda_retained_host_numeric_bytes(const MockPlan&) {
  ++plan_capacity_samples;actions.push_back("snapshot-plan");
  return 3000+100*builds;
}
namespace cuda_execution {
std::size_t host_batch_numeric_bytes(const MockHost&) {
  ++candidate_capacity_samples;actions.push_back("snapshot-candidate");
  return 4000+100*builds;
}
}
struct Item {generativeqc_status status=GENERATIVEQC_STATUS_SUCCESS;ScfResult scf;};
void fill_global_failure(std::vector<Item>& out,generativeqc_status status) {
  for(auto& row:out)row.status=status;
}
generativeqc_status cuda_status(int) {return GENERATIVEQC_STATUS_INTERNAL_ERROR;}
unsigned blocks_for(std::size_t n){return static_cast<unsigned>(n);}
void launch_copy_selected_matrices_kernel(unsigned,unsigned,int,int,int,int,int n,
                                         const std::uint8_t* active,const double* from,double* to) {
  actions.push_back("copy");require(*active==1,"retry lost active selection");
  std::copy_n(from,n*n,to);
}
generativeqc_status multiply_matrices(const double*,bool,const double*,double*) {
  ++products;actions.push_back("product");
  return mode==10 ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE : GENERATIVEQC_STATUS_SUCCESS;
}
generativeqc_status launch_solver(int,int,int,int,double*,double*,double*,int,int*,
                                 const std::uint8_t* active) {
  ++solves;actions.push_back("solve");require(*active==1,"inactive solver");
  return mode==11 ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE : GENERATIVEQC_STATUS_SUCCESS;
}
void launch_inspect_solver_kernel(unsigned,unsigned,int,int,int,const int*,
                                 std::uint8_t* active,std::uint8_t* failed,std::uint8_t* converged) {
  if(mode==12){*active=0;*failed=1;*converged=0;}
}
void launch_compute_energy_kernel(unsigned,unsigned,int,int,int,int,const double* density,
                                 const double*,const double* fock,const double*,
                                 const std::uint8_t* active,double* energy) {
  ++energies;actions.push_back("energy");
  if(*active){require(*density==22 && *fock==44,"energy has mismatched P/F");*energy=-99;}
}
namespace generativeqc::runtime::df_progress {
struct Scope {
  Scope(const char*,const char*){}
  static void label(const char*,const char*){}
  template<class T> static void number(const char*,T){}
};
}
namespace runtime=generativeqc::runtime;
std::vector<Item> run() {
  std::vector<Item> outputs(1);
  ScfOptions options;options.export_physical_reference=true;
  struct {int stream_=0;std::size_t reference_peak_bytes_=1000,reference_eri_bytes_=0;
          int eigensolver_view(){return 0;}} resources;
  const auto reference_phase_peak=resources.reference_peak_bytes_;
  struct MockResidentValues {
    bool active() const {return false;}
    generativeqc_status audit(){return GENERATIVEQC_STATUS_SUCCESS;}
    void observe_completed(std::size_t) const {}
  };
  std::unique_ptr<MockResidentValues> resident_values;
  const MockHost host;
  const std::size_t nbf=2,batch_size=1,spin_count=1,spin_matrix_elements=4;
  const unsigned threads=32,matrix_reduction_threads=32;
  double overlap[4]{1,2,3,4},hcore[4]{},fock[4]{33},coefficients[4]{},density[4]{11};
  double next_density[4]{22},eigenvalues[2]{},orthogonalizer[4]{},temporary[4]{},eigensystem[4]{};
  double energy[1]{-77},energy_change[1]{1e-14},density_rms[1]{1e-13},nuclear_repulsion[1]{};
  std::uint8_t converged[1]{1},failed[1]{0},active[1]{1};std::uint32_t iterations[1]{5};
  int solver_info[1]{},ordinary_eigensolver_family=0,lwork=0,cuda_error=0;
  generativeqc_status status=GENERATIVEQC_STATUS_SUCCESS;
  const bool reuse_converged_fock=mode!=4,mixed_precision_fock=false;
  std::uint32_t host_final_fock_rebuild_count=mode==3 ? 1U : 0U;
  std::uint32_t post_scf_physical_fock_builds=0,post_scf_final_eigen_solves=1;
  const bool persistent_eri=false,geometry_changed=false,quartet_direct=false,bounded_direct_streaming=false;
  const std::size_t eri_elements=0,pair_count=0,total_shell_pairs=0,total_shell_quartets=0;
  if(mode==8)energy[0]=std::numeric_limits<double>::quiet_NaN();
  if(mode==13)converged[0]=0;
  if(mode==14)failed[0]=1;
  const auto launch_fock_builder=[&](const double* input,bool mixed,bool incremental) {
    ++builds;actions.push_back("build");
    require(*input==22 && !mixed && !incremental,"retry operator provenance");
    if(mode==9)return 1;
    fock[0]=44;return 0;
  };
"""

_MAIN = r"""
  return outputs;
}
int main(int argc,char**argv) {
  if(argc!=2)return 99;mode=std::stoi(argv[1]);
  try {
    std::vector<Item> result;
    std::string caught;
    try {result=run();}
    catch(const std::invalid_argument&){caught="shape";}
    catch(const std::bad_alloc&){caught="allocation";}
    catch(const std::runtime_error& error){caught=error.what();}
    require(pending.empty(),"exception left asynchronous host borrows pending");
    if(mode==0 || mode==1) {
      require(caught.empty() && result.size()==1,"expected publication");
      const auto& r=result[0];
      require(r.status==GENERATIVEQC_STATUS_SUCCESS && r.scf.converged && r.scf.reference,
              "successful final reference was not published");
      require(r.scf.iterations==5 && r.scf.reference->numeric_capacity_bytes==1000,
              "lost reference metadata");
      require(r.scf.reference->energy==(mode==1 ? -99 : -77),"stale reference energy");
      require(r.scf.precision.operator_work_counters_valid==1 &&
                  r.scf.precision.strict_stage_fock_builds==5 &&
                  r.scf.precision.post_scf_fock_builds==static_cast<unsigned>(mode) &&
                  r.scf.precision.skipped_final_fock_builds==static_cast<unsigned>(1-mode),
              "incorrect published work counts");
      require(validations==mode+1 && builds==mode && solves==mode && products==3*mode &&
                  energies==mode,"retry work is not bounded");
      require(plan.reference_admitted_plan_host_bytes==3000+100*mode &&
                  plan.reference_admitted_candidate_host_bytes==4000+100*mode &&
                  plan_capacity_samples==1 && candidate_capacity_samples==1,
              "admission snapshots did not capture final validated capacities exactly once");
      const auto expected_actions=mode==0 ?
          std::vector<std::string>{"validate","snapshot-plan","snapshot-candidate"} :
          std::vector<std::string>{"validate","copy","build","product","product","solve",
                                   "product","energy","validate","snapshot-plan","snapshot-candidate"};
      require(actions==expected_actions,"validation/retry/snapshot ordering");
    } else if(mode==2 || mode==3 || mode==4) {
      require(caught=="canonicality" && result.empty(),"validation rejection was swallowed");
      require(builds==(mode==2 ? 1 : 0) && validations==(mode==2 ? 2 : 1),"unbounded/forbidden retry");
    } else if(mode==5 || mode==6 || mode==7 || mode==8 || mode==15) {
      const std::string expected=mode==5 ? "shape" : mode==6 ? "allocation" :
                                 mode==8 ? "nonfinite CUDA RHF energy" : "transport";
      require(caught==expected && result.empty(),"wrong failure category or publication");
      require(builds==(mode==15 ? 1 : 0),"retried a non-validation failure");
    } else {
      require(caught.empty() && result.size()==1 && !result[0].scf.reference &&
                  !result[0].scf.converged,"failed operation published reference");
      const auto expected=mode==13 ? GENERATIVEQC_STATUS_NOT_CONVERGED :
          (mode==9 || mode==16) ? GENERATIVEQC_STATUS_INTERNAL_ERROR : GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
      require(result[0].status==expected,"lost failure status");
      require(builds==((mode==13 || mode==14) ? 0 : 1),"incorrect failure-path build count");
      require(validations==((mode==13 || mode==14) ? 0 : 1),"validated failed retry");
    }
    if(mode!=0 && mode!=1) {
      require(plan.reference_admitted_plan_host_bytes==101 &&
                  plan.reference_admitted_candidate_host_bytes==202 &&
                  plan_capacity_samples==0 && candidate_capacity_samples==0,
              "rejection or failure changed admission snapshots");
    }
    std::cout<<"scenario "<<mode<<" passed\n";
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def reference_retry_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires a host C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=5)
    helper = (ROOT / "src/scf/cuda/reference_export.cuh").read_text()
    helper = helper[helper.index("namespace generativeqc::scf::reference_detail {") :]
    publication = _publication((ROOT / "src/scf/cuda_rhf.cpp").read_text())
    folder = tmp_path_factory.mktemp("rhf-reference-retry")
    source, object_file, executable = (
        folder / "probe.cpp",
        folder / "probe.o",
        folder / "probe",
    )
    source.write_text(_PREFIX + helper + _ACTIONS + publication + _MAIN)
    compiled = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(source),
            "-o",
            str(object_file),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert compiled.returncode == 0, compiled.stderr
    linked = subprocess.run(
        [compiler, str(object_file), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert linked.returncode == 0, linked.stderr
    return executable


@pytest.mark.parametrize(
    "scenario",
    range(17),
    ids=[
        "retained-success",
        "one-validated-rebuild",
        "second-validation-failure",
        "already-rebuilt",
        "reuse-disabled",
        "shape-error",
        "allocation-error",
        "transport-error",
        "nonfinite-energy",
        "fock-error",
        "product-error",
        "solver-status-error",
        "solver-info-error",
        "not-converged",
        "failed-scf",
        "retry-download-error",
        "retry-kernel-error",
    ],
)
def test_reference_retry_publishes_only_validated_state(
    reference_retry_probe: Path, scenario: int
) -> None:
    result = subprocess.run(
        [str(reference_retry_probe), str(scenario)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
