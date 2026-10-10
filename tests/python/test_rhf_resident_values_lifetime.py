"""Exercise phase-only ownership and optional refusals without initializing CUDA."""

import json
import subprocess
from pathlib import Path

from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_phase_values_retire_graphs_before_source_and_preserve_failures(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Run the real phase owner with injected allocator/provider/stream outcomes."""

    def stub(name: str, contents: str) -> None:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    stub(
        "cuda_runtime_api.h",
        r"""
#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorUnknown=999;
constexpr int cudaMemcpyDeviceToHost=2;
int cudaStreamSynchronize(cudaStream_t);
int cudaMemGetInfo(std::size_t*, std::size_t*);
int cudaGetLastError();
int cudaPeekAtLastError();
int cudaMemcpyAsync(void*, const void*, std::size_t, int, cudaStream_t);
""",
    )
    stub(
        "runtime/resource_cuda.cuh",
        r"""
#pragma once
#include "cuda_runtime_api.h"
namespace generativeqc::runtime {
int resource_cuda_malloc(void**, std::size_t);
int resource_cuda_free(void*);
}
""",
    )
    stub(
        "scf/cuda/runtime_support.hpp",
        r"""
#pragma once
#include "generativeqc/generativeqc.h"
#include "cuda_runtime_api.h"
namespace generativeqc::scf::cuda_execution {
inline generativeqc_status cuda_status(int status) {
  return status==0 ? GENERATIVEQC_STATUS_SUCCESS : status==2
    ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
}
}
""",
    )
    stub(
        "scf/cuda/df_scf_kernels.hpp",
        r"""
#pragma once
#include "cuda_runtime_api.h"
namespace generativeqc::scf::cuda_df {
void launch_assemble_rhf_fock_kernel(unsigned, unsigned, std::size_t, cudaStream_t,
                                    std::size_t, const double*, const double*,
                                    const double*, double*);
}
""",
    )
    harness = tmp_path / "probe.cpp"
    harness.write_text(r"""
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "scf/cuda/rhf_resident_values.hpp"
#include "molecule/basis.hpp"
using namespace generativeqc;
static int allocations=0, plans=0, fences=0, creates=0, mode=0, mallocs=0;
static std::size_t direct_dimension=64;
static bool graph_live=false;
static std::vector<int> teardown;
static cudaStream_t stream=reinterpret_cast<void*>(123);
static void require(bool value) { if(!value) throw std::logic_error("lifetime invariant"); }
int cudaStreamSynchronize(cudaStream_t value) { require(value==stream); ++fences; return 0; }
int cudaMemGetInfo(std::size_t* available, std::size_t* total) {
  *available=*total=mode==1 ? 0 : 16ULL<<30; return mode==2 ? 999 : 0;
}
int cudaGetLastError() { return mode==3 ? 2 : mode==4 ? 999 : 0; }
int cudaPeekAtLastError() { return 0; }
int cudaMemcpyAsync(void* to, const void* from, std::size_t bytes, int, cudaStream_t value) {
  require(value==stream); std::memcpy(to,from,bytes); return 0;
}
namespace generativeqc::runtime {
int resource_cuda_malloc(void** output, std::size_t bytes) {
  if((mode==3 || mode==4) && ++mallocs==2) return 2;
  *output=std::malloc(bytes); ++allocations; return 0;
}
int resource_cuda_free(void* pointer) {
  require(!graph_live && fences>0); teardown.push_back(3);
  std::free(pointer); --allocations; return 0;
}
}
namespace generativeqc::molecule {
std::size_t ao_count(const core::System& system) noexcept { return system.shells.size(); }
}
namespace generativeqc::scf {
struct CudaDirectJkPlan { bool resident=false; };
FockBuildSpec make_hf_fock_spec(FockSpin spin, FockApproximation approximation) {
  FockBuildSpec spec; spec.spin=spin;
  spec.coulomb.approximation=spec.exchange.approximation=approximation;
  return spec;
}
std::size_t cuda_direct_coulomb_device_bytes(std::size_t,std::size_t,std::size_t,
                                            std::size_t,std::size_t,unsigned,bool) { return 4096; }
generativeqc_status create_cuda_direct_jk_plan_on_stream(
    int, const std::vector<core::System>& systems, unsigned order, double screening,
    std::size_t, cudaStream_t value, CudaDirectJkPlan** output,
    CudaDirectJkDiagnostic& diagnostic, std::string&) {
  require(value==stream && order==0 && screening==0 && systems.size()==1);
  ++creates;
  if(mode==5) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if(mode==6) return GENERATIVEQC_STATUS_CUDA_ERROR;
  *output=new CudaDirectJkPlan; ++plans;
  diagnostic.device_bytes=4096; diagnostic.host_bytes=100;
  diagnostic.host_preparation_bytes=100;
  return GENERATIVEQC_STATUS_SUCCESS;
}
void destroy_cuda_direct_jk_plan(CudaDirectJkPlan* plan) noexcept {
  require(!graph_live && fences>0); teardown.push_back(2); delete plan; --plans;
}
std::size_t cuda_direct_jk_resident_value_bytes(const CudaDirectJkPlan*) { return 800; }
std::size_t cuda_direct_jk_compensation_elements(const CudaDirectJkPlan*) {
  return 2*direct_dimension*direct_dimension;
}
generativeqc_status prepare_cuda_direct_jk_resident_values(CudaDirectJkPlan* plan,
                                                         std::size_t, std::string&) {
  if(mode==7 || mode==12) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if(mode==8) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  plan->resident=true; ++fences; return GENERATIVEQC_STATUS_SUCCESS;
}
CudaDirectJkDiagnostic cuda_direct_jk_plan_diagnostic(const CudaDirectJkPlan* plan) noexcept {
  CudaDirectJkDiagnostic diagnostic;
  if(plan && mode==12) {
    diagnostic.resident_values_submitted=100;
    diagnostic.resident_values_completed=100;
  }
  if(plan && plan->resident) { diagnostic.resident_value_bytes=800; diagnostic.resident_value_count=100; }
  return diagnostic;
}
generativeqc_status enqueue_cuda_direct_jk_compensated_device(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double*, std::size_t elements,
    double*, double*, double*, std::size_t correction, int* error, double threshold,
    std::uint64_t* census, std::string&, bool resident) {
  require(plan->resident && resident && spec.derivative_order==0 && elements==64*64 &&
          correction==2*64*64 && threshold==0);
  *error=mode==9; census[0]=mode==10 ? 99 : 100; census[1]=mode==11;
  return GENERATIVEQC_STATUS_SUCCESS;
}
}
namespace generativeqc::scf::cuda_execution {
RhfIterationGraphs::~RhfIterationGraphs() { reset(); }
void RhfIterationGraphs::reset() noexcept { graph_live=false; teardown.push_back(1); }
}
namespace generativeqc::scf::cuda_df {
void launch_assemble_rhf_fock_kernel(unsigned,unsigned,std::size_t,cudaStream_t value,
                                    std::size_t,const double*,const double*,const double*,double*) {
  require(value==stream);
}
}
int main(int argc, char** argv) {
  require(argc==2);
  setenv("GENERATIVEQC_DF_PROGRESS_TRACE",argv[1],1);
  setenv("GENERATIVEQC_RHF_RESIDENT_VALUES","1",1);
  using scf::cuda_execution::RhfResidentValues;
  core::System system; system.atoms.resize(1); system.shells.resize(64);
  for(auto& shell:system.shells) shell.primitives.push_back({1,1});
  std::vector<core::System> systems{system};
  for(mode=0;mode<13;++mode) {
    fences=1; mallocs=0; teardown.clear();
    const bool fatal=mode==2 || mode==4 || mode==6 || mode==8;
    const bool admitted=mode==0 || (mode>=9 && mode<=11);
    {
      RhfResidentValues owner(stream);
      auto status=owner.prepare(0,systems,100,64,1000,16ULL<<30);
      require((status!=GENERATIVEQC_STATUS_SUCCESS)==fatal);
      require(owner.active()==admitted);
      if(mode==12) require(owner.capacity_bytes()>800 && owner.value_bytes()==0);
      if(admitted) {
        require(owner.capacity_bytes()>800 && owner.value_bytes()==800);
        double matrix[64*64]{};
        require(owner.enqueue(matrix,matrix,matrix)==GENERATIVEQC_STATUS_SUCCESS);
        require((owner.audit()==GENERATIVEQC_STATUS_SUCCESS)==(mode==0));
        graph_live=true;
      }
    }
    require(allocations==0 && plans==0 && !graph_live);
    if(std::find(teardown.begin(),teardown.end(),2)!=teardown.end())
      require(std::find(teardown.begin(),teardown.end(),1)<
              std::find(teardown.begin(),teardown.end(),2) || fatal || !admitted);
  }
  mode=0;
  for(const char* selection:{"0","bad"}) {
    setenv("GENERATIVEQC_RHF_RESIDENT_VALUES",selection,1);
    const auto before=creates;
    RhfResidentValues owner(stream);
    auto status=owner.prepare(0,systems,100,64,1000,16ULL<<30);
    require(!owner.active() && creates==before);
    require((status==GENERATIVEQC_STATUS_SUCCESS)==(selection[0]=='0'));
  }
  unsetenv("GENERATIVEQC_RHF_RESIDENT_VALUES");
  {
    RhfResidentValues automatic(stream);
    const auto before=creates;
    require(automatic.prepare(0,systems,100,64,1000,16ULL<<30)==GENERATIVEQC_STATUS_SUCCESS);
    require(!automatic.active() && creates==before);
    require(automatic.prepare(0,systems,100,64,1000,16ULL<<30)==GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  }
  systems.front().shells.front().angular_momentum=3;
  direct_dimension=128;
  for(const char* selection:{"auto", "unset"}) {
    if(selection[0]=='u') unsetenv("GENERATIVEQC_RHF_RESIDENT_VALUES");
    else setenv("GENERATIVEQC_RHF_RESIDENT_VALUES",selection,1);
    {
      RhfResidentValues automatic(stream);
      require(automatic.prepare(0,systems,100,128,1000,16ULL<<30,true,true)==GENERATIVEQC_STATUS_SUCCESS);
      require(automatic.active());
    }
    require(allocations==0 && plans==0);
    const auto before=creates;
    for(const auto context:{0,1,2}) {
      RhfResidentValues refused_auto(stream);
      require(refused_auto.prepare(0,systems,100,context==2 ? 127 : 128,1000,16ULL<<30,
                                  context!=0,context!=1)==GENERATIVEQC_STATUS_SUCCESS);
      require(!refused_auto.active() && creates==before);
    }
  }
  setenv("GENERATIVEQC_RHF_RESIDENT_VALUES","1",1);
  RhfResidentValues refused(stream);
  const auto before=creates;
  require(refused.prepare(0,systems,100,64,1000,1000)==GENERATIVEQC_STATUS_SUCCESS);
  require(!refused.active() && creates==before);
}
""")
    executable = tmp_path / "probe"
    native_cxx.build_executable(
        [ROOT / "src/scf/cuda/rhf_resident_values.cpp", harness],
        executable,
        compile_args=(
            "-std=c++20",
            f"-I{tmp_path}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        link_args=("-pthread",),
    )
    journal = tmp_path / "progress.jsonl"
    subprocess.run([str(executable), str(journal)], check=True, timeout=10)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    discarded = next(
        record
        for record in records
        if record.get("key") == "source_values_completed" and record["value"] == 100
    )
    completion = next(
        record
        for record in records
        if record["id"] == discarded["id"] and record["event"] == "END"
    )
    assert completion["status"] == "refused"
