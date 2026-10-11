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
static int allocations=0, plans=0, fences=0, creates=0, mode=0, mallocs=0, pending=0, assemblies=0;
static int layout_queries=0;
static int memory_queries=0;
static std::size_t direct_dimension=64;
static bool graph_live=false;
static std::vector<int> teardown;
static cudaStream_t stream=reinterpret_cast<void*>(123);
static void require(bool value) { if(!value) throw std::logic_error("lifetime invariant"); }
int cudaStreamSynchronize(cudaStream_t value) {
  require(value==stream); ++fences; return mode==15 || mode==19 ? 999 : 0;
}
int cudaMemGetInfo(std::size_t* available, std::size_t* total) {
  ++memory_queries;
  if(mode==26 || mode==27 || mode==28) {
    const auto initial=(256ULL<<20)+5120+4*64*64*sizeof(double)+20+400;
    *available=*total=initial-(memory_queries==2 ? 4096 : 0);
    if(mode==28 && memory_queries==2) *available-=625;
    return mode==27 && memory_queries==2 ? 999 : 0;
  }
  *available=*total=mode==1 ? 0 : 16ULL<<30; return mode==2 ? 999 : 0;
}
int cudaGetLastError() { const auto value=pending; pending=0; return value; }
int cudaPeekAtLastError() { return pending; }
int cudaMemcpyAsync(void* to, const void* from, std::size_t bytes, int, cudaStream_t value) {
  require(value==stream); std::memcpy(to,from,bytes); return 0;
}
namespace generativeqc::runtime {
int resource_cuda_malloc(void** output, std::size_t bytes) {
  if((mode==3 || mode==4) && ++mallocs==2) { pending=mode==3 ? 2 : 999; return 2; }
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
generativeqc_status create_cuda_direct_jk_resident_plan_on_stream(
    int, const std::vector<core::System>& systems, double screening,
    std::size_t budget, cudaStream_t value, CudaDirectJkPlan** output,
    CudaDirectJkDiagnostic& diagnostic, std::string&) {
  const bool screened=mode==21 || mode>=26;
  require(value==stream && screening==(screened ? 1e-12 : 0.0) && systems.size()==1);
  require(budget==(screened ? 5120 : 4096));
  ++creates;
  if(mode==5) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  // Real provider metadata/scratch cudaMalloc failures leave last-error pending.
  if(mode>=13 && mode<=15) {
    pending=mode==14 ? 999 : 2;
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  if(mode>=17 && mode<=19) {
    pending=mode==17 ? 999 : 0;
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
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
generativeqc_status cuda_direct_jk_resident_layout_device_bytes(
    std::size_t, std::size_t, std::size_t& bytes, std::string&) {
  ++layout_queries;
  bytes=1024;
  if(mode==22) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if(mode==23) return GENERATIVEQC_STATUS_CUDA_ERROR;
  if(mode==24 || mode==25) {
    pending=mode==24 ? 2 : 999;
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}
generativeqc_status prepare_cuda_direct_jk_resident_layout(CudaDirectJkPlan*, std::string&) {
  return GENERATIVEQC_STATUS_SUCCESS;
}
std::size_t cuda_direct_jk_compensation_elements(const CudaDirectJkPlan*) {
  return 2*direct_dimension*direct_dimension;
}
generativeqc_status prepare_cuda_direct_jk_resident_values(CudaDirectJkPlan* plan,
                                                         std::size_t, std::string&) {
  if(mode==7 || mode==12) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  if(mode==16) { pending=999; return GENERATIVEQC_STATUS_OUT_OF_MEMORY; }
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
          correction==2*64*64 && threshold==((mode==20 || mode==21) ? 1e-12 : 0.0));
  if(mode==20) require(spec.exchange.coefficient==-0.125);
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
  require(value==stream); ++assemblies;
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
  for(mode=0;mode<20;++mode) {
    fences=1; mallocs=0; pending=0; teardown.clear();
    const bool fatal=mode==2 || mode==4 || mode==6 || mode==8 ||
                     mode==14 || mode==15 || mode==16 || mode==17 || mode==19;
    const bool admitted=mode==0 || (mode>=9 && mode<=11);
    {
      RhfResidentValues owner(stream);
      auto status=owner.prepare(0,systems,100,64,1000,16ULL<<30);
      require((status!=GENERATIVEQC_STATUS_SUCCESS)==fatal);
      require(owner.active()==admitted);
      if(!fatal && cudaPeekAtLastError()!=0)
        throw std::logic_error("optional refusal left CUDA last-error pending");
      if(mode>=14 && fatal) require(status==GENERATIVEQC_STATUS_CUDA_ERROR);
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
  mode=20;
  direct_dimension=64;
  {
    scf::cuda_execution::DirectJkSourceLease source(stream);
    require(source.prepare(0,systems,100,64,1000,16ULL<<30,"1","generic-test")
            ==GENERATIVEQC_STATUS_SUCCESS && source.active());
    const auto before_assembly=assemblies;
    auto spec=scf::make_hf_fock_spec(scf::FockSpin::Restricted);
    spec.derivative_order=0; spec.exchange.coefficient=-0.125;
    double matrix[64*64]{}; int error=0; std::string detail;
    require(source.enqueue(spec,matrix,source.coulomb(),source.exchange(),&error,1e-12,detail)
            ==GENERATIVEQC_STATUS_SUCCESS);
    require(source.audit()==GENERATIVEQC_STATUS_SUCCESS && assemblies==before_assembly);
  }
  for(mode=21;mode<=25;++mode) {
    scf::cuda_execution::DirectJkSourceLease screened(stream);
    const auto before_queries=layout_queries;
    const auto before_creates=creates;
    const auto status=screened.prepare(0,systems,100,64,1000,16ULL<<30,"1","screened-test",
                                       false,false,1e-12,16ULL<<30);
    require(layout_queries==before_queries+1);
    require((status==GENERATIVEQC_STATUS_SUCCESS)==(mode!=23 && mode!=25));
    require(screened.active()==(mode==21) && creates==before_creates+(mode==21));
    require(pending==0);
  }
  require(allocations==0 && plans==0);
  for(mode=26;mode<=28;++mode) {
    memory_queries=0;
    scf::cuda_execution::DirectJkSourceLease bounded_live(stream);
    const auto status=bounded_live.prepare(0,systems,100,64,1000,16ULL<<30,"1","live-memory-test",
                                          false,false,1e-12,16ULL<<30);
    require(memory_queries==2 && bounded_live.active()==(mode==26));
    require((status==GENERATIVEQC_STATUS_SUCCESS)==(mode!=27));
  }
  require(allocations==0 && plans==0);
  mode=29;
  {
    scf::cuda_execution::DirectJkSourceLease borrowed_outputs(stream);
    require(borrowed_outputs.prepare(0,systems,100,64,1000,16ULL<<30,"1","borrowed-output-test",
                                     false,false,1e-12,16ULL<<30,false)
            ==GENERATIVEQC_STATUS_SUCCESS && borrowed_outputs.active());
    require(borrowed_outputs.coulomb()==nullptr && borrowed_outputs.exchange()==nullptr);
    require(borrowed_outputs.device_bytes()==2*64*64*sizeof(double)+sizeof(int)+2*sizeof(std::uint64_t));
  }
  require(allocations==0 && plans==0);
  mode=21;
  direct_dimension=128;
  systems.front().shells.front().angular_momentum=2;
  for(const auto context:{0,1,2,3,4,5,6}) {
    const auto before_creates=creates;
    const auto before_queries=layout_queries;
    const auto before_memory=memory_queries;
    {
      scf::cuda_execution::DirectJkSourceLease automatic(stream);
      require(automatic.prepare(0,systems,context==4 ? 7 : 100,context==3 ? 127 : 128,
                                1000,context==5 ? 1000 : 16ULL<<30,"auto","ks-auto-test",
                                context!=1,false,context==2 ? 0.0 : 1e-12,16ULL<<30,false,
                                context==6 ? scf::cuda_execution::DirectJkSourceAutoPolicy::generic_f
                                           : scf::cuda_execution::DirectJkSourceAutoPolicy::screened_cold)
              ==GENERATIVEQC_STATUS_SUCCESS);
      require(automatic.active()==(context==0));
      require(creates==before_creates+(context==0));
      if(context!=0 && context!=5)
        require(layout_queries==before_queries && memory_queries==before_memory);
    }
    require(allocations==0 && plans==0);
  }
  direct_dimension=64;
  mode=0;
  RhfResidentValues refused(stream);
  const auto before=creates;
  require(refused.prepare(0,systems,100,64,1000,1000)==GENERATIVEQC_STATUS_SUCCESS);
  require(!refused.active() && creates==before);
  mode=12;
  scf::cuda_execution::DirectJkSourceLease discarded(stream);
  require(discarded.prepare(0,systems,100,64,1000,16ULL<<30,"1","discarded-test")
          ==GENERATIVEQC_STATUS_SUCCESS);
  require(!discarded.active() && discarded.completed_value_count()==100);
}
""")
    executable = tmp_path / "probe"
    native_cxx.build_executable(
        [
            ROOT / "src/scf/cuda/rhf_resident_values.cpp",
            ROOT / "src/scf/cuda/direct_jk_source_lease.cpp",
            harness,
        ],
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
