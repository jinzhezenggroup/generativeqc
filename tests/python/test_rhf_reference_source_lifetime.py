"""Fault-inject the production CUDA reference adapter and resident source on host.

Only CUDA submission/completion and the bucket solver are replaced. Deferred
streams expose ownership errors without a GPU, integrals, or SCF execution.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    text = (ROOT / "src/scf/cuda_hf_entry.cpp").read_text()
    source_class = text[text.index("class CudaRhfReferenceInteractionSource final") :]
    source_class = source_class[: source_class.index("\n}  // namespace")]
    adapters = text[
        text.index("ScfResult run_rhf_cuda(") : text.index("ScfResult run_uhf_cuda(")
    ]
    helper = (ROOT / "src/methods/correlated_cuda_reference.hpp").read_text()
    helper = helper[
        helper.index("template <class State, class Operation>") : helper.index(
            "/** Method-layer"
        )
    ]
    directory = tmp_path_factory.mktemp("rhf-reference-source")
    source, obj, exe = (
        directory / "probe.cpp",
        directory / "probe.o",
        directory / "probe",
    )
    source.write_text(PREFIX + source_class + adapters + helper + MAIN)
    result = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run(
        [compiler, str(obj), "-o", str(exe)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return exe


@pytest.mark.parametrize("mode", range(21))
def test_reference_source_exception_lifetime(source_probe: Path, mode: int) -> None:
    result = subprocess.run(
        [str(source_probe), str(mode)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_terminal_cuda_failure_retains_unfenced_plan(source_probe: Path) -> None:
    result = subprocess.run(
        [str(source_probe), "21"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
int mode=0, live=0, destroyed=0, unsafe=0, events=0, launches=0, records=0;
int submitted[2]{}, completed[2]{};
bool fail_allocation=false;
void* operator new(std::size_t n) {
  if (std::exchange(fail_allocation,false)) throw std::bad_alloc();
  if (auto* p=std::malloc(n?n:1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
int pending() { return submitted[0]+submitted[1]-completed[0]-completed[1]; }
using cudaStream_t = void*;
int stream_index(cudaStream_t s) { return s==reinterpret_cast<void*>(8)?0:1; }
struct Event { int stream=0, generation=0; };
using cudaEvent_t = Event*;
constexpr int cudaSuccess=0, cudaEventDisableTiming=1;
int cudaSetDevice(int) { return 0; }
int cudaPeekAtLastError() { return mode==7?1:0; }
int cudaEventCreateWithFlags(cudaEvent_t* e,int) {
  if(mode==1 || (mode==12 && launches)) return 2;
  *e=new Event; ++events;
  if(mode==3 || (mode==13 && launches)) fail_allocation=true;
  return 0;
}
int cudaEventRecord(cudaEvent_t e,cudaStream_t stream) {
  ++records;
  if(mode==2 || mode==9 || (mode==4 && records==2)) return 3;
  e->stream=stream_index(stream); e->generation=submitted[e->stream]; return 0;
}
int cudaEventSynchronize(cudaEvent_t e) {
  if(mode==10 || mode==21) return 4;
  completed[e->stream]=std::max(completed[e->stream],e->generation); return 0;
}
int cudaStreamSynchronize(cudaStream_t s) {
  if(mode==9 || mode==21) return 5;
  completed[stream_index(s)]=submitted[stream_index(s)]; return 0;
}
int cudaDeviceSynchronize() { if(mode==21) return 6; completed[0]=submitted[0];completed[1]=submitted[1]; return 0; }
int cudaEventDestroy(cudaEvent_t e) { delete e; --events; return 0; }
namespace core { struct System { std::vector<int> ecp_terms; }; }
namespace runtime { template<class T> std::size_t vector_bytes(const std::vector<T>& v) { return v.capacity()*sizeof(T); } }
namespace molecule { std::size_t ao_count(const core::System&) { return 2; } }
namespace posthf {
std::size_t checked_mul(std::size_t a,std::size_t b) { return a*b; }
std::size_t checked_add(std::size_t a,std::size_t b) {
 if(b>std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("capacity overflow");
 return a+b;
}
std::size_t source_capacity(const core::System&) { return 256; }
}
namespace integrals {
enum class ElectronInteractionOperator { eri };
struct DeviceInteractionTarget { int device; void* stream; double* values; std::size_t capacity; };
class ElectronInteractionSource {
 public:
 using Operator=ElectronInteractionOperator;
 virtual ~ElectronInteractionSource()=default;
 virtual const core::System& orbital() const=0;
 virtual std::size_t nbf() const=0;
 virtual std::size_t naux() const=0;
 virtual std::size_t retained_numeric_bytes() const=0;
 virtual bool supports(Operator) const noexcept=0;
 virtual bool supports_host_read(Operator) const noexcept=0;
 virtual bool supports_device_read(Operator,int) const noexcept=0;
 virtual void read(Operator,const std::array<std::size_t,4>&,const std::array<std::size_t,4>&,double*,std::size_t) const=0;
 virtual void read_device(Operator,const std::array<std::size_t,4>&,const std::array<std::size_t,4>&,DeviceInteractionTarget,std::size_t) const=0;
};
}
struct ScfOptions { bool hooks=false,strict_initial_density=false; };
struct CudaRhfBucketPlan {
 struct Resources { int device_id_=0; double* reference_eri_=reinterpret_cast<double*>(4); } resources;
 CudaRhfBucketPlan(){ ++live; }
 ~CudaRhfBucketPlan(){ --live; ++destroyed; if(pending()) ++unsafe; }
};
CudaRhfBucketPlan* last_plan=nullptr;
void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* p){ delete p; }
using RhfPlanOwner=std::unique_ptr<CudaRhfBucketPlan,void(*)(CudaRhfBucketPlan*)>;
namespace cuda_execution {
void launch_copy_resident_eri_tile(cudaStream_t stream,const double*,std::size_t,const std::array<std::size_t,4>&,const std::array<std::size_t,4>&,std::size_t,double*){
 ++submitted[stream_index(stream)]; ++launches;
 if(mode==8) throw std::runtime_error("post-submission launch failure");
}
}
using generativeqc_status=int;
constexpr int GENERATIVEQC_STATUS_SUCCESS=0,GENeric=1,
 GENERATIVEQC_STATUS_OUT_OF_MEMORY=2,GENeric2=3,
 GENERATIVEQC_STATUS_INVALID_ARGUMENT=4,GENeric3=5,
 GENERATIVEQC_STATUS_NOT_IMPLEMENTED=6,GENeric4=7,
 GENERATIVEQC_STATUS_SCF_NOT_CONVERGED=8;
struct Error:std::runtime_error { Error(int,const char* s):std::runtime_error(s){} };
struct ScfResult { bool converged=true; std::shared_ptr<int> reference=std::make_shared<int>(1); };
struct RhfBucketItem { generativeqc_status status=0; ScfResult scf; bool execution_plan_reused=false; };
std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(CudaRhfBucketPlan** p,const std::vector<core::System>&,const ScfOptions&,const std::vector<const std::vector<double>*>&,int){
 const bool reused=*p!=nullptr;
 if(!*p) last_plan=*p=new CudaRhfBucketPlan;
 if(mode==5) throw std::length_error("driver capacity failure after allocation");
 if(mode==6) throw std::bad_alloc();
 RhfBucketItem item; item.execution_plan_reused=reused;
 if(mode==15) {item.scf.converged=false;item.status=GENERATIVEQC_STATUS_SCF_NOT_CONVERGED;}
 return {item};
}
std::vector<RhfBucketItem> run_rhf_cuda_bucket(const std::vector<core::System>&,const ScfOptions&,const std::vector<const std::vector<double>*>&,int){ return {RhfBucketItem{}}; }
bool exact_reference_source_identity(const CudaRhfBucketPlan*,const core::System&,const ScfOptions&,int){ return true; }
std::size_t hf_cuda_retained_numeric_bytes(const CudaRhfBucketPlan*) {
 if(mode==11) fail_allocation=true;
 return mode==14?std::numeric_limits<std::size_t>::max():1000;
}
ScfResult run_rhf_cuda_cached(CudaRhfBucketPlan**,const core::System&,const ScfOptions&,int,
 const std::vector<double>*,bool*,std::shared_ptr<const integrals::ElectronInteractionSource>*);
"""

MAIN = r"""
void read(const integrals::ElectronInteractionSource& source,int stream=0) {
 double values[16]{};
 source.read_device(integrals::ElectronInteractionOperator::eri,{0,0,0,0},{2,2,2,2},
                    {0,reinterpret_cast<void*>(stream?16:8),values,16},16);
}
int main(int argc,char** argv) {
 try {
  require(argc==2,"mode"); mode=std::atoi(argv[1]);
  core::System system;
  if(mode==21) {
   CudaRhfBucketPlan* plan=nullptr;
   std::shared_ptr<const integrals::ElectronInteractionSource> source;
   (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,nullptr,&source);
   auto* retained=last_plan;
   read(*source);
   require(!reclaim_rhf_cuda_reference_plan(&plan,source) && source && !plan,
           "terminal fence failure reclaimed a live borrow");
   source.reset();
   require(live==1 && destroyed==0 && unsafe==0 && pending()==1 && events==0,
           "terminal fence failure released live storage");
   // Simulate explicit runtime recovery only to clean up this test's retained
   // object; production deliberately does not guess that a failed fence completed.
   mode=0; cudaDeviceSynchronize(); destroy_rhf_cuda_bucket_plan(retained);
  } else if(mode==17) {
   system.ecp_terms.resize(4096);
   CudaRhfBucketPlan* plan=nullptr;
   std::shared_ptr<const integrals::ElectronInteractionSource> source;
   (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,nullptr,&source);
   require(!plan && source->retained_numeric_bytes()==1256+4096*sizeof(int),"ECP system copy missing");
   source.reset();
  } else if(mode==18 || mode==19 || mode==20) {
   struct State {
    std::shared_ptr<const integrals::ElectronInteractionSource> reference_interaction_source;
    struct {std::size_t reference_retained_bytes=1356;} problem;
   } state;
   CudaRhfBucketPlan* plan=nullptr;
   (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,nullptr,&state.reference_interaction_source);
   int attempts=0;
   bool caught=false;
   try {
    run_with_rccsd_reference_source(state,[&] {
     ++attempts;
     if(mode==19) throw std::invalid_argument("scientific failure");
     if(attempts==1 || mode==20) throw std::length_error("source pressure");
     require(!plan && state.problem.reference_retained_bytes==100 && live==0,"source retry overcharged");
    });
   } catch(const std::exception&) {caught=true;}
   require(caught==(mode!=18),"failure propagation");
   require(attempts==(mode==19?1:2),"source retry count");
   require(bool(state.reference_interaction_source)==(mode==19),"numerical failure retired owner");
  } else if(mode==5 || mode==6) {
   std::shared_ptr<const integrals::ElectronInteractionSource> source;
   bool caught=false;
   try { (void)run_rhf_cuda(system,ScfOptions{},0,nullptr,&source); }
   catch(const std::exception&) { caught=true; }
   require(caught && !source && live==0 && destroyed==1,"throwing bucket leaked or published");
  } else if(mode==11 || mode==14 || mode==15) {
   CudaRhfBucketPlan* plan=nullptr;
   std::shared_ptr<const integrals::ElectronInteractionSource> source;
   (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,nullptr,&source);
   require(!source,"optional failure published source");
   require(mode==14 ? plan!=nullptr : plan==nullptr,"failure ownership transition");
   destroy_rhf_cuda_bucket_plan(plan);
   require(live==0 && destroyed==1,"failure cleanup");
  } else if(mode==16) {
   struct State {
    std::shared_ptr<const integrals::ElectronInteractionSource> reference_interaction_source;
    struct {std::size_t reference_retained_bytes=100;} problem;
   } state;
   CudaRhfBucketPlan* plan=nullptr;
   (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,nullptr,&state.reference_interaction_source);
   require(!plan && state.reference_interaction_source->retained_numeric_bytes()==1256,"full source query");
   state.problem.reference_retained_bytes+=1256;
   int attempts=0;
   run_with_rccsd_reference_source(state,[&] {
    ++attempts;
    require(!plan,"double owner during source retry");
    if(attempts==1) { require(live==1 && state.problem.reference_retained_bytes==1356,"initial reservation"); throw std::bad_alloc(); }
    require(live==0 && state.problem.reference_retained_bytes==100,"restored reservation");
   });
   require(attempts==2 && !state.reference_interaction_source,"source fallback");
  } else {
   CudaRhfBucketPlan* plan=nullptr;
   std::shared_ptr<const integrals::ElectronInteractionSource> source;
   bool reused=true;
   (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,&reused,&source);
   require(source && !plan && !reused && live==1,"single transfer owner");
   require(source->retained_numeric_bytes()==1256,"complete retained query plus system");
   bool caught=false;
   try {
    read(*source);
    if(mode==4) read(*source);
    if(mode==12 || mode==13) read(*source,1);
   } catch(const std::exception&) { caught=true; }
   require(caught==(mode!=0 && mode!=10),"fault did not reach expected path");
   if(mode==1 || mode==3) require(launches==0,"bookkeeping failure after launch");
   if(mode==2 || mode==4 || mode==7 || mode==8 || mode==9)
    require(pending()==0,"exception left consumer borrow pending");
   if(mode==0) {
    auto alias=source;
    require(!reclaim_rhf_cuda_reference_plan(&plan,source) && !plan,"reclaimed aliased source");
    alias.reset();
    require(reclaim_rhf_cuda_reference_plan(&plan,source),"exclusive reclaim failed");
    require(!source && plan && live==1 && pending()==0 && events==0,"reclaim did not fence source");
    (void)run_rhf_cuda_cached(&plan,system,ScfOptions{},0,nullptr,&reused,&source);
    require(reused && !plan && source && live==1,"replay lost executable");
   }
   source.reset();
   require(!plan && live==0 && destroyed==1 && unsafe==0 && pending()==0 && events==0,"unsafe source teardown");
  }
  require(live==0 && unsafe==0 && pending()==0 && events==0,"final ownership invariant");
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
"""
