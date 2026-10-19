"""Execute the production MD source launch/error boundary with a CUDA double.

Only CUDA launch punctuation and device execution are replaced. Assertions cover
observable failure propagation and submission termination, not source spelling,
internal call counts, or GPU arithmetic. Native CUDA CI still compiles the owner.
"""

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _host_probe(cuda_source: str, adapter_source: str) -> str:
    definitions = [
        cpp_function_definition(cuda_source, "check_md_j_cuda"),
        cpp_function_definition(
            cuda_source, "source_workspace_width", include_template=True
        ),
        cpp_function_definition(
            cuda_source, "launch_source_kernel", include_template=True
        ),
        cpp_function_definition(
            cuda_source, "launch_source_class", include_template=True
        ),
        cpp_function_definition(cuda_source, "launch_md_source_jk"),
        cpp_function_definition(adapter_source, "direct_jk_check"),
        cpp_function_definition(
            adapter_source, "direct_jk_guard", include_template=True
        ),
    ]
    definitions.insert(
        3,
        cpp_function_definition(
            cuda_source, "launch_masked_packed_source", include_template=True
        ),
    )
    definitions.insert(
        3,
        cpp_function_definition(
            cuda_source, "launch_masked_tile_source", include_template=True
        ),
    )
    # Forward the actual launch configuration to the recording kernel double.
    # No branch, status operation, or return statement is replaced.
    production = re.sub(r"<<<([^;]+?)>>>\s*\(", r"(\1, ", "\n".join(definitions))
    return PREFIX + production + DRIVER


def test_md_source_launch_propagates_first_failure(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    cpp = tmp_path / "md_source_status.cpp"
    cpp.write_text(
        _host_probe(
            (ROOT / "src/scf/cuda/direct_md_jk.cu").read_text(),
            (ROOT / "src/scf/cuda/direct_jk.cpp").read_text(),
        )
    )
    executable = native_cxx.build_executable(
        [cpp],
        tmp_path / "md_source_status",
        compile_args=["-std=c++17", "-O2", f"-I{ROOT / 'src'}"],
    )
    subprocess.run([executable], check=True, timeout=10)


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <set>
#include <string>
#include <new>
#include <stdexcept>
#include <vector>
#define __host__
#define __device__
#include "scf/cuda/md_source_tiles.cuh"
using namespace generativeqc::scf::cuda_execution;
using cudaError_t = int;
using cudaStream_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorLaunchFailure=2;
constexpr unsigned kSourceCandidates=128, kSourceWorkers=4096;
struct DeviceBatch { std::size_t nbf=4; };
struct MdJView {
  unsigned active_pair_offsets[6]{0,1,2,3,4,5};
  unsigned long long* source_cursor=nullptr;
  void* work_counts=nullptr;
  bool residual_tiled=false;
  bool masked_residual=false;
  unsigned masked_tiles=0;
  void* masked_pairs=nullptr;
  std::size_t pair_count=1;
  unsigned long long residual_candidate_count=15;
};
template<unsigned A,unsigned B> struct SourceSchedule {
  static constexpr unsigned groups=A+B<=2?16:1, block_threads=A+B<=2?128:32;
};
template<class T,unsigned A> struct CoulombAuxiliary {
  static constexpr unsigned kStateCount=100;
};
template<unsigned BraAngular,unsigned KetAngular,bool Cached=false> struct MdMaskedWorkspace {
  double storage[1024];
};
constexpr unsigned source_components(unsigned a) { return a==0?1:a==1?3:a==2?9:a==3?18:36; }
constexpr unsigned md_j_hermite_count(unsigned a) { return (a+1)*(a+2)*(a+3)/6; }
struct Submission { char kind; void* pointer; std::size_t bytes; };
std::vector<Submission> submissions;
std::size_t fail_at=0;
bool failed=false;
cudaError_t sticky=cudaSuccess;
unsigned drains=0;
void reset(std::size_t fail=0,cudaError_t prior=cudaSuccess) {
  submissions.clear();fail_at=fail;failed=false;sticky=prior;drains=0;
}
bool submit(char kind,void* pointer=nullptr,std::size_t bytes=0) {
  // No further operation may be submitted after a failed reset or kernel launch.
  assert(!failed);
  submissions.push_back({kind,pointer,bytes});
  failed=fail_at && submissions.size()==fail_at;
  return failed;
}
cudaError_t cudaMemsetAsync(void* pointer,int value,std::size_t bytes,cudaStream_t stream) {
  assert(value==0 && stream==7);
  return submit('m',pointer,bytes)?cudaErrorInvalidValue:cudaSuccess;
}
cudaError_t cudaGetLastError() {
  auto error=sticky;sticky=cudaSuccess;return error;
}
template<unsigned A,unsigned B,bool Count,bool Tiled,bool Masked,bool Cached=false,class... Args>
void consume_source_tasks(unsigned workers,unsigned threads,std::size_t shared_bytes,
                          cudaStream_t stream,Args...) {
  assert(workers>0 && threads>0 && shared_bytes>0 && stream==7);
  if(submit('k')) sticky=cudaErrorLaunchFailure;
}
template<class... Args>
void md_masked_refresh(unsigned workers,unsigned threads,std::size_t shared_bytes,
                       cudaStream_t stream,Args...) {
  assert(workers && threads && stream==7);
  if(submit('k'))sticky=cudaErrorLaunchFailure;
}
template<bool Coarse> struct MdMaskedTileFallback {};
template<unsigned KetAngular> struct MdMaskedPackedFallback {};
template<unsigned KetAngular,bool Count,class Fallback,class... Args>
void md_masked_packed_sources(unsigned workers,unsigned threads,std::size_t shared_bytes,
                              cudaStream_t stream,Args...) {
  assert(workers && threads==128 && shared_bytes==0 && stream==7);
  if(submit('k'))sticky=cudaErrorLaunchFailure;
}
template<bool Count,class Fallback,bool Primitive=false,class... Args>
void md_masked_tiles(unsigned workers,unsigned threads,std::size_t shared_bytes,
                     cudaStream_t stream,Args...) {
  assert(workers && threads==128 && shared_bytes==0 && stream==7);
  if(submit('k'))sticky=cudaErrorLaunchFailure;
}
using generativeqc_status=int;
constexpr int GENERATIVEQC_STATUS_SUCCESS=0, GENERATIVEQC_STATUS_OUT_OF_MEMORY=3;
constexpr int GENERATIVEQC_STATUS_INVALID_ARGUMENT=4;
struct DirectJkFailure { int status; std::string detail; };
struct CudaDirectJkPlan { cudaStream_t stream=7; };
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  assert(stream==7);++drains;return cudaSuccess;
}
int source_cuda_status(int status) { return status; }
const char* cudaGetErrorString(int) { return "injected runtime failure"; }

"""

DRIVER = r"""
int main() {
  DeviceBatch batch;MdJView md;unsigned long long cursor=0;md.source_cursor=&cursor;
  double j[96]{},alpha[96]{},beta[96]{};
  CudaDirectJkPlan plan;std::string detail;
  auto run=[&](bool want_j,bool want_k,bool unrestricted) {
    return direct_jk_guard(&plan,detail,[&] {
      launch_md_source_jk(7,batch,md,2,3,want_j,want_k,unrestricted,
                         1e-10,nullptr,nullptr,nullptr,j,alpha,beta);
      direct_jk_check(cudaGetLastError());
    });
  };
  for(bool want_j:{false,true})for(bool want_k:{false,true})
  for(bool unrestricted:{false,true})for(bool counted:{false,true})
  for(bool coarse:{false,true})for(bool masked:{false,true})for(bool cached:{false,true})for(unsigned tiled:{0U,1U,2U}) {
    md.work_counts=counted?&cursor:nullptr;
    md.residual_tiled=coarse;
    md.masked_residual=masked;
    md.masked_tiles=tiled;
    md.masked_pairs=cached?&cursor:nullptr;
    reset();assert(run(want_j,want_k,unrestricted)==cudaSuccess);
    assert(drains==0 && detail.empty());
    const auto successful=submissions;
    assert(!successful.empty());
    std::set<void*> cleared;
    for(const auto& event:successful)if(event.kind=='m') {
      cleared.insert(event.pointer);
      if(event.pointer==&cursor) assert(event.bytes==sizeof(cursor));
      else assert(event.bytes==3*16*sizeof(double));
    }
    std::set<void*> expected{&cursor};
    if(want_j) expected.insert(j+32);
    if(want_k) { expected.insert(alpha+32);if(unrestricted)expected.insert(beta+32); }
    assert(cleared==expected);
    // Discover failure points from observable successful submissions, so harmless
    // batching/refactoring does not pin an internal call count or source shape.
    for(std::size_t stage=1;stage<=successful.size();++stage) {
      reset(stage);
      const auto expected_error=successful[stage-1].kind=='m'?
          cudaErrorInvalidValue:cudaErrorLaunchFailure;
      assert(run(want_j,want_k,unrestricted)==expected_error);
      assert(failed && submissions.size()==stage && drains>0 && !detail.empty());
    }
  }
  // A valid empty J residual still resets the requested output; a runtime error
  // during that reset is not mistaken for an empty successful operation.
  md.residual_candidate_count=0;
  reset();assert(run(true,false,false)==cudaSuccess);
  assert(std::none_of(submissions.begin(),submissions.end(),[](auto e){return e.kind=='k';}));
  reset(1);assert(run(true,false,false)==cudaErrorInvalidValue);
  reset(0,cudaErrorLaunchFailure);assert(run(true,false,false)==cudaErrorLaunchFailure);
  assert(sticky==cudaSuccess);
  // No active class is also a valid empty schedule; preserve the former caller's
  // final sticky-error check even when there is no kernel launch to check.
  md.residual_candidate_count=15;
  for(auto& offset:md.active_pair_offsets)offset=0;
  reset();assert(run(true,true,true)==cudaSuccess);
  assert(std::none_of(submissions.begin(),submissions.end(),[](auto e){return e.kind=='k';}));
  reset(0,cudaErrorLaunchFailure);assert(run(true,true,true)==cudaErrorLaunchFailure);
  reset(1);assert(run(true,true,true)==cudaErrorInvalidValue);
}
"""
