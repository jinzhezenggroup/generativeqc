"""Compile the real CUDA owner's construction path with injected API failures."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner, write_df_cpu_headers

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_owner_unwinds_every_setup_failure(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/cc/cuda_solver.cu").read_text()
    helpers = source[
        source.index("std::size_t checked_mul(") : source.index(
            "__global__ void damped_advance"
        )
    ]
    # Extract the live production members/constructor/destructor, not a copied
    # model. Kernel methods are irrelevant to constructor unwind and excluded.
    owner = source[
        source.index("struct Layout {") : source.index("  void virtual_corrections()")
    ]
    owner += source[
        source.index("  void cleanup() noexcept {") : source.index(
            "  template <class Output>"
        )
    ]
    provider = (ROOT / "src/tensor/cuda_contraction.cuh").read_text()
    provider = provider[
        provider.index("class CudaContractionContext {") : provider.index(
            "template <class T>"
        )
    ]
    # Compile the real shared preparation/cleanup owner against injected CUDA
    # APIs. Generated numerical tables are irrelevant to setup unwinding.
    support = (ROOT / "src/cc/cuda_solver_support.cuh").read_text()
    state = support[
        support.index("struct CudaState {") : support.index(
            "struct DeviceIterationOutputs"
        )
    ]
    write_df_cpu_headers(tmp_path)
    cpp = tmp_path / "owner.cpp"
    cpp.write_text(
        PREFIX
        + "namespace generativeqc::tensor {\n"
        + provider
        + "struct PreparedContractions { static constexpr std::size_t storage_bytes(std::size_t n) {return 128*n;} void release() {} };\n"
        + "}\n"
        + OPEN_CC
        + state
        + GENERATED
        + helpers
        + owner
        + "};\n"
        + MAIN
    )
    exe = tmp_path / "owner"
    compile_owner(compiler, tmp_path, [cpp], exe)
    result = subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include "cc/solver.hpp"
#include "cc/df_plan.hpp"
#include "generated_rccsd_cpu.hpp"
#include "generated_df_ccsd_spectator_pairs_cpu.hpp"
#include "runtime/allocation_measurement.hpp"
#include "solver/diis_ring.hpp"
#include <functional>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaStreamNonBlocking = 1, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToDevice = 2;
int calls = 0, fail_at = 0, streams = 0, events = 0, allocations = 0, handles = 0, device = 7;
int provider_alloc_failures = 0, arena_alloc_failures = 0;
int history_alloc_failures=0,history_alloc_error=2,numeric_attempts=0,history_attempts=0;
int free_error_once=0,sync_error_once=0;
std::size_t memory_limit=std::numeric_limits<std::size_t>::max(),live_bytes=0;
std::unordered_map<void*,std::size_t> allocation_bytes;
int step() { return ++calls == fail_at ? 999 : 0; }
using cublasHandle_t = void*;
constexpr int cudaErrorMemoryAllocation=2, CUBLAS_STATUS_ALLOC_FAILED=3;
constexpr int CUBLAS_POINTER_MODE_HOST=0, CUBLAS_PEDANTIC_MATH=0, CUBLAS_OP_N=0, CUBLAS_OP_T=1;
int cublasCreate(cublasHandle_t* p) {
  if (const int error=step()) return error;
  if (provider_alloc_failures) { --provider_alloc_failures; return CUBLAS_STATUS_ALLOC_FAILED; }
  *p=new int(1); ++handles; return 0;
}
int cublasDestroy(cublasHandle_t p) { delete static_cast<int*>(p); --handles; return 0; }
int cublasSetStream(cublasHandle_t, cudaStream_t) { return step(); }
int cublasSetPointerMode(cublasHandle_t, int) { return step(); }
int cublasSetMathMode(cublasHandle_t, int) { return step(); }
int cublasSetWorkspace(cublasHandle_t, void*, std::size_t) { return step(); }
int cudaMemGetInfo(std::size_t* free, std::size_t* total) {
  *free=*total=1ULL<<30; return step();
}
int cudaGetLastError() { return 0; }
int cublasDgemm(cublasHandle_t,int,int,int,int,int,const double*,const double*,int,
                 const double*,int,const double*,double*,int) {
  throw std::logic_error("ownership test must not execute numerical callback");
}
void audit_df_matrix(const double*,std::size_t,int*) {
  throw std::logic_error("ownership test must not execute numerical callback");
}
void blas_check(int code) { if (code) throw std::runtime_error("injected CUDA failure"); }
int cudaGetDevice(int* p) { *p = device; return 0; }
int cudaSetDevice(int d) { device = d; return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* p, int) {
  if (const int error = step()) return error;
  *p = new int(1); ++streams; return 0;
}
int cudaEventCreate(cudaEvent_t* p) {
  if (const int error = step()) return error;
  *p = new int(1); ++events; return 0;
}
int cudaEventDestroy(cudaEvent_t p) { delete static_cast<int*>(p); --events; return 0; }
int cudaMalloc(void** p, std::size_t bytes) {
  ++numeric_attempts;
  if (const int error = step()) return error;
  if (arena_alloc_failures) { --arena_alloc_failures; return cudaErrorMemoryAllocation; }
  if(allocations==1) {
    ++history_attempts;
    if(history_alloc_failures) { --history_alloc_failures; return history_alloc_error; }
  }
  if(bytes>memory_limit-live_bytes) return cudaErrorMemoryAllocation;
  *p = new unsigned char[bytes]; ++allocations;
  allocation_bytes[*p]=bytes;live_bytes+=bytes;return 0;
}
int cudaMemcpyAsync(void* d, const void* s, std::size_t n, int, cudaStream_t) {
  if (const int error = step()) return error;
  std::memcpy(d, s, n); return 0;
}
int cudaMemsetAsync(void* d, int value, std::size_t n, cudaStream_t) {
  if (const int error=step()) return error;
  std::memset(d,value,n); return 0;
}
"""

PREFIX += r"""
int cudaStreamSynchronize(cudaStream_t) {
  if(sync_error_once) { const auto result=sync_error_once;sync_error_once=0;return result; }
  return step();
}
int cudaFree(void* p) {
  if(free_error_once) { const auto result=free_error_once;free_error_once=0;return result; }
  live_bytes-=allocation_bytes.at(p);allocation_bytes.erase(p);
  delete[] static_cast<unsigned char*>(p); --allocations; return 0;
}
int cudaStreamDestroy(cudaStream_t p) { delete static_cast<int*>(p); --streams; return 0; }
void cuda_check(int code) { if (code) throw std::runtime_error("injected CUDA failure"); }
using cudaStreamCaptureStatus = int;
constexpr int cudaStreamCaptureStatusNone=0;
int cudaStreamIsCapturing(cudaStream_t, int* p) { *p=0; return step(); }
int cublasGetVersion(cublasHandle_t, int* p) { *p=120900; return step(); }
int cudaRuntimeGetVersion(int* p) { *p=12090; return step(); }
namespace generativeqc::runtime {
// Keep the extracted production owner on the same injected allocation APIs.
int resource_cuda_malloc(void** pointer,std::size_t bytes,bool* host_oom) {
  *host_oom=false; return cudaMalloc(pointer,bytes);
}
int resource_cuda_free(void* pointer) { return cudaFree(pointer); }
struct CudaDeviceScope {
  int previous;
  template<class Check> CudaDeviceScope(int selected,Check check) {
    check(cudaGetDevice(&previous)); check(cudaSetDevice(selected));
  }
  ~CudaDeviceScope() { (void)cudaSetDevice(previous); }
};
}
namespace generativeqc_tensor {
using ::cuda_check;
using ::blas_check;
struct DeviceAllocationError : std::runtime_error { using std::runtime_error::runtime_error; };
}
"""
OPEN_CC = r"""
namespace generativeqc::cc {
constexpr std::size_t kContractionProviderAllowance=96ULL<<20;
namespace generated {
"""
GENERATED = r"""
void prepare_iteration_contractions(CudaState&,tensor::CudaContractionContext&,std::size_t&,std::size_t&) {}
namespace dfcore {
struct CudaState : generated::CudaState {
  const double *df_virtual_singles{}, *df_virtual_doubles{};
};
}
namespace df {
struct CudaState {
  std::size_t o{}, v{};
  cudaStream_t stream{};
  int* error{};
  double* response_arena{};
};
struct ReplayCudaState : CudaState {
  tensor::PreparedContractions contractions;
};
constexpr std::size_t replay_binding_host_bytes() { return 128; }
void prepare_virtual_replay(ReplayCudaState&,tensor::CudaContractionContext&,
                            std::size_t&,std::size_t&) {}
}
namespace dfhoist {
struct CudaState : dfcore::CudaState {
  double *prepare_arena{}, *auxiliary_arena{};
};
constexpr std::size_t contraction_host_bytes(std::size_t variants) { return 1024+variants*512; }
std::size_t prepared_batch=0,prepared_tail=0;
void prepare_contractions(CudaState&,tensor::CudaContractionContext&,std::size_t batch,
                          std::size_t tail,std::size_t&,std::size_t&) {
  prepared_batch=batch; prepared_tail=tail;
}
}
namespace dfpairs {
struct CudaState : dfhoist::CudaState {
  tensor::PreparedContractions paired_contractions, paired_batched_contractions;
};
constexpr std::size_t contraction_host_bytes(std::size_t variants) { return 768+variants*512; }
void prepare_contractions(CudaState&,tensor::CudaContractionContext&,std::size_t,
                          std::size_t,std::size_t&,std::size_t&) {}
}
}
std::size_t problem_host_bytes(const Problem&) { return 128; }
std::uint64_t denominator_identity(const Problem&) { return 1; }
"""
MAIN = r"""
}  // namespace generativeqc::cc
int main() {
  generativeqc::cc::Problem p; p.nocc = p.nvir = 1;
  for (auto* v : {&p.foo, &p.fov, &p.fvv, &p.ovov, &p.ovvo, &p.oovv,
                 &p.ovvv, &p.ovoo, &p.oooo, &p.vvvv, &p.d1, &p.d2,
                 &p.initial_t1, &p.initial_t2}) v->push_back(1.0);
  // Compile the production owner once, then exercise disabled, one-slot and
  // ordinary DIIS. Event creation participates in the same failure sequence.
  bool saw_matrix=false;
  for (const unsigned naux : {0U, 2U, 5U, 10U, 15U, 16U}) {
  p.naux = naux; p.df_bov.assign(naux, 0.1); p.df_bvv.assign(naux, 0.1);
  for (const unsigned history : {0U, 1U, 6U}) {
  for (const bool packed : {false, true}) {
    generativeqc::cc::SolverOptions options;
    options.diis_size = history;
    options.packed_diis = packed;
    calls = 0; fail_at = 0;
    int constructor_calls = 0;
    { generativeqc::cc::Owner good(p, options, 0); constructor_calls = calls;
      saw_matrix = saw_matrix || good.plan.matrix_gemm || good.conventional_prepared;
      if (handles != (good.plan.matrix_gemm || good.conventional_prepared ? 1 : 0)) return 10;
      const auto detached = (good.n1 + good.n2) * sizeof(double);
      if (good.diagnostic.numeric_capacity_bytes < 128 + good.layout.total + detached) {
        std::cerr << "CUDA detached result storage was not reserved\n"; return 8;
      }
      // Both numeric allocations coexist with prepared host descriptors.
      auto expected_capacity = 128 + good.layout.total + good.layout.history_bytes + detached;
      if (good.plan.matrix_gemm) {
        const auto batch=good.plan.auxiliary_batch_size,tail=naux%batch;
        const auto variants=batch>1 ? 1+(tail>1) : 0;
        expected_capacity+=generativeqc::cc::kContractionProviderAllowance+
          generativeqc::cc::generated::dfhoist::contraction_host_bytes(variants);
        if(generativeqc::cc::generated::dfhoist::prepared_batch!=batch ||
           generativeqc::cc::generated::dfhoist::prepared_tail!=tail) return 17;
      }
      if (good.replay_matrix)
        expected_capacity+=generativeqc::cc::generated::df::replay_binding_host_bytes();
      if (good.pairs_enabled) {
        const auto batch=good.plan.auxiliary_batch_size,tail=naux%batch;
        expected_capacity+=generativeqc::cc::generated::dfpairs::contraction_host_bytes(
          batch>1 ? 1+(tail>1) : 0);
      }
      if (good.conventional_prepared) {
        expected_capacity+=generativeqc::cc::kContractionProviderAllowance+
          generativeqc::tensor::PreparedContractions::storage_bytes(
            generativeqc::cc::generated::iteration_prepared_contractions);
      }
      if(good.diagnostic.numeric_capacity_bytes!=expected_capacity) return 18;
      if (events != (history ? 2 : 0)) return 9;
    }
    if (streams || events || allocations || handles || device != 7 || constructor_calls < 18) return 1;
    for (int failure = 1; failure <= constructor_calls; ++failure) {
      calls = 0; fail_at = failure;
      try { generativeqc::cc::Owner broken(p, options, 0); return 2; }
      catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "injected CUDA failure") return 3;
      }
      if (streams || events || allocations || handles || device != 7) {
        std::cerr << "leaked owners after setup operation " << failure << '\n';
        return 4;
      }
      calls = 0; fail_at = 0;
      { generativeqc::cc::Owner retry(p, options, 0); }
      if (streams || events || allocations || handles || device != 7) return 5;
    }
    if (!naux) {
      options.max_bytes = 8ULL << 20;
      { generativeqc::cc::Owner bounded(p, options, 0);
        if (bounded.conventional_prepared || handles ||
            bounded.diagnostic.numeric_capacity_bytes > options.max_bytes) return 12;
      }
      if (streams || events || allocations || handles || device != 7) return 13;
    }
    options.max_bytes = 1; calls = 0;
    try { generativeqc::cc::Owner over_budget(p, options, 0); return 6; }
    catch (const std::length_error&) {}
    if (calls || streams || events || allocations || handles || device != 7) return 7;
    std::cout << "DIIS " << history << ": setup failures and retries checked: "
              << constructor_calls << '\n';
  }
  }
  }
  if (!saw_matrix) return 11;
  p.naux=2; p.df_bov.assign(2,0.1); p.df_bvv.assign(2,0.1);
  // Allocation rejection exercises the actual production retry chain: a Q
  // tile may lose its arena while the admitted matrix provider stays usable.
  for (int failures : {0, 1, 2}) {
    calls = fail_at = 0;
    arena_alloc_failures = failures;
    generativeqc::cc::SolverOptions options;
    options.df_replay_matrix_gemm=false;
    options.df_occupied_pairs=false;
    { generativeqc::cc::Owner retry(p, options, 0);
      if (retry.plan.matrix_gemm != (failures < 2)) return 12;
      if (retry.plan.auxiliary_batch_size != (failures == 0 ? 2U : 1U)) return 13;
    }
    if (streams || events || allocations || handles || device != 7) return 14;
  }
  // Replay admission is optional after the primal tile: its OOM retry must
  // keep the provider and Q batch before trying the existing primal fallbacks.
  for (int failures : {0, 1, 2, 3}) {
    calls = fail_at = 0;
    arena_alloc_failures = failures;
    generativeqc::cc::SolverOptions options;
    options.df_occupied_pairs=false;
    { generativeqc::cc::Owner retry(p, options, 0);
      if (retry.replay_matrix != (failures == 0)) return 37;
      if (retry.plan.matrix_gemm != (failures < 3)) return 38;
      if (retry.plan.auxiliary_batch_size != (failures < 2 ? 2U : 1U)) return 39;
    }
    if (streams || events || allocations || handles || device != 7) return 40;
  }
  calls = fail_at = 0;
  provider_alloc_failures = 1;
  { generativeqc::cc::SolverOptions options;
    generativeqc::cc::Owner retry(p, options, 0);
    if (retry.plan.matrix_gemm || retry.plan.auxiliary_batch_size != 1) return 15;
  }
  if (streams || events || allocations || handles || device != 7) return 16;
  for (const int refusal : {0,1,2}) {
    calls=fail_at=0;
    generativeqc::cc::SolverOptions options;
    options.packed_diis=true;
    { generativeqc::cc::Owner owner(p,options,0);
      if (!owner.packed) return 17;
      if (refusal==1) options.max_bytes=owner.non_history_capacity;
      if (refusal==2) arena_alloc_failures=1;
      owner.refuse_packed_history(options);
      if (owner.packed || !owner.diagnostic.packed_diis_refused) return 18;
      if ((owner.history.capacity()==0)!=(refusal!=0)) return 19;
      if (allocations != (refusal ? 1 : 2)) return 20;
    }
    if (streams || events || allocations || handles || device != 7) return 21;
  }
  p.nocc=2;p.nvir=6;p.naux=16;
  p.foo.assign(4,1.);p.fov.assign(12,1.);p.fvv.assign(36,1.);
  p.ovov.assign(144,1.);p.ovvo.assign(144,1.);p.oovv.assign(144,1.);
  p.ovvv.assign(432,1.);p.ovoo.assign(48,1.);p.oooo.assign(16,1.);p.vvvv.assign(1296,1.);
  p.d1.assign(12,-2.);p.d2.assign(144,-4.);p.initial_t1.assign(12,0.);p.initial_t2.assign(144,0.);
  p.df_bov.assign(192,.1);p.df_bvv.assign(576,.1);
  // New optional pair storage yields before the original replay and tile.
  for (int failures : {0,1,2,3,4}) {
    calls=fail_at=0;arena_alloc_failures=failures;
    generativeqc::cc::SolverOptions options;
    {generativeqc::cc::Owner retry(p,options,0);
      if(retry.pairs_enabled!=(failures==0)) return 43;
      if(retry.replay_matrix!=(failures<2)) return 44;
      if(retry.plan.matrix_gemm!=(failures<4)) return 45;
      if(retry.plan.auxiliary_batch_size!=(failures<3 ? 8U : 1U)) return 46;
      if(retry.diagnostic.df_pair_resource_refused!=(failures>0)) return 47;
    }
    if(streams || events || allocations || handles || device!=7) return 48;
  }
  for(const bool packed:{false,true}) {
    generativeqc::cc::SolverOptions options;options.diis_size=8;options.packed_diis=packed;
    options.df_replay_matrix_gemm=false;
    options.df_occupied_pairs=false;
    std::size_t wide_base=0,history_bytes=0,narrow_total=0,wide_non_history_capacity=0;
    {generativeqc::cc::Owner wide(p,options,0);
      wide_base=wide.layout.total;history_bytes=wide.layout.history_bytes;
      wide_non_history_capacity=wide.diagnostic.numeric_capacity_bytes-history_bytes;
      if(wide.plan.auxiliary_batch_size!=8) return 22;
    }
    options.df_auxiliary_batch_limit=1;
    {generativeqc::cc::Owner narrow(p,options,0);
      narrow_total=narrow.layout.total+narrow.layout.history_bytes;
    }
    options.df_auxiliary_batch_limit=8;
    // The optional wide base fits, but its second numeric allocation does not.
    // The complete one-Q pair must still be retried, including full histories.
    memory_limit=wide_base+history_bytes-1;
    if(narrow_total>memory_limit) return 23;
    try {
      generativeqc::cc::Owner retry(p,options,0);
      if(!retry.plan.matrix_gemm || retry.plan.auxiliary_batch_size!=1 ||
         live_bytes!=narrow_total || retry.packed!=packed) return 24;
      if(retry.diagnostic.owned_device_bytes<wide_base+generativeqc::cc::kContractionProviderAllowance ||
         retry.diagnostic.numeric_capacity_bytes<wide_non_history_capacity ||
         retry.diagnostic.synchronizations<2) return 32;
    } catch(const std::runtime_error& error) {
      std::cerr<<"history OOM bypassed fitting one-Q fallback, packed="<<packed
               <<": "<<error.what()<<'\n';return 25;
    }
    memory_limit=std::numeric_limits<std::size_t>::max();
    if(live_bytes || streams || events || allocations || handles || device!=7) return 26;
    for(const int failures:{1,2,3}) {
      history_alloc_failures=failures;numeric_attempts=history_attempts=0;
      try {
        generativeqc::cc::Owner retry(p,options,0);
        if(failures==3 || retry.plan.matrix_gemm!=(failures==1) ||
           retry.plan.auxiliary_batch_size!=1 || retry.packed!=packed) return 27;
      } catch(const std::runtime_error&) {if(failures!=3) return 28;}
      if(history_attempts!=std::min(failures+1,3) || numeric_attempts!=2*history_attempts ||
         history_alloc_failures || live_bytes || streams || events || allocations ||
         handles || device!=7) return 29;
    }
    // A history driver error, failed drain or failed partial free must propagate,
    // never turn into an optional-resource retry. Cleanup gets one safe retry.
    for(const int error:{0,1,2}) {
      history_alloc_failures=1;numeric_attempts=history_attempts=0;
      history_alloc_error=error==0 ? 999 : cudaErrorMemoryAllocation;
      sync_error_once=error==1 ? 999 : 0;free_error_once=error==2 ? 999 : 0;
      try {generativeqc::cc::Owner broken(p,options,0);return 30;}
      catch(const std::runtime_error&) {}
      if(numeric_attempts!=2 || history_attempts!=1 || live_bytes || streams || events ||
         allocations || handles || device!=7 || sync_error_once || free_error_once) return 31;
    }
    history_alloc_error=cudaErrorMemoryAllocation;
  }
  // Conventional preparation shares the two-allocation history retry. A
  // refused second allocation must release the provider, retain its prior
  // peak, and try the complete scalar pair exactly once.
  p.naux=0;p.df_bov.clear();p.df_bvv.clear();
  for(const bool packed:{false,true}) {
    std::size_t prepared_base=0,prepared_non_history=0;
    for(const int failures:{0,1,2}) {
      generativeqc::cc::SolverOptions options;options.diis_size=8;options.packed_diis=packed;
      history_alloc_failures=failures;numeric_attempts=history_attempts=0;
      try {
        generativeqc::cc::Owner retry(p,options,0);
        if(failures==2 || retry.conventional_prepared!=(failures==0) ||
           retry.plan.matrix_gemm || retry.packed!=packed) return 33;
        if(!failures) {
          prepared_base=retry.layout.total;
          prepared_non_history=retry.diagnostic.numeric_capacity_bytes-retry.layout.history_bytes;
        } else if(retry.diagnostic.owned_device_bytes<prepared_base+generativeqc::cc::kContractionProviderAllowance ||
                  retry.diagnostic.numeric_capacity_bytes<prepared_non_history ||
                  retry.diagnostic.conventional_provider_capacity_bytes ||
                  retry.diagnostic.conventional_binding_host_bytes ||
                  retry.diagnostic.synchronizations<2) return 34;
      } catch(const std::runtime_error&) {if(failures!=2) return 35;}
      if(history_attempts!=std::min(failures+1,2) || numeric_attempts!=2*history_attempts ||
         history_alloc_failures || live_bytes || streams || events || allocations ||
         handles || device!=7) return 36;
    }
  }
  // Post-execution packed refusal retains the conventional provider while
  // replacing histories. Its allowance coexists with the larger full payload.
  { generativeqc::cc::SolverOptions options;options.diis_size=8;options.packed_diis=true;
    generativeqc::cc::Owner owner(p,options,0);
    const auto packed_bytes=owner.layout.history_bytes;
    const auto before=owner.diagnostic.owned_device_bytes;
    if(!owner.conventional_prepared || !owner.packed || !handles) return 37;
    owner.refuse_packed_history(options);
    const auto expected=owner.layout.total+owner.layout.history_bytes+
        owner.diagnostic.conventional_provider_capacity_bytes;
    if(owner.packed || !owner.conventional_prepared || !handles ||
       owner.layout.history_bytes<=packed_bytes || expected<=before ||
       owner.diagnostic.owned_device_bytes!=expected ||
       owner.diagnostic.numeric_capacity_bytes!=owner.non_history_capacity+owner.layout.history_bytes)
      return 38;
  }
  if(live_bytes || streams || events || allocations || handles || device!=7) return 39;
  std::cout<<"Full/packed history allocation pairs retain one-Q/scalar retry and error gates\n";
}
"""
