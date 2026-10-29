#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <utility>

#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/provider_initialization_runtime.hpp"

namespace {
int fail_after = -1, allocations = 0, bucket_deletions = 0;
void* watched_bucket{};

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
[[noreturn]] void unexpected_hardware() { std::abort(); }
}  // namespace

void* operator new(std::size_t bytes) {
  ++allocations;
  if (fail_after == 0) {
    fail_after = -1;
    throw std::bad_alloc();
  }
  if (fail_after > 0) --fail_after;
  if (auto* result = std::malloc(bytes ? bytes : 1)) return result;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept {
  if (pointer && pointer == watched_bucket) {
    ++bucket_deletions;
    watched_bucket = nullptr;
  }
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

// Only the external ABI is replaced. Every path is inert before first creation.
#define CUDA_ABORT(name, args) \
  cudaError_t name args { unexpected_hardware(); }
CUDA_ABORT(cudaMemcpy, (void*, const void*, std::size_t, cudaMemcpyKind))
CUDA_ABORT(cudaMemcpyAsync, (void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t))
CUDA_ABORT(cudaMalloc, (void**, std::size_t))
CUDA_ABORT(cudaMallocAsync, (void**, std::size_t, cudaStream_t))
CUDA_ABORT(cudaFree, (void*))
CUDA_ABORT(cudaFreeAsync, (void*, cudaStream_t))
CUDA_ABORT(cudaGetDevice, (int*))
CUDA_ABORT(cudaRuntimeGetVersion, (int*))
CUDA_ABORT(cudaSetDevice, (int))
CUDA_ABORT(cudaStreamSynchronize, (cudaStream_t))
CUDA_ABORT(cudaStreamDestroy, (cudaStream_t))
CUDA_ABORT(cudaEventCreateWithFlags, (cudaEvent_t*, unsigned))
CUDA_ABORT(cudaEventSynchronize, (cudaEvent_t))
CUDA_ABORT(cudaEventDestroy, (cudaEvent_t))
CUDA_ABORT(cudaGraphExecDestroy, (cudaGraphExec_t))
CUDA_ABORT(cudaGraphDestroy, (cudaGraph_t))
CUDA_ABORT(cudaStreamBeginCapture, (cudaStream_t, int))
CUDA_ABORT(cudaStreamEndCapture, (cudaStream_t, cudaGraph_t*))
CUDA_ABORT(cudaGraphInstantiate, (cudaGraphExec_t*, cudaGraph_t, unsigned long long))
CUDA_ABORT(cudaGraphUpload, (cudaGraphExec_t, cudaStream_t))
CUDA_ABORT(cudaGraphLaunch, (cudaGraphExec_t, cudaStream_t))
#undef CUDA_ABORT

cublasStatus_t cublasCreate(cublasHandle_t*) { unexpected_hardware(); }
cublasStatus_t cublasDestroy(cublasHandle_t) { unexpected_hardware(); }
#define SOLVER_ABORT(name, args) \
  cusolverStatus_t name args { unexpected_hardware(); }
SOLVER_ABORT(cusolverDnCreate, (cusolverDnHandle_t*))
SOLVER_ABORT(cusolverDnDestroy, (cusolverDnHandle_t))
SOLVER_ABORT(cusolverDnSetStream, (cusolverDnHandle_t, cudaStream_t))
SOLVER_ABORT(cusolverDnCreateParams, (cusolverDnParams_t*))
SOLVER_ABORT(cusolverDnDestroyParams, (cusolverDnParams_t))
SOLVER_ABORT(cusolverDnCreateSyevjInfo, (syevjInfo_t*))
SOLVER_ABORT(cusolverDnDestroySyevjInfo, (syevjInfo_t))
SOLVER_ABORT(cusolverDnXsyevjSetTolerance, (syevjInfo_t, double))
SOLVER_ABORT(cusolverDnXsyevjSetMaxSweeps, (syevjInfo_t, int))
SOLVER_ABORT(cusolverDnXsyevjSetSortEig, (syevjInfo_t, int))
#undef SOLVER_ABORT

namespace generativeqc::runtime {
CudaProviderIdentityObservation observe_cuda_provider_identity(std::uint8_t,
                                                               const std::array<const void*, 3>&) {
  unexpected_hardware();
}
}  // namespace generativeqc::runtime
namespace generativeqc::scf {
std::size_t hf_cuda_retained_host_numeric_bytes(const CudaRhfBucketPlan&) noexcept {
  return 0;  // The fixture never prepares numeric/topology payload.
}
}  // namespace generativeqc::scf

int main() {
  using namespace generativeqc::scf;
  int attachment_allocations = -1;
  // The control discovers the actual allocation population. Every later pass
  // fails one allocation without depending on Holder size or a fixed call count.
  for (int fault = -1; fault == -1 || fault < attachment_allocations; ++fault) {
    auto owner =
        make_pending_reference_provider_owner(ProviderInitializationIntent(512), 0, 1024, 1024);
    const std::weak_ptr<PendingReferenceProviderOwner> lifetime = owner;
    auto* raw = new CudaRhfBucketPlan{};
    watched_bucket = raw;
    bucket_deletions = 0;
    bool allocation_failed = false;
    allocations = 0;
    fail_after = fault;
    try {
#include "attach_rollback.inc"
    } catch (const std::bad_alloc&) {
      allocation_failed = true;
    } catch (...) {
      require(false, "attachment replaced its original allocation exception");
    }
    fail_after = -1;
    if (fault == -1) {
      attachment_allocations = allocations;
      require(attachment_allocations > 0, "attachment exercised no allocations");
      require(!allocation_failed && owner->bucket() == raw, "control did not attach");
      require(bucket_deletions == 0, "control deleted its live bucket");
    } else {
      require(allocation_failed, "fault did not reach an attachment allocation");
      require(owner->bucket() == nullptr, "failed attachment kept a published bucket");
      require(bucket_deletions == 1, "rollback did not delete the inert bucket exactly once");
    }
    require(owner->close_current().confirmed(), "owner could not close after attachment");
    require(owner->closed() && !owner->bucket(), "closed owner retained its bucket");
    require(owner->close_current().confirmed(), "repeated close lost confirmation");
    require(bucket_deletions == 1, "close changed the single bucket deletion count");
    owner.reset();
    require(lifetime.expired(), "retention cell kept the closed owner alive");
  }
  std::cout << "attachment control and " << attachment_allocations
            << " allocation faults passed without CUDA/vendor calls\n";
}
