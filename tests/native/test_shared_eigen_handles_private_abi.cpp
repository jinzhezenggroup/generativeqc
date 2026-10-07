// The real embedded private status ABI must never meet the official enum ABI
// in one translation unit. Only the shared ABI-free owner crosses that seam.
#include <cassert>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

#include "runtime/nvidia_host_api.h"
#include "runtime/types.hpp"
#include "solver/cuda/symmetric_eigen_handles.hpp"

namespace shared = generativeqc::solver::cuda;
static_assert(std::is_same_v<cusolverStatus_t, std::uint32_t>);

std::uint32_t private_handle_create(shared::PreparedSymmetricEigenHandles& owner) {
  const cusolverStatus_t status = owner.create();
  return status;
}

std::uint32_t private_handle_bind(shared::PreparedSymmetricEigenHandles& owner, void* stream) {
  const cusolverStatus_t status = owner.bind_stream(stream);
  return status;
}

shared::SymmetricEigenResources private_handle_view(
    const shared::PreparedSymmetricEigenHandles& owner) {
  const auto resources = owner.view();
  const cusolverDnHandle_t solver = static_cast<cusolverDnHandle_t>(resources.solver);
  const cusolverDnParams_t parameters = static_cast<cusolverDnParams_t>(resources.parameters);
  const syevjInfo_t jacobi = static_cast<syevjInfo_t>(resources.jacobi);
  return {solver, parameters, jacobi};
}

std::uint32_t trace_blas_create(void**);
std::uint32_t trace_blas_bind(void*, void*);
std::uint32_t trace_blas_destroy(void*);
void trace_external(const char*);
void begin_teardown_trace();
cudaError_t trace_release(void*, bool);

extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  void* result{};
  const auto status = trace_blas_create(&result);
  *handle = static_cast<cublasHandle_t>(result);
  return status;
}
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
  return trace_blas_bind(handle, stream);
}
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  return trace_blas_destroy(handle);
}

namespace runtime {
std::mutex allocation_measurement_mutex;
cudaError_t resource_cuda_free(void* pointer) { return trace_release(pointer, false); }
cudaError_t resource_cuda_free_async(void* pointer, cudaStream_t) {
  return trace_release(pointer, true);
}
}  // namespace runtime

struct Prepared {
  ~Prepared() { trace_external("prepared"); }
};
bool ensure_cuda_gfn2_parameters(int, std::string&) {
  trace_external("gfn2-parameters");
  return true;
}
std::string cuda_error_message(const char* text, cudaError_t) { return text; }

struct CudaResources {
  int device_id_{3};
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  shared::PreparedSymmetricEigenHandles eigen_handles_;
  void *reference_eri_{}, *solver_workspace_{}, *direct_tile_validation_{}, *arena_{};
  void* reference_fock_correction_{};
  void* solver_host_workspace_{};
  ~CudaResources();
};
struct OrdinaryStreamEigensolver {
  int device_{3};
  struct Resources {
    cudaStream_t stream_{};
    void* solver_workspace_{};
  } resources_;
  shared::PreparedSymmetricEigenHandles handles_;
  void cleanup() noexcept;
  ~OrdinaryStreamEigensolver();
};

#include "shared_eigen_lifetimes.inc"

void* private_gfn2_create(void* stream) {
  auto* owner = new Gfn2Owner;
  owner->stream = static_cast<cudaStream_t>(stream);
  return owner;
}
std::int32_t private_gfn2_ensure(void* pointer, std::string& error) {
  return static_cast<Gfn2Owner*>(pointer)->ensure_handles(error);
}
bool private_gfn2_published(void* pointer) {
  const auto& owner = *static_cast<Gfn2Owner*>(pointer);
  const auto view = owner.eigen_handles.view();
  return owner.handles_created && owner.blas && view.solver && view.parameters && view.jacobi;
}
bool private_gfn2_empty(void* pointer) {
  const auto& owner = *static_cast<Gfn2Owner*>(pointer);
  const auto view = owner.eigen_handles.view();
  return !owner.handles_created && !owner.blas && !view.solver && !view.parameters && !view.jacobi;
}
void private_gfn2_destroy(void* pointer) {
  auto* owner = static_cast<Gfn2Owner*>(pointer);
  owner->prepared = std::make_unique<Prepared>();
  delete owner;
}

void private_scf_lifetime(const std::string& mode, void* stream) {
  assert(cudaSetDevice(3) == cudaSuccess);
  if (mode == "ordinary") {
    OrdinaryStreamEigensolver owner;
    owner.resources_ = {static_cast<cudaStream_t>(stream), reinterpret_cast<void*>(0x200)};
    assert(owner.handles_.create() == 0);
    assert(owner.handles_.bind_stream(stream) == 0);
    assert(owner.handles_.create_parameters() == 0);
    begin_teardown_trace();
  } else {
    CudaResources owner;
    owner.stream_ = static_cast<cudaStream_t>(stream);
    owner.reference_eri_ = reinterpret_cast<void*>(0x100);
    owner.reference_fock_correction_ = reinterpret_cast<void*>(0x500);
    owner.solver_workspace_ = reinterpret_cast<void*>(0x200);
    owner.direct_tile_validation_ = reinterpret_cast<void*>(0x300);
    owner.arena_ = reinterpret_cast<void*>(0x400);
    assert(cublasCreate(&owner.blas_) == 0);
    assert(owner.eigen_handles_.create() == 0);
    assert(owner.eigen_handles_.bind_stream(stream) == 0);
    if (mode == "rhf-jacobi")
      assert(owner.eigen_handles_.configure_jacobi(1.0e-13, 100, 1) == 0);
    else {
      assert(mode == "rhf-generic");
      assert(owner.eigen_handles_.create_parameters() == 0);
    }
    begin_teardown_trace();
  }
}
