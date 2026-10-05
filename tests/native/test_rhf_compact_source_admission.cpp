#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "integrals/electron_interaction_source.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"
#include "scf/cuda/rhf_source_handoff.hpp"
int mode = 0, allocations = 0, frees = 0, copies = 0, fences = 0;
void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}
cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaMalloc(void** result, std::size_t size) {
  if (mode == 1) return cudaErrorMemoryAllocation;
  *result = std::malloc(size ? size : 1);
  ++allocations;
  return cudaSuccess;
}
cudaError_t cudaFree(void* value) {
  if (value) {
    std::free(value);
    ++frees;
  }
  return cudaSuccess;
}
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA status"; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
  *stream = reinterpret_cast<void*>(8);
  return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++fences;
  return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* to, const void* from, std::size_t count, cudaMemcpyKind,
                            cudaStream_t) {
  if (mode == 2) return cudaErrorInvalidValue;
  std::memcpy(to, from, count);
  ++copies;
  return cudaSuccess;
}
namespace generativeqc::scf::cuda_execution {
CudaResources::~CudaResources() = default;
RhfIterationGraphs::~RhfIterationGraphs() = default;
GeneratedCoulombPlan::~GeneratedCoulombPlan() = default;
GeneratedExchangePlan::~GeneratedExchangePlan() = default;
}  // namespace generativeqc::scf::cuda_execution
namespace generativeqc::scf {
CudaDirectJkPlan::~CudaDirectJkPlan() {
  for (auto* allocation : allocations) runtime::resource_cuda_free(allocation);
}
generativeqc_status enqueue_cuda_direct_eri_tile(CudaDirectJkPlan*, std::size_t,
                                                 const std::array<std::size_t, 4>&,
                                                 const std::array<std::size_t, 4>&, double*,
                                                 std::size_t, cudaStream_t, std::string&) {
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace generativeqc::scf
int main() {
  try {
    using namespace generativeqc;
    scf::CudaRhfBucketPlan owner;
    owner.initialized = true;
    owner.batch_size = 1;
    owner.nbf = 2;
    owner.total_atoms = 1;
    owner.options.export_physical_reference = true;
    owner.resources.device_id_ = 0;
    alignas(double) unsigned char arena[128]{};
    const double positions[3]{1, 2, 3};
    std::memcpy(arena, positions, sizeof positions);
    owner.layout.bytes = sizeof arena;
    owner.layout.positions = 0;
    owner.resources.arena_ = arena;
    owner.resources.stream_ = reinterpret_cast<void*>(4);
    core::System system;
    system.atoms.push_back({2, {1, 2, 3}});
    system.ecp_terms.resize(17);
    constexpr std::size_t peak = 8000;
    auto copy = system;
    auto value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, 1000000);
    require(
        value.source && value.device_copy_bytes == sizeof positions && copies == 1 && fences == 1,
        "complete compaction not fenced");
    require(value.numeric_peak_bytes == peak + value.retained_numeric_bytes &&
                value.required_peak_bytes == value.numeric_peak_bytes,
            "simultaneous retained budget mismatch");
    const auto retained = value.retained_numeric_bytes, exact = value.required_peak_bytes;
    require(value.source->orbital().ecp_terms.size() == 17, "moved ECP snapshot lost");
    require(value.source->retained_numeric_bytes() == retained, "source capacity differs");
    value.source.reset();
    require(allocations == frees, "success allocation leak");
    copy = system;
    value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, exact);
    require(value.source && !value.resource_fallback, "exact capacity rejected");
    value.source.reset();
    const auto before = allocations;
    copy = system;
    value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, exact - 1);
    require(!value.source && value.resource_fallback && value.numeric_peak_bytes == peak &&
                value.required_peak_bytes == exact && allocations == before,
            "short budget allocated or lost reference");
    copy = system;
    copy.ecp_terms.clear();
    copy.ecp_terms.shrink_to_fit();
    value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, 1000000);
    require(value.retained_numeric_bytes + system.ecp_terms.capacity() * sizeof(core::EcpTerm) ==
                retained,
            "ECP capacity omitted");
    value.source.reset();
    mode = 1;
    copy = system;
    value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, 1000000);
    require(!value.source && value.resource_fallback, "OOM did not preserve fallback");
    mode = 2;
    copy = system;
    bool caught = false;
    try {
      value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, 1000000);
    } catch (const std::runtime_error&) {
      caught = true;
    }
    require(caught && allocations == frees, "transport error hidden or allocation leaked");
    mode = 0;
    owner.layout.bytes = sizeof(double);
    copy = system;
    caught = false;
    try {
      value = scf::detach_rhf_cuda_source(owner, std::move(copy), peak, 1000000);
    } catch (const std::logic_error&) {
      caught = true;
    }
    require(caught && allocations == frees, "malformed view did not fail safely");
    std::cout << "actual compact source: exact/short budget, ECP, copy/fence, OOM, transport, "
                 "malformed extent pass\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
