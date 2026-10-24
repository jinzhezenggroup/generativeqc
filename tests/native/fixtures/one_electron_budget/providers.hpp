#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
using cudaStream_t = void*;
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorUnknown = 999,
  cudaErrorNotSupported = 801,
  cudaErrorInvalidResourceHandle = 400
};
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone };
inline constexpr int cudaMemcpyHostToDevice = 1;
namespace stub {
inline std::size_t live_allocations = 0, last_allocation = 0, last_integral_budget = 0;
}
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA mock error"; }
inline cudaError_t cudaGetDevice(int* p) {
  *p = 0;
  return cudaSuccess;
}
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* p) {
  *p = cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, std::size_t bytes) {
  *p = std::malloc(bytes);
  if (!*p) return cudaErrorMemoryAllocation;
  ++stub::live_allocations;
  stub::last_allocation = bytes;
  return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) {
  if (p) {
    std::free(p);
    --stub::live_allocations;
  }
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, int,
                                   cudaStream_t) {
  std::memcpy(dst, src, bytes);
  return cudaSuccess;
}

#pragma once
#include <cstddef>
#include <string>
#include <vector>
enum generativeqc_status {
  GENERATIVEQC_STATUS_SUCCESS = 0,
  GENERATIVEQC_STATUS_OUT_OF_MEMORY = 1,
  GENERATIVEQC_STATUS_INVALID_ARGUMENT = 2
};
namespace generativeqc {
namespace core {
inline constexpr int kSphericalBasis = 1;
struct Atom {
  int ionic_charge() const { return 1; }
};
struct Shell {
  unsigned angular_momentum, atom_index;
  std::vector<int> primitives;
};
struct System {
  int basis_representation = 0;
  std::vector<Atom> atoms{{}, {}};
  std::vector<Shell> shells{{0, 0, {1, 2}}, {0, 1, {1, 2}}};
};
}  // namespace core
namespace methods {
class PreparedBatch {};
}  // namespace methods
namespace dft {
struct CudaKsFinalStateToken {};
struct Primary {
  core::System data;
  const core::System& system() const { return data; }
};
inline Primary primary;
struct CudaKsResidentFockBinding {
  Primary* primary = &dft::primary;
  int device_id = 0;
  void* stream = reinterpret_cast<void*>(1);
  std::size_t nao = 2;
  explicit operator bool() const { return primary; }
  bool operator==(const CudaKsResidentFockBinding&) const = default;
};
struct Shape {
  std::size_t atoms = 2, primitives = 4, nao = 2;
};
struct Lease {
  const double* basis;
  int device_id;
  cudaStream_t stream;
  cudaError_t (*authenticate)(void*) noexcept;
  void* owner;
};
class CudaKsGridReference {
 public:
  CudaKsGridReference(methods::PreparedBatch&, std::size_t, const CudaKsFinalStateToken&) {}
  generativeqc_status prepare(std::string&) { return GENERATIVEQC_STATUS_SUCCESS; }
  Shape shape() const { return {}; }
  Lease reference() {
    static double basis[100]{};
    return {basis, 0, reinterpret_cast<void*>(1), +[](void*) noexcept { return cudaSuccess; },
            this};
  }
};
}  // namespace dft
namespace methods::detail {
inline generativeqc_status dft_cuda_resident_fock_sources(PreparedBatch&, std::size_t,
                                                          const dft::CudaKsFinalStateToken&,
                                                          dft::CudaKsResidentFockBinding& out,
                                                          std::string&) {
  out = {};
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace methods::detail
namespace molecule {
inline unsigned cartesian_count(unsigned a) { return (a + 1) * (a + 2) / 2; }
}  // namespace molecule
}  // namespace generativeqc

#include "integrals/cuda_one_electron_response.hpp"

namespace generativeqc::integrals {
// The provider models one mandatory arena plus an optional bounded schedule.
// No arithmetic, GPU scheduling or live scientific state is simulated here.
inline constexpr std::size_t mandatory_bytes = 256, optional_bytes = 512;
cudaError_t cuda_one_electron_workspace(const CudaOneElectronShape&,
                                        std::span<const CudaOneElectronShell>,
                                        const CudaOneElectronOptions& options,
                                        std::size_t& bytes) noexcept {
  bytes = 0;
  if (options.max_workspace_bytes < mandatory_bytes || options.max_jobs == 1)
    return cudaErrorInvalidValue;
  bytes = options.parallel_streams && options.max_workspace_bytes >= optional_bytes
              ? optional_bytes
              : mandatory_bytes;
  return cudaSuccess;
}
struct CudaOneElectronResponse::Impl {
  CudaOneElectronWork work;
};
CudaOneElectronResponse::CudaOneElectronResponse() : impl_(std::make_unique<Impl>()) {}
CudaOneElectronResponse::~CudaOneElectronResponse() = default;
const CudaOneElectronWork& CudaOneElectronResponse::work() const noexcept { return impl_->work; }
cudaError_t CudaOneElectronResponse::prepare(const CudaOneElectronShape& shape,
                                             std::span<const CudaOneElectronShell> shells,
                                             const CudaOneElectronOptions& options,
                                             const CudaOneElectronReference&, void*,
                                             std::size_t bytes) noexcept {
  std::size_t required{};
  const auto error = cuda_one_electron_workspace(shape, shells, options, required);
  if (error != cudaSuccess || required != bytes) return cudaErrorInvalidValue;
  impl_->work.workspace_bytes = bytes;
  return cudaSuccess;
}
cudaError_t CudaOneElectronResponse::apply(const double*, const CudaOneElectronWeights&,
                                           const CudaOneElectronOutput&, cudaStream_t) noexcept {
  return cudaSuccess;
}
}  // namespace generativeqc::integrals
