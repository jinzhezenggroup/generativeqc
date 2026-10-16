#pragma once

#include <cuda_runtime_api.h>

#include <new>
#include <stdexcept>

namespace generativeqc::runtime {

inline void cuda_resource_check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** Own a temporary change to CUDA's thread-local current device.
 * The throwing resource API always selects the requested device, preserving
 * its CUDA initialization boundary. The status-returning API avoids changing
 * an already selected device and rejects negative ordinals before any query.
 * A failed explicit restore retains the lease so cleanup can retry it.
 */
class CudaDeviceScope {
 public:
  using StatusChecker = void (*)(cudaError_t);
  enum class Operation { kQuery, kSelect };

  explicit CudaDeviceScope(int device, StatusChecker check = cuda_resource_check)
      : selected_(device) {
    select(true);
    check(status_);
  }

  CudaDeviceScope(int device, std::nothrow_t) noexcept : selected_(device) {
    if (device < 0) {
      operation_ = Operation::kSelect;
      status_ = cudaErrorInvalidDevice;
      return;
    }
    select(false);
  }

  CudaDeviceScope(const CudaDeviceScope&) = delete;
  CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

  ~CudaDeviceScope() {
    if (restore() != cudaSuccess) (void)cudaGetLastError();
  }

  [[nodiscard]] cudaError_t status() const noexcept { return status_; }
  [[nodiscard]] Operation operation() const noexcept { return operation_; }
  [[nodiscard]] int previous_device() const noexcept { return previous_; }
  [[nodiscard]] int selected_device() const noexcept { return selected_; }

  /** Restore explicitly when the caller must report cleanup failure.
   * Success is idempotent; failure leaves restoration pending and does not
   * consume CUDA's last error, so the boundary can apply its own error policy.
   */
  [[nodiscard]] cudaError_t restore() noexcept {
    if (!restore_pending_) return cudaSuccess;
    const cudaError_t result = cudaSetDevice(previous_);
    if (result == cudaSuccess) restore_pending_ = false;
    return result;
  }

 private:
  void select(bool reselect_current) noexcept {
    status_ = cudaGetDevice(&previous_);
    if (status_ != cudaSuccess || (!reselect_current && previous_ == selected_)) return;
    operation_ = Operation::kSelect;
    status_ = cudaSetDevice(selected_);
    restore_pending_ = status_ == cudaSuccess;
  }

  int previous_{-1};
  int selected_{-1};
  cudaError_t status_{cudaSuccess};
  Operation operation_{Operation::kQuery};
  bool restore_pending_{false};
};

}  // namespace generativeqc::runtime
