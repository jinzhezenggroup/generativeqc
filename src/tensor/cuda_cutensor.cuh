#pragma once

// Optional provider implementation, included by the shared executor only when
// the build explicitly enables and links cuTENSOR.
#include <cutensor.h>

#include <chrono>
#include <string>
#include <type_traits>

#include "tensor/cuda_affine_audit.cuh"
#include "tensor/cuda_runtime.cuh"
#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

class CutensorError : public std::runtime_error {
 public:
  explicit CutensorError(cutensorStatus_t status)
      : std::runtime_error(cutensorGetErrorString(status)), status(status) {}
  cutensorStatus_t status;
};

inline void cutensor_check(cutensorStatus_t status) {
  if (status != CUTENSOR_STATUS_SUCCESS) throw CutensorError(status);
}

/** One immutable affine contraction plan on a caller-owned stream.
 * Native preparation supplies provider resource reservations; scientific
 * consumers never name cuTENSOR. No JIT, global plan cache or incremental
 * autotuning is enabled. execute only binds borrowed addresses and enqueues.
 * This object is not thread-safe and must be destroyed before its stream.
 * cuTENSOR exposes exact device workspace but no opaque host allocation query:
 * host_bytes is an externally qualified reservation, not a measured footprint.
 */
class CudaCutensorContraction {
 public:
  /** Stable preparation facts, excluding addresses and mutable replay counts.
   * The request carries resolved modes/extents/strides, arithmetic and alpha/beta;
   * version/architecture plus the fixed algorithm and kernel rank identify the
   * provider choice. Identity strings remain borrowed from the compiler artifact,
   * which must outlive this record. This description is not a cache key. */
  struct Provenance {
    ContractionRequest request;
    std::size_t provider_version{}, workspace_bytes{};
    int runtime_version{}, architecture{};
    cutensorAlgo_t algorithm{};
    std::int32_t kernel_rank{};
  };

  CudaCutensorContraction() = default;
  CudaCutensorContraction(const CudaCutensorContraction&) = delete;
  CudaCutensorContraction& operator=(const CudaCutensorContraction&) = delete;
  ~CudaCutensorContraction() { reset(); }

  bool prepare(const ContractionRequest& request, cudaStream_t stream, std::size_t workspace_limit,
               std::size_t provider_limit, std::size_t host_reservation) {
    if (handle_) throw std::logic_error("cuTENSOR binding is already prepared");
    request.validate_affine();
    require_uncaptured(stream);
    rejection_ = {};
    if (host_reservation < sizeof(*this)) {
      rejection_ = "host metadata reservation is insufficient";
      return false;
    }
    const auto started = std::chrono::steady_clock::now();
    ++prepare_calls_;
    request_ = request;
    audit_.rank = request.operands[2].rank;
    for (std::size_t axis = 0; axis < audit_.rank; ++axis) {
      audit_.shape[axis] = request.operands[2].shape[axis];
      audit_.strides[axis] = request.operands[2].strides[axis];
    }
    stream_ = stream;
    generativeqc_tensor::cuda_check(cudaGetDevice(&device_));
    version_ = cutensorGetVersion();
    if (version_ / 10000 != 2 || version_ < 20800) {
      rejection_ = "native provider requires cuTENSOR 2.8 or later in 2.x";
      return false;
    }
    generativeqc_tensor::cuda_check(cudaRuntimeGetVersion(&runtime_version_));
    int major{}, minor{};
    generativeqc_tensor::cuda_check(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    generativeqc_tensor::cuda_check(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_));
    architecture_ = 10 * major + minor;
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    try {
      std::size_t before{}, after{}, total{};
      generativeqc_tensor::cuda_check(cudaMemGetInfo(&before, &total));
      cutensor_check(cutensorCreate(&handle_));
      cutensor_check(cutensorHandleResizePlanCache(handle_, 0));
      const bool fp64 = request.precision.storage_dtype == PrecisionDtype::Fp64;
      const auto dtype = fp64 ? CUDA_R_64F : CUDA_R_32F;
      for (std::size_t operand = 0; operand < 3; ++operand) {
        const auto& view = request.operands[operand];
        std::array<std::int64_t, ContractionOperand::kMaximumRank> extents{}, strides{};
        for (std::size_t axis = 0; axis < view.rank; ++axis) {
          extents[axis] = view.shape[axis];
          strides[axis] = view.strides[axis];
        }
        cutensor_check(cutensorCreateTensorDescriptor(handle_, &tensors_[operand], view.rank,
                                                      extents.data(), strides.data(), dtype,
                                                      fp64 ? sizeof(double) : sizeof(float)));
      }
      cutensor_check(cutensorCreateContraction(
          handle_, &operation_, tensors_[0], request.operands[0].modes.data(), CUTENSOR_OP_IDENTITY,
          tensors_[1], request.operands[1].modes.data(), CUTENSOR_OP_IDENTITY, tensors_[2],
          request.operands[2].modes.data(), CUTENSOR_OP_IDENTITY, tensors_[2],
          request.operands[2].modes.data(),
          fp64 ? CUTENSOR_COMPUTE_DESC_64F : CUTENSOR_COMPUTE_DESC_32F));
      cudaDataType_t scalar_type{};
      cutensor_check(cutensorOperationDescriptorGetAttribute(
          handle_, operation_, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, &scalar_type,
          sizeof(scalar_type)));
      if (scalar_type != dtype) throw std::logic_error("cuTENSOR changed requested scalar dtype");
      // DEFAULT hides the chosen algorithm/kernel and 2.8 exposes no plan query
      // for either. A fixed family and rank give reproducible provider provenance.
      // Unsupported shapes reject preparation and retain the caller's fallback.
      cutensor_check(cutensorCreatePlanPreference(handle_, &preference_, CUTENSOR_ALGO_GETT,
                                                  CUTENSOR_JIT_MODE_NONE));
      const std::int32_t rank = 0;
      cutensor_check(cutensorPlanPreferenceSetAttribute(
          handle_, preference_, CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK, &rank, sizeof(rank)));
      const auto autotune = CUTENSOR_AUTOTUNE_MODE_NONE;
      const auto cache = CUTENSOR_CACHE_MODE_NONE;
      cutensor_check(cutensorPlanPreferenceSetAttribute(handle_, preference_,
                                                        CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
                                                        &autotune, sizeof(autotune)));
      cutensor_check(cutensorPlanPreferenceSetAttribute(
          handle_, preference_, CUTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cache, sizeof(cache)));
      cutensor_check(cutensorCreatePlan(handle_, &plan_, operation_, preference_, workspace_limit));
      cutensor_check(cutensorPlanPreferenceGetAttribute(
          handle_, preference_, CUTENSOR_PLAN_PREFERENCE_ALGO, &algorithm_, sizeof(algorithm_)));
      cutensor_check(cutensorPlanPreferenceGetAttribute(handle_, preference_,
                                                        CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK,
                                                        &kernel_rank_, sizeof(kernel_rank_)));
      if (algorithm_ != CUTENSOR_ALGO_GETT || kernel_rank_ != rank)
        throw std::logic_error("cuTENSOR changed the fixed algorithm/kernel contract");
      std::uint64_t workspace{};
      cutensor_check(cutensorPlanGetAttribute(handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                              &workspace, sizeof(workspace)));
      if (workspace > workspace_limit) throw std::logic_error("cuTENSOR exceeded workspace limit");
      workspace_bytes_ = workspace;
      if (workspace_bytes_)
        generativeqc_tensor::cuda_check(cudaMalloc(&workspace_, workspace_bytes_));
      generativeqc_tensor::cuda_check(cudaMemGetInfo(&after, &total));
      const auto retained = before > after ? before - after : 0;
      provider_bytes_ = retained > workspace_bytes_ ? retained - workspace_bytes_ : 0;
      if (provider_bytes_ > provider_limit) {
        rejection_ = "cuTENSOR retained device storage exceeds reservation";
        clear_locked(true);
        return false;
      }
      host_bytes_ = host_reservation;
      prepared_ = true;
      prepare_seconds_ =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return true;
    } catch (const CutensorError& error) {
      clear_locked(true);
      if (error.status != CUTENSOR_STATUS_NOT_SUPPORTED &&
          error.status != CUTENSOR_STATUS_ALLOC_FAILED &&
          error.status != CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
        throw;
      if (error.status == CUTENSOR_STATUS_ALLOC_FAILED) (void)cudaGetLastError();
      rejection_ = "cuTENSOR has no supported plan within resources";
      return false;
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      clear_locked(true);
      (void)cudaGetLastError();
      rejection_ = "cuTENSOR workspace allocation failed";
      return false;
    } catch (...) {
      clear_locked(false);
      throw;
    }
  }

  template <class T>
  void execute(cudaStream_t stream, const T* a, const T* b, T* output, int* error) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if (!prepared_ || stream != stream_) throw std::logic_error("stale cuTENSOR binding or stream");
    require_uncaptured(stream);
    int current{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&current));
    if (current != device_) throw std::logic_error("cuTENSOR binding device changed");
    if ((std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32) !=
        request_.publication_dtype)
      throw std::invalid_argument("cuTENSOR pointer dtype differs from the prepared request");
    if (!a || !b || !output || !error)
      throw std::invalid_argument("null cuTENSOR execution pointer");
    const auto overlaps = [](const void* a, std::size_t as, const void* b, std::size_t bs) {
      const auto left = reinterpret_cast<std::uintptr_t>(a),
                 right = reinterpret_cast<std::uintptr_t>(b);
      return left <= right ? right - left < as : left - right < bs;
    };
    for (std::size_t operand = 0; operand < 2; ++operand)
      if (overlaps(operand == 0 ? a : b, request_.operands[operand].storage_elements() * sizeof(T),
                   output, request_.operands[2].storage_elements() * sizeof(T)))
        throw std::invalid_argument("cuTENSOR output aliases an input");
    for (const auto* pointer : {a, b, static_cast<const T*>(output)})
      if (reinterpret_cast<std::uintptr_t>(pointer) % sizeof(T))
        throw std::invalid_argument("cuTENSOR pointer alignment differs from prepared layout");
    const T alpha = static_cast<T>(request_.coefficient), beta = static_cast<T>(request_.beta);
    if (!std::isfinite(alpha) || !std::isfinite(beta))
      throw std::invalid_argument("cuTENSOR scalar overflow");
    // Only preparation can choose a fallback. Any execution error propagates.
    cutensor_check(cutensorContract(handle_, plan_, &alpha, a, b, &beta, output, output, workspace_,
                                    workspace_bytes_, stream));
    const auto count = request_.affine_output_elements();
    audit_affine_contraction<<<generativeqc_tensor::blocks(count, 256), 256, 0, stream>>>(
        output, audit_, count, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    if (calls_ == std::numeric_limits<std::size_t>::max())
      throw std::length_error("cuTENSOR call counter overflow");
    ++calls_;
  }

  void reset() noexcept {
    if (!handle_ && !workspace_) return;
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    int previous = device_;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device_);
    clear_locked(false);
    (void)cudaSetDevice(previous);
  }
  /** Checked live release for transactional preparation/fallback. The owner
   * may retry another provider only after all pending work and cleanup succeed.
   * Destructors instead use best-effort reset(). */
  void release() {
    if (!handle_ && !workspace_) return;
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    int previous{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&previous));
    generativeqc_tensor::cuda_check(cudaSetDevice(device_));
    try {
      clear_locked(true);
    } catch (...) {
      (void)cudaSetDevice(previous);
      throw;
    }
    generativeqc_tensor::cuda_check(cudaSetDevice(previous));
  }
  std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
  std::size_t provider_bytes() const noexcept { return provider_bytes_; }
  std::size_t host_bytes() const noexcept { return host_bytes_; }
  std::size_t provider_version() const noexcept { return version_; }
  int runtime_version() const noexcept { return runtime_version_; }
  std::size_t prepare_calls() const noexcept { return prepare_calls_; }
  std::size_t calls() const noexcept { return calls_; }
  double prepare_seconds() const noexcept { return prepare_seconds_; }
  std::string_view rejection() const noexcept { return rejection_; }
  Provenance provenance() const {
    if (!prepared_) throw std::logic_error("cuTENSOR plan provenance requires preparation");
    return {request_,      version_,   workspace_bytes_, runtime_version_,
            architecture_, algorithm_, kernel_rank_};
  }

 private:
  static void require_uncaptured(cudaStream_t stream) {
    cudaStreamCaptureStatus status{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone)
      throw std::logic_error("cuTENSOR binding capture is not qualified");
  }
  // Caller holds the allocation mutex and owning device. Live fallback must
  // propagate cleanup/driver failures; destructor cleanup is best effort.
  void clear_locked(bool checked) {
    cudaError_t cuda_status = cudaSuccess;
    cutensorStatus_t status = CUTENSOR_STATUS_SUCCESS;
    auto retain = [&](auto result) {
      if (status == CUTENSOR_STATUS_SUCCESS) status = result;
    };
    // nullptr is the valid default CUDA stream and must be drained too.
    if (handle_) cuda_status = cudaStreamSynchronize(stream_);
    if (plan_) retain(cutensorDestroyPlan(plan_));
    if (preference_) retain(cutensorDestroyPlanPreference(preference_));
    if (operation_) retain(cutensorDestroyOperationDescriptor(operation_));
    for (auto tensor : tensors_)
      if (tensor) retain(cutensorDestroyTensorDescriptor(tensor));
    if (handle_) retain(cutensorDestroy(handle_));
    if (workspace_) {
      const auto result = cudaFree(workspace_);
      if (cuda_status == cudaSuccess) cuda_status = result;
    }
    handle_ = nullptr;
    plan_ = nullptr;
    preference_ = nullptr;
    operation_ = nullptr;
    tensors_ = {};
    workspace_ = nullptr;
    stream_ = nullptr;
    prepared_ = false;
    workspace_bytes_ = provider_bytes_ = host_bytes_ = 0;
    if (checked) {
      generativeqc_tensor::cuda_check(cuda_status);
      cutensor_check(status);
    }
  }
  ContractionRequest request_;
  AffineAuditView audit_;
  cudaStream_t stream_{};
  int device_{}, runtime_version_{}, architecture_{};
  cutensorAlgo_t algorithm_{CUTENSOR_ALGO_GETT};
  std::int32_t kernel_rank_{};
  cutensorHandle_t handle_{};
  std::array<cutensorTensorDescriptor_t, 3> tensors_{};
  cutensorOperationDescriptor_t operation_{};
  cutensorPlanPreference_t preference_{};
  cutensorPlan_t plan_{};
  void* workspace_{};
  std::size_t workspace_bytes_{}, provider_bytes_{}, host_bytes_{}, version_{}, prepare_calls_{},
      calls_{};
  double prepare_seconds_{};
  bool prepared_{};
  std::string_view rejection_;
};

}  // namespace generativeqc::tensor
