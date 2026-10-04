#pragma once

#include <cublasLt.h>

#include <chrono>
#include <type_traits>

#include "tensor/cuda_affine_audit.cuh"
#include "tensor/native_cublaslt.hpp"

namespace generativeqc::tensor {

class CublasLtError : public std::runtime_error {
 public:
  explicit CublasLtError(cublasStatus_t status)
      : std::runtime_error("cuBLASLt operation failed: " + std::to_string(int(status))),
        status(status) {}
  cublasStatus_t status;
};
inline void cublaslt_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw CublasLtError(status);
}

/** One prepared matmul; discovery is bounded and never repeated by execute.
 * Borrows the stream and tensor addresses, owns descriptors/algorithm/workspace.
 * Opaque host/global heuristic-cache state needs an external qualified host
 * reservation. We query cache capacity but never mutate process-global policy.
 * This owner is not thread-safe and must be released before its borrowed stream.
 */
class CudaCublasLtContraction {
 public:
  static constexpr int kMaximumHeuristics = 8;
  struct Algorithm {
    std::int32_t id{}, split_k{};
    std::uint32_t tile{}, reduction{}, swizzle{}, custom{}, stages{};
    std::uint16_t inner_shape{}, cluster_shape{};
    bool operator==(const Algorithm&) const = default;
  };
  /** Diagnostic identity is meaningful together with request/version/target;
   * neither a bare algorithm ID nor an opaque algorithm blob is a cache key. */
  struct Provenance {
    ContractionRequest request;
    Algorithm algorithm;
    std::size_t provider_version{}, workspace_bytes{}, heuristic_cache_capacity{};
    int runtime_version{}, architecture{};
  };

  CudaCublasLtContraction() = default;
  CudaCublasLtContraction(const CudaCublasLtContraction&) = delete;
  CudaCublasLtContraction& operator=(const CudaCublasLtContraction&) = delete;
  ~CudaCublasLtContraction() { reset(); }

  bool prepare(const ContractionRequest& request, cudaStream_t stream, std::size_t workspace_limit,
               std::size_t provider_limit, std::size_t host_reservation) {
    if (handle_) throw std::logic_error("cuBLASLt binding is already prepared");
    const auto recipe = CublasLtMatrixRecipe::from(request);
    require_uncaptured(stream);
    rejection_ = {};
    // Include the bounded temporary heuristic array, simultaneous with the owner.
    if (host_reservation <
        sizeof(*this) + kMaximumHeuristics * sizeof(cublasLtMatmulHeuristicResult_t)) {
      rejection_ = "cuBLASLt host metadata reservation is insufficient";
      return false;
    }
    const auto started = std::chrono::steady_clock::now();
    request_ = request;
    stream_ = stream;
    audit_.rank = request.operands[2].rank;
    for (std::size_t axis = 0; axis < audit_.rank; ++axis) {
      audit_.shape[axis] = request.operands[2].shape[axis];
      audit_.strides[axis] = request.operands[2].strides[axis];
    }
    generativeqc_tensor::cuda_check(cudaGetDevice(&device_));
    generativeqc_tensor::cuda_check(cudaRuntimeGetVersion(&runtime_version_));
    int major{}, minor{};
    generativeqc_tensor::cuda_check(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    generativeqc_tensor::cuda_check(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_));
    architecture_ = 10 * major + minor;
    version_ = cublasLtGetVersion();
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    try {
      std::size_t before{}, after{}, total{};
      generativeqc_tensor::cuda_check(cudaMemGetInfo(&before, &total));
      cublaslt_check(cublasLtCreate(&handle_));
      cublaslt_check(cublasLtHeuristicsCacheGetCapacity(&cache_capacity_));
      const bool fp64 = request.publication_dtype == PrecisionDtype::Fp64;
      const auto dtype = fp64 ? CUDA_R_64F : CUDA_R_32F;
      const auto compute = fp64 ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_32F_PEDANTIC;
      cublaslt_check(cublasLtMatmulDescCreate(&operation_, compute, dtype));
      const cublasOperation_t no_transpose = CUBLAS_OP_N;
      cublaslt_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &no_transpose, sizeof(no_transpose)));
      cublaslt_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &no_transpose, sizeof(no_transpose)));
      const auto epilogue = CUBLASLT_EPILOGUE_DEFAULT;
      cublaslt_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue, sizeof(epilogue)));
      for (std::size_t i = 0; i < 3; ++i) {
        const auto& layout = recipe.layouts[i];
        cublaslt_check(cublasLtMatrixLayoutCreate(&layouts_[i], dtype, layout.rows, layout.columns,
                                                  layout.ld));
        const auto order = layout.row_major ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL;
        cublaslt_check(cublasLtMatrixLayoutSetAttribute(layouts_[i], CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &order, sizeof(order)));
        const auto batches = std::int32_t(recipe.batches);
        const auto stride = std::int64_t(layout.batch_stride);
        cublaslt_check(cublasLtMatrixLayoutSetAttribute(
            layouts_[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
        cublaslt_check(cublasLtMatrixLayoutSetAttribute(
            layouts_[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
      }
      cublaslt_check(cublasLtMatmulPreferenceCreate(&preference_));
      cublaslt_check(cublasLtMatmulPreferenceSetAttribute(
          preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
          sizeof(workspace_limit)));
      const std::uint32_t alignment = fp64 ? sizeof(double) : sizeof(float);
      for (const auto attr :
           {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        cublaslt_check(
            cublasLtMatmulPreferenceSetAttribute(preference_, attr, &alignment, sizeof(alignment)));
      std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> results{};
      int count{};
      ++heuristic_calls_;
      cublaslt_check(cublasLtMatmulAlgoGetHeuristic(handle_, operation_, layouts_[0], layouts_[1],
                                                    layouts_[2], layouts_[2], preference_,
                                                    results.size(), results.data(), &count));
      if (count < 0 || count > kMaximumHeuristics)
        throw std::logic_error("cuBLASLt heuristic count exceeds bound");
      bool found = false;
      for (int i = 0; i < count; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > workspace_limit)
          continue;
        cublasLtMatmulHeuristicResult_t checked{};
        const auto status =
            cublasLtMatmulAlgoCheck(handle_, operation_, layouts_[0], layouts_[1], layouts_[2],
                                    layouts_[2], &results[i].algo, &checked);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) continue;
        cublaslt_check(status);
        if (checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > workspace_limit)
          continue;
        algorithm_ = results[i].algo;
        workspace_bytes_ = checked.workspaceSize;
        found = true;
        break;
      }
      if (!found) {
        rejection_ = "cuBLASLt has no bounded heuristic for this request";
        clear_locked(true);
        return false;
      }
      algorithm_facts_ = read_algorithm();
      if (workspace_bytes_)
        generativeqc_tensor::cuda_check(cudaMalloc(&workspace_, workspace_bytes_));
      generativeqc_tensor::cuda_check(cudaMemGetInfo(&after, &total));
      const auto retained = before > after ? before - after : 0;
      provider_bytes_ = retained > workspace_bytes_ ? retained - workspace_bytes_ : 0;
      if (provider_bytes_ > provider_limit) {
        rejection_ = "cuBLASLt retained device storage exceeds reservation";
        clear_locked(true);
        return false;
      }
      host_bytes_ = host_reservation;
      prepared_ = true;
      prepare_seconds_ =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return true;
    } catch (const CublasLtError& error) {
      clear_locked(true);
      if (error.status != CUBLAS_STATUS_NOT_SUPPORTED && error.status != CUBLAS_STATUS_ALLOC_FAILED)
        throw;
      if (error.status == CUBLAS_STATUS_ALLOC_FAILED) (void)cudaGetLastError();
      rejection_ = "cuBLASLt preparation unavailable within resources";
      return false;
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      clear_locked(true);
      (void)cudaGetLastError();
      rejection_ = "cuBLASLt workspace allocation failed";
      return false;
    } catch (...) {
      clear_locked(false);
      throw;
    }
  }

  template <class T>
  void execute(cudaStream_t stream, const T* a, const T* b, T* output, int* error) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    if (!prepared_ || stream != stream_) throw std::logic_error("stale cuBLASLt binding or stream");
    require_uncaptured(stream);
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != device_) throw std::logic_error("cuBLASLt binding device changed");
    if ((std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32) !=
        request_.publication_dtype)
      throw std::invalid_argument("cuBLASLt pointer dtype differs from request");
    if (!a || !b || !output || !error)
      throw std::invalid_argument("null cuBLASLt execution pointer");
    for (const auto* pointer : {a, b, static_cast<const T*>(output)})
      if (reinterpret_cast<std::uintptr_t>(pointer) % sizeof(T))
        throw std::invalid_argument("cuBLASLt pointer alignment differs from preparation");
    const auto overlaps = [](const void* x, std::size_t xs, const void* y, std::size_t ys) {
      const auto left = reinterpret_cast<std::uintptr_t>(x),
                 right = reinterpret_cast<std::uintptr_t>(y);
      return left <= right ? right - left < xs : left - right < ys;
    };
    for (std::size_t i = 0; i < 2; ++i)
      if (overlaps(i == 0 ? a : b, request_.operands[i].storage_elements() * sizeof(T), output,
                   request_.operands[2].storage_elements() * sizeof(T)))
        throw std::invalid_argument("cuBLASLt output aliases an input");
    const T alpha = T(request_.coefficient), beta = T(request_.beta);
    if (!std::isfinite(alpha) || !std::isfinite(beta))
      throw std::invalid_argument("cuBLASLt scalar overflow");
    if (calls_ == std::numeric_limits<std::size_t>::max())
      throw std::length_error("cuBLASLt call counter overflow");
    // Explicit algorithm prevents implicit heuristic lookup. No fallback is
    // legal here: execution may already have published part of the output.
    cublaslt_check(cublasLtMatmul(handle_, operation_, &alpha, a, layouts_[0], b, layouts_[1],
                                  &beta, output, layouts_[2], output, layouts_[2], &algorithm_,
                                  workspace_, workspace_bytes_, stream));
    const auto count = request_.affine_output_elements();
    audit_affine_contraction<<<generativeqc_tensor::blocks(count, 256), 256, 0, stream>>>(
        output, audit_, count, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++calls_;
  }

  Provenance provenance() const {
    if (!prepared_) throw std::logic_error("cuBLASLt provenance requires a prepared binding");
    return {request_,        algorithm_facts_, version_,     workspace_bytes_,
            cache_capacity_, runtime_version_, architecture_};
  }
  std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
  std::size_t provider_bytes() const noexcept { return provider_bytes_; }
  std::size_t host_bytes() const noexcept { return host_bytes_; }
  std::size_t heuristic_calls() const noexcept { return heuristic_calls_; }
  std::size_t calls() const noexcept { return calls_; }
  double prepare_seconds() const noexcept { return prepare_seconds_; }
  std::string_view rejection() const noexcept { return rejection_; }

  void release() {
    if (!handle_) return;
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
  void reset() noexcept {
    if (!handle_) return;
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    int previous = device_;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device_);
    clear_locked(false);
    (void)cudaSetDevice(previous);
  }

 private:
  static void require_uncaptured(cudaStream_t stream) {
    cudaStreamCaptureStatus status{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone)
      throw std::logic_error("cuBLASLt capture is not qualified");
  }
  Algorithm read_algorithm() const {
    Algorithm out;
    const auto read = [&](auto attribute, auto& value) {
      std::size_t written{};
      cublaslt_check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm_, attribute, &value,
                                                          sizeof(value), &written));
      if (written != sizeof(value))
        throw std::logic_error("cuBLASLt algorithm attribute size changed");
    };
    read(CUBLASLT_ALGO_CONFIG_ID, out.id);
    read(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, out.split_k);
    read(CUBLASLT_ALGO_CONFIG_TILE_ID, out.tile);
    read(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, out.reduction);
    read(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, out.swizzle);
    read(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, out.custom);
    read(CUBLASLT_ALGO_CONFIG_STAGES_ID, out.stages);
    read(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, out.inner_shape);
    read(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, out.cluster_shape);
    return out;
  }
  // Caller holds the allocation mutex and owning device. Preserve cleanup
  // failures for live re-selection; destructors remain best effort.
  void clear_locked(bool checked) {
    auto cuda_status = handle_ ? cudaStreamSynchronize(stream_) : cudaSuccess;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    const auto retain = [&](auto value) {
      if (status == CUBLAS_STATUS_SUCCESS) status = value;
    };
    if (preference_) retain(cublasLtMatmulPreferenceDestroy(preference_));
    if (operation_) retain(cublasLtMatmulDescDestroy(operation_));
    for (auto layout : layouts_)
      if (layout) retain(cublasLtMatrixLayoutDestroy(layout));
    if (handle_) retain(cublasLtDestroy(handle_));
    if (workspace_) {
      const auto freed = cudaFree(workspace_);
      if (cuda_status == cudaSuccess) cuda_status = freed;
    }
    handle_ = nullptr;
    preference_ = nullptr;
    operation_ = nullptr;
    layouts_ = {};
    workspace_ = nullptr;
    stream_ = nullptr;
    prepared_ = false;
    workspace_bytes_ = provider_bytes_ = host_bytes_ = 0;
    if (checked) {
      generativeqc_tensor::cuda_check(cuda_status);
      cublaslt_check(status);
    }
  }
  ContractionRequest request_;
  AffineAuditView audit_;
  Algorithm algorithm_facts_;
  cublasLtMatmulAlgo_t algorithm_{};
  cublasLtHandle_t handle_{};
  cublasLtMatmulDesc_t operation_{};
  cublasLtMatmulPreference_t preference_{};
  std::array<cublasLtMatrixLayout_t, 3> layouts_{};
  cudaStream_t stream_{};
  void* workspace_{};
  std::size_t workspace_bytes_{}, provider_bytes_{}, host_bytes_{}, cache_capacity_{}, version_{},
      heuristic_calls_{}, calls_{};
  int device_{}, runtime_version_{}, architecture_{};
  double prepare_seconds_{};
  bool prepared_{};
  std::string_view rejection_;
};

}  // namespace generativeqc::tensor
