#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensor/cuda_runtime.cuh"
#include "tensor/native_contraction.hpp"

#if GENERATIVEQC_HAS_CUTENSOR
#include "tensor/cuda_cutensor.cuh"
#endif
#if GENERATIVEQC_HAS_CUBLASLT
#include "tensor/cuda_cublaslt.cuh"
#endif
#if GENERATIVEQC_HAS_CUTLASS
#include "tensor/cuda_cutlass.cuh"
#endif

namespace generativeqc::tensor {

/** Shared native provider resources, prepared once outside iteration/capture.
 * The conservative retained allowance is charged by the enclosing arena owner;
 * observed handle growth is bounded as well. No implicit BLAS workspace is
 * allowed. Allocation failure alone admits the caller's scalar schedule.
 */
class CudaContractionContext {
 public:
  static constexpr std::size_t kProviderAllowance = 96ULL << 20;
  CudaContractionContext() = default;
  CudaContractionContext(const CudaContractionContext&) = delete;
  CudaContractionContext& operator=(const CudaContractionContext&) = delete;
  ~CudaContractionContext() { reset(); }

  bool prepare(cudaStream_t stream) {
    if (handle_ || prepared_) throw std::logic_error("contraction context is already prepared");
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("contraction preparation is forbidden during capture");
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    try {
      generativeqc_tensor::cuda_check(cudaGetDevice(&device_));
      std::size_t before{}, after{}, total{};
      generativeqc_tensor::cuda_check(cudaMemGetInfo(&before, &total));
      const auto status = cublasCreate(&handle_);
      if (status == CUBLAS_STATUS_ALLOC_FAILED) {
        release_locked();
        (void)cudaGetLastError();
        return false;
      }
      generativeqc_tensor::blas_check(status);
      stream_ = stream;
      generativeqc_tensor::blas_check(cublasSetStream(handle_, stream_));
      generativeqc_tensor::blas_check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
      generativeqc_tensor::blas_check(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
      generativeqc_tensor::blas_check(cublasSetWorkspace(handle_, nullptr, 0));
      generativeqc_tensor::blas_check(cublasGetVersion(handle_, &provider_version_));
      generativeqc_tensor::cuda_check(cudaRuntimeGetVersion(&runtime_version_));
      generativeqc_tensor::cuda_check(cudaMemGetInfo(&after, &total));
      retained_bytes_ = before > after ? before - after : 0;
      if (before > after && before - after > kProviderAllowance) {
        release_locked();
        return false;
      }
      ++generation_;
      prepared_ = true;
      return true;
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      release_locked();
      (void)cudaGetLastError();
      return false;
    } catch (...) {
      reset_locked();
      throw;
    }
  }

  /** Bind the complete generated fallback on the same caller-owned stream.
   * Used only during preparation after an optional provider is unavailable. */
  void prepare_generated(cudaStream_t stream) {
    if (handle_ || prepared_) throw std::logic_error("contraction context already prepared");
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("generated contraction preparation during capture");
    generativeqc_tensor::cuda_check(cudaGetDevice(&device_));
    stream_ = stream;
    provider_version_ = 0;
    generativeqc_tensor::cuda_check(cudaRuntimeGetVersion(&runtime_version_));
    retained_bytes_ = 0;
    prepared_ = true;
    ++generation_;
  }

  void reset() noexcept {
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    reset_locked();
  }

  // Optional-resource fallback is a live operation, unlike destructor cleanup:
  // a driver or destroy failure must propagate instead of admitting a retry.
  // The caller holds allocation_measurement_mutex and the prepared device.
  void release_locked() {
    if (handle_) {
      if (stream_) generativeqc_tensor::cuda_check(cudaStreamSynchronize(stream_));
      generativeqc_tensor::blas_check(cublasDestroy(handle_));
    }
    handle_ = nullptr;
    stream_ = nullptr;
    prepared_ = false;
    retained_bytes_ = 0;
    ++generation_;
  }

  // The arena owner already holds allocation_measurement_mutex when releasing
  // all retained storage together. This avoids recursively taking that lock.
  void reset_locked() noexcept {
    if (handle_) {
      int previous = device_;
      (void)cudaGetDevice(&previous);
      (void)cudaSetDevice(device_);
      if (stream_) (void)cudaStreamSynchronize(stream_);
      (void)cublasDestroy(handle_);
      (void)cudaSetDevice(previous);
    }
    handle_ = nullptr;
    stream_ = nullptr;
    prepared_ = false;
    retained_bytes_ = 0;
    ++generation_;
  }

  cublasHandle_t handle() const noexcept { return handle_; }
  cudaStream_t stream() const noexcept { return stream_; }
  int device() const noexcept { return device_; }
  int provider_version() const noexcept { return provider_version_; }
  int runtime_version() const noexcept { return runtime_version_; }
  std::size_t generation() const noexcept { return generation_; }
  std::size_t retained_bytes() const noexcept { return retained_bytes_; }
  bool prepared() const noexcept { return prepared_; }

 private:
  cublasHandle_t handle_{};
  cudaStream_t stream_{};
  int device_{}, provider_version_{}, runtime_version_{};
  std::size_t generation_{};
  std::size_t retained_bytes_{};
  bool prepared_{};
};

template <class T>
static __global__ void audit_contraction(const T* values, std::size_t count, int* error,
                                         std::size_t columns = 0,
                                         std::size_t leading_dimension = 0) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    if (!isfinite(values[columns ? (i / columns) * leading_dimension + i % columns : i]))
      atomicCAS(error, 0, 1);
}

/** Backend implementation identities, supplied only by compiler/provider
 * preparation. Scientific request metadata never includes this choice. */
enum class ContractionAlgorithm : std::uint8_t {
  PedanticBlas,
  GeneratedOrdered,
  CutensorAffine,
  CublasLtMatmul,
  CutlassAot
};

/** Optional-provider rejection is distinct from malformed science or execution
 * failures. Only this exception authorizes preparation of an admitted fallback. */
class ContractionPreparationUnavailable : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/** Per-plan ceilings, supplied by provider qualification and admitted by the
 * enclosing owner before preparation. Zero defaults intentionally admit no
 * opaque optional-provider host storage. There is no production reservation heuristic.
 * host_bytes includes the native plan object and opaque provider host storage;
 * descriptor/pointer tables are charged separately by storage_bytes().
 * cache_bytes covers context-retained AOT modules and is not reclaimed when
 * an optional descriptor or an unpublished batch is released. */
struct ContractionProviderReservation {
  std::size_t workspace_bytes{}, provider_bytes{}, host_bytes{}, cache_bytes{};

  std::size_t total_bytes(std::size_t plans) const {
    auto result = checked_add(checked_add(workspace_bytes, provider_bytes), cache_bytes);
    return contraction_product(checked_add(result, host_bytes), plans);
  }

  static std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::length_error("native contraction resource overflow");
    return a + b;
  }
};

#if defined(GENERATIVEQC_TEST_HOOKS)
// Provider-layer qualification controls, absent from production builds and
// method APIs. Negative means no injection; zero rejects the next preparation.
inline thread_local ContractionProviderReservation cutensor_reservation_for_test;
inline thread_local ContractionProviderReservation cublaslt_reservation_for_test;
inline thread_local int cutensor_preparations_before_rejection_for_test = -1;
inline thread_local int cublaslt_preparations_before_rejection_for_test = -1;
inline thread_local int cutlass_preparations_before_rejection_for_test = -1;
#endif

/** Resource evidence is independent of build availability. No production
 * cuTENSOR resource profile is qualified yet, including lazy execution storage.
 * Endpoint qualification may inject explicit test reservations without turning
 * synthetic limits into production defaults. */
inline ContractionProviderReservation qualified_cutensor_reservation() noexcept {
#if GENERATIVEQC_HAS_CUTENSOR && defined(GENERATIVEQC_TEST_HOOKS)
  return cutensor_reservation_for_test;
#else
  return {};
#endif
}

inline std::size_t cutensor_provider_version() noexcept {
#if GENERATIVEQC_HAS_CUTENSOR
  return cutensorGetVersion();
#else
  return 0;
#endif
}

/** cuBLASLt remains unavailable to production selection without measured
 * simultaneous host/cache/provider and lazy-execution resource bounds. */
inline ContractionProviderReservation qualified_cublaslt_reservation() noexcept {
#if GENERATIVEQC_HAS_CUBLASLT && defined(GENERATIVEQC_TEST_HOOKS)
  return cublaslt_reservation_for_test;
#else
  return {};
#endif
}

inline std::size_t cublaslt_provider_version() noexcept {
#if GENERATIVEQC_HAS_CUBLASLT
  return cublasLtGetVersion();
#else
  return 0;
#endif
}

template <class T>
__device__ T contraction_multiply(T a, T b) {
  if constexpr (std::is_same_v<T, double>)
    return __dmul_rn(a, b);
  else
    return __fmul_rn(a, b);
}
template <class T>
__device__ T contraction_add(T a, T b) {
  if constexpr (std::is_same_v<T, double>)
    return __dadd_rn(a, b);
  else
    return __fadd_rn(a, b);
}

/** Exact increasing-k matrix lowering of the validated semantic einsum.
 * Padding is never visited. beta==0 never reads output; every scalar multiply
 * and add has explicit RN semantics, including FP32 accumulation. */
template <class T>
static __global__ void generated_contraction(const T* a, const T* b, T* output, std::size_t count,
                                             std::size_t m, std::size_t n, std::size_t k,
                                             std::size_t lda, std::size_t ldb, std::size_t ldc,
                                             bool ta, bool tb, T alpha, T beta, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = i / (m * n), row = i / n % m, col = i % n;
    T value = 0;
    for (std::size_t reduction = 0; reduction < k; ++reduction) {
      const auto ai = batch * m * k + (ta ? reduction * lda + row : row * lda + reduction);
      const auto bi = batch * k * n + (tb ? col * ldb + reduction : reduction * ldb + col);
      value = contraction_add(value, contraction_multiply(a[ai], b[bi]));
    }
    const auto ci = batch * m * n + row * ldc + col;
    value = contraction_multiply(alpha, value);
    if (beta != T(0)) value = contraction_add(value, contraction_multiply(beta, output[ci]));
    if (!isfinite(value)) atomicCAS(error, 0, 1);
    output[ci] = value;
  }
}

/** Fused batch weighting preserves the incumbent increasing-k FMA chain.
 * Only the weighted publication is checked, as in the original fused consumer;
 * invalid publications become zero while the device error remains sticky. */
template <class T>
static __global__ void generated_batch_scaled_contraction(
    const T* left, const T* right, const T* scale, T* output, std::size_t count, std::size_t rows,
    std::size_t columns, std::size_t reduction_extent, std::size_t left_stride,
    std::size_t right_stride, std::size_t output_stride, bool transpose_left, bool transpose_right,
    T coefficient, int* error) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = index / (rows * columns), row = index / columns % rows,
               column = index % columns;
    T value = 0;
    for (std::size_t reduction = 0; reduction < reduction_extent; ++reduction) {
      const auto left_index =
          batch * rows * reduction_extent +
          (transpose_left ? reduction * left_stride + row : row * left_stride + reduction);
      const auto right_index =
          batch * reduction_extent * columns +
          (transpose_right ? column * right_stride + reduction : reduction * right_stride + column);
      if constexpr (std::is_same_v<T, double>)
        value = __fma_rn(left[left_index], right[right_index], value);
      else
        value = __fmaf_rn(left[left_index], right[right_index], value);
    }
    value = contraction_multiply(scale[batch], contraction_multiply(coefficient, value));
    if (!isfinite(value)) {
      atomicCAS(error, 0, 1);
      value = 0;
    }
    output[batch * rows * columns + row * output_stride + column] = value;
  }
}

/** Library output is an unobservable intermediate in the publication buffer.
 * Combine scaling and finite checks in one pass, with no extra numeric cache. */
template <class T>
static __global__ void publish_batch_scaled_contraction(T* output, const T* scale,
                                                        std::size_t count, std::size_t rows,
                                                        std::size_t columns,
                                                        std::size_t output_stride, int* error) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = index / (rows * columns), row = index / columns % rows,
               column = index % columns;
    const auto address = batch * rows * columns + row * output_stride + column;
    T value = contraction_multiply(scale[batch], output[address]);
    if (!isfinite(value)) {
      atomicCAS(error, 0, 1);
      value = 0;
    }
    output[address] = value;
  }
}

/** Execute the compiler's existing scalar helper at every increasing-k step.
 * The helper owns rounding and input/update checks. Invalid intermediate work
 * cannot be hidden by later cancellation or a zero publication weight. */
template <class Step>
static __global__ void generated_checked_contraction(
    const double* left, const double* right, const double* scale, double* output, std::size_t count,
    std::size_t rows, std::size_t columns, std::size_t reduction_extent, std::size_t left_stride,
    std::size_t right_stride, std::size_t output_stride, bool transpose_left, bool transpose_right,
    int* error) {
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const auto batch = index / (rows * columns), row = index / columns % rows,
               column = index % columns;
    double value = 0;
    bool valid = true;
    for (std::size_t reduction = 0; reduction < reduction_extent; ++reduction) {
      const auto ai =
          batch * rows * reduction_extent +
          (transpose_left ? reduction * left_stride + row : row * left_stride + reduction);
      const auto bi =
          batch * reduction_extent * columns +
          (transpose_right ? column * right_stride + reduction : reduction * right_stride + column);
      if (!Step::update(left[ai], right[bi], value)) {
        valid = false;
        break;
      }
    }
    if (valid && scale) {
      double published = 0;
      valid = Step::publish(scale[batch], value, published);
      value = published;
    }
    if (!valid) {
      atomicCAS(error, 0, 1);
      value = 0;
    }
    output[batch * rows * columns + row * output_stride + column] = value;
  }
}

/** Prepared projection of the canonical compiler requests for one AOT stage.
 * Each stage has at most a full batch and a tail batch. All descriptor storage,
 * validation and provider setup occur before execution; replay only selects an
 * already prepared shape and changes borrowed tensor addresses. The enclosing
 * native owner keeps both the table and its borrowed context alive together.
 */
class PreparedContractions {
 public:
  // Graph launches need their own replay work accounting. Until that owner is
  // connected, reject capture instead of counting only the capture enqueue.
  static constexpr bool supports_capture = false;
  // Include owner records, bounded variant storage and a conservative second
  // descriptor copy live during construction. No cache grows during replay.
  static constexpr std::size_t storage_bytes(std::size_t requests, std::size_t variants = 1) {
    return sizeof(PreparedContractions) + 2 * sizeof(Variant) +
           2 * variants * requests *
               (sizeof(ContractionRequest) + sizeof(ContractionAlgorithm) +
                sizeof(ContractionOperand)
#if GENERATIVEQC_HAS_CUTENSOR
                + sizeof(std::unique_ptr<CudaCutensorContraction>)
#endif
#if GENERATIVEQC_HAS_CUBLASLT
                + sizeof(std::unique_ptr<CudaCublasLtContraction>)
#endif
#if GENERATIVEQC_HAS_CUTLASS
                + sizeof(std::unique_ptr<CudaCutlassContraction>)
#endif
               );
  }

  /** Bind one shape transactionally. Optional provider plans are fully prepared
   * before publishing the variant; rejection releases all provisional plans and
   * leaves executable variants unchanged. Context-retained module charges survive
   * rejection and release, and must remain in the enclosing owner's budget.
   * A hard retaining-load failure quarantines this table, including its older
   * variants. Neither release nor a fresh plan can readmit an unknown overrun.
   * The caller owns fallback selection and budgets
   * reservation.total_bytes(number_of_optional_plans) in addition to the table.
   * A mixed table must supply ceilings valid for every selected provider.
   * Pure affine requests need no matrix recipe when that provider is selected.
   * CUTLASS requires the actual AOT artifact digest as execution provenance. */
  void add(std::size_t o, std::size_t v, std::size_t q, std::vector<ContractionRequest> requests,
           CudaContractionContext& context, std::size_t& calls, std::size_t& summands,
           std::vector<ContractionAlgorithm> algorithms = {},
           ContractionProviderReservation reservation = {}, std::string_view artifact = {},
           std::vector<ContractionOperand> batch_scales = {}) {
    require_safe_retained_cache();
    if (variants_.size() == 2) throw std::length_error("native contraction batch variant bound");
    if (!context.prepared()) throw std::logic_error("native contraction provider is not prepared");
    if (algorithms.empty()) algorithms.assign(requests.size(), ContractionAlgorithm::PedanticBlas);
    if (algorithms.size() != requests.size() || algorithms.capacity() > algorithms.size())
      throw std::invalid_argument("native contraction algorithm table bound");
    if (batch_scales.empty()) batch_scales.resize(requests.size());
    if (batch_scales.size() != requests.size() || batch_scales.capacity() > batch_scales.size())
      throw std::invalid_argument("native contraction batch-scale table bound");
    for (std::size_t slot = 0; slot < requests.size(); ++slot) {
      requests[slot].validate_batch_scale(batch_scales[slot]);
      if (!requests[slot].checked_update_identity.empty() &&
          algorithms[slot] != ContractionAlgorithm::GeneratedOrdered)
        throw ContractionPreparationUnavailable(
            "provider does not implement ordered scalar checks");
      if (batch_scales[slot].rank && algorithms[slot] != ContractionAlgorithm::GeneratedOrdered &&
          algorithms[slot] != ContractionAlgorithm::PedanticBlas)
        throw ContractionPreparationUnavailable("provider does not implement weighted publication");
    }
    for (auto algorithm : algorithms) {
      if ((algorithm != ContractionAlgorithm::PedanticBlas &&
           algorithm != ContractionAlgorithm::GeneratedOrdered &&
           algorithm != ContractionAlgorithm::CutensorAffine &&
           algorithm != ContractionAlgorithm::CublasLtMatmul &&
           algorithm != ContractionAlgorithm::CutlassAot) ||
          (algorithm == ContractionAlgorithm::PedanticBlas && !context.handle()))
        throw std::invalid_argument("native contraction provider unavailable");
#if !GENERATIVEQC_HAS_CUTENSOR
      if (algorithm == ContractionAlgorithm::CutensorAffine)
        throw ContractionPreparationUnavailable("cuTENSOR was not enabled in this build");
#endif
#if !GENERATIVEQC_HAS_CUTLASS
      if (algorithm == ContractionAlgorithm::CutlassAot)
        throw ContractionPreparationUnavailable("CUTLASS was not enabled in this build");
#endif
#if !GENERATIVEQC_HAS_CUBLASLT
      if (algorithm == ContractionAlgorithm::CublasLtMatmul)
        throw ContractionPreparationUnavailable("cuBLASLt was not enabled in this build");
#endif
    }
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(context.stream(), &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("native descriptor preparation is forbidden during capture");
    for (const auto& variant : variants_)
      if (variant.o == o && variant.v == v && variant.q == q)
        throw std::invalid_argument("duplicate native contraction batch variant");
    for (std::size_t i = 0; i < requests.size(); ++i) {
      if (algorithms[i] == ContractionAlgorithm::CutensorAffine ||
          algorithms[i] == ContractionAlgorithm::CublasLtMatmul ||
          algorithms[i] == ContractionAlgorithm::CutlassAot)
        requests[i].validate_affine();
      else
        requests[i].validate();
    }
    if (variants_.empty()) variants_.reserve(2);
    if (variants_.capacity() > 2 || requests.capacity() > requests.size())
      throw std::length_error("native descriptor allocation exceeds admitted bound");
    if (context_ && (context_ != &context || generation_ != context.generation()))
      throw std::invalid_argument("native contraction table cannot mix contexts");
    if (calls_ && (calls_ != &calls || summands_ != &summands))
      throw std::invalid_argument("native contraction table cannot mix work counters");
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != context.device()) throw std::logic_error("native preparation device changed");
    if (retained_cache_bytes_ && device != retained_cache_device_)
      throw std::logic_error("native contraction retained module device changed");
    Variant variant{o, v, q, std::move(requests), std::move(algorithms), std::move(batch_scales)};
#if GENERATIVEQC_HAS_CUTENSOR || GENERATIVEQC_HAS_CUBLASLT || GENERATIVEQC_HAS_CUTLASS
    // All optional families can be live together during preparation. Include
    // retained cache charges from earlier attempts before allocating new plans.
    const auto count =
        std::count_if(variant.algorithms.begin(), variant.algorithms.end(), [](auto algorithm) {
          return algorithm == ContractionAlgorithm::CutensorAffine ||
                 algorithm == ContractionAlgorithm::CublasLtMatmul ||
                 algorithm == ContractionAlgorithm::CutlassAot;
        });
    (void)ContractionProviderReservation::checked_add(retained_cache_bytes_,
                                                      reservation.total_bytes(count));
#if GENERATIVEQC_HAS_CUTENSOR || GENERATIVEQC_HAS_CUBLASLT
    const auto prepare_optional = [&]<class Plan>(std::vector<std::unique_ptr<Plan>>& plans,
                                                  ContractionAlgorithm algorithm) {
      if (std::find(variant.algorithms.begin(), variant.algorithms.end(), algorithm) ==
          variant.algorithms.end())
        return;
      if (reservation.host_bytes < sizeof(Plan))
        throw ContractionPreparationUnavailable(
            "optional provider host reservation is insufficient");
      plans.resize(variant.requests.size());
      if (plans.capacity() > plans.size())
        throw std::length_error("native provider pointer table exceeds admitted bound");
      for (std::size_t i = 0; i < variant.requests.size(); ++i) {
        if (variant.algorithms[i] != algorithm) continue;
#if defined(GENERATIVEQC_TEST_HOOKS)
        auto& remaining = algorithm == ContractionAlgorithm::CutensorAffine
                              ? cutensor_preparations_before_rejection_for_test
                              : cublaslt_preparations_before_rejection_for_test;
        if (remaining == 0)
          throw ContractionPreparationUnavailable("injected optional provider rejection");
        if (remaining > 0) --remaining;
#endif
        auto& plan = plans[i];
        plan = std::make_unique<Plan>();
        if (!plan->prepare(variant.requests[i], context.stream(), reservation.workspace_bytes,
                           reservation.provider_bytes, reservation.host_bytes))
          throw ContractionPreparationUnavailable(std::string(plan->rejection()));
      }
    };
#endif
    try {
#if GENERATIVEQC_HAS_CUTENSOR
      prepare_optional(variant.cutensor, ContractionAlgorithm::CutensorAffine);
#endif
#if GENERATIVEQC_HAS_CUBLASLT
      prepare_optional(variant.cublaslt, ContractionAlgorithm::CublasLtMatmul);
#endif
#if GENERATIVEQC_HAS_CUTLASS
      if (std::find(variant.algorithms.begin(), variant.algorithms.end(),
                    ContractionAlgorithm::CutlassAot) != variant.algorithms.end()) {
        if (reservation.host_bytes < sizeof(CudaCutlassContraction) || !reservation.cache_bytes)
          throw ContractionPreparationUnavailable("CUTLASS host/module reservation missing");
        variant.cutlass.resize(variant.requests.size());
        if (variant.cutlass.capacity() > variant.cutlass.size())
          throw std::length_error("native CUTLASS pointer table exceeds admitted bound");
        for (std::size_t i = 0; i < variant.requests.size(); ++i) {
          if (variant.algorithms[i] != ContractionAlgorithm::CutlassAot) continue;
#if defined(GENERATIVEQC_TEST_HOOKS)
          auto& remaining = cutlass_preparations_before_rejection_for_test;
          if (remaining == 0)
            throw ContractionPreparationUnavailable("injected CUTLASS preparation rejection");
          if (remaining > 0) --remaining;
#endif
          auto& plan = variant.cutlass[i];
          plan = std::make_unique<CudaCutlassContraction>();
          const auto remember_modules = [&] {
            const auto charge = plan->module_bytes();
            if (!charge) return;
            retained_cache_device_ = device;
            if (charge > std::numeric_limits<std::size_t>::max() - retained_cache_bytes_) {
              retained_cache_bytes_ = std::numeric_limits<std::size_t>::max();
              retained_cache_quarantined_ = true;
              throw std::length_error("native contraction retained module accounting overflow");
            }
            retained_cache_bytes_ += charge;
          };
          bool ready{};
          try {
            ready = plan->prepare(variant.requests[i], context.stream(), artifact,
                                  reservation.host_bytes, reservation.cache_bytes);
          } catch (...) {
            // No fallback may assume that a failed loader returned its modules
            // to CUDA or stayed inside its reservation. A fresh plan publishes
            // module_bytes only once loading begins. Preserve the failure and
            // known charge floor before this provisional owner disappears.
            if (plan->module_bytes()) retained_cache_quarantined_ = true;
            remember_modules();
            throw;
          }
          remember_modules();
          if (!ready) throw ContractionPreparationUnavailable("CUTLASS preparation unavailable");
        }
      }
#else
      (void)artifact;
#endif
    } catch (...) {
      // Failure of the second family must also drain the first family's plans.
      release_variant(variant);
      throw;
    }
#else
    (void)reservation;
    (void)artifact;
#endif
    variants_.push_back(std::move(variant));
    context_ = &context;
    generation_ = context.generation();
    calls_ = &calls;
    summands_ = &summands;
  }

  explicit operator bool() const noexcept { return !variants_.empty(); }

  /** Unsafe retained loading requires enclosing CUDA-context reconciliation.
   * The owner must carry both this state and the charge across table destruction;
   * release() does not clear it. There is no local reset/requalification API. */
  bool retained_cache_quarantined() const noexcept { return retained_cache_quarantined_; }

  /** Exact prepared workspace and observed provider device growth. Opaque
   * library host storage keeps its reservation; CUTLASS reports its exact native
   * plan size. Both exclude storage_bytes(). Cache charges include prior failed
   * preparations and released plans, not only the executable variants. Under
   * quarantine cache_bytes is only the known charge floor, not a measured bound
   * on all retained memory; SIZE_MAX also represents an unrepresentable sum. */
  ContractionProviderReservation optional_resources() const {
    ContractionProviderReservation result;
    result.cache_bytes = retained_cache_bytes_;
#if GENERATIVEQC_HAS_CUTENSOR || GENERATIVEQC_HAS_CUBLASLT
    const auto accumulate = [&](const auto& plans) {
      for (const auto& plan : plans) {
        if (!plan) continue;
        result.workspace_bytes = ContractionProviderReservation::checked_add(
            result.workspace_bytes, plan->workspace_bytes());
        result.provider_bytes = ContractionProviderReservation::checked_add(result.provider_bytes,
                                                                            plan->provider_bytes());
        result.host_bytes =
            ContractionProviderReservation::checked_add(result.host_bytes, plan->host_bytes());
      }
    };
    for (const auto& variant : variants_) {
#if GENERATIVEQC_HAS_CUTENSOR
      accumulate(variant.cutensor);
#endif
#if GENERATIVEQC_HAS_CUBLASLT
      accumulate(variant.cublaslt);
#endif
    }
#endif
#if GENERATIVEQC_HAS_CUTLASS
    for (const auto& variant : variants_)
      for (const auto& plan : variant.cutlass)
        if (plan)
          result.host_bytes =
              ContractionProviderReservation::checked_add(result.host_bytes, plan->host_bytes());
#endif
    return result;
  }

#if GENERATIVEQC_HAS_CUTENSOR
  /** Inspect each prepared optional plan without search, allocation or replay.
   * Logical variant dimensions and slot identify the owning compiler operation;
   * the callback receives resolved shape/precision and actual fixed plan policy.
   * No plan pointer or borrowed tensor address is exposed as an identity. */
  template <class F>
  void visit_optional_provenance(F&& consume) const {
    require_safe_retained_cache();
    if (!variants_.empty() &&
        (!context_ || !context_->prepared() || generation_ != context_->generation()))
      throw std::logic_error("stale native contraction context; prepare again");
    for (const auto& variant : variants_)
      for (std::size_t slot = 0; slot < variant.cutensor.size(); ++slot)
        if (variant.cutensor[slot])
          consume(variant.o, variant.v, variant.q, slot, variant.cutensor[slot]->provenance());
  }
#endif

#if GENERATIVEQC_HAS_CUBLASLT
  /** Provider-typed visitor keeps algorithm provenance available without
   * forcing existing cuTENSOR visitors to understand a different record type. */
  template <class F>
  void visit_matmul_provenance(F&& consume) const {
    require_safe_retained_cache();
    if (!variants_.empty() &&
        (!context_ || !context_->prepared() || generation_ != context_->generation()))
      throw std::logic_error("stale native contraction context; prepare again");
    for (const auto& variant : variants_)
      for (std::size_t slot = 0; slot < variant.cublaslt.size(); ++slot)
        if (variant.cublaslt[slot])
          consume(variant.o, variant.v, variant.q, slot, variant.cublaslt[slot]->provenance());
  }
#endif
#if GENERATIVEQC_HAS_CUTLASS
  /** Inspect resolved AOT provenance without retaining execution addresses. */
  template <class F>
  void visit_aot_provenance(F&& consume) const {
    require_safe_retained_cache();
    if (!variants_.empty() &&
        (!context_ || !context_->prepared() || generation_ != context_->generation()))
      throw std::logic_error("stale native contraction context; prepare again");
    for (const auto& variant : variants_)
      for (std::size_t slot = 0; slot < variant.cutlass.size(); ++slot)
        if (variant.cutlass[slot])
          consume(variant.o, variant.v, variant.q, slot, variant.cutlass[slot]->provenance());
  }
#endif
  /** Drain executable plans before fallback. optional_resources().cache_bytes
   * remains charged; the enclosing owner must carry it across table destruction
   * too if that owner continues using the same CUDA context. Quarantine also
   * survives release, so unsafe retaining failures cannot use this fallback. */
  void release() {
    for (auto& variant : variants_) release_variant(variant);
    variants_.clear();
    context_ = nullptr;
    calls_ = summands_ = nullptr;
  }

  template <class T, class CheckedStep = void>
  void execute(std::size_t slot, std::size_t o, std::size_t v, std::size_t q, cudaStream_t stream,
               const T* a, const T* b, T* output, int* error,
               const T* batch_scale = nullptr) const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    require_safe_retained_cache();
    if (!context_ || !context_->prepared() || generation_ != context_->generation() ||
        stream != context_->stream())
      throw std::logic_error("stale native contraction context; prepare again");
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("native contraction capture requires replay work accounting");
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != context_->device()) throw std::logic_error("native contraction device changed");
    const Variant* selected = nullptr;
    for (const auto& variant : variants_)
      if (variant.o == o && variant.v == v && variant.q == q) selected = &variant;
    if (!selected || slot >= selected->requests.size())
      throw std::logic_error("native contraction shape changed; prepare again");
    const auto& r = selected->requests[slot];
    if constexpr (std::is_void_v<CheckedStep>) {
      if (!r.checked_update_identity.empty())
        throw std::invalid_argument("checked contraction requires its compiler scalar helper");
    } else {
      static_assert(std::is_same_v<T, double>);
      if (r.checked_update_identity.empty() ||
          r.checked_update_identity != CheckedStep::update_identity ||
          r.checked_publication_identity != CheckedStep::publication_identity)
        throw std::invalid_argument("scalar helper differs from the prepared contraction");
    }
    if (bool(selected->batch_scales[slot].rank) != bool(batch_scale))
      throw std::invalid_argument("contraction batch-scale input differs from prepared region");
    // Semantic work remains valid for both matrix and general affine layouts.
    const auto work = r.affine_summands();
    if (*calls_ == std::numeric_limits<std::size_t>::max() ||
        work > std::numeric_limits<std::size_t>::max() - *summands_)
      throw std::length_error("native contraction diagnostic counter overflow");
#if GENERATIVEQC_HAS_CUTENSOR
    if (selected->algorithms[slot] == ContractionAlgorithm::CutensorAffine) {
      selected->cutensor[slot]->execute(stream, a, b, output, error);
      ++*calls_;
      *summands_ += work;
      return;
    }
#endif
#if GENERATIVEQC_HAS_CUBLASLT
    if (selected->algorithms[slot] == ContractionAlgorithm::CublasLtMatmul) {
      selected->cublaslt[slot]->execute(stream, a, b, output, error);
      ++*calls_;
      *summands_ += work;
      return;
    }
#endif
#if GENERATIVEQC_HAS_CUTLASS
    if (selected->algorithms[slot] == ContractionAlgorithm::CutlassAot) {
      selected->cutlass[slot]->execute(stream, a, b, output, error);
      ++*calls_;
      *summands_ += work;
      return;
    }
#endif
    constexpr auto dtype = std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32;
    if (r.precision.storage_dtype != dtype || !a || !b || !output || !error)
      throw std::invalid_argument("native contraction buffer dtype/address mismatch");
    const auto overlaps = [](const void* left, std::size_t left_bytes, const void* right,
                             std::size_t right_bytes) {
      const auto l = reinterpret_cast<std::uintptr_t>(left);
      const auto r = reinterpret_cast<std::uintptr_t>(right);
      return l <= r ? r - l < left_bytes : l - r < right_bytes;
    };
    if (overlaps(a, r.operands[0].storage_elements() * sizeof(T), output,
                 r.operands[2].storage_elements() * sizeof(T)) ||
        overlaps(b, r.operands[1].storage_elements() * sizeof(T), output,
                 r.operands[2].storage_elements() * sizeof(T)) ||
        (batch_scale && overlaps(batch_scale, r.batches * sizeof(T), output,
                                 r.operands[2].storage_elements() * sizeof(T))))
      throw std::invalid_argument("native contraction output must not alias its inputs");
    const T alpha = static_cast<T>(r.coefficient), beta = static_cast<T>(r.beta);
    if (!std::isfinite(alpha) || !std::isfinite(beta))
      throw std::invalid_argument("contraction coefficient is not representable");
    const auto ta = r.a_trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto tb = r.b_trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto m = int(r.m), n = int(r.n), k = int(r.k), batches = int(r.batches);
    const int lda = int(r.leading_dimension(0)), ldb = int(r.leading_dimension(1)),
              ldc = int(r.leading_dimension(2));
    // The matrix layout implements the original einsum: C^T = op(B)^T op(A)^T.
    // Batches are explicitly materialized by TensorIR, including broadcasts.
    if (selected->algorithms[slot] == ContractionAlgorithm::GeneratedOrdered) {
      if constexpr (!std::is_void_v<CheckedStep>) {
        generated_checked_contraction<CheckedStep>
            <<<generativeqc_tensor::blocks(r.output_elements(), 128), 128, 0, stream>>>(
                a, b, batch_scale, output, r.output_elements(), r.m, r.n, r.k, lda, ldb, ldc,
                r.a_trans == 'T', r.b_trans == 'T', error);
      } else if (batch_scale) {
        generated_batch_scaled_contraction<<<generativeqc_tensor::blocks(r.output_elements(), 128),
                                             128, 0, stream>>>(
            a, b, batch_scale, output, r.output_elements(), r.m, r.n, r.k, lda, ldb, ldc,
            r.a_trans == 'T', r.b_trans == 'T', alpha, error);
      } else {
        generated_contraction<<<generativeqc_tensor::blocks(r.output_elements(), 256), 256, 0,
                                stream>>>(a, b, output, r.output_elements(), r.m, r.n, r.k, lda,
                                          ldb, ldc, r.a_trans == 'T', r.b_trans == 'T', alpha, beta,
                                          error);
      }
      generativeqc_tensor::cuda_check(cudaGetLastError());
    } else {
      cublasStatus_t status;
      if constexpr (std::is_same_v<T, double>) {
        status = batches == 1 ? cublasDgemm(context_->handle(), tb, ta, n, m, k, &alpha, b, ldb, a,
                                            lda, &beta, output, ldc)
                              : cublasDgemmStridedBatched(
                                    context_->handle(), tb, ta, n, m, k, &alpha, b, ldb, r.k * r.n,
                                    a, lda, r.m * r.k, &beta, output, n, r.m * r.n, batches);
      } else {
        status = batches == 1 ? cublasSgemm(context_->handle(), tb, ta, n, m, k, &alpha, b, ldb, a,
                                            lda, &beta, output, ldc)
                              : cublasSgemmStridedBatched(
                                    context_->handle(), tb, ta, n, m, k, &alpha, b, ldb, r.k * r.n,
                                    a, lda, r.m * r.k, &beta, output, n, r.m * r.n, batches);
      }
      // Execution and arithmetic failures propagate. Only preparation-time OOM
      // may change schedule; a failed enqueue must never silently replay work.
      generativeqc_tensor::blas_check(status);
      const auto count = r.output_elements();
      if (batch_scale) {
        publish_batch_scaled_contraction<<<generativeqc_tensor::blocks(count, 256), 256, 0,
                                           stream>>>(output, batch_scale, count, r.m, r.n, ldc,
                                                     error);
      } else {
        audit_contraction<<<generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(count),
                                                        256),
                            256, 0, stream>>>(output, count, error, r.n, ldc);
      }
      generativeqc_tensor::cuda_check(cudaGetLastError());
    }
    ++*calls_;
    *summands_ += work;
  }

 private:
  void require_safe_retained_cache() const {
    if (retained_cache_quarantined_)
      throw std::logic_error("native contraction retained module loading requires reconciliation");
  }
  struct Variant {
    std::size_t o, v, q;
    std::vector<ContractionRequest> requests;
    std::vector<ContractionAlgorithm> algorithms;
    std::vector<ContractionOperand> batch_scales;
#if GENERATIVEQC_HAS_CUTENSOR
    std::vector<std::unique_ptr<CudaCutensorContraction>> cutensor;
#endif
#if GENERATIVEQC_HAS_CUBLASLT
    std::vector<std::unique_ptr<CudaCublasLtContraction>> cublaslt;
#endif
#if GENERATIVEQC_HAS_CUTLASS
    std::vector<std::unique_ptr<CudaCutlassContraction>> cutlass;
#endif
  };
  static void release_variant(Variant& variant) {
#if GENERATIVEQC_HAS_CUTENSOR
    for (auto& plan : variant.cutensor)
      if (plan) plan->release();
#endif
#if GENERATIVEQC_HAS_CUBLASLT
    for (auto& plan : variant.cublaslt)
      if (plan) plan->release();
#endif
#if GENERATIVEQC_HAS_CUTLASS
    for (auto& plan : variant.cutlass)
      if (plan) plan->release();
#endif
    (void)variant;
  }
  std::vector<Variant> variants_;
  CudaContractionContext* context_{};
  std::size_t generation_{};
  std::size_t retained_cache_bytes_{};
  int retained_cache_device_{};
  bool retained_cache_quarantined_{};
  std::size_t *calls_{}, *summands_{};
};

}  // namespace generativeqc::tensor
