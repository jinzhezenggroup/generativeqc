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
enum class ContractionAlgorithm : std::uint8_t { PedanticBlas, GeneratedOrdered, CutensorAffine };

/** Optional-provider rejection is distinct from malformed science or execution
 * failures. Only this exception authorizes preparation of an admitted fallback. */
class ContractionPreparationUnavailable : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/** Per-plan ceilings, supplied by provider qualification and admitted by the
 * enclosing owner before preparation. Zero defaults intentionally admit no
 * opaque cuTENSOR host storage. There is no production reservation heuristic.
 * host_bytes includes the native plan object and opaque provider host storage;
 * descriptor/pointer tables are charged separately by storage_bytes(). */
struct ContractionProviderReservation {
  std::size_t workspace_bytes{}, provider_bytes{}, host_bytes{};

  std::size_t total_bytes(std::size_t plans) const {
    auto result = checked_add(workspace_bytes, provider_bytes);
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
inline thread_local int cutensor_preparations_before_rejection_for_test = -1;
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
               (sizeof(ContractionRequest) + sizeof(ContractionAlgorithm)
#if GENERATIVEQC_HAS_CUTENSOR
                + sizeof(std::unique_ptr<CudaCutensorContraction>)
#endif
               );
  }

  /** Bind one shape transactionally. Optional cuTENSOR plans are fully prepared
   * before publishing the variant; rejection releases all provisional plans and
   * leaves the table unchanged. The caller owns fallback selection and budgets
   * reservation.total_bytes(number_of_cutensor_plans) in addition to the table.
   * Pure affine requests need no matrix recipe when that provider is selected. */
  void add(std::size_t o, std::size_t v, std::size_t q, std::vector<ContractionRequest> requests,
           CudaContractionContext& context, std::size_t& calls, std::size_t& summands,
           std::vector<ContractionAlgorithm> algorithms = {},
           ContractionProviderReservation reservation = {}) {
    if (variants_.size() == 2) throw std::length_error("native contraction batch variant bound");
    if (!context.prepared()) throw std::logic_error("native contraction provider is not prepared");
    if (algorithms.empty()) algorithms.assign(requests.size(), ContractionAlgorithm::PedanticBlas);
    if (algorithms.size() != requests.size() || algorithms.capacity() > algorithms.size())
      throw std::invalid_argument("native contraction algorithm table bound");
    for (auto algorithm : algorithms) {
      if ((algorithm != ContractionAlgorithm::PedanticBlas &&
           algorithm != ContractionAlgorithm::GeneratedOrdered &&
           algorithm != ContractionAlgorithm::CutensorAffine) ||
          (algorithm == ContractionAlgorithm::PedanticBlas && !context.handle()))
        throw std::invalid_argument("native contraction provider unavailable");
#if !GENERATIVEQC_HAS_CUTENSOR
      if (algorithm == ContractionAlgorithm::CutensorAffine)
        throw ContractionPreparationUnavailable("cuTENSOR was not enabled in this build");
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
      if (algorithms[i] == ContractionAlgorithm::CutensorAffine)
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
    Variant variant{o, v, q, std::move(requests), std::move(algorithms)};
#if GENERATIVEQC_HAS_CUTENSOR
    const auto count = std::count(variant.algorithms.begin(), variant.algorithms.end(),
                                  ContractionAlgorithm::CutensorAffine);
    if (count) {
      // Check aggregate arithmetic before allocating any provider storage.
      (void)reservation.total_bytes(count);
      if (reservation.host_bytes < sizeof(CudaCutensorContraction))
        throw ContractionPreparationUnavailable("cuTENSOR host reservation is insufficient");
      variant.cutensor.resize(variant.requests.size());
      if (variant.cutensor.capacity() > variant.cutensor.size())
        throw std::length_error("native provider pointer table exceeds admitted bound");
      try {
        for (std::size_t i = 0; i < variant.requests.size(); ++i) {
          if (variant.algorithms[i] != ContractionAlgorithm::CutensorAffine) continue;
#if defined(GENERATIVEQC_TEST_HOOKS)
          if (cutensor_preparations_before_rejection_for_test == 0)
            throw ContractionPreparationUnavailable("injected optional provider rejection");
          if (cutensor_preparations_before_rejection_for_test > 0)
            --cutensor_preparations_before_rejection_for_test;
#endif
          auto& plan = variant.cutensor[i];
          plan = std::make_unique<CudaCutensorContraction>();
          if (!plan->prepare(variant.requests[i], context.stream(), reservation.workspace_bytes,
                             reservation.provider_bytes, reservation.host_bytes))
            throw ContractionPreparationUnavailable(std::string(plan->rejection()));
        }
      } catch (...) {
        // A live cleanup error propagates as a hard failure, never as optional
        // rejection. Destructors still release remaining plans best effort.
        for (auto& plan : variant.cutensor)
          if (plan) plan->release();
        throw;
      }
    }
#else
    (void)reservation;
#endif
    variants_.push_back(std::move(variant));
    context_ = &context;
    generation_ = context.generation();
    calls_ = &calls;
    summands_ = &summands;
  }

  explicit operator bool() const noexcept { return !variants_.empty(); }

  /** Exact prepared workspace and observed provider device growth; host bytes
   * remain the externally supplied reservation, excluding storage_bytes(). */
  ContractionProviderReservation optional_resources() const {
    ContractionProviderReservation result;
#if GENERATIVEQC_HAS_CUTENSOR
    for (const auto& variant : variants_)
      for (const auto& plan : variant.cutensor) {
        if (!plan) continue;
        result.workspace_bytes = ContractionProviderReservation::checked_add(
            result.workspace_bytes, plan->workspace_bytes());
        result.provider_bytes = ContractionProviderReservation::checked_add(result.provider_bytes,
                                                                            plan->provider_bytes());
        result.host_bytes =
            ContractionProviderReservation::checked_add(result.host_bytes, plan->host_bytes());
      }
#endif
    return result;
  }

  /** Drain and release a live table before re-admitting a fallback. */
  void release() {
#if GENERATIVEQC_HAS_CUTENSOR
    for (auto& variant : variants_)
      for (auto& plan : variant.cutensor)
        if (plan) plan->release();
#endif
    variants_.clear();
    context_ = nullptr;
    calls_ = summands_ = nullptr;
  }

  template <class T>
  void execute(std::size_t slot, std::size_t o, std::size_t v, std::size_t q, cudaStream_t stream,
               const T* a, const T* b, T* output, int* error) const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
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
                 r.operands[2].storage_elements() * sizeof(T)))
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
      generated_contraction<<<generativeqc_tensor::blocks(r.output_elements(), 256), 256, 0,
                              stream>>>(a, b, output, r.output_elements(), r.m, r.n, r.k, lda, ldb,
                                        ldc, r.a_trans == 'T', r.b_trans == 'T', alpha, beta,
                                        error);
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
      audit_contraction<<<generativeqc_tensor::blocks(static_cast<generativeqc_tensor::I>(count),
                                                      256),
                          256, 0, stream>>>(output, count, error, r.n, ldc);
      generativeqc_tensor::cuda_check(cudaGetLastError());
    }
    ++*calls_;
    *summands_ += work;
  }

 private:
  struct Variant {
    std::size_t o, v, q;
    std::vector<ContractionRequest> requests;
    std::vector<ContractionAlgorithm> algorithms;
#if GENERATIVEQC_HAS_CUTENSOR
    std::vector<std::unique_ptr<CudaCutensorContraction>> cutensor;
#endif
  };
  std::vector<Variant> variants_;
  CudaContractionContext* context_{};
  std::size_t generation_{};
  std::size_t *calls_{}, *summands_{};
};

}  // namespace generativeqc::tensor
