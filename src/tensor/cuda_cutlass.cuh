#pragma once

#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/version.h>

#include <memory>
#include <type_traits>

#if defined(CUTLASS_ENABLE_SYNCLOG)
#error "CUTLASS synclog introduces unbudgeted replay allocation"
#endif

#include "tensor/cuda_affine_audit.cuh"
#include "tensor/native_matrix_view.hpp"

namespace generativeqc::tensor {

#if defined(GENERATIVEQC_TEST_HOOKS)
// Exercise cleanup after CUDA has had an opportunity to retain module storage.
inline thread_local bool cutlass_fail_after_module_load_for_test = false;
#endif

/** Optional AOT SIMT contraction owner. The canonical request defines science;
 * a fixed scalar-aligned kernel family implements its proven matrix views.
 * No tensor-core mode, split-K, packing, heuristic or JIT is introduced.
 * Instances serialize replay and borrow a stream that must outlive them.
 */
class CudaCutlassContraction {
  struct Plan {
    virtual ~Plan() = default;
    virtual void execute(const void*, const void*, void*, cudaStream_t) = 0;
    virtual void load() = 0;
    virtual std::size_t bytes() const = 0;
  };

  template <bool Row>
  using Layout = std::conditional_t<Row, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  template <class T, unsigned Order>
  struct TypedPlan final : Plan {
    // Fixed small tiles keep shared storage below 48 KiB, avoiding the runtime
    // attribute mutation in CUTLASS run(). Scalar accesses admit odd dimensions
    // and the exact alignment promised by the canonical tensor views.
    using Gemm = cutlass::gemm::device::GemmBatched<
        T, Layout<(Order & 1) != 0>, T, Layout<(Order & 2) != 0>, T, Layout<(Order & 4) != 0>, T,
        cutlass::arch::OpClassSimt, cutlass::arch::Sm50, cutlass::gemm::GemmShape<32, 64, 8>,
        cutlass::gemm::GemmShape<16, 32, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<T, 1, T, T>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, 1, 1>;
    static_assert(sizeof(typename Gemm::GemmKernel::SharedStorage) < (48 << 10));
    static_assert(Gemm::GemmKernel::kThreadCount == 128);
    typename Gemm::Arguments args;
    Gemm gemm;

    TypedPlan(const ContractionRequest& request, const MatrixContractionRecipe& recipe)
        : args({int(recipe.layouts[0].rows), int(recipe.layouts[1].columns),
                int(recipe.layouts[0].columns)},
               {nullptr, int(recipe.layouts[0].ld)}, recipe.layouts[0].batch_stride,
               {nullptr, int(recipe.layouts[1].ld)}, recipe.layouts[1].batch_stride,
               {nullptr, int(recipe.layouts[2].ld)}, recipe.layouts[2].batch_stride,
               {nullptr, int(recipe.layouts[2].ld)}, recipe.layouts[2].batch_stride,
               {T(request.coefficient), T(request.beta)}, int(recipe.batches)) {
      if (!std::isfinite(T(request.coefficient)) || !std::isfinite(T(request.beta)))
        throw std::invalid_argument("CUTLASS scalar coefficient is not representable");
      check(Gemm::can_implement(args));
      if (Gemm::get_workspace_size(args))
        throw std::logic_error("CUTLASS AOT family unexpectedly requires workspace");
      check(gemm.initialize(args));
    }
    void load() override {
      cudaFuncAttributes attributes{};
      // Resolve lazy module loading during preparation, never on first replay.
      generativeqc_tensor::cuda_check(
          cudaFuncGetAttributes(&attributes, cutlass::Kernel<typename Gemm::GemmKernel>));
      generativeqc_tensor::cuda_check(
          cudaFuncGetAttributes(&attributes, audit_affine_contraction<T>));
    }
    void execute(const void* a, const void* b, void* output, cudaStream_t stream) override {
      args.ref_A.reset(static_cast<const T*>(a));
      args.ref_B.reset(static_cast<const T*>(b));
      args.ref_C.reset(static_cast<const T*>(output));
      args.ref_D.reset(static_cast<T*>(output));
      // update only replaces borrowed pointers in the already prepared Params.
      check(gemm.update(args));
      check(gemm.run(stream));
    }
    std::size_t bytes() const override { return sizeof(*this); }
  };

 public:
  static constexpr std::string_view kFamily = "simt-32x64x8-v1";
  struct Provenance {
    ContractionRequest request;
    std::array<char, 64> artifact_identity{};
    std::array<int, 3> tile{32, 64, 8};
    int version{CUTLASS_VERSION}, architecture{}, runtime_version{};
    unsigned layout_order{};
  };

  CudaCutlassContraction() = default;
  CudaCutlassContraction(const CudaCutlassContraction&) = delete;
  CudaCutlassContraction& operator=(const CudaCutlassContraction&) = delete;
  ~CudaCutlassContraction() {
    if (plan_) {
      int previous = device_;
      (void)cudaGetDevice(&previous);
      (void)cudaSetDevice(device_);
      (void)cudaStreamSynchronize(stream_);
      (void)cudaSetDevice(previous);
    }
  }

  /** Prepare one shape. artifact must reference the actual source/toolchain/
   * flags/header-content identity held by the enclosing build/artifact owner.
   * module_bytes is an externally qualified reservation for context-retained
   * AOT modules, not a claim of zero allocation or a reclaimable plan buffer.
   * Rejection before loading returns false without increasing the charge.
   * Once loading begins, the reservation survives every exception and release.
   * A module-growth violation throws: CUDA may retain that module until context
   * destruction, so the enclosing owner must not treat this as a free fallback.
   */
  bool prepare(const ContractionRequest& request, cudaStream_t stream, std::string_view artifact,
               std::size_t host_limit, std::size_t module_bytes) {
    if (plan_) throw std::logic_error("CUTLASS contraction already prepared");
    const auto recipe = MatrixContractionRecipe::from(request);
    if (!valid_digest(artifact)) throw std::invalid_argument("CUTLASS artifact identity missing");
    require_uncaptured(stream);
    if (!module_bytes || module_bytes < module_bytes_) return false;
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (module_bytes_ && device != device_)
      throw std::logic_error("CUTLASS retained module device changed");
    device_ = device;
    int major{}, minor{};
    generativeqc_tensor::cuda_check(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    generativeqc_tensor::cuda_check(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_));
    if (major < 5) return false;
    // CUTLASS's fixed batched swizzle maps matrix columns and batch count to
    // CUDA grid y/z. Reject before module loading or submitting any work.
    const auto kernel_columns =
        recipe.layouts[2].row_major ? recipe.layouts[2].columns : recipe.layouts[2].rows;
    if ((kernel_columns + 63) / 64 > 65535 || recipe.batches > 65535) return false;
    unsigned order{};
    for (unsigned i = 0; i < 3; ++i)
      if (recipe.layouts[i].row_major) order |= 1U << i;
    std::unique_ptr<Plan> prepared;
    try {
      prepared = request.publication_dtype == PrecisionDtype::Fp64
                     ? make<double>(order, request, recipe, host_limit)
                     : make<float>(order, request, recipe, host_limit);
    } catch (const std::bad_alloc&) {
      return false;
    }
    if (!prepared) return false;
    std::lock_guard<std::mutex> lock(runtime::allocation_measurement_mutex);
    std::size_t before{}, after{}, total{};
    generativeqc_tensor::cuda_check(cudaMemGetInfo(&before, &total));
    // Loading is not transactional: even a failed attribute query can leave
    // previously loaded kernels in the CUDA context. Reserve before the first
    // such call, while descriptor-only host rejection above stays reclaimable.
    module_bytes_ = module_bytes;
    prepared->load();
#if defined(GENERATIVEQC_TEST_HOOKS)
    if (cutlass_fail_after_module_load_for_test)
      throw std::runtime_error("injected failure after CUTLASS module loading");
#endif
    generativeqc_tensor::cuda_check(cudaMemGetInfo(&after, &total));
    if (before > after && before - after > module_bytes) {
      // Preserve the observed excess for diagnostics even though admission is
      // invalid and must abort. Reporting the smaller ceiling hides real cost.
      module_bytes_ = before - after;
      throw std::length_error("CUTLASS AOT module exceeds qualified cache reservation");
    }
    int runtime{};
    generativeqc_tensor::cuda_check(cudaRuntimeGetVersion(&runtime));
    provenance_ = {request, {}, {32, 64, 8}, CUTLASS_VERSION, 10 * major + minor, runtime, order};
    std::copy(artifact.begin(), artifact.end(), provenance_.artifact_identity.begin());
    for (std::size_t axis = 0; axis < request.operands[2].rank; ++axis) {
      audit_.shape[axis] = request.operands[2].shape[axis];
      audit_.strides[axis] = request.operands[2].strides[axis];
    }
    audit_.rank = request.operands[2].rank;
    stream_ = stream;
    plan_ = std::move(prepared);
    ++preparations_;
    return true;
  }

  template <class T>
  void execute(cudaStream_t stream, const T* a, const T* b, T* output, int* error) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if (!plan_ || stream != stream_) throw std::logic_error("stale CUTLASS plan or stream");
    require_uncaptured(stream);
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != device_) throw std::logic_error("CUTLASS plan device changed");
    const auto& request = provenance_.request;
    if ((std::is_same_v<T, double> ? PrecisionDtype::Fp64 : PrecisionDtype::Fp32) !=
        request.publication_dtype)
      throw std::invalid_argument("CUTLASS pointer dtype changed");
    if (!a || !b || !output || !error)
      throw std::invalid_argument("CUTLASS null execution pointer");
    for (const auto* pointer : {a, b, static_cast<const T*>(output)})
      if (reinterpret_cast<std::uintptr_t>(pointer) % sizeof(T))
        throw std::invalid_argument("CUTLASS pointer alignment changed");
    for (std::size_t i = 0; i < 2; ++i) {
      const auto x = reinterpret_cast<std::uintptr_t>(i ? b : a);
      const auto y = reinterpret_cast<std::uintptr_t>(output);
      if (x <= y ? y - x < request.operands[i].storage_elements() * sizeof(T)
                 : x - y < request.operands[2].storage_elements() * sizeof(T))
        throw std::invalid_argument("CUTLASS output aliases an input");
    }
    const auto work = request.affine_summands();
    if (calls_ == std::numeric_limits<std::size_t>::max() ||
        work > std::numeric_limits<std::size_t>::max() - summands_)
      throw std::length_error("CUTLASS work counter overflow");
    plan_->execute(a, b, output, stream);
    const auto count = request.affine_output_elements();
    audit_affine_contraction<<<generativeqc_tensor::blocks(count, 256), 256, 0, stream>>>(
        output, audit_, count, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++calls_;
    summands_ += work;
  }

  const Provenance& provenance() const {
    if (!plan_) throw std::logic_error("CUTLASS provenance requires preparation");
    return provenance_;
  }
  std::size_t host_bytes() const { return sizeof(*this) + (plan_ ? plan_->bytes() : 0); }
  std::size_t module_bytes() const { return module_bytes_; }
  std::size_t calls() const { return calls_; }
  std::size_t summands() const { return summands_; }
  std::size_t preparations() const { return preparations_; }
  void release() {
    if (!plan_) return;
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != device_) throw std::logic_error("CUTLASS release device changed");
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(stream_));
    plan_.reset();
    // Module reservation survives release: the CUDA context still owns it.
  }

 private:
  static void check(cutlass::Status status) {
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error("CUTLASS operation failed: " + std::to_string(int(status)));
  }
  static bool valid_digest(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
             return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
  }
  static void require_uncaptured(cudaStream_t stream) {
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("CUTLASS capture work accounting is not qualified");
  }
  template <class T, unsigned Order = 0>
  static std::unique_ptr<Plan> make(unsigned order, const ContractionRequest& request,
                                    const MatrixContractionRecipe& recipe, std::size_t limit) {
    if (order == Order) {
      if (limit < sizeof(CudaCutlassContraction) + sizeof(TypedPlan<T, Order>)) return {};
      return std::make_unique<TypedPlan<T, Order>>(request, recipe);
    }
    if constexpr (Order < 7) return make<T, Order + 1>(order, request, recipe, limit);
    throw std::logic_error("invalid CUTLASS physical layout code");
  }
  std::unique_ptr<Plan> plan_;
  Provenance provenance_{};
  AffineAuditView audit_{};
  cudaStream_t stream_{};
  int device_{};
  std::size_t module_bytes_{}, calls_{}, summands_{}, preparations_{};
};

}  // namespace generativeqc::tensor
