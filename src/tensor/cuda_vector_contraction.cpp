#include "tensor/cuda_vector_contraction.hpp"

#include <chrono>
#include <cstdio>

#include "generativeqc/generativeqc.hpp"

namespace generativeqc::tensor {
namespace {
generativeqc_status status(cublasStatus_t error) {
  return error == CUBLAS_STATUS_SUCCESS        ? GENERATIVEQC_STATUS_SUCCESS
         : error == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                               : GENERATIVEQC_STATUS_CUDA_ERROR;
}
void check(cublasStatus_t error) {
  if (error != CUBLAS_STATUS_SUCCESS)
    throw Error(status(error), "vector contraction provider preparation failed");
}
void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw Error(error == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                   : GENERATIVEQC_STATUS_CUDA_ERROR,
                "vector contraction CUDA preparation failed");
}
}  // namespace

CudaVectorContraction::CudaVectorContraction(
    const runtime::NativeLoweringRequest& request,
    std::array<runtime::NativeLoweringCandidate, 2> candidates, std::string_view target,
    std::string_view compilation, const ContractionRequest& resolved,
    std::size_t qualified_incumbent, cublasHandle_t handle, cudaStream_t stream)
    : resolved_(resolved), handle_(handle), stream_(stream) {
  const auto started = std::chrono::steady_clock::now();
  resolved.validate();
  for (std::size_t i = 0; i != operand_bytes_.size(); ++i)
    operand_bytes_[i] = resolved.operands[i].storage_elements() * sizeof(double);
  if (!handle || !stream || resolved.n != 1 || resolved.b_trans != 'N' ||
      resolved.coefficient != 1.0 || (resolved.beta != 0.0 && resolved.beta != 1.0) ||
      request.inputs != (resolved.beta == 0.0 ? 2U : 3U) || resolved.leading_dimension(1) != 1 ||
      resolved.leading_dimension(2) != 1 || !resolved.precision.is_strict_fp64() ||
      resolved.publication_dtype != PrecisionDtype::Fp64 ||
      resolved.scientific_identity != request.scientific_identity ||
      resolved.semantic_template_identity != request.semantic_identity ||
      request.precisions.size() != 1 || !request.precisions[0].arithmetic.is_strict_fp64() ||
      !request.precisions[0].casts.empty() || !request.precisions[0].refinement.empty() ||
      !request.precisions[0].audit.empty() ||
      resolved.precision_identity != request.precisions[0].identity)
    throw std::invalid_argument("incompatible vector contraction request");
  cudaStreamCaptureStatus capture{};
  check(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("vector contraction preparation cannot capture");
  cudaStream_t actual_stream{};
  cublasPointerMode_t pointer_mode{};
  check(cublasGetStream(handle, &actual_stream));
  check(cublasGetPointerMode(handle, &pointer_mode));
  if (actual_stream != stream || pointer_mode != CUBLAS_POINTER_MODE_HOST)
    throw std::invalid_argument("vector contraction borrowed handle modes changed");
  check(cudaGetDevice(&diagnostic_.device));
  check(cudaDeviceGetAttribute(&diagnostic_.compute_major, cudaDevAttrComputeCapabilityMajor,
                               diagnostic_.device));
  check(cudaDeviceGetAttribute(&diagnostic_.compute_minor, cudaDevAttrComputeCapabilityMinor,
                               diagnostic_.device));
  check(cudaRuntimeGetVersion(&diagnostic_.runtime_version));
  check(cublasGetVersion(handle, &diagnostic_.provider_version));
  const int length =
      std::snprintf(version_.data(), version_.size(), "%d", diagnostic_.provider_version);
  if (length <= 0 || std::size_t(length) >= version_.size())
    throw std::runtime_error("vector contraction provider version unavailable");
  diagnostic_.request = request;
  for (auto& candidate : candidates) {
    candidate.provider_version = version_.data();
    candidate.host_bytes = sizeof(*this);
    if (candidate.algorithm == "gemv") {
      if (resolved.batches != 1) candidate.rejection = "GEMV requires one contiguous vector batch";
    } else if (candidate.algorithm != "gemm-strided-batched-vector") {
      candidate.rejection = "vector algorithm has no executable binding";
    }
  }
  const auto decision = runtime::select_native_lowering(request, candidates, target, compilation, 1,
                                                        qualified_incumbent);
  diagnostic_.candidates = candidates;
  diagnostic_.selected = decision.selected;
  diagnostic_.retained_incumbent = decision.retained_incumbent;
  for (const auto& rejection : decision.rejections)
    diagnostic_.rejections[rejection.candidate] = rejection.reason;
  gemv_ = candidates[decision.selected].algorithm == "gemv";
  diagnostic_.prepare_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - started)
                               .count();
}

generativeqc_status CudaVectorContraction::launch(const double* matrix, const double* vector,
                                                  double* output) const {
  if (!matrix || !vector || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  // The result is disjoint from the two product inputs. An admitted SSA update
  // donates its seed/result allocation, which is represented by output itself.
  // Reject partial aliasing before submission; product inputs may share storage.
  const auto overlaps = [](const void* a, std::size_t na, const void* b, std::size_t nb) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na : x - y < nb;
  };
  const auto bytes = [this](std::size_t i) { return operand_bytes_[i]; };
  if (overlaps(output, bytes(2), matrix, bytes(0)) || overlaps(output, bytes(2), vector, bytes(1)))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  int device{};
  if (cudaGetDevice(&device) != cudaSuccess) return GENERATIVEQC_STATUS_CUDA_ERROR;
  if (device != diagnostic_.device) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto& r = resolved_;
  // A row-major matrix is the transpose of its column-major library view.
  // A vector's row/column storage is identical. Keep the incumbent m-by-1
  // library recipe, including batch strides, rather than changing GEMM order.
  const auto operation = r.a_trans == 'T' ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int rows = r.a_trans == 'T' ? r.m : r.k;
  const int columns = r.a_trans == 'T' ? r.k : r.m;
  if (gemv_)
    return status(cublasDgemv(handle_, operation, rows, columns, &r.coefficient, matrix,
                              r.leading_dimension(0), vector, 1, &r.beta, output, 1));
  return status(cublasDgemmStridedBatched(handle_, operation, CUBLAS_OP_N, r.m, 1, r.k,
                                          &r.coefficient, matrix, r.leading_dimension(0), r.m * r.k,
                                          vector, r.k, r.k, &r.beta, output, r.m, r.m, r.batches));
}

}  // namespace generativeqc::tensor
