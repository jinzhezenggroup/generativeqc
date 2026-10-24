#pragma once

#include "tensor/cuda_contraction.cuh"
#include "tensor/layout_contraction.hpp"

namespace generativeqc::tensor {

struct LayoutCopyView {
  std::size_t shape[2], source[2], destination[2];
};

static __global__ void copy_layout_contraction(const double* source, double* destination,
                                               LayoutCopyView view, std::size_t count) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const auto row = i / view.shape[1], column = i % view.shape[1];
    destination[row * view.destination[0] + column * view.destination[1]] =
        source[row * view.source[0] + column * view.source[1]];
  }
}

/** Prepared copy/contract/publication sequence on an existing shared context.
 * Scratch is precharged by the caller and stream-exclusive until completion.
 * The owner cannot move while its table borrows its diagnostic counters.
 * Capture remains explicitly unsupported like the underlying prepared table.
 */
class CudaLayoutContraction {
 public:
  static constexpr bool supports_capture = false;
  CudaLayoutContraction(const LayoutContractionPlan& plan,
                        const runtime::NativeLoweringRequest& request,
                        const runtime::NativeLoweringCandidate& candidate, std::string_view target,
                        std::string_view compilation, CudaContractionContext& context,
                        std::size_t device_allowance, std::size_t host_allowance)
      : plan_(plan), context_(context), generation_(context.generation()) {
    plan_.validate_binding(request, candidate, target, compilation, device_allowance,
                           host_allowance);
    if (candidate.host_bytes < storage_bytes())
      throw std::length_error("layout contraction host owner is not reserved");
    const bool library = candidate.algorithm == LayoutContractionPlan::kBlas;
    if (library && (!context.handle() ||
                    candidate.provider_bytes < CudaContractionContext::kProviderAllowance))
      throw ContractionPreparationUnavailable(
          "layout contraction provider is unavailable or uncharged");
    table_.add(
        1, 1, 1, {plan_.physical}, context_, calls_, summands_,
        {library ? ContractionAlgorithm::PedanticBlas : ContractionAlgorithm::GeneratedOrdered});
  }
  CudaLayoutContraction(const CudaLayoutContraction&) = delete;
  CudaLayoutContraction& operator=(const CudaLayoutContraction&) = delete;

  static std::size_t storage_bytes() {
    return runtime::lowering_add(sizeof(CudaLayoutContraction),
                                 PreparedContractions::storage_bytes(1));
  }
  std::size_t scratch_bytes() const noexcept { return plan_.scratch_bytes(); }
  std::size_t calls() const noexcept { return calls_; }
  std::size_t summands() const noexcept { return summands_; }
  std::size_t copy_launches() const noexcept { return copy_launches_; }
  std::size_t transfer_bytes() const noexcept { return transfer_bytes_; }
  std::size_t transfer_elements() const noexcept { return transfer_elements_; }

  void execute(cudaStream_t stream, const double* a, std::size_t a_bytes, const double* b,
               std::size_t b_bytes, double* output, std::size_t output_bytes, void* scratch,
               std::size_t capacity, int* error) {
    // Every host-checkable failure precedes all copies and publication.
    if (!context_.prepared() || context_.generation() != generation_ || stream != context_.stream())
      throw std::logic_error("layout contraction context changed; prepare again");
    int device{};
    generativeqc_tensor::cuda_check(cudaGetDevice(&device));
    if (device != context_.device()) throw std::logic_error("layout contraction device changed");
    cudaStreamCaptureStatus capture{};
    generativeqc_tensor::cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::logic_error("layout contraction capture/replay is not qualified");
    const std::array<const void*, 5> pointers{a, b, output, scratch, error};
    const std::array<std::size_t, 5> bytes{
        contraction_product(plan_.original.operands[0].storage_elements(), sizeof(double)),
        contraction_product(plan_.original.operands[1].storage_elements(), sizeof(double)),
        contraction_product(plan_.original.operands[2].storage_elements(), sizeof(double)),
        plan_.scratch_bytes(), sizeof(int)};
    if (a_bytes < bytes[0] || b_bytes < bytes[1] || output_bytes < bytes[2] || capacity < bytes[3])
      throw std::length_error("layout contraction buffer capacity is insufficient");
    for (std::size_t i = 0; i != pointers.size(); ++i) {
      const auto address = reinterpret_cast<std::uintptr_t>(pointers[i]);
      if (!address || address % (i == 4 ? alignof(int) : alignof(double)) ||
          bytes[i] > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("layout contraction pointer/alignment/range is invalid");
    }
    for (std::size_t i = 0; i != pointers.size(); ++i)
      for (std::size_t j = i + 1; j != pointers.size(); ++j) {
        if (i == 0 && j == 1) continue;  // readonly inputs may share storage
        const auto x = reinterpret_cast<std::uintptr_t>(pointers[i]);
        const auto y = reinterpret_cast<std::uintptr_t>(pointers[j]);
        if (x <= y ? y - x < bytes[i] : x - y < bytes[j])
          throw std::invalid_argument("layout contraction buffers must be disjoint");
      }
    for (const auto pointer : pointers) {
      cudaPointerAttributes attributes{};
      const auto status = cudaPointerGetAttributes(&attributes, pointer);
      if (status == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        throw std::invalid_argument("layout contraction requires device allocations");
      }
      generativeqc_tensor::cuda_check(status);
      if (attributes.type != cudaMemoryTypeDevice || attributes.device != context_.device())
        throw std::invalid_argument("layout contraction allocation device changed");
    }
    (void)runtime::lowering_add(copy_launches_, plan_.copy_launches());
    (void)runtime::lowering_add(transfer_bytes_, plan_.transfer_bytes());
    (void)runtime::lowering_add(transfer_elements_, plan_.transfer_elements());
    (void)runtime::lowering_add(calls_, 1);
    (void)runtime::lowering_add(summands_, plan_.physical.summands());
    std::array<double*, 3> values{const_cast<double*>(a), const_cast<double*>(b), output};
    for (std::size_t i = 0; i != values.size(); ++i)
      if (plan_.converted(i))
        values[i] = reinterpret_cast<double*>(static_cast<std::byte*>(scratch) + plan_.offset(i));
    for (std::size_t i = 0; i != 3; ++i)
      if (plan_.converted(i) && (i != 2 || plan_.original.beta != 0))
        copy(i, i == 0 ? a : i == 1 ? b : output, values[i], false, stream);
    // No allocation, heuristic, provider search or retry can occur after copies.
    table_.execute(0, 1, 1, 1, stream, values[0], values[1], values[2], error);
    if (plan_.converted(2)) copy(2, values[2], output, true, stream);
  }

 private:
  void copy(std::size_t operand, const double* source, double* destination, bool publication,
            cudaStream_t stream) {
    const auto& from =
        publication ? plan_.physical.operands[operand] : plan_.original.operands[operand];
    const auto& to =
        publication ? plan_.original.operands[operand] : plan_.physical.operands[operand];
    LayoutCopyView view{{from.shape[0], from.shape[1]},
                        {from.strides[0], from.strides[1]},
                        {to.strides[0], to.strides[1]}};
    copy_layout_contraction<<<generativeqc_tensor::blocks(from.elements(), 256), 256, 0, stream>>>(
        source, destination, view, from.elements());
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++copy_launches_;
    transfer_elements_ += from.elements();
    transfer_bytes_ += from.elements() * 2 * sizeof(double);
  }
  LayoutContractionPlan plan_;
  CudaContractionContext& context_;
  std::size_t generation_{}, calls_{}, summands_{};
  std::size_t copy_launches_{}, transfer_elements_{}, transfer_bytes_{};
  PreparedContractions table_;
};

}  // namespace generativeqc::tensor
