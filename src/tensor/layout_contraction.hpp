#pragma once

#include <bit>

#include "runtime/lowering_binding.hpp"
#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

/** Fixed-shape execution projection of compiler LayoutTransfer obligations.
 * This first executor copies ordinary positive, unbatched FP64 matrix views.
 * It owns no science, provider, allocation, search or approximation policy.
 */
struct LayoutContractionPlan {
  static constexpr std::string_view kGenerated = "affine-copy-v1-generated";
  static constexpr std::string_view kBlas = "affine-copy-v1-blas";
  const std::string_view request_identity, candidate_identity;
  const ContractionRequest original, physical;

  LayoutContractionPlan(std::string_view request, std::string_view candidate,
                        ContractionRequest source, ContractionRequest packed)
      : request_identity(request),
        candidate_identity(candidate),
        original(source),
        physical(packed) {
    if (!runtime::lowering_digest(request) || !runtime::lowering_digest(candidate))
      throw std::invalid_argument("layout contraction requires canonical identities");
    original.validate_affine();
    physical.validate();
    if (!physical.precision.is_strict_fp64() || physical.batches != 1 || physical.a_trans != 'N' ||
        physical.b_trans != 'N' || physical.leading_dimensions != std::array<std::size_t, 3>{} ||
        !original.checked_update_identity.empty() || !physical.checked_update_identity.empty() ||
        original.checked_transpose_pair_role || physical.checked_transpose_pair_role ||
        original.scientific_identity != physical.scientific_identity ||
        original.semantic_template_identity != physical.semantic_template_identity ||
        original.precision_identity != physical.precision_identity ||
        !original.precision.is_strict_fp64() ||
        original.precision.qualification != physical.precision.qualification ||
        original.publication_dtype != physical.publication_dtype ||
        std::bit_cast<std::uint64_t>(original.coefficient) !=
            std::bit_cast<std::uint64_t>(physical.coefficient) ||
        std::bit_cast<std::uint64_t>(original.beta) != std::bit_cast<std::uint64_t>(physical.beta))
      throw std::invalid_argument("layout contraction changes science or unsupported arithmetic");
    for (std::size_t i = 0; i != 3; ++i) {
      const auto& a = original.operands[i];
      const auto& b = physical.operands[i];
      if (a.rank != 2 || b.rank != 2 || a.modes != b.modes || a.shape != b.shape ||
          a.dtype != b.dtype)
        throw std::invalid_argument("layout contraction changes logical matrix views");
      converted_[i] = a.strides != b.strides;
      offsets_[i] = scratch_bytes_;
      if (!converted_[i]) continue;
      buffer_bytes_[i] = contraction_product(b.storage_elements(), sizeof(double));
      scratch_bytes_ = runtime::lowering_add(scratch_bytes_, buffer_bytes_[i]);
      const std::size_t copies = i == 2 ? 1 + std::size_t(original.beta != 0) : 1;
      const auto elements = contraction_product(b.elements(), copies);
      transfer_elements_ = runtime::lowering_add(transfer_elements_, elements);
      transfer_bytes_ =
          runtime::lowering_add(transfer_bytes_, contraction_product(elements, 2 * sizeof(double)));
      copy_launches_ += copies;
    }
    if (!copy_launches_) throw std::invalid_argument("layout contraction has no physical transfer");
  }

  bool converted(std::size_t operand) const { return converted_.at(operand); }
  std::size_t offset(std::size_t operand) const { return offsets_.at(operand); }
  std::size_t scratch_bytes() const noexcept { return scratch_bytes_; }
  std::size_t transfer_bytes() const noexcept { return transfer_bytes_; }
  std::size_t transfer_elements() const noexcept { return transfer_elements_; }
  std::size_t copy_launches() const noexcept { return copy_launches_; }

  void validate_binding(const runtime::NativeLoweringRequest& request,
                        const runtime::NativeLoweringCandidate& candidate, std::string_view target,
                        std::string_view compilation, std::size_t device_allowance,
                        std::size_t host_allowance) const {
    // This is a complete strict candidate; no unmeasured-incumbent escape.
    (void)runtime::select_native_lowering(request, std::span(&candidate, 1), target, compilation);
    const auto& precision = request.precisions[candidate.precision];
    const auto extra = runtime::lowering_add(
        runtime::lowering_add(candidate.workspace_bytes, candidate.provider_bytes),
        runtime::lowering_add(candidate.temporary_bytes, candidate.cache_bytes));
    if (request.identity != request_identity || candidate.identity != candidate_identity ||
        request.scientific_identity != original.scientific_identity ||
        request.semantic_identity != original.semantic_template_identity ||
        precision.identity != original.precision_identity ||
        !runtime::strict_requested_precision(request, precision) ||
        !precision.arithmetic.is_strict_fp64() || !precision.casts.empty() ||
        !precision.refinement.empty() || !precision.audit.empty() || candidate.capture_safe ||
        (candidate.algorithm == kBlas &&
         candidate.determinism == runtime::LoweringDeterminism::ExactOrder) ||
        candidate.temporary_bytes < scratch_bytes_ || candidate.cost.pack_bytes < transfer_bytes_ ||
        extra > device_allowance || candidate.host_bytes > host_allowance ||
        !((candidate.algorithm == kGenerated && candidate.provider == "generated.cuda") ||
          (candidate.algorithm == kBlas && candidate.provider == "cublas")))
      throw std::invalid_argument("layout contraction binding or complete resource budget changed");
  }

 private:
  std::array<bool, 3> converted_{};
  std::array<std::size_t, 3> offsets_{}, buffer_bytes_{};
  std::size_t scratch_bytes_{}, transfer_bytes_{}, transfer_elements_{}, copy_launches_{};
};

}  // namespace generativeqc::tensor
