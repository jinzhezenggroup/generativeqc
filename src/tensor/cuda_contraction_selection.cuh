#pragma once

#include "runtime/lowering_binding.hpp"
#include "tensor/cuda_contraction.cuh"

namespace generativeqc::tensor {

#if defined(GENERATIVEQC_TEST_HOOKS)
// Qualification can remove optional implementations at the provider boundary.
// This is absent from production and is not a scientific method option.
inline thread_local bool contraction_libraries_unavailable_for_test = false;
#endif

/** One provider implementation of a compiler-owned contraction region. The
 * region owns numerical conversions/traversal; the provider owns all handles
 * and immutable per-operation plans. Resource bounds cover simultaneous plans,
 * including provisional preparation before an optional fallback. */
struct ContractionRegionPlan {
  ContractionAlgorithm algorithm{};
  ContractionProviderReservation reservation;
  std::size_t selected{}, binding_bytes{};
  bool retained_incumbent{};
};

/** Resolve a bounded emitted region portfolio before any method allocation.
 * No shape or method heuristic is used. Until complete measured phase costs
 * exist, preserve a legal incumbent; resource failure admits generated execution.
 * Test-only scores exercise optional implementations without production promotion.
 */
template <std::size_t N>
ContractionRegionPlan select_contraction_region(
    const runtime::NativeLoweringRequest& request,
    const std::array<runtime::NativeLoweringCandidate, N>& candidates, std::string_view target,
    std::string_view compilation, std::size_t plans, std::size_t host_bytes,
    std::size_t maximum_bytes, bool library_available = true) {
  static_assert(N > 0 && N <= 256);
  if (!plans) throw std::invalid_argument("empty contraction region");
#if defined(GENERATIVEQC_TEST_HOOKS)
  library_available = library_available && !contraction_libraries_unavailable_for_test;
#endif
  auto offers = candidates;
  const auto reservation = qualified_cutensor_reservation();
  const auto version = cutensor_provider_version();
  std::array<std::size_t, N> complete_bytes{};
  std::optional<std::size_t> library, generated;
  for (std::size_t i = 0; i < N; ++i) {
    auto& offer = offers[i];
    if (offer.precision >= request.precisions.size())
      throw std::invalid_argument("contraction region precision index is invalid");
    if (offer.provider == "cublas") {
      offer.provider_bytes = CudaContractionContext::kProviderAllowance;
      offer.host_bytes = host_bytes;
    } else if (offer.provider == "generated.cuda") {
      offer.host_bytes = host_bytes;
    } else if (offer.provider == "cutensor") {
      offer.workspace_bytes = contraction_product(plans, reservation.workspace_bytes);
      offer.provider_bytes = contraction_product(plans, reservation.provider_bytes);
      offer.host_bytes = ContractionProviderReservation::checked_add(
          host_bytes, contraction_product(plans, reservation.host_bytes));
      if (!reservation.host_bytes || version < 20800 || version / 10000 != 2)
        offer.rejection = "qualified cuTENSOR resource profile unavailable";
    } else {
      offer.rejection = "contraction region has no prepared implementation";
    }
    if (!library_available && offer.provider != "generated.cuda")
      offer.rejection = "optional provider preparation unavailable";
    complete_bytes[i] = ContractionProviderReservation::checked_add(
        offer.host_bytes,
        ContractionProviderReservation::checked_add(offer.workspace_bytes, offer.provider_bytes));
    complete_bytes[i] = ContractionProviderReservation::checked_add(
        complete_bytes[i],
        ContractionProviderReservation::checked_add(offer.temporary_bytes, offer.cache_bytes));
    if (complete_bytes[i] > maximum_bytes)
      offer.rejection = "contraction binding exceeds remaining complete budget";
    if (offer.rejection.empty() &&
        runtime::strict_requested_precision(request, request.precisions[offer.precision])) {
      if (offer.provider == "cublas") library = i;
      if (offer.provider == "generated.cuda") generated = i;
    }
#if defined(GENERATIVEQC_TEST_HOOKS)
    if (library_available && reservation.host_bytes) {
      offer.cost.source = "test-only-provider-ranking";
      offer.cost.prepare_ns = offer.cost.cast_ns = offer.cost.pack_ns = 0;
      offer.cost.refinement_ns = offer.cost.audit_ns = offer.cost.fallback_ns = 0;
      offer.cost.kernel_ns = offer.provider == "cutensor" ? 1 : 100;
    }
#endif
  }
  if (!generated) throw std::length_error("contraction bindings exceed complete budget");
  const auto decision = runtime::select_native_lowering(request, offers, target, compilation, 1,
                                                        library ? library : generated);
  const auto i = decision.selected;
  const auto algorithm = offers[i].provider == "cutensor" ? ContractionAlgorithm::CutensorAffine
                         : offers[i].provider == "cublas" ? ContractionAlgorithm::PedanticBlas
                                                          : ContractionAlgorithm::GeneratedOrdered;
  return {algorithm, reservation, i, complete_bytes[i], decision.retained_incumbent};
}

}  // namespace generativeqc::tensor
