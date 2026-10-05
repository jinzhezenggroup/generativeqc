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

/** Provider-owned lifetime of one homogeneous compiler region. The compiler
 * supplies immutable descriptors and a semantic variant key; this layer owns
 * context setup, transactional preparation, same-precision fallback and replay.
 * Factories run only during construction and are never retained for replay. */
class PreparedContractionRegion {
 public:
  static std::size_t storage_bytes(std::size_t requests) {
    const auto base = PreparedContractions::storage_bytes(0);
    const auto per_request = PreparedContractions::storage_bytes(1) - base;
    return ContractionProviderReservation::checked_add(
        sizeof(PreparedContractionRegion), ContractionProviderReservation::checked_add(
                                               base, contraction_product(requests, per_request)));
  }

  template <std::size_t N, class MakeRequests>
  PreparedContractionRegion(ContractionRegionPlan admitted,
                            const runtime::NativeLoweringRequest& request,
                            const std::array<runtime::NativeLoweringCandidate, N>& candidates,
                            std::string_view target, std::string_view compilation,
                            std::size_t requests, std::array<std::size_t, 3> shape,
                            cudaStream_t stream, std::size_t& calls, std::size_t& summands,
                            MakeRequests&& make_requests)
      : plan_(admitted), shape_(shape) {
    auto minimum = storage_bytes(requests);
    if (!requests || std::any_of(shape.begin(), shape.end(), [](auto n) { return n == 0; }))
      throw std::invalid_argument("empty prepared contraction region");
    if (plan_.algorithm == ContractionAlgorithm::PedanticBlas)
      minimum = ContractionProviderReservation::checked_add(
          minimum, CudaContractionContext::kProviderAllowance);
    else if (plan_.algorithm == ContractionAlgorithm::CutensorAffine)
      minimum = ContractionProviderReservation::checked_add(
          minimum, plan_.reservation.total_bytes(requests));
    else if (plan_.algorithm != ContractionAlgorithm::GeneratedOrdered)
      throw std::invalid_argument("invalid prepared contraction region algorithm");
    if (minimum > plan_.binding_bytes)
      throw std::length_error("prepared contraction region exceeds binding reservation");
    const auto fallback = [&] {
      plan_ = select_contraction_region(request, candidates, target, compilation, requests,
                                        storage_bytes(requests), plan_.binding_bytes, false);
    };
    if (plan_.algorithm == ContractionAlgorithm::PedanticBlas) {
      if (!context_.prepare(stream)) {
        context_.prepare_generated(stream);
        fallback();
      }
    } else {
      context_.prepare_generated(stream);
    }
    const auto bind = [&] {
      auto descriptors = make_requests();
      if (descriptors.size() != requests)
        throw std::logic_error("prepared region descriptor count differs from admission");
      table_.add(shape_[0], shape_[1], shape_[2], std::move(descriptors), context_, calls, summands,
                 std::vector<ContractionAlgorithm>(requests, plan_.algorithm), plan_.reservation);
    };
    try {
      bind();
    } catch (const ContractionPreparationUnavailable&) {
      table_.release();
      fallback();
      bind();
    }
  }

  template <class T>
  void execute(std::size_t slot, cudaStream_t stream, const T* a, const T* b, T* c, int* error) {
    table_.execute(slot, shape_[0], shape_[1], shape_[2], stream, a, b, c, error);
  }
  ContractionProviderReservation optional_resources() const { return table_.optional_resources(); }
  std::size_t retained_provider_bytes() const {
    return ContractionProviderReservation::checked_add(context_.retained_bytes(),
                                                       table_.optional_resources().provider_bytes);
  }
  const ContractionRegionPlan& selected() const noexcept { return plan_; }
  std::size_t provider_version() const noexcept {
    return plan_.algorithm == ContractionAlgorithm::CutensorAffine
               ? cutensor_provider_version()
               : std::size_t(context_.provider_version());
  }

 private:
  ContractionRegionPlan plan_;
  std::array<std::size_t, 3> shape_;
  CudaContractionContext context_;
  PreparedContractions table_;
};

}  // namespace generativeqc::tensor
