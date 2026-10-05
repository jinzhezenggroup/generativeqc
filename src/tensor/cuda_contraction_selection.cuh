#pragma once

#include "runtime/lowering_binding.hpp"
#include "tensor/cuda_contraction.cuh"

namespace generativeqc::tensor {

/** Evidence for one compiled AOT family. The profile must qualify simultaneous
 * per-plan host/module bounds for these exact artifact bytes and header version.
 * No production profile is installed; a build option alone is insufficient. */
struct CutlassRegionQualification {
  ContractionProviderReservation reservation;
  std::string_view artifact;
  std::size_t version{};
};

#if defined(GENERATIVEQC_TEST_HOOKS)
// Qualification can remove optional implementations at the provider boundary.
// This is absent from production and is not a scientific method option.
inline thread_local bool contraction_libraries_unavailable_for_test = false;
inline thread_local CutlassRegionQualification cutlass_region_qualification_for_test;
#endif

inline constexpr std::string_view kCutlassRegionUnavailable =
    "CUTLASS region requires a context-lifetime retention owner";

inline CutlassRegionQualification qualified_cutlass_region() noexcept {
  // This region and its provider context are call-local. A failed constructor
  // destroys the table while CUDA can retain unmeasured modules. Neither a
  // resource profile nor test hooks establish an owner across subsequent calls.
  // Keep this capability closed until that context/build lifetime is explicit.
  return {};
}

inline std::size_t cutlass_provider_version() noexcept {
#if GENERATIVEQC_HAS_CUTLASS
  return CUTLASS_VERSION;
#else
  return 0;
#endif
}

/** One provider implementation of a compiler-owned contraction region. The
 * region owns numerical conversions/traversal; the provider owns all handles
 * and immutable per-operation plans. Resource bounds cover simultaneous plans,
 * including provisional preparation before an optional fallback. */
struct ContractionRegionPlan {
  ContractionAlgorithm algorithm{};
  ContractionProviderReservation reservation;
  std::size_t selected{}, binding_bytes{};
  bool retained_incumbent{};
  // Own the digest: the qualification record may be temporary after admission.
  std::array<char, 64> artifact_identity{};
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
  const auto cutensor_reservation = qualified_cutensor_reservation();
  const auto cublaslt_reservation = qualified_cublaslt_reservation();
  const auto cutlass = qualified_cutlass_region();
  std::array<ContractionProviderReservation, N> reservations{};
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
    } else if (offer.provider == "cutensor" || offer.provider == "cublaslt") {
      const auto reservation =
          offer.provider == "cutensor" ? cutensor_reservation : cublaslt_reservation;
      reservations[i] = reservation;
      offer.workspace_bytes = contraction_product(plans, reservation.workspace_bytes);
      offer.provider_bytes = contraction_product(plans, reservation.provider_bytes);
      offer.host_bytes = ContractionProviderReservation::checked_add(
          host_bytes, contraction_product(plans, reservation.host_bytes));
      const auto version =
          offer.provider == "cutensor" ? cutensor_provider_version() : cublaslt_provider_version();
      const bool version_available =
          offer.provider == "cutensor" ? version >= 20800 && version / 10000 == 2 : version != 0;
      if (!reservation.host_bytes || !version_available)
        offer.rejection = "qualified optional contraction resource profile unavailable";
    } else if (offer.provider == "cutlass-aot") {
      const auto& reservation = cutlass.reservation;
      reservations[i] = reservation;
      offer.workspace_bytes = offer.provider_bytes = 0;
      offer.cache_bytes = contraction_product(plans, reservation.cache_bytes);
      offer.host_bytes = ContractionProviderReservation::checked_add(
          host_bytes, contraction_product(plans, reservation.host_bytes));
      if (!reservation.host_bytes || !reservation.cache_bytes || reservation.workspace_bytes ||
          reservation.provider_bytes || !runtime::lowering_digest(cutlass.artifact) ||
          !cutlass.version || cutlass.version != cutlass_provider_version())
        offer.rejection = kCutlassRegionUnavailable;
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
    if (library_available && (cutensor_reservation.host_bytes || cublaslt_reservation.host_bytes ||
                              cutlass.reservation.host_bytes)) {
      offer.cost.source = "test-only-provider-ranking";
      offer.cost.prepare_ns = offer.cost.cast_ns = offer.cost.pack_ns = 0;
      offer.cost.refinement_ns = offer.cost.audit_ns = offer.cost.fallback_ns = 0;
      // Qualification expects an explicit generated fallback ranking. Equal
      // synthetic scores would let unrelated candidate hashes break the tie.
      offer.cost.kernel_ns = reservations[i].host_bytes           ? 1
                             : offer.provider == "generated.cuda" ? 100
                                                                  : 200;
    }
#endif
  }
  if (!generated) throw std::length_error("contraction bindings exceed complete budget");
  const auto decision = runtime::select_native_lowering(request, offers, target, compilation, 1,
                                                        library ? library : generated);
  const auto i = decision.selected;
  const auto algorithm = offers[i].provider == "cutensor"   ? ContractionAlgorithm::CutensorAffine
                         : offers[i].provider == "cublaslt" ? ContractionAlgorithm::CublasLtMatmul
                         : offers[i].provider == "cutlass-aot" ? ContractionAlgorithm::CutlassAot
                         : offers[i].provider == "cublas" ? ContractionAlgorithm::PedanticBlas
                                                          : ContractionAlgorithm::GeneratedOrdered;
  ContractionRegionPlan result{algorithm, reservations[i], i, complete_bytes[i],
                               decision.retained_incumbent};
  if (algorithm == ContractionAlgorithm::CutlassAot)
    std::copy(cutlass.artifact.begin(), cutlass.artifact.end(), result.artifact_identity.begin());
  return result;
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
    // Reject forged/stale admission too, before context setup or module loading.
    if (plan_.algorithm == ContractionAlgorithm::CutlassAot)
      throw std::invalid_argument(std::string(kCutlassRegionUnavailable));
    auto minimum = storage_bytes(requests);
    if (!requests || std::any_of(shape.begin(), shape.end(), [](auto n) { return n == 0; }))
      throw std::invalid_argument("empty prepared contraction region");
    if (plan_.algorithm == ContractionAlgorithm::PedanticBlas)
      minimum = ContractionProviderReservation::checked_add(
          minimum, CudaContractionContext::kProviderAllowance);
    else if (plan_.algorithm == ContractionAlgorithm::CutensorAffine ||
             plan_.algorithm == ContractionAlgorithm::CublasLtMatmul ||
             plan_.algorithm == ContractionAlgorithm::CutlassAot)
      minimum = ContractionProviderReservation::checked_add(
          minimum, plan_.reservation.total_bytes(requests));
    else if (plan_.algorithm != ContractionAlgorithm::GeneratedOrdered)
      throw std::invalid_argument("invalid prepared contraction region algorithm");
    if (minimum > plan_.binding_bytes)
      throw std::length_error("prepared contraction region exceeds binding reservation");
    const auto fallback = [&] {
      // Descriptor cleanup does not free CUDA modules. Admit fallback against
      // the unconsumed budget and retain those bytes in its complete binding.
      const auto cache = table_.optional_resources().cache_bytes;
      const auto admitted_bytes = plan_.binding_bytes;
      if (cache > plan_.binding_bytes)
        throw std::length_error("retained contraction modules exceed fallback budget");
      plan_ =
          select_contraction_region(request, candidates, target, compilation, requests,
                                    storage_bytes(requests), plan_.binding_bytes - cache, false);
      plan_.binding_bytes = ContractionProviderReservation::checked_add(plan_.binding_bytes, cache);
      // Preserve the preparation ceiling when a partial AOT load occurred:
      // descriptors and modules coexisted before cleanup. Final live storage
      // alone is not a complete bound for this preparation-and-replay endpoint.
      if (cache) plan_.binding_bytes = std::max(plan_.binding_bytes, admitted_bytes);
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
                 std::vector<ContractionAlgorithm>(requests, plan_.algorithm), plan_.reservation,
                 std::string_view(plan_.artifact_identity.data(), plan_.artifact_identity.size()));
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
    const auto resources = table_.optional_resources();
    return ContractionProviderReservation::checked_add(
        context_.retained_bytes(), ContractionProviderReservation::checked_add(
                                       resources.provider_bytes, resources.cache_bytes));
  }
  const ContractionRegionPlan& selected() const noexcept { return plan_; }
  std::size_t provider_version() const noexcept {
    return plan_.algorithm == ContractionAlgorithm::CutensorAffine   ? cutensor_provider_version()
           : plan_.algorithm == ContractionAlgorithm::CublasLtMatmul ? cublaslt_provider_version()
           : plan_.algorithm == ContractionAlgorithm::CutlassAot
               ? cutlass_provider_version()
               : std::size_t(context_.provider_version());
  }

 private:
  ContractionRegionPlan plan_;
  std::array<std::size_t, 3> shape_;
  CudaContractionContext context_;
  PreparedContractions table_;
};

}  // namespace generativeqc::tensor
