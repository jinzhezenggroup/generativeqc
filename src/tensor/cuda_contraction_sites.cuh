#pragma once

#include <chrono>

#include "tensor/contraction_sites.hpp"
#include "tensor/cuda_contraction.cuh"

namespace generativeqc::tensor {

#if defined(GENERATIVEQC_TEST_HOOKS)
// Qualification only: each bit supplies evidence for one compiler site. There
// is no production method option or shape heuristic that names a provider.
inline thread_local unsigned contraction_sites_qualification_for_test = 0;
inline thread_local bool contraction_sites_unavailable_for_test = false;
#endif

/** Independently selected semantic sites sharing one stream/provider owner.
 * All exact full/tail descriptors are prepared together. Iteration changes only
 * borrowed addresses and a fixed slot; it performs no allocation or selection.
 * The generated incumbent remains until complete endpoint evidence exists. */
template <std::size_t Sites>
class PreparedContractionSites {
 public:
  static_assert(Sites > 0 && Sites <= 8);
  static constexpr std::size_t host_reservation = (64U + 16U * Sites) << 10;

  template <std::size_t Offers>
  PreparedContractionSites(const std::array<ContractionSite<Offers>, Sites>& sites,
                           cudaStream_t stream, std::size_t device_budget) {
    static_assert(Offers > 0 && Offers <= 5);
    static_assert(sizeof(PreparedContractionSites) + 3 * sizeof(sites) +
                      2 * PreparedContractions::storage_bytes(Sites) + 8192 <=
                  host_reservation);
    const auto started = std::chrono::steady_clock::now();
    // Reject malformed descriptors before optional resource acquisition. The
    // donated output of beta=1 is represented in the canonical SSA request.
    for (const auto& site : sites) {
      site.resolved.validate();
      const auto& r = site.request;
      if (site.resolved.scientific_identity != r.scientific_identity ||
          site.resolved.semantic_template_identity != r.semantic_identity ||
          r.inputs != (site.resolved.beta == 0 ? 2U : 3U) ||
          (site.resolved.beta != 0 && site.resolved.beta != 1) ||
          !site.resolved.precision.is_strict_fp64() ||
          site.resolved.publication_dtype != PrecisionDtype::Fp64 || r.precisions.size() != 1 ||
          !r.precisions[0].arithmetic.is_strict_fp64() || !r.precisions[0].casts.empty() ||
          !r.precisions[0].refinement.empty() || !r.precisions[0].audit.empty() ||
          site.resolved.precision_identity != r.precisions[0].identity)
        throw std::invalid_argument("site differs from its strict canonical contraction");
    }
    std::vector<ContractionAlgorithm> algorithms(Sites);
    const auto select = [&](bool available) {
      bool needs_provider = false;
      for (std::size_t slot = 0; slot < Sites; ++slot) {
        const auto& site = sites[slot];
        auto offers = site.candidates;
        std::optional<std::size_t> incumbent;
        std::string_view rejection;
        bool qualified = false;
#if defined(GENERATIVEQC_TEST_HOOKS)
        qualified = (contraction_sites_qualification_for_test & (1U << slot)) != 0;
        available = available && !contraction_sites_unavailable_for_test;
#endif
        for (std::size_t i = 0; i < Offers; ++i) {
          auto& offer = offers[i];
          offer.host_bytes = host_reservation;
          if (offer.provider == "generated.cuda") {
            incumbent = i;
          } else if (offer.provider == "cublas") {
            offer.provider_bytes = CudaContractionContext::kProviderAllowance;
            if (!qualified)
              offer.rejection = "complete site endpoint profile is not qualified";
            else if (!available)
              offer.rejection = "shared provider preparation unavailable";
            else if (device_budget < offer.provider_bytes)
              offer.rejection = "shared provider reservation exceeds device budget";
            rejection = offer.rejection;
            if (offer.rejection.empty()) {
              offer.cost.source = "test-only-site-qualification";
              offer.cost.prepare_ns = offer.cost.cast_ns = offer.cost.pack_ns = 0;
              offer.cost.kernel_ns = offer.cost.refinement_ns = offer.cost.audit_ns = 0;
              offer.cost.fallback_ns = 0;
            }
          } else {
            offer.rejection = "site executor lacks a qualified optional plan reservation";
          }
        }
        if (!incumbent) throw std::invalid_argument("site has no generated fallback");
        // Explicit qualification preserves the common selector's cost rules:
        // compare complete finite scores, never known versus unknown cost.
        if (qualified) {
          offers[*incumbent].cost.source = "test-only-site-qualification";
          offers[*incumbent].cost.prepare_ns = offers[*incumbent].cost.cast_ns = 0;
          offers[*incumbent].cost.pack_ns = offers[*incumbent].cost.refinement_ns = 0;
          offers[*incumbent].cost.audit_ns = offers[*incumbent].cost.fallback_ns = 0;
          offers[*incumbent].cost.kernel_ns = 1;
        }
        const auto decision = runtime::select_native_lowering(site.request, offers, site.target,
                                                              site.compilation, 1, incumbent);
        const auto selected = offers[decision.selected];
        const bool library = selected.provider == "cublas";
        algorithms[slot] =
            library ? ContractionAlgorithm::PedanticBlas : ContractionAlgorithm::GeneratedOrdered;
        diagnostics_[slot] = {site.resolved, selected, rejection};
        std::copy(offers.begin(), offers.end(), diagnostics_[slot].offers.begin());
        diagnostics_[slot].offer_count = Offers;
        diagnostics_[slot].selected = decision.selected;
        diagnostics_[slot].retained_incumbent = decision.retained_incumbent;
        needs_provider |= library;
      }
      return needs_provider;
    };
    bool library = select(true);
    if (library && !context_.prepare(stream)) library = select(false);
    if (!library) context_.prepare_generated(stream);
    provider_bytes_ = library ? CudaContractionContext::kProviderAllowance : 0;
    std::vector<ContractionRequest> requests(Sites);
    for (std::size_t i = 0; i < Sites; ++i) requests[i] = sites[i].resolved;
    table_.add(1, 1, 1, std::move(requests), context_, calls_, summands_, std::move(algorithms));
    prepare_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  }

  void execute(std::size_t slot, cudaStream_t stream, const double* a, const double* b,
               double* output, int* error) {
    if (slot >= Sites) throw std::out_of_range("unknown prepared contraction site");
    auto& diagnostic = diagnostics_[slot];
    const auto next_calls = runtime::lowering_add(diagnostic.calls, 1);
    const auto next_work =
        runtime::lowering_add(diagnostic.summands, diagnostic.resolved.summands());
    table_.execute(slot, 1, 1, 1, stream, a, b, output, error);
    diagnostic.calls = next_calls;
    diagnostic.summands = next_work;
  }
  const auto& diagnostics() const noexcept { return diagnostics_; }
  std::size_t provider_bytes() const noexcept { return provider_bytes_; }
  std::size_t retained_provider_bytes() const noexcept { return context_.retained_bytes(); }
  int provider_version() const noexcept { return context_.provider_version(); }
  int runtime_version() const noexcept { return context_.runtime_version(); }
  double prepare_seconds() const noexcept { return prepare_seconds_; }

 private:
  CudaContractionContext context_;
  PreparedContractions table_;
  std::array<ContractionSiteDiagnostic, Sites> diagnostics_;
  std::size_t calls_{}, summands_{}, provider_bytes_{};
  double prepare_seconds_{};
};

}  // namespace generativeqc::tensor
