#pragma once

#include "runtime/lowering_binding.hpp"
#include "tensor/native_contraction.hpp"

namespace generativeqc::tensor {

/** One compiler site and its resolved physical view. Template identities and
 * candidate strings borrow immutable generated storage, never caller buffers. */
template <std::size_t Offers>
struct ContractionSite {
  const runtime::NativeLoweringRequest& request;
  const std::array<runtime::NativeLoweringCandidate, Offers>& candidates;
  std::string_view target, compilation;
  ContractionRequest resolved;
  // Optional compiler-owned input: one scalar per leading batch. It is part of
  // the canonical region, not a method-selected implementation epilogue.
  ContractionOperand batch_scale{};
};

/** Per-operation execution provenance; host/device reservations belong to the
 * shared owner, so four sites using one provider do not charge four handles. */
struct ContractionSiteDiagnostic {
  ContractionRequest resolved;
  runtime::NativeLoweringCandidate candidate;
  std::string_view alternative_rejection;
  std::size_t calls{}, summands{};
  // Keep every bounded offer, including unsupported optional implementations.
  std::array<runtime::NativeLoweringCandidate, 5> offers{};
  std::size_t offer_count{}, selected{};
  bool retained_incumbent{};
  ContractionOperand batch_scale{};
  std::size_t scaled_elements{}, publication_passes{};
};

}  // namespace generativeqc::tensor
