#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "runtime/execution_precision.hpp"

namespace generativeqc::runtime {

/** Native projection of the compiler's lowering contracts, not a scientific IR.
 * A generated adapter supplies canonical template/candidate identities, admitted
 * precision variants and resolved physical facts. Backend preparation supplies
 * target/version capability facts. Method controllers do not name providers.
 * All selection and allocation here are preparation-time, bounded operations.
 */
inline constexpr std::string_view kLoweringBindingSchema =
    "generativeqc.compiler.lowering-binding.v1";

enum class LoweringDeterminism : std::uint8_t { Unspecified, Reproducible, ExactOrder };

struct NativeLoweringPrecision {
  std::string_view identity;
  PrecisionDirective arithmetic;
  std::array<PrecisionDtype, 4> input_dtypes{};
  std::size_t inputs{};
  PrecisionDtype publication_dtype{PrecisionDtype::Fp64};
  // Identities reference compiler-owned cast/refinement/audit obligations.
  // A provider must implement the entire variant; a cast is not implicit.
  std::string_view casts, refinement, audit;
};

struct NativeLoweringConstraints {
  std::optional<std::size_t> workspace_bytes, provider_bytes, additional_device_bytes, host_bytes;
  LoweringDeterminism determinism{LoweringDeterminism::Unspecified};
  bool capture_required{};
  std::size_t maximum_candidates{256};
};

struct NativeLoweringRequest {
  // Complete canonical request identity includes precision admission and effects;
  // semantic identity alone only identifies the underlying operation.
  std::string_view scientific_identity, semantic_identity, identity;
  PrecisionDtype dtype{PrecisionDtype::Fp64}, accumulation_dtype{PrecisionDtype::Fp64};
  std::array<PrecisionDtype, 4> input_dtypes{};
  std::size_t inputs{};
  std::span<const NativeLoweringPrecision> precisions;
  NativeLoweringConstraints constraints;
};

struct NativeLoweringCost {
  std::string_view source;
  bool measured{};
  // Same non-overlapping phases as common.lowering_contract.LoweringCost.
  // Missing evidence is nullopt, not a free phase. Fused work stays in kernel.
  std::optional<std::uint64_t> prepare_ns, kernel_ns, cast_ns, pack_ns;
  std::optional<std::uint64_t> refinement_ns, audit_ns, fallback_ns;
  std::uint64_t cast_bytes{}, pack_bytes{}, refinement_bytes{}, audit_bytes{}, launches{};
};

struct NativeLoweringCandidate {
  std::string_view identity, semantic_identity, request_identity, precision_identity;
  std::string_view provider, provider_version, algorithm, layout_identity, fusion_identity;
  std::string_view target_identity, compilation_identity;
  std::size_t precision{};
  NativeLoweringCost cost;
  std::size_t workspace_bytes{}, provider_bytes{}, temporary_bytes{}, cache_bytes{}, host_bytes{};
  LoweringDeterminism determinism{LoweringDeterminism::Unspecified};
  bool capture_safe{}, capabilities_available{};
  // Unavailable/unsupported offers remain present as explicit negative evidence.
  std::string_view rejection;
};

struct NativeLoweringRejection {
  std::size_t candidate;
  std::string_view reason;
};

struct NativeLoweringDecision {
  std::size_t selected{};
  std::vector<std::size_t> fallbacks;
  std::vector<NativeLoweringRejection> rejections;
  bool retained_incumbent{};
};

inline bool lowering_digest(std::string_view value) noexcept {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

inline bool lowering_dtype(PrecisionDtype value) noexcept {
  return value == PrecisionDtype::Fp32 || value == PrecisionDtype::Fp64;
}

inline std::uint64_t lowering_add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    throw std::overflow_error("native lowering resource/cost overflow");
  return a + b;
}

inline std::uint64_t lowering_multiply(std::uint64_t a, std::uint64_t b) {
  if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
    throw std::overflow_error("native lowering cost overflow");
  return a * b;
}

inline bool strict_requested_precision(const NativeLoweringRequest& request,
                                       const NativeLoweringPrecision& precision) {
  return precision.arithmetic.storage_dtype == request.dtype &&
         precision.arithmetic.compute_dtype == request.dtype &&
         precision.arithmetic.accumulation_dtype == request.accumulation_dtype &&
         precision.inputs == request.inputs &&
         std::equal(request.input_dtypes.begin(), request.input_dtypes.begin() + request.inputs,
                    precision.input_dtypes.begin());
}

/** Select a complete candidate and same-precision/strict fallbacks.
 * This mirrors common.lowering_selection: complete prepare + expected replay
 * cost ranks candidates; lexical compiler identities break ties, never registry
 * order. Runtime shape compatibility and executable lifetime stay with the
 * typed backend binding. No kernel, heuristic, allocation retry or tuning runs.
 * An optional already-qualified incumbent is retained if its cost is unknown;
 * legality and a strict fallback are still required. This explicit compatibility
 * path never treats unknown work as free or claims a performance promotion.
 */
inline NativeLoweringDecision select_native_lowering(
    const NativeLoweringRequest& request, std::span<const NativeLoweringCandidate> candidates,
    std::string_view target_identity, std::string_view compilation_identity,
    std::uint64_t expected_replays = 1,
    std::optional<std::size_t> qualified_incumbent = std::nullopt) {
  constexpr std::size_t maximum = 256;
  if (!expected_replays || !request.constraints.maximum_candidates ||
      candidates.size() > std::min(maximum, request.constraints.maximum_candidates) ||
      request.precisions.empty() || request.precisions.size() > 16 ||
      request.inputs > request.input_dtypes.size() ||
      !lowering_digest(request.scientific_identity) ||
      !lowering_digest(request.semantic_identity) || !lowering_digest(request.identity) ||
      !lowering_digest(target_identity) || !lowering_digest(compilation_identity) ||
      !lowering_dtype(request.dtype) || !lowering_dtype(request.accumulation_dtype) ||
      request.constraints.determinism > LoweringDeterminism::ExactOrder ||
      (qualified_incumbent && *qualified_incumbent >= candidates.size()))
    throw std::invalid_argument("invalid or unbounded native lowering request");
  for (std::size_t i = 0; i != request.inputs; ++i)
    if (!lowering_dtype(request.input_dtypes[i]))
      throw std::invalid_argument("invalid native lowering input dtype");
  for (std::size_t i = 0; i != request.precisions.size(); ++i) {
    const auto& precision = request.precisions[i];
    const auto& arithmetic = precision.arithmetic;
    if (!lowering_digest(precision.identity) || precision.inputs != request.inputs ||
        precision.publication_dtype != request.dtype || !lowering_dtype(arithmetic.storage_dtype) ||
        !lowering_dtype(arithmetic.compute_dtype) ||
        !lowering_dtype(arithmetic.accumulation_dtype) ||
        arithmetic.math_mode != kStrictPrecisionMathMode ||
        (!strict_requested_precision(request, precision) && arithmetic.qualification.empty()) ||
        (arithmetic.storage_dtype != precision.publication_dtype && precision.casts.empty()) ||
        (!precision.casts.empty() && !lowering_digest(precision.casts)))
      throw std::invalid_argument("invalid native lowering precision contract");
    for (std::size_t input = 0; input != precision.inputs; ++input)
      if (!lowering_dtype(precision.input_dtypes[input]))
        throw std::invalid_argument("invalid native lowering precision input dtype");
    for (std::size_t j = 0; j != i; ++j)
      if (request.precisions[j].identity == precision.identity)
        throw std::invalid_argument("duplicate native lowering precision identity");
  }
  NativeLoweringDecision result;
  result.fallbacks.reserve(candidates.size());
  result.rejections.reserve(candidates.size());
  struct Ranked {
    std::uint64_t cost;
    std::size_t candidate;
  };
  std::array<Ranked, maximum> ranked{};
  std::array<bool, maximum> legal{}, complete{};
  std::size_t count{};
  bool has_strict = false;
  const auto exceeds = [](std::size_t bytes, const std::optional<std::size_t>& limit) {
    return limit && bytes > *limit;
  };
  for (std::size_t i = 0; i != candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    if (candidate.request_identity != request.identity ||
        candidate.semantic_identity != request.semantic_identity ||
        candidate.precision >= request.precisions.size() || !lowering_digest(candidate.identity) ||
        candidate.determinism > LoweringDeterminism::ExactOrder)
      throw std::invalid_argument("native candidates must consume the same admitted request");
    // The index is only a lookup aid: it cannot reinterpret another variant.
    // Offers without typed execution remain explicit negative evidence.
    if ((!candidate.precision_identity.empty() || candidate.rejection.empty()) &&
        candidate.precision_identity != request.precisions[candidate.precision].identity)
      throw std::invalid_argument("native candidate precision identity differs from the request");
    for (std::size_t j = 0; j != i; ++j)
      if (candidates[j].identity == candidate.identity)
        throw std::invalid_argument("duplicate native lowering candidate identity");
    auto rejection = candidate.rejection;
    const auto& precision = request.precisions[candidate.precision];
    const auto& cost = candidate.cost;
    if (rejection.empty() && (candidate.target_identity != target_identity ||
                              candidate.compilation_identity != compilation_identity))
      rejection = "candidate context differs from the binding target/compiler";
    if (rejection.empty() && (candidate.provider.empty() || candidate.provider_version.empty() ||
                              candidate.algorithm.empty() || candidate.layout_identity.empty()))
      rejection = "provider algorithm/layout/version identity is unavailable";
    if (rejection.empty() && !candidate.capabilities_available)
      rejection = "required provider capability is unavailable";
    if (rejection.empty() && request.constraints.capture_required && !candidate.capture_safe)
      rejection = "candidate does not implement capture/replay";
    if (rejection.empty() && candidate.determinism < request.constraints.determinism)
      rejection = "candidate does not implement the required reduction order";
    if (rejection.empty() &&
        (exceeds(candidate.workspace_bytes, request.constraints.workspace_bytes) ||
         exceeds(candidate.provider_bytes, request.constraints.provider_bytes) ||
         exceeds(candidate.host_bytes, request.constraints.host_bytes) ||
         exceeds(lowering_add(lowering_add(candidate.workspace_bytes, candidate.provider_bytes),
                              lowering_add(candidate.temporary_bytes, candidate.cache_bytes)),
                 request.constraints.additional_device_bytes)))
      rejection = "candidate exceeds the simultaneous resource budget";
    legal[i] = rejection.empty();
    complete[i] = !cost.source.empty() && cost.prepare_ns && cost.kernel_ns && cost.cast_ns &&
                  cost.pack_ns && cost.refinement_ns && cost.audit_ns && cost.fallback_ns;
    if (rejection.empty() &&
        (cost.source.empty() || !cost.prepare_ns || !cost.kernel_ns || !cost.cast_ns ||
         !cost.pack_ns || !cost.refinement_ns || !cost.audit_ns || !cost.fallback_ns))
      rejection = "complete prepare/cast/pack/kernel/refinement/audit/fallback cost is unavailable";
    if (!rejection.empty()) {
      result.rejections.push_back({i, rejection});
      continue;
    }
    const auto replay = lowering_add(lowering_add(lowering_add(*cost.kernel_ns, *cost.cast_ns),
                                                  lowering_add(*cost.pack_ns, *cost.refinement_ns)),
                                     lowering_add(*cost.audit_ns, *cost.fallback_ns));
    ranked[count++] = {lowering_add(*cost.prepare_ns, lowering_multiply(expected_replays, replay)),
                       i};
    has_strict = has_strict || strict_requested_precision(request, precision);
  }
  if (qualified_incumbent) {
    const auto incumbent = *qualified_incumbent;
    if (!legal[incumbent])
      throw std::invalid_argument("qualified incumbent fails binding admission");
    if (!complete[incumbent]) {
      bool strict = false;
      result.selected = incumbent;
      result.retained_incumbent = true;
      const auto selected = candidates[incumbent].precision;
      for (std::size_t i = 0; i != candidates.size(); ++i) {
        if (!legal[i]) continue;
        const auto precision = candidates[i].precision;
        const bool is_strict = strict_requested_precision(request, request.precisions[precision]);
        strict = strict || is_strict;
        if (i != incumbent && (precision == selected || is_strict)) result.fallbacks.push_back(i);
      }
      if (!strict)
        throw std::invalid_argument("incumbent retention requires a legal strict candidate");
      std::sort(result.fallbacks.begin(), result.fallbacks.end(),
                [&](auto a, auto b) { return candidates[a].identity < candidates[b].identity; });
      return result;
    }
  }
  if (!count || !has_strict)
    throw std::invalid_argument(
        "native lowering requires a complete strict requested-precision candidate");
  std::sort(ranked.begin(), ranked.begin() + count, [&](const auto& a, const auto& b) {
    return a.cost != b.cost ? a.cost < b.cost
                            : candidates[a.candidate].identity < candidates[b.candidate].identity;
  });
  result.selected = ranked[0].candidate;
  const auto& selected_precision = request.precisions[candidates[result.selected].precision];
  for (std::size_t i = 1; i != count; ++i) {
    const auto candidate = ranked[i].candidate;
    const auto& precision = request.precisions[candidates[candidate].precision];
    if (precision.identity == selected_precision.identity ||
        strict_requested_precision(request, precision))
      result.fallbacks.push_back(candidate);
  }
  return result;
}

}  // namespace generativeqc::runtime
