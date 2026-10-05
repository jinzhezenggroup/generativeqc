#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace generativeqc::runtime {

/** Native projection of common.precision.ExecutionPrecisionSchedule.
 *
 * Keep the schema and arithmetic-mode tokens byte-identical to the Python
 * compiler contract. Method controllers own numerical qualification and decide
 * which regions are admitted; this record only carries execution semantics.
 */
inline constexpr std::string_view kExecutionPrecisionSchema =
    "generativeqc.compiler.execution-precision.v1";
inline constexpr std::string_view kStrictPrecisionMathMode = "ieee-rn-no-tf32";

/** The method requests an admitted iteration or the strict publication audit;
 * implementation arithmetic is resolved once by the prepared binding. */
enum class PrecisionPhase : std::uint8_t { Admitted, StrictAudit };

enum class PrecisionDtype : std::uint8_t { Fp32, Fp64 };

struct PrecisionDirective {
  PrecisionDtype storage_dtype{PrecisionDtype::Fp64};
  PrecisionDtype compute_dtype{PrecisionDtype::Fp64};
  PrecisionDtype accumulation_dtype{PrecisionDtype::Fp64};
  std::string_view qualification{};
  std::string_view math_mode{kStrictPrecisionMathMode};

  constexpr bool is_strict_fp64() const noexcept {
    return storage_dtype == PrecisionDtype::Fp64 && compute_dtype == PrecisionDtype::Fp64 &&
           accumulation_dtype == PrecisionDtype::Fp64;
  }
};

inline constexpr PrecisionDirective strict_fp64_precision() noexcept { return {}; }

inline PrecisionDirective fp32_compute_fp64_accumulation(std::string_view qualification) {
  if (qualification.empty())
    throw std::invalid_argument("mixed execution precision requires qualification");
  return {PrecisionDtype::Fp64, PrecisionDtype::Fp32, PrecisionDtype::Fp64, qualification,
          kStrictPrecisionMathMode};
}

struct PrecisionRegion {
  std::string_view name{};
  PrecisionDirective directive{};
};

class ExecutionPrecisionSchedule {
 public:
  static constexpr std::size_t kMaximumRegions = 16;

  void add_region(std::string_view name, PrecisionDirective directive) {
    if (name.empty()) throw std::invalid_argument("precision region requires a name");
    if (directive.math_mode != kStrictPrecisionMathMode)
      throw std::invalid_argument("unsupported native precision arithmetic mode");
    if (!directive.is_strict_fp64() && directive.qualification.empty())
      throw std::invalid_argument("lower-precision region requires qualification");
    if (find(name) != nullptr) throw std::invalid_argument("duplicate precision region");
    if (size_ == regions_.size()) throw std::length_error("precision region capacity exceeded");
    regions_[size_++] = {name, directive};
  }

  const PrecisionDirective* find(std::string_view name) const noexcept {
    for (std::size_t index = 0; index != size_; ++index)
      if (regions_[index].name == name) return &regions_[index].directive;
    return nullptr;
  }

  bool uses_lower_precision(std::string_view name) const noexcept {
    const auto* directive = find(name);
    return directive != nullptr && !directive->is_strict_fp64();
  }

  bool any_lower_precision() const noexcept {
    for (std::size_t index = 0; index != size_; ++index)
      if (!regions_[index].directive.is_strict_fp64()) return true;
    return false;
  }

  bool is_strict_fp64() const noexcept { return !any_lower_precision(); }

  /** Intersect admitted arithmetic with an execution capability without
   * discarding region identity, retained directives or the strict audit policy.
   * The predicate sees only lower-precision regions; rejection restores that
   * region to the common FP64 fallback. This cannot introduce lower precision
   * or mutate the reusable requested schedule. Numerical qualification and
   * provider/layout capability remain the caller's responsibility. */
  template <typename Qualified>
  ExecutionPrecisionSchedule filter_lower_precision(Qualified&& qualified) const {
    auto execution = *this;
    for (std::size_t index = 0; index != size_; ++index) {
      const auto& region = regions_[index];
      if (!region.directive.is_strict_fp64() && !qualified(region))
        execution.regions_[index].directive = strict_fp64_precision();
    }
    return execution;
  }

  std::size_t size() const noexcept { return size_; }
  PrecisionDtype strict_audit_dtype() const noexcept { return strict_audit_dtype_; }
  std::string_view audit_owner() const noexcept { return audit_owner_; }
  std::string_view math_mode() const noexcept { return math_mode_; }

 private:
  std::array<PrecisionRegion, kMaximumRegions> regions_{};
  std::size_t size_{};
  PrecisionDtype strict_audit_dtype_{PrecisionDtype::Fp64};
  std::string_view audit_owner_{"method-controller"};
  std::string_view math_mode_{kStrictPrecisionMathMode};
};

}  // namespace generativeqc::runtime
