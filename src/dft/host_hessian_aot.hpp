#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace generativeqc::dft::host_hessian_aot {

/** Allocation-free linked CPU scalar AD, not a complete Hessian capability.
 * Panels are input/root-major with contiguous point lanes. Callers own both
 * panels, must avoid aliasing them, and discard the complete output on a nonzero
 * status. Negative status refuses ABI admission; positive status identifies a
 * nonfinite output root. No Python, compiler, device or reference engine runs.
 */
using Evaluate = int (*)(const double*, std::size_t, double*, std::size_t, std::size_t) noexcept;

struct Program {
  std::string_view name;
  std::string_view identity;
  std::span<const std::string_view> inputs;
  std::size_t outputs{};
  Evaluate evaluate{};
};

/** Exact scientific identity and ABI metadata accompany every linked leaf. */
std::span<const Program> programs() noexcept;
/** Missing name/identity pairs never request runtime compilation or fallback. */
const Program* find(std::string_view name, std::string_view identity) noexcept;

/** The physical point provider is part of an exact method binding, not a fallback
 * when a linked scalar leaf happens to report a numerical error. */
enum class PointDomain {
  linked_ad,
  scaled_pbe0_rks,
  lda_rks,
  pbe_rks,
  scaled_pbe0_uks,
  pbe_uks,
  lda_uks,
};

/** Physical fields have four Cartesian components per density block. */
constexpr std::size_t physical_spins(PointDomain domain) noexcept {
  if (domain == PointDomain::linked_ad) return 0;
  return domain == PointDomain::scaled_pbe0_uks || domain == PointDomain::pbe_uks ||
                 domain == PointDomain::lda_uks
             ? 2
             : 1;
}

/** The canonical semilocal bindings use unit XC scales and no Fock exchange. */
constexpr bool semilocal_only(PointDomain domain) noexcept {
  return domain == PointDomain::lda_rks || domain == PointDomain::pbe_rks ||
         domain == PointDomain::pbe_uks || domain == PointDomain::lda_uks;
}

/** One complete fixed-grid scalar closure bound to native KS model metadata.
 * This authenticates composition/spin and linked programs, not a stationary
 * state, integral provider or complete derivative endpoint. The optional
 * field_response_identity may be empty; nuclear consumers must additionally
 * require that exact linked leaf rather than inferring it from fixed-grid AD.
 */
struct MethodBinding {
  std::string_view identity;
  std::uint32_t functional{}, scf_domain{};
  std::size_t spins{};
  double exact_exchange{};
  std::string_view xc_identity, coefficient_response_identity;
  std::string_view field_response_identity;
  /** Optional independent frozen-energy mixed closure; not response or assembly. */
  std::string_view energy_mixed_identity;
  PointDomain point_domain{PointDomain::linked_ad};
};

std::span<const MethodBinding> method_bindings() noexcept;
const MethodBinding* find_method(std::string_view identity) noexcept;

}  // namespace generativeqc::dft::host_hessian_aot
