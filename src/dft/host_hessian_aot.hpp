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
};

std::span<const MethodBinding> method_bindings() noexcept;
const MethodBinding* find_method(std::string_view identity) noexcept;

}  // namespace generativeqc::dft::host_hessian_aot
