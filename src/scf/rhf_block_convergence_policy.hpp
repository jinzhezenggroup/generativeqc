#pragma once

namespace generativeqc::scf::cuda_policy {

/** Private default-off matrix-walk selection, frozen before graph admission. */
enum class RhfBlockConvergenceRequest { Off, On, Invalid };
RhfBlockConvergenceRequest parse_rhf_block_convergence_request(const char* value) noexcept;
RhfBlockConvergenceRequest rhf_block_convergence_requested() noexcept;
bool resolve_rhf_block_convergence(RhfBlockConvergenceRequest request, bool unrestricted,
                                   bool mixed_precision, bool incremental, bool quartet_direct,
                                   bool physical_reference) noexcept;

}  // namespace generativeqc::scf::cuda_policy
