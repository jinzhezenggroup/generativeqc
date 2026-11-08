#pragma once

namespace generativeqc::cc {

/** Diagnostic-only exclusions for matched DF contraction qualification.
 * Production callers leave this empty: compiler shapes, complete storage and
 * prepared Tensor capabilities admit each region independently. Exclusions
 * retain the original ordered graph; they never promote a provider, change
 * precision, reuse primal residuals or relax a scientific acceptance gate.
 */
struct DFContractionQualification {
  bool ordered_iteration{};
  bool ordered_replay{};
  bool ordered_response{};
  bool ordered_audit{};
  bool ordered_primal{};
};

}  // namespace generativeqc::cc
