# Decision: share native CUDA transport and complete stationary source reduction

Status: implemented on the unmerged #2357 branch; real-GPU requalification pending
Date: 2026-10-13

## Problem

The existing MethodIR graph dispatch removed name-based scientific admission,
but every native stationary CUDA profile still emitted identical device copies
of its integral-gradient transport and its final ordered source sum. These are
not XC functional mathematics or Becke/AO derivatives. They do not require a
functional- or spin-specialized kernel.

## Decision

The native C++ force owner now links exactly one CUDA implementation of:

- Existing source transport, including signed force-to-gradient conversion,
  double-channel J/K behavior, missing exchange-slot refusal and finite checks.
- Full ordered source reduction, using the compiler's ProgramIR source ordering.

The compiler explicitly rejects native shared-reduction admission unless the
complete ProgramIR gradient is an ordered unit-coefficient sum. All XC/exact-K
weights remain computed by their original source owners. The shared reduction
keeps the left-to-right FP64 accumulation and nonfinite publication guard; it
does not fuse, reorder or omit a source.

Python stationary AOT callers keep their existing emitted reduction. Native
CUDA profile wrappers no longer emit the reduction kernel, and native transport
calls the single shared launcher. Backend/provider, lifecycle, memory allowance,
fresh final-state token, stream fences and atomic output publication remain
with the original prepared C++ owner.

## Rejected alternatives

- A generic callable CUDA XC/Becke/AO evaluator on this change: these have
  family-specific jet storage and nonlinear point-domain policies; promotion
  without real-GPU independent force and whole-endpoint performance evidence
  risks silent numerical and register-pressure regressions.
- Summing sources on CPU: unnecessary device-to-host movement and an alternate
  scientific publication owner.
- One source-reduction kernel per MethodIR label: repeats identical arithmetic
  and makes additional functionals inflate device executable size.

## Evidence and next gate

Compiler tests require complete source coverage and unit coefficients for every
admitted graph and preserve the original profile wrapper's XC/AO/Becke source.
Standalone CLI static-link checks require both shared CUDA launchers. The
existing ten-method independent E/gradient and reconverged finite-difference
tests remain mandatory real-GPU gates, along with sanitizer and cold/warm
timings before any performance claim. CI compilation without a GPU is not
numerical qualification. No PR merge, tag, or Release is authorized.

The next numerical refactor should separate method-independent Becke and AO
geometry stages, then compare whole-endpoint behavior against the preserved
graph-specialized baseline before changing the release default.
