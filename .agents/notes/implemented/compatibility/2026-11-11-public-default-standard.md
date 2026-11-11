# Decision: public precision defaults select the measured Standard endpoint

Status: implemented
Date: 2026-11-11

## Problem

The README benchmark explicitly selected Standard, and ordinary Calculator
already used the shared native Standard controls. However, the public
PrecisionCalculator facade and a default-constructed PrecisionPolicy still
selected Balanced. Balanced requests an independent observable audit, adding
two or three complete endpoints. A user omitting the policy therefore did not
execute the measured endpoint, even after HF/KS convergence defaults aligned.

## Decision

Both public precision constructors default to Standard. The README Standard
arm now omits the policy argument instead of supplying a benchmark-only choice.
Method, device, orbital representation/basis and explicit Direct/DF auxiliary
inputs still identify the compared physical model. Convergence, initial guess,
grid, scheduling, precision profile and optional environment selectors are not
overridden by the benchmark's Standard constructor.

Balanced, Strict, audit_auto and explicit targets remain opt-in audits with
their complete comparisons and acceptance gates unchanged. Consumers that
intentionally test these audits now request Balanced explicitly. No numerical
gate, GPU4PySCF timed control, final-state proof or resource fallback changes.

## Evidence and invariants

The permanent public-boundary regression exercises all eight README methods
in Direct/DF mode with both an omitted policy and PrecisionPolicy(). It checks
the same resolved controls as the benchmark constructor and one FP64 endpoint
without inventing observable-budget evidence. Existing explicit-audit fallback,
invalid-reference and independent native-oracle tests retain their full work
and acceptance requirements.

The policy implementation participates in the complete source identity even
though no C++/CUDA numerical implementation changes. A new clean source/image
and complete benchmark acquisition are therefore required for the replacement
publication; the earlier source-bound campaign remains retained and is never
relabeled. This default correction does not establish a causal timing gain.

## References

- `docs/user/application_precision.md`
- `python/generativeqc/precision_policy.py`
- `benchmarks/readme_defaults.py`
- `.agents/notes/implemented/numerics/2026-11-11-df-hf-orbital-stopping.md`
