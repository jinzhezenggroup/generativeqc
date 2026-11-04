# Decision: one native runtime observable-comparison owner

Status: implemented
Date: 2026-11-04

## Problem

Native precision scheduling still called Python/NumPy to measure energy/force
errors and resolve runtime relative allowances. A separate SDK implementation or
an optional query with a NumPy fallback would retain competing scientific owners.

## Decision

The required selected-library `generativeqc_observable_comparison_query_v1`
measures all declared energy-absolute, force-maximum and force-RMS requirements
in one allocation-free, transactional receipt. Native code owns reference norms,
absolute-plus-relative allowances and inclusive comparisons, with no added slack.
At most two passes over each 3N force vector are needed. Shape/representation
validation, model/scope/convergence admission and reference uncertainty remain
explicit concerns of their respective transport/endpoint/evidence owners.

Copy the descriptor before reading arrays; complete every borrowed read before
publishing output. Invalid descriptors, dimensions, nonfinite inputs and overflow
leave output untouched. Missing/refused support cannot select a Python reduction.
Scalar and prepared precision require support before electronic/fleet work.
Public `compare_observables` uses the same numerical owner via the normal loader.
Only ABI transport makes noncontiguous/unaligned force inputs contiguous; ordinary
result-shape validation does not allocate an unused contiguous copy.

## Floating-point policy

Scale force components before squaring, then compensate their accumulation.
This preserves small positive contributions without overflow in an unscaled sum
of squares. RMS cannot exceed its largest component, so bound any rounded mean
overshoot by one. Zero and subnormal vectors retain their ordinary FP64 meaning.

This is not a byte-for-byte reproduction of NumPy's platform/length-dependent
pairwise bracket: final RMS rounding can differ. The acceptance gate uses exact
input conversion and independent 100-digit Decimal square-root norms, with at
most four binary64 ULPs for RMS and exact scalar/max comparisons. Tests include
tiny/huge scales and exact threshold neighbors; runtime never adds those test
ULPs to a caller allowance. Force/energy difference or allowance overflow rejects.
Relative multiplication is explicitly rounded before addition, retaining the
incumbent Python allowance bracket rather than silently contracting to an FMA.

Typed `ObservableTarget.allowance` and `AccuracyAssessment` scalar views remain
portable offline evidence metadata, including loaded-status consistency checks.
They are not production array reducers or precision execution fallbacks. Keep
them distinct from model qualification and observable/reference uncertainty.

## Boundaries and revisit conditions

The change retires production `_measure_observables`, not complete precision
qualification, gradient admission, initial prepared fleet execution or independent
seed eligibility/history. Existing physical HF/KS energy/full-force and derivative
gates remain; no new method/backend/provider/derivative scope follows from array
comparison. Complete-endpoint matched timing and reported-work parity are still
required, not query microbenchmarks. Real-GPU qualification must use an available
permitted Slurm allocation; unavailable or drained maintenance nodes are no reason
to bypass the scheduler. Full #2227 R1–R6 remain incomplete.

## References

- `docs/user/accuracy.md`
- `docs/user/application_precision.md`
- `tests/python/test_native_observable_comparison.py`
- `tests/native/test_observable_comparison.cpp`
- #2227 / #2972
- [Earlier bounded execution-order decision](../architecture/2026-11-04-native-precision-execution-order.md)
