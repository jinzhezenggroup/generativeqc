# Decision: preserve accepted-trial reuse when integrating static TensorIR reuse

Status: implemented
Date: 2026-10-07

## Integration boundary

The common TensorIR invariant proof and first native candidate were prepared on
`21f6314f5a812e9e83a41b3c9921549d6e944778`. While publication was pending, master
merged #2039's accepted-trial output reuse. Integration with
`c3bb6df4468ad75d34a2e2effb6e2450f2838a5a` preserves that behavior rather than
reinstating duplicate evaluations to match old benchmark counts.

Trial outputs borrow the current iteration arena. They survive until consumed
because DIIS modifies separately owned vectors, and physical replay has a distinct
arena. A carried output is accepted only when `update.modified` is false. Static
reference slots are globally exclusive and cannot be overwritten by dynamic
scratch. No-DIIS uses two host vectors and evaluates each accepted state once.

## Validation contract

Tests and the benchmark capacity helper use the current two-vector no-DIIS
formula. Diagnostic work derives from actual graph calls: preparation once,
dynamic work per executed evaluator, and nine saved transforms per later
execution. Carried-output consumption is not an execution. Default-off and
opt-in must preserve exact final energies/amplitudes and identical work/control
counts, with independent physical replay unchanged.

## Historical measurements

The October 6 measurements remain frozen at the original source checkpoint
`3e60fcaa66fb5b36f7b59b10bc0f80cf89852aeb`. Its formatting-only restoration patch
must be applied there. New integration checks are correctness evidence, not a
new performance result. Do not relabel nine historical evaluations or 72 saved
operations as measurements of integrated master.

## Integrated evidence

On the integrated source, eight native tests passed across
`test_rccsd_iteration_reuse.py`, `test_rccsd_numeric_capacity.py`,
`test_cc_canonical_denominators.py` and `test_host_diis_gram_symmetry.py`.
The actual compiler launcher was verified ccache 4.14.1 (10 cacheable calls:
3 hits and 7 misses). Seventy-eight shared dependency, canonicalization,
prepared-lifecycle and default-inventory tests also passed. The bounded native
iteration loop through result return is byte-identical to the selected master;
new assertions cover forced accepted-trial carry, no-DIIS work and exact budgets.
CUDA runtime/device execution was not run.
