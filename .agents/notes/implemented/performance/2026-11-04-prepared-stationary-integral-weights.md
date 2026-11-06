# Decision: response-local prepared stationary integral weights

Status: implemented
Date: 2026-11-04

## Problem

The exact CPU ERI orbit schedule removes redundant integral differentiation,
but arbitrary ordered shell cotangents must still be visited. The composite
consumer reconstructed the same TensorIR integral block on every callback,
evaluated both fixed and response weights when only one was requested, and
repeated public-to-Cartesian density transforms on every exchange shell block.
These are invariant mathematical/data preparations, not new scientific work.

## Decision

Prepare one private weight owner for each stationary source contribution. Borrow
its fixed density and response snapshot, pull both into the existing Cartesian
workspace once and reuse immutable TensorIR programs by term count. Evaluate
only the requested fixed cotangent or its density-response JVP. Build bounded
Cartesian component indices with an array operation rather than a Python list
of every component tuple. Migrate the full-range and range-separated exchange
callers directly; remove the obsolete per-callback exchange helper.

Do not derive another J/K energy or response formula here. The existing compiler
programs still own MethodIR fractions, spin sums, same-spin exchange, overlap
signs and product rules. Keep shell weights ephemeral, without retaining a
molecular rank-four tensor, component-index numeric cache, integral cache or
geometry/state replay facility. The source-local owner expires after its first
and second integral passes. Existing whole-response identity/lifetime checks
remain in the shared executor.

## Scientific validation

Independent explicit scalar density-product oracles cover B3LYP, PBE0,
CAM-B3LYP and WB97M-V in both spin layouts, asymmetric and repeated shells,
fixed/moving one-electron and Pulay weights, arbitrary linear public-space
pullbacks and repeated callback replay. Fixed gates are 2e-14 absolute/relative.
The new tests plus the existing stationary gradient-plan suite pass 48 cases
in 3.09 seconds. Retain the initial four fixture failures caused by asking an
HVP plan for a gradient-plan-only accessor; the oracle now reads range fractions
directly from physical MethodIR primitives rather than adding an API alias.

After the change, the two independent asymmetric custom-hybrid sentinels and
complete WB97M-V RKS/UKS/direct/full-rank-DF Hessian files pass 41 cases in
955.60 seconds (two JUnit record-property warnings). All applicable hooks pass.
The preceding larger orbit-schedule regression separately passed 153 cases in
2389.42 seconds with two intentionally unexecuted CUDA cases and eight JUnit
warnings; it did not collect this later weight-preparation change. Keep these
source receipts distinct rather than presenting one as an all-current test run.

Full molecular and installed/public promotion are separate gates. In particular,
the preceding Ni trial's larger restart converged the same nuclear response in
257 iterations/514 actions, with true residual 1.052e-11, without relaxing the
1e-11 absolute/1e-10 relative tolerances. That trial nevertheless terminated by
signal 11 during one-electron second-source assembly. An isolated source probe
under GDB retains its own receipt; neither convergence nor this scheduling
change turns the failed complete HVP into a pass.

The isolated source probe did not reproduce that signal before failing the
finite 300-second C++ compilation deadline for a nuclear-attraction d/f tile
with unit nuclear charge. Do not label the original signal's cause as solved.
The probe's compiler children were reaped; the next work item is bounded
second-source compilation, not relaxed scientific gates. A separate experiment
with the existing guarded diagonal preconditioner and physical orbital gaps
converged the response in 17 iterations/34 actions, true residual 8.111e-12;
that experiment deliberately stopped before HVP assembly and does not change
the production response policy in this decision.

## Consequences and revisit conditions

This reduces repeated program/transform preparation, not the ordered-weight
domain or mathematical term count. There is no endpoint speedup claim. Revisit
cross-source integral reuse only with explicit resource ownership and complete
endpoint work/time/memory evidence, rather than retaining unbounded weights.

## References

- Issues #2224 and #2843; draft PR #2977.
- `2026-11-04-shared-cpu-eri-orbit-schedule.md` in this directory.
- `tests/python/test_composite_integral_weights.py`.
- Local receipts under `/tmp/gqc-2843-2849/`: `prepared-weights-final.{log,xml}`,
  `ni-independent-response.log`, `ni-hvp-large-restart-v2.log`,
  `ni-second-gdb.log`.
