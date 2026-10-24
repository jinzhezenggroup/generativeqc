# Decision: bind native prepared response plans to their logical problem

Status: implemented
Date: 2026-10-11

## Problem

The existing method-neutral native response boundary prepared only a raw
`GmresPlan`. Its single-solve check compared dimensions; its batch check
recomputed numeric admission. Neither distinguished different same-sized
operators or a newly reconstructed problem. A regression with two different
2-by-2 matrices demonstrated that a plan for the first silently solved the
second. Empty and workspace-refused batches likewise carried no binding check.

## Decision

- Give each constructed `LinearResponseProblem` a private opaque binding token.
  Copies preserve the logical binding; moves transfer it; reconstruction starts
  a new binding even when the callback and dimensions match.
- Return a `ResponsePlan` from `prepare_response` and store it in batch plans.
  It extends the existing numeric `GmresPlan`, preserving method resource
  estimators without adding a solver or a second derivative schema.
- Keep the binding private and check it before single-solve execution or batch
  admission/publication, including empty and refused batches.
- Retain only the token in plans. Keeping it alive prevents address-reuse aliasing
  without extending callback, provider or scientific-buffer ownership.

## Boundaries and rejected alternatives

A dimension or callback-address comparison cannot distinguish logical problem
bindings. An exposed identifier could be rewritten alongside public numeric
plan fields. Retaining the complete problem in the plan would unexpectedly
retain method callback storage and alter endpoint ownership/accounting.

This is an E1 prepared-binding safety slice of #932, not a complete generation
or derivative-capability implementation. Method adapters remain responsible for
converged terminal-state proof, scientific/provider identity and borrowed lease
validation. Mutating state captured by a callback does not create a fresh
binding automatically. No new CUDA capability, resident Krylov cutover or
performance improvement is claimed.

## Invariants

- Method equations, symmetry declarations, true residuals and numeric admission
  remain under their existing owners and retain their previous behavior.
- Raw `GmresPlan` values remain usable with the low-level solver, but cannot
  bypass bound `solve_response` entry points.
- Tokens are descriptor/ownership overhead, not dimension-sized numeric payload.
- Existing MP2 and RCCSD Lambda CPU/CUDA consumers use the bound return type
  without changes to their scientific actions.

## Evidence

The standalone native GMRES executable checks unrelated same-sized operators,
reconstructed and moved-from problems, copy/move preservation, fresh-plan
recovery, callback-owner release, raw-plan non-convertibility, and
empty/refused/admitted batches. Existing independent dense-matrix residual,
resident-controller pairing, ragged-request and exact-budget checks remain.
Bound and raw legacy solves are also paired for identical solutions, residuals,
iteration/operator counts and numeric workspace diagnostics.
The Python response-contract suite compiles and executes the native suite
through its required cached compiler fixture, without CUDA or a native library.

Local qualification on the change based on `master@4889e74ce` passes:

- CPU-only CMake library build and native GMRES, Lambda preconditioner and MP2
  gradient CTests (3/3).
- Focused response/CC ownership, resource, Python API and module-boundary tests
  (33 passed, 54 subtests passed).
- Native GMRES contracts under ASan/UBSan with leak detection.
- Applicable pre-commit hooks and the full Sphinx warnings-as-errors build.

Compiler commands use the verified ccache launcher; before/after shared-cache
statistics and validation logs remain in ignored local build artifacts. No
real-CUDA execution evidence is claimed by this prepared-binding slice.

## References

- #932 E1; #925; #181.
- [Original native boundary decision](2026-09-22-cpp-linear-response-problem.md).
- `src/response/linear_problem.hpp`; `src/response/solve.hpp`.
- `tests/native/test_native_gmres.cpp`.
- `tests/python/test_response_operator_contract.py`.
