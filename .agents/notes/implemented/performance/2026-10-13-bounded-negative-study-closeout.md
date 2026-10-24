# Decision: conclude bounded studies when promotion is declined

Status: implemented
Date: 2026-10-13

## Problem

The user asked to prioritize genuine issue completion and questioned whether
acceptance requirements had expanded beyond the underlying work. The initial
#2054 report made successful completion of every attempted arm a prerequisite
for closing a comparison study. Its original scope permits unsupported arms
and requires retaining losing regimes. Similarly, #2053 explicitly permits a
negative promotion decision, but its full positive-qualification matrix had
become an unconditional obligation.

## Decision and evidence

Conclude these investigations with explicit, limited outcomes:

- **#2053:** keep adaptive CUDA KS incremental Direct J/K opt-in. The retained
  [repair campaign](../../../../benchmarks/results/pbe0-incremental-refinement-20261007/README.md)
  has same-binary complete cold E+force medians of 214.07210 s OFF and
  214.25545 s ON, two samples per arm. Its existing verifier rechecks 15 accepted
  experimental endpoints, independent numerical gates and full/delta/final
  build accounting. This establishes no benefit, not equivalence or a universal
  regression. The qualified repair samples use a zero density threshold; an
  optimized adaptive gate remains unestablished. Missing repeated moved/48-atom
  populations and valid quartet census are not reported as qualified. The
  separate [PRO 6000 observation](../../rejected/2026-10-09-pbe0-incremental-endpoint-promotion.md)
  remains a distinct source and device, without pooling or claiming fresh raw
  verification.
- **#2054:** no qualified large-system crossover or provider recommendation
  follows from the [installed campaign](../../../../benchmarks/results/hybrid-provider-crossover-20261013/README.md).
  Both formaldehyde pairs pass their provider-matched pilot gates. Direct48
  references fail convergence; Direct96 forces run out of memory; DF-JK96
  exceeds the declared workspace; DF-J/exact-K is unsupported. All four water
  pair summaries stay `INCOMPLETE`. The auxiliary approximation differs
  substantially from exact Direct. Missing complete quartet accounting and
  process peak memory remain missing.

The original positive promotion requirements are retained as future
prerequisites, not checked off. Parent #1895 continues to own large-system
performance improvement; no duplicate follow-up issue is created just to move
an identical checklist. The issue closure comments record the dispositions.

## Invariants and rejected alternatives

No runtime default, numerical tolerance, physical model, resource limit,
validator or raw receipt changes. Declining promotion requires less evidence
than asserting a successful or universally losing optimization. Missing data
cannot become PASS or zero. These finite outcomes do not close positive product
deliverables such as #1189, #1190 or #1191, nor do they certify the frozen FP64
matrix owned by #1186/#1187.

Requiring new provider implementation and resource repairs just to finish an
inconclusive comparison would conflate investigation with feature development.
Conversely, calling the unsuccessful populations performance qualification
would erase the scientific question and is rejected.

## Revisit when

A materially different density/work guard, provider capability, auxiliary
approximation, resource plan or converged independent reference justifies a
new experiment. Any positive claim still needs matched complete endpoints,
independent numerical acceptance, valid work/memory accounting and applicable
lifecycle and statistical evidence. Repeating an unchanged campaign merely to
obtain successful labels is not a reason to reopen the study.
