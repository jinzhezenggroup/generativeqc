# Decision: share density screening with mixed canonical Fock work

Status: implemented
Date: 2026-10-16

## Problem

The canonical FP32 value route reused shell-local primitive geometry and the
incumbent component Schwarz predicate, but omitted the strict generated
traversal's density-conditioned shell screening. In a traced 96-atom endpoint,
mixed J/K took about 28.2 seconds per Fock versus 12.8 seconds for strict J/K.
Changing the scalar type did not make these different execution plans comparable.
The traced endpoint is diagnostic, not clean performance evidence.

## Decision

Before a mixed whole-shell value traversal, reduce the current projected
Cartesian density into the geometry owner's already charged shell-pair scratch.
The reduction and subsequent consumers use the owning stream. Recompute it on
every call, including signed densities and calls that change selected outputs;
never borrow stale maxima from an earlier generated J/K traversal.

Reuse `direct_shell_quartet_survives_screening` for independent positive raw-J
and raw-K predicates, taking their union for a combined consumer. In particular,
raw K must use the full exchange bound, not restricted HF's `J-K/2` bound. SR/LR
K retains the existing full-range Schwarz envelope and its compiler-owned radial
operator. A uniform CTA vote rejects tasks before shared recurrence barriers.
Actual mixed recurrence counters still count evaluated AO-ERI components, not
candidate shell tasks, requested flags or a neighboring caller counter slot.

This changes neither recurrence algebra nor precision: retained components use
FP32 recurrence with FP64 primitive contraction, density products and scatter.
It does not enable the generated weighted-FP32-product contract accidentally.
Fixed-mask/compensated response work retains its geometry-only predicate.
Absent optional shell metadata, the bounded canonical/independent fallback stays
available. No new device allocations, host copies or synchronization fences are
introduced; stream ordering and the existing optional-owner accounting remain.

For an admitted nonlocal budget solve, finish one bounded frozen-potential
proposal in mixed precision before rebuilding a fresh strict target. Preserve
the old eleven-step initializer allowance separately: the extra mixed budget
belongs to the target's inner proposal, not to solving the incomplete seed more
accurately. Reserve the existing fresh-target tail and preserve the finite
iteration ceiling. Explicit strict tolerances, ordinary AUTO, numerical-failure
recovery and all physical/observable publication gates retain their contracts.
The [long-initializer candidate](../../rejected/2026-10-16-long-mixed-nonlocal-initializer.md)
was slower and is not relabeled as this result.

Both changes are selected by shared operator/precision/solver capabilities,
never by a functional name, a benchmark geometry or a new AO-size whitelist.

## Clean endpoint evidence

Slurm 8924 uses one RTX 5090, UUID
`GPU-d79031c2-173e-535f-3563-5520bab0293b`, driver 580.95.05 and four CPU threads.
Every sample starts a fresh process, calculator and hcore density, includes
complete moving-grid forces, export and teardown, and retains the continuous
observed span as well as the explicitly excluded build-provenance gap.
Artifacts are warmed explicitly; no converged density or live CUDA owner is
shared. Both sizes use direct RKS, spherical def2-SVP, `GridSpec(48,16,32)` and
a 180-iteration ceiling.

| Endpoint | Incumbent Fast | New Fast | Ratio |
| --- | ---: | ---: | ---: |
| 12 atoms / 96 AO, three-repeat median | 17.909884 s | 15.781075 s | 1.135x |
| 96 atoms / 768 AO, one pilot pair | 875.672482 s | 788.818552 s | 1.110x |

The corresponding elapsed reductions are 11.9% and 9.9%, not a claim of a large
universal FP32 speedup. The 12-atom samples alternate arm order in the middle
repeat; individual times and separate artifact warmups are retained.

On the same frozen candidate build and GPU, the separate fresh-owner FP64
96-atom endpoint takes 1093.846727 seconds versus Fast's 788.818552 seconds:
1.387x, or 27.9% less elapsed time. FP64 executes 34 Focks, Fast 21; each
executes three nonlocal pair sweeps and one exact cache hit. This compares the
complete precision policies, including their different numerical convergence
budgets, not isolated FP32-versus-FP64 kernel throughput. Both arms pass their
respective independent gates; FP64 errors are `6.83e-12 Eh`, maximum force
`1.25e-9 Eh/bohr` and RMS force `6.90e-10 Eh/bohr`.

For 12 atoms, total Focks stay at 18, mixed/strict counts change from 12/6 to
16/2, and nonlocal pair sweeps drop from three to two. For 96 atoms, Focks change
from 19 to 21 and mixed/strict counts from 12/7 to 17/4; both execute three pair
sweeps and one exact cache hit. The primary shared J/full-range-K mixed AO-ERI
census increases from 17,717,684,496 to 20,390,627,797 because more Focks use that
route. Its mean per mixed Fock decreases from about 1.477 billion to 1.199 billion.
These are different evolving densities, not a fixed-density causal experiment;
do not describe the larger total count as less total recurrence work.

## Independent accuracy and safety

- The 96-atom result passes an authenticated retained independent accuracy
  oracle bound to the exact geometry, basis bytes, moving quadrature, density
  domains and original reference source/driver. Historical timing is not reused.
  Errors are `6.68e-10 Eh`, maximum force `3.86e-6 Eh/bohr`, and RMS force
  `1.17e-6 Eh/bohr`, below unchanged `1e-5 / 1e-4 / 3e-5` observable gates.
- Six independent matrix groups cover displaced 3/12-atom geometries, positive,
  negative and zero densities, pure J, pure K, combined J/K, SR/LR and optional
  whole-shell fallback. The newly added full-range direct checks keep the
  `2e-6` matrix gate; the largest observed error is `1.205e-6`.
- All sixteen shared-Fock endpoint cases pass: eight XC ingredient families,
  four range-source selections and four nonlocal strict-anchor regressions.
  Explicit tight energy/density/screening inputs remain covered. All eight
  families retain genuine single budget solves and independent complete E/F
  gates, not a relabeled strict fallback.
- The native mixed-work/strict-K suite passes. Slurm 1022 on node4 also passes
  native memcheck, native synccheck and complete 3-atom nonlocal endpoint memcheck,
  each with zero sanitizer errors. These instrumented durations are not timing
  evidence. Slurm 8927 could not start due to `InvalidAccount` and was canceled;
  its scheduler log is retained, not counted as validation.
- The configured host selection passes 1231 tests with 24 skips; formatting,
  whitespace and a no-op rebuild pass with the existing ccache. No cache is
  cleared or disabled.

The broader GPU run has 30 passes and seven failures in the unchanged
`test_application_fast_cuda.py`: six expect the superseded
`qualified_mixed_gradient` label instead of `qualified_budget_fock`, and one
expects two rather than three fallback solves. The incumbent build independently
fails the same seven assertions under Slurm 1022. Retain both XML files and the
comparison; do not weaken these tests or report the whole selection as green.

## Identities and reproducibility

Final source identity:
`d308384eaba42e70ed1e6411cadbad6ced87a1361408544a20ac6377cdcbbeeb`.
Core SHA256:
`bd8b0725e55edd54739d5525bb143fa664b34444cec48c99a742ed5f3384987f`.
Independent native-probe SHA256:
`122906ff32bc22786ec7229221798bc55c16aa23ca7d330de47745712352faee`.
The six existing force packages keep their unchanged, separately qualified
asset identities; their manifests are not rewritten to match the core hash.

Raw freezes, patches, binaries, matrix fixtures/results, XML, scheduler logs,
independent accuracy authentication and per-arm receipts are retained under
`../evidence-fast-followup-20261016/`. Relevant entries include
`candidate-density-v2-freeze.json`, `density-v2-fast-comparison-summary.json`,
`density-v2-96-all-arms-accuracy.json`, `density-v2-fp64-fast-summary.json`,
`stale-test-baseline-comparison.json`,
`run-density-comparison.sh`, `run-variant.sh`, `check-workers.py` and
`candidate-density-v2-*-slurm.log`. The intermediate density-v1 freeze is
unmeasured and must not supply qualification or speedup claims.

## Remaining limitation

The mixed canonical and generated strict paths still have different recurrence
and scheduling costs, and the expensive nonlocal/force work remains in the
complete endpoint. Extending generated FP32 product arithmetic needs its own
explicit contract, actual-work inventory and independent qualification. Merely
removing the strict-generated guard is not this optimization.
