# Decision: reuse compact MD stages for cooperative Direct force roots

Status: implemented (qualified in the existing default cooperative consumer)
Date: 2026-11-05

## Problem

Issue #2976 tracks the remaining complete-endpoint gap against GPU4PySCF.
An A100 diagnostic on master `a970014e8de544aa54c6e0662eda32ebc5e5388f`
identifies order-six Direct forces as a substantial remaining owner for the
96-atom water r²SCAN-3c energy-plus-analytic-force endpoint. Intrusive tracing
and census report 2.428 s in `direct_force_order6_relative` within a 7.431 s
stationary force; these are inclusive, instrumentation-affected observations,
not clean performance evidence or additive region timings.

The admitted cooperative order-six/seven source interprets a degree-eight
1991-node, 24-level spatial DAG for each primitive product. Total order six only
needs degree-seven force roots. Existing scalar MD-J code already owns the
same Cartesian recurrence with a compact previous-stage dependency schedule.
The new degree-seven/eight schedules respectively evaluate 322/486 recurrence
destinations (plus their unchanged Boys seeds), versus 1991 interpreted DAG
instructions. The radial shared allocation becomes 339 doubles rather than
1991; these are generated work/storage counts, not measured endpoint speedups.

## Decision

Share the existing MD stage builder and expression construction between scalar
and collective emitters. The scalar lowering retains descending in-place
stores. The cooperative lowering uses two bounded shared-memory buffers and
publishes every previous stage with a CTA barrier before reuse. The caller's
existing degree-eight Boys ladder remains unchanged. Order-six consumers use
degree-seven roots; order-seven consumers retain degree-eight roots.

This changes neither the integral model nor contraction/scatter ownership.
It adds no GPU-name dispatch, production CPU/reference dependency, environment
opt-in, device-batch field, or native ABI. Existing domain/resource fallbacks
remain authoritative. Selection is unconditional only inside the already-admitted
cooperative consumer; the real-GPU gates below qualify that default.

## Rejected alternatives

- Do not import the whole old research tree: projection caches, convolution,
  warp force consumers, and pair-mask changes have separate ownership and
  qualification costs. Isolate radial source work first.
- Do not optimize toward the old 5.7 s reference observation as if it were a
  newly measured current-master reference. The fresh A100 reference attempt
  produced no complete samples before its finite cap; its cause is unresolved.
- Do not loosen convergence, alter grids, replace Direct with DF, or select an
  A100-specific kernel by product name to manufacture a comparison.

## Invariants and acceptance gates

- Preserve independent 80-digit closed-derivative radial oracles and executable
  generated-code publication/canary checks, including narrow stages.
- Preserve independent Libcint Separate/Combined J′/K′ agreement for RKS/UKS
  and Cartesian/spherical representations on a real scheduled GPU.
- Qualify complete cold plus three frozen-density warm energy/force endpoints
  on actual A100 and RTX 5090, with tracing/census disabled in timed processes.
- Native/native: energy per atom ≤ 1e-9 Eh, force maximum ≤ 1e-7 Eh/Bohr,
  force RMS ≤ 3e-8 Eh/Bohr; preserve semantic work counts and actual settings.
- Independent reference: energy per atom ≤ 1e-5 Eh, force maximum ≤ 1e-4
  Eh/Bohr, force RMS ≤ 3e-5 Eh/Bohr. Never compare device-crossing timings.
- Each corresponding 5090 warm endpoint candidate/baseline ratio ≤ 1.05.
  A performance failure must not be hidden behind an unqualified default.

## Evidence so far

- Candidate base: the API-verified master commit above; source identity
  `b78434b9416f124db8bf73b3b339bae894027959ed0220f6731e1bb1b10c2293`.
- One-off byte comparison confirms scalar generated radial sources at maximum
  orders 8, 9, and 11 are unchanged. This is a migration check, not a permanent
  source-spelling test.
- 66 focused cooperative radial/schedule/density-compiler host tests pass.
- 1378 additional MD math/reference/emission tests pass; 73 GPU/upstream cases
  skip and are not GPU acceptance. The first invocation omitted the repository
  root from `PYTHONPATH`; rerunning with `PYTHONPATH=python:.` resolves those
  test-harness import failures without changing production code.
- Ruff checks and formatting and compiler-structure checks pass.
- A100 diagnostic job 225599 completes native cold/warm, but its reference
  exits 137 without any complete sample. Bootstrap jobs 225596 and 225598 do
  not execute endpoints and are not accepted evidence.
- The first normal dual-architecture candidate build hits its 1800 s CPU
  compilation cap (exit 124) after 32/43 reported actions; it produces no
  authenticated candidate image and launches no GPU qualification. A bounded
  3600 s incremental continuation reuses completed objects and the existing
  sccache. Both attempts must remain visible in build provenance.
- Raw task receipts are retained under ignored
  `.artifacts/direct-gap-master-20261105/`; publication requires compact,
  full-precision replay evidence and updating issue #2976.

## Accepted endpoint evidence

Retained full-precision populations, replay, scheduler receipts, build/cache
provenance and failed diagnostic boundaries are published at
`benchmarks/results/direct-cooperative-md-20261105/`.

- A100 LSF job 225682 completes successfully. Cold is 18.829707883298397 →
  18.356019165366888 s; median warm is 8.462727900594473 →
  7.974124159663916 s (5.774% less latency, approximately 0.489 s saved).
- RTX 5090 Slurm job 16004 completes successfully. Cold is
  29.551986195147038 → 29.451527167111635 s; median warm is
  11.473440650850534 → 11.283032014966011 s (1.660% less latency).
  Each corresponding warm ratio is below one, and hence below the 1.05 gate.
- Both populations retain 10/10 cold SCF iterations/Focks and 1/1 for each
  warm endpoint, unchanged incremental work counters, 1,078,336 XC grid points,
  and resolved 1024-point SCF/force tiles. No trace/census runs in timed calls.
- All 32 native/native sample pairings pass strict gates; maximum force
  difference is approximately 1.005e-11 Eh/Bohr. All 64 observable comparisons
  against the fresh 5090 GPU4PySCF population pass common gates; A100 uses this
  reference for science only, never for cross-device timing claims.
- All 24 scheduled Libcint Separate/Combined source tests pass on 5090 with
  the authenticated identical SM80+SM120 core used on A100. The 5090 reference
  completes; the earlier A100 reference failure remains unresolved and visible.

The measured images remain based on `a970014e8de544aa54c6e0662eda32ebc5e5388f`.
Before publication the working branch is synchronized to master
`30657763a1f79c44e1b897b524ae9f54c96928cd` (#2991), then
`051a157c64061eac4fd9ee90d7becfa982f80161` (#3011). The sole native source
change concerns ordinary CPU UKS final closure; ECP fixtures/documentation and
qualification metadata account for the other #2991 changes. The #3011 update
adds public CPU full-range Hessian admission and its qualification, not a new
CUDA energy/force path. No measured CUDA force consumer, generation source, or
endpoint runner changes in either update. The 66 focused host gates pass again
after final synchronization; the old images are not relabeled as newly rebuilt
master images.

This is a scoped improvement, not closure of the entire A100 gap or the earlier
1.576 s target. The intrusive diagnostic still identifies generated low-order
forces as another substantial owner. No new same-A100 GPU4PySCF ranking follows
from the incomplete reference attempt, and no repeated broad population is run
to rescue or amplify the observed improvement.

## Revisit when

Real-GPU parity, endpoint timing, resource bounds, or 5090 non-regression fails.
Further force-contraction optimizations require a new isolated work-count and
complete-endpoint qualification rather than reusing this radial-only result.
