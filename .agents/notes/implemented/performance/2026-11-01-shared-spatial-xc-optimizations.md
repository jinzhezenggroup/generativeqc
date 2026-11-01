# Decision: apply native XC optimizations to source-derived spatial programs

Status: implemented
Date: 2026-11-01

## Problem

The generic Libxc spatial adapter already evaluated each point once, but its
potential assembly and integration remained scalar and its point executions
were serialized by tile. Imported functionals, including MN15's registered
semilocal pair, did not receive the compact matrix, batching, compensated
reduction, optional mixed matrix or parameter-package optimizations available
to other native consumers.

This supersedes the execution limitations in
[the original Libxc adapter note](2026-11-01-libxc-spatial-point-programs.md).
Neither note grants full hybrid or force capability to the spatial E/V API.

## Decision

- Share the existing native compact panel algebra and tiled potential body.
  The spatial binding changes coefficient addressing only; it retains the
  existing tau-half and symmetric-pair conventions. A one-off comparison found
  both original native matrix and batch emission byte-for-byte unchanged.
- Use a fixed-warp compensated FP64 integration reduction. Batch accumulation
  retains three high/low pairs across flushes, folding only on publication.
- Give each prepared consumer an optional bounded queue. Default admission is
  at most eight tiles and 32 MiB, with a one-tile resource alternative. The
  native grid owns queues and destroys them before its stream. Python wrappers
  release their individual queue when closed/dropped, even if the grid survives.
- Copy live AO/features/maps and producer error witnesses into queue slots before
  the next collocation overwrites them. Keep original tiles and point counts;
  only reduction scratch is padded. Reuse the native ordered segmented scatter
  over overlapping maps without floating atomics. A reset starts a new execution
  and clears accumulated totals, potential and sticky errors.
- Expose `SpatialXcOptions.matrix_precision`, default FP64. Explicit FP32 uses
  the existing float compact potential contraction, while AO/density production,
  nonlinear point evaluation, storage, integration and global scatter stay FP64.
  It is a potential-assembly option, not a claim of whole-XC FP32 qualification.
- Enable the existing AO radial/axis reuse emitter for spatial programs, with
  an explicit scalar compilation alternative.
- Reuse native exact parameter-worker discovery for compatible single-component
  imported registrations. Both spin ABIs must agree. Each compiled package has
  immutable coefficient rows and a per-call selector; no mutable global CUDA
  coefficients are shared between owners. Scientific registration/precision
  identities remain separate from common package identity. MN15's distinct
  graph retains its own worker and shares the surrounding execution machinery.

The private grid XC call migrated directly to v3 for its explicit precision
argument. All repository callers were updated; no hypothetical ABI alias was
retained. The compact descriptor now has a lightweight shared header, included
in JIT source identity, avoiding the full native XC owner dependency.

## Resource and failure invariants

The base local owner retains eleven coefficient rows plus one weight row and
six integration words: `8 * (12*T + 6)` bytes. Existing dead contracted AO work
is reused for compact panels. For `B` queued tiles, `T` point capacity, `A` active
AO capacity and `J=min(4,jets)`, additional device storage is

```text
8 * (B*((J+8)*T*A + 28*T + 2*A*A + A) + 6)
+ 56*B + 4*(B+1)
```

Host descriptor storage is `56*B`. Every product is checked before native
allocation, and the runtime verifies the planner's exact byte receipt. Tiny or
out-of-domain shapes, tight budgets and optional allocation failure retain
single-tile execution. Device/numerical errors never authorize a CPU fallback.
The task lease and grid artifact remain checked before staging; native queues
also reject inconsistent selectors or spin bindings within one execution.

Statistics expose actual batch capacity/bytes/flushes, total coefficient-cache
capacity, point evaluations and complete owned device bytes. CUDA context/module
storage and allocator rounding retain their explicit resource-plan exclusions.
The profitability model conservatively charges panel traffic/liveness rather
than forcing the device candidate to rank first.

## Numerical evidence

Ignored `.artifacts/spatial-xc-optimization-evidence/` retains source inventories,
commands, logs, binaries and complete endpoint JSON. All real device work uses
finite Slurm allocations on n1's RTX 5090, preserving scheduler visibility.
Compiler caching uses verified sccache 0.16.0, with before/after counters.

- Host compact/policy/ownership regression: 819 passing tests; shared native
  matrix/batch/radial regression: 213 passed, one opt-in GPU test skipped.
- Independent Libxc checks cover all 73 shared registrations in 12 exact worker
  groups, both spins: 146 cases, at `3e-10` absolute/relative tolerance.
- The expanded spatial GPU suite passed 121 tests; the combined spatial GPU
  regression passed 241 tests, and Slurm memcheck passed 61 selected
  compact/batch/parameter cases with zero errors. The final integrated source
  replay passed the same 241 tests in 415.63 seconds and 61 memcheck cases in
  37.37 seconds, again with zero errors. Coverage includes six functional
  families/pairs, both spins, minimal/tail tiles, signed
  weights, repeated and orbital execution, FP64/FP32 matrices, heterogeneous
  overlapping maps, parameter-owner isolation, expired leases, invalid-point
  recovery, owner-before-queue close and optional allocation fallback.
- Spatial fixture gates are `5e-10` absolute/relative for FP64 E/V, and `2e-6`
  for the explicit FP32 potential only. These are fixture acceptance gates,
  not universal error guarantees or molecular-method admission.

## Endpoint evidence and tradeoffs

The initial matched fixed-density run (Slurm 15081) used 32 analytic Gaussian
AOs, 4096 points, two spins, 128-point tiles, identical full feature production,
one cold and seven warm calls. Every variant evaluated exactly 4096 point
programs. These are complete E/V calls including density upload, AO/features,
XC, scatter and final host publication, not isolated kernels or full SCF timings.

| Functional | Prior E/V median | FP64 batched median | Ratio |
| --- | ---: | ---: | ---: |
| LDA_C_PZ | 11.710 ms | 8.279 ms | 1.41x |
| GGA_X_B88 | 17.574 ms | 8.873 ms | 1.98x |
| GGA_C_LYP | 15.595 ms | 8.545 ms | 1.82x |
| MGGA_X_MVS | 54.907 ms | 42.988 ms | 1.28x |
| MN15 semilocal | 25.224 ms | 11.077 ms | 2.28x |
| M06-2X semilocal | 31.353 ms | 12.571 ms | 2.49x |

The final integrated-source repeat (Slurm 15086) retained the same workload,
baseline snapshot and seven-warm-call protocol:

| Functional | Prior E/V median | FP64 batched median | Ratio |
| --- | ---: | ---: | ---: |
| LDA_C_PZ | 11.578 ms | 9.012 ms | 1.28x |
| GGA_X_B88 | 17.083 ms | 8.749 ms | 1.95x |
| GGA_C_LYP | 15.317 ms | 9.859 ms | 1.55x |
| MGGA_X_MVS | 54.476 ms | 42.086 ms | 1.29x |
| MN15 semilocal | 25.069 ms | 10.825 ms | 2.32x |
| M06-2X semilocal | 30.913 ms | 12.497 ms | 2.47x |

Each call still evaluates 4096 points; an eight-tile queue publishes four
flushes. Maximum independent energy/potential errors across these FP64 cases
are `2.91e-11` / `1.64e-11`. The two runs expose normal timing variability;
neither is a claim about full SCF speedup.

The queue adds 3,508,756 device bytes in this fixture; complete owned storage is
8,195,860 bytes. All padding, copies and retained slots are charged. The explicit
FP32 option brings only small additional gains here and increases maximum
absolute potential error (up to about 0.00206 on the large-magnitude MVS matrix,
while satisfying the relative gate). FP64 remains the default. Structural
package discovery adds cold CPU work; compatible program construction reuses
its plan and compiled artifact. Singleton/composed programs can explicitly
select specialized compilation without discovery.

## Native MN15 local AO admission

The separate split-hybrid manifest now includes MN15's physical FP64 local AO
consumer. Mixed AO/density arithmetic and response are not inferred from this
flag. Existing map resource gates, derivative coverage, immutable geometry,
1e-16 screening predicate and dense FP64 fallbacks remain authoritative.

A source-matched Release/sm120 native build used the portable AOT profile,
MN15 RKS/UKS stationary profiles and sccache. Slurm 15087 passed all four
`test_all_curated_native_force_routes` MN15 CUDA cases: RKS/UKS times direct/DF,
independent PySCF energy/forces and final density/weighted-density states,
cold/warm execution, invalid-input recovery and moved geometry. Python force
orchestration is forbidden during these tests. Acceptance is `2e-8` Eh energy,
`3e-7` Eh/bohr forces, and `4e-7` final-state matrices. These complete molecular
checks qualify the native admission separately from the spatial E/V fixtures.

The same allocation passed 16 full RKS energy/force endpoints on four separated
STO-3G waters (28 AOs, 9304 points, Treutler level zero), comparing explicit
dense/local AO selection for direct and DF at cold, warm, moved and moved-warm
geometries. Both choices use the same native binary and scientific controls.
Maximum independent errors were `7.96e-13` Eh and `1.84e-10` Eh/bohr. Selection
was observed, not inferred: every local call reports `selected=1` and nonzero
XC work. Per-evaluation point/AO-pair work falls from 7,294,336 to 5,288,920
(27.49%); point/AO visits fall from 260,512 to 221,080. Active counts are 21--26
instead of 28 across 37 tiles. Discovery evaluates 1,042,048 AO jet values and
transfers 4,292 bytes; its roughly 2.0--2.2 ms is retained in the endpoint record.

| Route/phase | Dense E/F | Local E/F |
| --- | ---: | ---: |
| Direct cold | 535.745 ms | 144.256 ms |
| Direct warm | 66.770 ms | 65.349 ms |
| Direct moved | 417.493 ms | 426.046 ms |
| Direct moved-warm | 67.232 ms | 65.397 ms |
| DF cold | 127.496 ms | 96.767 ms |
| DF warm | 26.025 ms | 25.404 ms |
| DF moved | 102.738 ms | 103.204 ms |
| DF moved-warm | 26.616 ms | 25.551 ms |

These are single qualification samples per phase, not a statistically qualified
SCF speedup. Cold preparation/cache effects differ, and moved endpoints are
slightly slower with local AO in this small fixture. Dense/local iterations
match (direct 10/1/10/1; DF 11/1/10/1); retain the full counters and timings
rather than extrapolating from reduced AO work. The native-library host
regressions additionally passed 115 cases with 29 skips.

## Catalog provenance

Retain the exact previous catalog summary from merged PR head `f0d80a2f8`
before acknowledging 221 compiler-identity changes. There are no stage
promotions/demotions in the historical automatic catalog. Its old native campaign
remains stale; new spatial evidence is not substituted for it. The separate
MN15 capability has its own native endpoint acceptance boundary.

## Revisit when

A workload justifies whole-XC mixed precision, density-provider batching,
additional derivative consumers, or larger resident queues. Require independent
oracles and full endpoint timing before changing precision/admission/defaults.
