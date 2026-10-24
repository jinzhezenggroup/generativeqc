# Decision: remove repeated strict endpoints and reuse auxiliary-g shell work

Status: implemented
Date: 2026-10-21

## Problem

The [endpoint diagnosis](../../proposed/2026-10-21-readme-slow-endpoint-causes.md)
separated real native costs from the historical reference's excessive SCF work.
Default Treutler/Lebedev/NWChem grids, default reference fitting auxiliaries,
full grid response and default reference convergence expose different costs
than the earlier README protocol. Restoring the earlier ranking by tightening
the reference again is not an acceptable optimization.

## Decisions and invariants

- Generate the sixteen orbital-s/p/d/f, auxiliary-g shell derivative classes
  with the existing auxiliary-g scalar geometry and raised-moment DAG. Keep
  the old 64-class tuning inventory and its Rys availability separate from
  these additional polynomial classes. Preserve generic, metric and bounded
  panel fallbacks; do not promote mixed auxiliary-g response.
- A Fast request that already executes a native FP64 Standard endpoint may
  retain it under Standard-or-tighter controls and a default-or-looser budget.
  Report `strict_standard_fallback`, not mixed arithmetic or measured error.
  Explicit targets and tighter budgets retain independent endpoint audits.
  Legacy fitted-HF counters remain unavailable; direct-HF warm final-Fock
  reuse remains an explicit native accounting case.
- Pure-J budget proposals prefer the existing generated strict J when its
  alternative is slower generic mixed recurrence. Require an actual strict
  J census plus executed admitted XC/density contractions, strict refinement
  and final physical checks. A strict final row alone cannot hide an
  unsupported approximate J row. A strict warm Fock can also accompany an
  independently qualified mixed gradient; the complete-model qualification
  must still pass.
- Numerical failures of prepared budget-Fock proposals use the existing
  bounded strict retry, as scalar calls do. Preserve the failed attempt and
  unknown work. Do not retry input or resource failures under this rule.
- Skip automatic full-range quartic residency for ordinary generated
  through-d strict hybrids, where measured construction did not amortize.
  Keep mixed, range and uncovered higher-angular routes. Automatic range
  sources select whole high-angular buckets within the existing live-resource
  checks, under a 16 GiB optional ceiling; unselected buckets remain streamed.
  Explicit masks retain their requested policy. Mixed and strict leases never
  coexist or certify one another.
- Screened incremental HF switches to a sticky full-density refinement after
  density stationarity. Otherwise screened full and delta Focks can alternate
  different omitted terms indefinitely at the energy gate. Write the sticky
  decision in the subsequent finalize kernel, avoiding a cross-block race.
  Unscreened exact increments retain their original cadence. This does not
  enable incremental J/K by default or relax final scientific checks.

## Qualification and observed effects

The base is master `339e3edb7c95261f946091586f7d7584a43338b6`. These are
diagnostic single cold/immutable-own-seed warm endpoints on Slurm node1 RTX
5090 allocations, not a new statistical README population. Each receipt
retains its source patch, file hashes, actual shared-library hash, environment,
complete E/F arrays and execution counts. No moved benchmark was added.

Candidate v4 independently qualifies 25 of 27 completed populations against
the unchanged dense-reference/common gates. HF DF12 fails the energy gate in
both Standard and Fast, exactly as the earlier native and reference default-DF
populations do; its fitting error is not repaired or relabelled as passing.

| Endpoint, 96 atoms | Earlier diagnostic / publication (s) | Candidate (s) |
| --- | ---: | ---: |
| PBE DF Standard warm | 29.37–29.88 | 8.08 |
| PBE0 DF Standard warm | 56.37 | 11.15 |
| PBE0 DF Fast warm | 176.98–178.81 | 11.22 |
| HF Direct Fast warm | 16.85 | 2.94 |
| PBE0 Direct Standard cold | 48.61–48.71 | 40.97 |
| r2SCAN-3c Direct Fast cold | 271.30–274.86 | 60.30 |
| WB97M-V Direct Standard cold (v2) | 642.38 | 290.00 |

PBE DF warm response falls from 24.32 s to 1.50 s. PBE0's separate warm K/J
responses now take 2.78/1.52 s; they are still separate consumers. The
WB97M-V v2 cold result passes the same independent gates, but warm remains
about 45 s. These observations do not establish universal superiority over
GPU4PySCF or a strict Fast speedup for every method and phase.

Thirty CUDA auxiliary-g tests pass against independent complete energies and
forces, including all six full/symmetric pair and warp/packed/compact schedule
combinations. Native incremental tests cover restricted/unrestricted, screened
and unscreened execution, adaptive density gates, geometry changes and owner
reuse. The 96-atom incremental HF endpoint now converges and passes the common
independent gate; it is about 24.6/5.5 s, versus ordinary HF about 25.8/2.9 s.
Thus enabling increments globally would regress warm behavior.

The host precision/owner/source-capacity tests pass (230), along with 89
additional transport, census, public-documentation and source-policy tests
(55 subtests). The compiler dependency checker covers 574 modules without
errors. Generated scalar/auxiliary-g compiler checks retain their independent
algebra coverage; the invalid-class diagnostic expectation now reflects g
admission. Prepared CUDA lifecycle qualification also retains independent
forces, frozen/moved seeds and explicit audit ownership: all nineteen cases
pass (fourteen in the v6 full run, five after correcting stale test assertions
and explicitly requesting forces with an energy-only audit target).

## Convergence investigation

Density convergence was already enabled. The disabled option was incremental
J/K construction from delta D, not the density stopping test. Frozen-baseline
Standard scans of HF/PBE0/r2SCAN-3c96 show that changing requested E/D to
1e-9/1e-6 or 1e-8/1e-5 does not change their iteration counts: the native
physical-final-state caps remain stricter. All 22 completed Standard calls
pass the common gate; baseline incremental HF fails to converge.

The Fast scan covers seven DFT models at 12 atoms and PBE0/r2SCAN-3c at 96.
All nine default populations pass. Tighter controls usually add iterations,
but PBE0/96 warm can take additional solves, and WB97M-V/12 at 1e-8/1e-5
fails the candidate final-state check. The bounded prepared retry addresses
the latter without weakening that check; its cold/warm retry results pass the
same independent gates. The finite static sweep does not
justify a universal new threshold. Standard defaults are unchanged.

## Remaining work and rejected shortcuts

Large DF source regeneration, two-factor whitening, separate J/K derivative
traversals and WB97M-V force-resource admission still require their own
changes and qualification. In particular, r2SCAN-3c96 has 1088 orbital and
6816 auxiliary functions: a complete FP64 packed tensor alone exceeds 32 GB.
Its streamed J currently traverses two complete ordered raw tensors per build.
Raising a resident cap cannot make this complete owner fit a 32 GB card.

Do not replace stable eigenfactor order with an explicit inverse-root product
merely to delete a GEMM, combine physical J/K output vectors without preserving
their one-shot scratch ownership, borrow unknown force reserves, or turn
missing work counters into zeros. Fast can legitimately equal Standard when
no approximation is admitted.

## Evidence retention

Ignored local `.artifacts/{convergence,convergence-fast,candidate,candidate-v3,
candidate-v4,candidate-v5,candidate-v6,build-evidence}` and corresponding
`n1:/data/jzzeng/scf-performance-20261021/` directories retain attempts,
scripts, qualifications and compiler-cache statistics. Baseline source/binary
remained immutable. Candidate v2 lacked separately packaged stationary-force
artifacts for several methods; v3 also encountered a relink while running.
Those failed attempts are retained, not counted as numerical failures or
publication timings. Subsequent runtime snapshots separate immutable loaded
libraries from the build directory. The v5 native library SHA-256 is
`7a0ca91c4f849a682a6f22283cc01e07c70c8ab15c31b11d511a155fc77a5c32`.

Revisit these policies with complete endpoints and independent observable
gates when native force capabilities, source profitability or memory envelopes
change. Preserve the source-specific receipts rather than relabelling these
diagnostics as a later clean master measurement.
