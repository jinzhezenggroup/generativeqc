# Decision: factor one dressing of the admitted paired DF ladder

Status: implemented
Date: 2026-10-10

Retain the one-sided factorization after exact proof, focused native qualification
and four fresh matched energy-only endpoints pass. This is scoped numerical and
timing evidence, not formal/global performance or force/Lambda/response promotion.
The merged copy-roundtrip PR #2184 remains the separate frozen baseline.

## Decision

For `D = t1.T @ Bov`, the supplied ladder DAG expands exactly to
`B tau B.T - D tau B.T - B tau D.T`. Combine its first two terms as
`(B-D) tau B.T`, retaining the other signed term from the original DAG. This
removes a complete three-GEMM low-rank dressing chain, at the cost of one small
geometry/amplitude dressing product and a virtual-matrix subtraction.

`factor_ladder_dressing` verifies exact polynomial equivalence and simultaneous
pair reflection with the existing shared CC expansion machinery. It selects the
remaining term by proof, not node numbers. Other auxiliary cuts retain their
original objects. Unsupported inventories retain the original graph. No symmetry
of the factor matrices is used to justify the equality.

The new left contraction must undergo the existing bounded binary reassociation
before matrix packing. A superficially smaller DAG containing a direct
three-/four-operand einsum raised scalar work by nearly a thousandfold despite
lower storage/node counts. The current packed and batched programs instead have
1,054,175,083 scalar summands per Q at o9/v221, versus 1,113,077,329: exactly
58,902,246 fewer, about 5.29% of the complete auxiliary work. Native work gates
must protect binary contractions and the two-virtual-axis storage bound.

## Exact-proof normalization issue

The existing reflection checker initially rejected the valid factorization.
Numeric comparison was not used to override the failed proof. Investigation
found that dummy ordinals depended on the alphabetical interleaving of occupied
and virtual dummy names. A new nested dressing introduces an occupied dummy
after virtual dummies, while the original tree introduces it earlier.

Normalize index-space group order independently of dummy spelling before
assigning ordinals; preserve signed rational coefficients and bounded expansion.
An optional reference comparison rejects a different symmetric operator, rather
than treating reflection alone as an equivalence certificate. Dedicated tests
cover the factored spelling and a symmetric coefficient-two counterexample.

## Range and fallback

Algebraic equality does not preserve every finite intermediate. With zero tau,
zero Bvv, sparse T1=1e140 and Bov=1e180, the old tau-first ladder is finite zero,
but forming D first overflows. This is a concrete reason for a production range
gate; do not simply catch a new physical arithmetic failure and clear it.

The retained schedule requires input magnitudes at most 2^128 and summed dimensions at
most 2^16. Including four input factors, reductions and signed additions leaves
new intermediate magnitudes below 2^570. Geometry bounds are maxima of the
already computed conservative row-L1 metadata: no new factor scan or reference
oracle is introduced. Tau maxima are captured from values already read by the
existing per-state projection kernel; T1 maxima were already captured there.

The projection summary grows by eight bytes; its host transfer and solver
capacity must remain honestly counted. Refusal selects the existing original
unpaired action, does not change physical sticky state, and retains the original
tau. This range envelope is **not** a rounding or convergence certificate.
Independent numerical and physical replay gates remain mandatory.

## Evidence

- Sixty host proof/range/adjacent tests pass, with the CUDA-only proof skipped.
  The eight final factorization tests additionally include the finite-original,
  overflowing-new-dressing counterexample. Seven tests overlap, so these counts
  must not be summed. Compiler structure checks report 509 modules with zero
  dependency errors; CUDA ownership checks cover 344 files.
- The paired generated CPU/CUDA actions change deliberately. Eleven existing
  nonpaired generated artifacts compare byte-identical. Original expanded
  replay and standard FP64 (T) remain untouched.
- GPU build and focused solver/action/projection qualification use finite Slurm
  job 2812 on node2, ccache and preserved device visibility. Nine targeted solver
  tests and eleven action/projection tests pass. Projection memcheck has zero
  errors. Native complexity auditing reports no matrix-chain failure.
- A two-endpoint feasibility pilot (job 2813) precedes four new complete endpoints
  in ABBA order (job 2814). At ethane230/o9/v221/Q488, FP64, Q8, ordinary DIIS8
  and 64-GiB budget, median complete wall decreases 160.801392 -> 157.290694 s
  (2.18%) and complete CCSD 64.686691 -> 61.154653 s (5.46%). There are only two
  fresh observations per selection: no statistical/global promotion follows.
- All independent total-energy 1e-8, (T)-energy 1e-10 and expanded physical replay
  1e-10 gates pass. Total energy is unchanged; (T) differs by one FP64 ulp
  (3.4694e-18 Eh). Iterations/evaluations, Q scope, ascending accumulation,
  setup H2D, geometry/projection work and zero pair refusals remain unchanged.
- Complete contraction/GEMM work decreases by exactly 1,092,283,249,824 summands.
  Two fewer GEMMs and three fewer Q operations per primary Q8 tile remove 4,636
  GEMM calls and 6,954 operations. Compiler transpose/broadcast metadata explains
  690,140,921,600 fewer logical packing bytes. These are not measured DRAM bytes.
  Solver numeric capacity decreases by 140,669,320 bytes, including eight more
  projection metadata bytes and fewer provider binding bytes.

The existing scalar readback diagnostic charges `sizeof(metadata)` for each
projection call. Its eight-byte growth across 38 calls implies 304 extra bytes
by source and observed call count; the endpoint does not expose this aggregate
as a direct receipt. Do not describe it as an independently measured D2H total.

The candidate is built from the actual measured copy-elided baseline's frozen
parent/archive/overlay, not from a different SCF or CLI master tree. This keeps
both complete endpoints source-matched outside the present transformation.
Source archives, failures and binary receipts remain in ignored scratch at
`.artifacts/df-cc-ladder-factorization-20261010/` and its node2 counterpart.
Compact receipts, every sample, reconstruction patches and recipes are retained
in `benchmarks/results/df-cc-ladder-dressing-factorization-20261010/`. The local
branch is rebased onto observed master `464df951f`; relevant production/compiler
files remain identical to the frozen measured overlay. Unrelated DFT/CLI/CI
updates do not justify repeating this campaign or relabeling it current-master
endpoint qualification.

The repository's aggregate evidence budget requires keeping this bundle compact,
not dropping any accepted ABBA sample or weakening a numerical gate. Provenance,
compiler plans, medians and compact feasibility-pilot receipts share the gzip
validation envelope; full transient pilot data remain in ignored scratch. The
candidate source patch is a zero-context delta atop the checksum-pinned existing
copy-elision patch, applied with `git apply --unidiff-zero`. Reconstruction from
the frozen parent verifies all five baseline overlay files and all thirteen
combined candidate overlay files byte-identical to the measured archives.

## Acceptance and revisit

Require independent total-energy 1e-8, (T)-energy 1e-10, and expanded physical
replay 1e-10 gates; unchanged solver trajectory counts and Q scope; exact executed
work deltas; focused range/refusal/canary/sanitizer checks; and fresh untraced
matched complete endpoints before a follow-up PR. Do not relabel frozen evidence
as latest-master qualification or rerun unrelated suites for new master commits.

Reject or revise the schedule if reassociation fails these gates, changes an
unrelated cut, defeats matrix lowering, exceeds the bounded range policy, or does
not improve the complete endpoint. Force/Lambda/response promotion would need
its own evidence and is not authorized by an energy-only observation.
