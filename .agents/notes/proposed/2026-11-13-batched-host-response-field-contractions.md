# Proposal: bounded BLAS scheduling for host response field contractions

Status: proposed; naive BLAS variant fails vacuum-boundary qualification, not production
Date: 2026-11-13

## Subsequent implemented alternative

The naive BLAS candidate remains rejected for production. An order-preserving
four-component-lane AVX2 execution template now reuses the existing admitted
coefficient-input arena and retains scalar capacity/ISA fallbacks. See
`../implemented/performance/2026-11-13-ordered-host-response-simd.md` for its
source-bound component results, exact vacuum-boundary controls, and the remaining
complete-endpoint gates. This does not retroactively qualify the BLAS schedule
or authorize reopening either public ECP saddle guard.

## Measured problem and scope

The explicit positive Full-moment production cohort now completes frozen
integral preparation but still times out at the unchanged complete 1800-second
cap. Nuclear columns zero/one take 167.629852575/152.978219656 seconds; column
two remains unfinished. Do not repeat this unchanged expensive endpoint or
attribute its remaining time to the now-completed matrix ERI preparation. See
`../implemented/performance/2026-11-13-positive-full-range-host-moments.md`.

`host-response-field-profile-20261113` reconstructs the actual native RKS D
from the saved occupied C, builds the exact native 81-AO/87458-point v4 grid,
and executes the actual frozen `HostGgaXcResponse`. A renamed, timer-instrumented
copy of the exact source returns array-equal outputs and identical semantic
point/tile/cache work for six interleaved streamed/cached actions, including
nonsymmetric directions. No native/reference SCF, JK action, nuclear column or
complete Hessian runs in this component experiment. There is no stationary W
substitution because fixed-grid XC response does not consume W.

For cached actions the complete incumbent component median is 3.522682219 s;
the instrumented median is 3.540344052 s. Instrumented field reduction and
potential pullback medians are 1.641545466/1.460448303 s; native AO reader time
is 0.404908421 s and scalar-feature time 0.006294915 s. Each dense bilinear
sweep has 573811938 row-pair/point terms; both mandatory AO/grid reads and all
exact primal-bank authentication remain. These are particular component
observations, not a full-response/column/endpoint timing decomposition. The
profile archive is sealed with SHA256
`743e348e2ada4a4c0a9d54464fed22e334558c32f0882111fc71252a75a4854c`.

## Test-only schedule

Use the existing `tensor::cpu_gemm` owner, not a new BLAS binding or scientific
recurrence. For a tile of row-major AO values V and a fully ordered matrix M,
compute V M and V M^T. Rho contracts V against the first product; gradient
fields contract each AO derivative against both products. This retains generic
nonsymmetric M semantics rather than reading one triangle or doubling one
gradient term. Apply the same contraction to reference and tangent densities.

For the adjoint potential, form a weighted AO panel containing one-half the
rho coefficient times V plus the three gradient coefficients times AO
derivatives. Two ordered GEMMs accumulate B^T V + V^T B directly into the
staging matrix; no triangle is copied and no response matrix is symmetrized.
The panels are two tile-by-AO FP64 arrays, reused between fields and pullback.
The current 81-by-256 prototype explicitly allocates 331776 additional bytes.

This changes floating-point reduction order. The test-only allocation is not
the production resource contract: the prototype's original `scratch_bytes`
does not count these extra per-action panels. Passing prototype numerical or
primal-prefix tests is NOT production capacity/lifetime/fallback qualification.

## Evidence

`host-response-blas-hypothesis-20261113` compares the test-only schedule against
the frozen core on the same six component actions. All return finite values;
maximum incumbent parity error is 3.6591823321385775e-18. Parity is a control,
not an independent oracle. Cached complete component medians are 3.527696224 s
(incumbent) and 0.635264407 s (candidate); streamed medians are 3.632556940 and
0.745049105 s. Provider is the existing OpenBLAS CPU owner with one thread;
before/after verified sccache evidence is retained. No molecular speedup is
claimed and the complete endpoint still has its separate negative receipt.

Thirty-six numerical/semantic cases pass; 32 compare against independent Libxc
potential differences at all three original steps/gates. Twenty-eight selected
existing numerical/identity/vacuum/
primal-prefix cases cover RKS/UKS, both gradient terms for nonsymmetric
densities/directions, cross-spin coupling and lazy vacuum/fail-closed behavior.
Twenty resource/lifecycle cases are intentionally deselected for this prototype,
not removed from permanent CI. Eight added cases cover 17/81 AOs, 257 points,
16/256-point tiles and nonsymmetric density/direction for both spins. They keep
the original 3e-9 absolute / 3e-8 relative independent response gates and raw
potential symmetry check; no reference SCF or molecular finite differences run.

## Promotion requirements and rejected shortcuts

### Observed disqualifying boundary failure

The additional frozen-core boundary control examines 792 requests over 4/17/81
AOs, eight deterministic densities and 33 neighboring density scales around
the linked B3LYP `density-vacuum-1e-18` boundary. It finds **108 classification
mismatches**, so its exit 1 is a real failed scientific control, not a test
harness error. The original 36 selected numerical/semantic passes and six
actual-grid parity actions do not qualify the naive schedule for production.

The first retained counterexample has four AOs and scale
3.146475374925623e-19. The frozen incumbent returns an active response with
maximum magnitude 417523277851.9949; the reassociated candidate returns exactly
zero. Input matrices/AO/weights for four counterexamples are retained as NPZs.
Changing summation order can cross the unchanged lazy vacuum branch; this is
not fixed by loosening a response tolerance or treating the difference as a
small floating-point error. No native/reference SCF runs in this control.

A future candidate needs a conservative cancellation/boundary admission test
and an ordinary-order fallback (or another independently justified schedule)
before numerical and resource promotion can be reconsidered. Preserve this
failed variant/evidence and use new artifacts for any corrected experiment.
Do not widen/smooth/clip the linked physical branch to obtain a pass. Do not
silently promote this prototype merely because its interior tests are green.

### Required production design

- Design explicit checked panel-size admission and owned lifetime in the native
  response owner; account every byte in `scratch_bytes` and the enclosing frame
  reservation. Reuse panels across actions rather than untracked per-action
  allocation. Retain the ordinary streamed field/pullback schedule when capacity
  or the qualified provider is unavailable, or optional allocation fails.
- Preserve exact-input primal-cache authentication and failed-first-action bank
  rollback. Changing a numerical schedule after cache creation requires an
  explicit invariant or rejection, not stale cached inputs/roots.
- Qualify severe cancellation, subnormal/overflow behavior and neighbors of the
  linked graph's vacuum boundary. In particular, one-half coefficient splitting
  and reassociated density products are not automatically equivalent at those
  floating-point edges. Do not add clipping, a tangent/density floor, or silently
  widen physical-field admission to make the candidate pass.
- Run all production numerical, failure injection, aliasing, finite budget,
  cleanup/reentry and provider/fallback contracts. Add independent larger-shape
  and endpoint derivative gates; a parity-only control is insufficient.
- Separate fixed-grid response from nuclear first/curvature and ECP column work.
  This profile does not explain the entire non-GMRES column time. Do not assume
  that batching this one consumer makes all 18 columns fit the original cap.
- Preserve compiler/native ownership and use a shared auditable contraction
  schedule if extending it to other generated nuclear field consumers. Do not
  duplicate response algebra, change method metadata, relax exact residual
  replay or lower any raw-H/mode/optimizer scientific acceptance gate.

No production response code or public ECP saddle guard changes here. Mixed
Ni/I/Cs, the two real stationary cases, practical 25--40 atoms and the requested
native Ni TS remain outside this component hypothesis. The full goal remains
active and incomplete.

## Retained hypothesis

The complete initial BLAS hypothesis, including its failed boundary control,
is sealed as `host-response-blas-hypothesis-20261113`, archive SHA256
`8ec24ffba71c3aa4cae7ebde7287a5457443b6b2403e8463ce8d7c7c9de3abbd`,
664741 bytes and 94 members. Do not edit or rerun this sealed cohort. A corrected
variant needs a new cohort, not overwritten or relabelled green receipts.
