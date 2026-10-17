# Decision: decouple FP32 XC products from point-functional arithmetic

Status: proposed
Date: 2026-10-17

## Problem

Budget Fast Fock can lower shared J/K and nonlocal work, but its XC matrix
products remained strict. Their existing FP32 implementations were selected
only by the experimental PBE point warmup. That is a policy coupling, not a
mathematical requirement: density-panel and symmetric potential products
consume the same typed AO panels and point coefficients for all functionals.

## Decision

Admit the shared matrix-product region for budget Fast, independently of point
capability or functional identity. Keep point algebra, AO production, analytic
forces, response, and strict audit unchanged. Products use FP32 operands and
inner accumulators with FP64 panel storage and inter-tile accumulation. The
existing indexed, compact, and ragged implementations preserve selected AO
maps and the original grid. Small nonempty FP32 tiles use their masked tiled
body; the strict-FP64 admission boundary does not change. Unsupported launch
shapes stay on the incumbent binding.

Report density and potential as two logical mixed matrix products per
nonempty admitted XC evaluation. Do not label strict point algebra as FP32 or
interpret logical products as physical launch counts. Strict handoff restores
all products to FP64.

## Rejected alternatives

- Add functional-name dispatch: these products have no functional-specific
  mathematics, and point accuracy must not gate an unrelated matrix backend.
- Blanket-cast the point graph: the independent generic-point experiment has
  retained failures. Matrix-product admission does not qualify float points.
- Drop local AO maps or batching: that would change scientific work or discard
  incumbent acceleration, obscuring the precision-only comparison.
- Change FP64 tiles or scheduling: outside the requested FP32-only scope.

## Invariants

Preserve independent-oracle gates and bounded strict recovery. Recovery is not
a successful Fast result. Endpoint qualification still requires genuine mixed
execution, solve count one, retry count zero, and unchanged E/maxF/RMS gates
of `1e-5 / 1e-4 / 3e-5`; strict gates remain `1e-7 / 1e-6 / 3e-6`.

## Evidence

Host policy/census tests pass (three tests). They distinguish strict point
work, FP32 inner accumulation, FP64 inter-tile reduction, and strict handoff
across point capability states and local/nonlocal compositions. CUDA oracle
tests and complete endpoint performance are pending; this is not a speed claim.

The isolated trial was initially based on master `8141cc7db`. Before any
candidate build or endpoint observation, it was deliberately rebased onto
`17fc57d69` so the authenticated Boys-integrated v19 runtime is the matched
baseline. The candidate contains the same Boys changes. Never compare an
XC-only `8141cc7db` build against Boys-integrated v19 and call it an XC gain.

## Revisit when

Promote only after independent CUDA products, native layout/handoff, and
matched 96-atom endpoint qualification. Retain regressions and failed cases;
do not loosen a gate to admit this trial.

## Pre-GPU ownership audit correction

The first source-only v20 trial inherited a coupling between mixed products
and point-total summation. Before its queued GPU test ran, that job and its
host controllers were canceled; no speed or numerical endpoint claim exists
for v20. Preserve that snapshot rather than rewriting its identity.

The corrected trial selects point-total precision independently. Generic
mixed products keep compensated FP64 point totals, while the existing
mixed-point warmup keeps its established FP32 totals. The census records the
density product's complete FP32 AO reduction separately from the potential
product's FP64 scatter across point tiles. Both use FP32 inner accumulation
and FP64 panel storage. Do not infer reduction precision merely from storage.

The initial v21 native fixture incorrectly demanded compact batching for
two-AO/seven-point domains, below the incumbent's 32-AO/32-point admission
boundary. Job3293 retains that failed harness assertion. Correct qualification
must explicitly check the tiny fallback and use a separate legal medium/large
domain for compact/ragged execution. Do not open the FP64 compact scheduler or
weaken a numerical gate to make the fixture pass. This test-only correction
does not change the production source identity.

## V21-r2 complete-endpoint rejection

Native job3294 passes its independent matrix E/V cases, 15 NumPy cases and
memcheck, but this is insufficient for production promotion. Family job3295
retains two MN15 Fast returns with one numerical retry each; the other 30
meet their E/F and retry gates, not a proof of the nonlocal XC census. The n4
three-pair population job1086 starts before a pending
cancellation check and stops at its first PBE96 candidate: one retry, despite
passing final E/F errors. Do not count these strict recoveries as Fast success,
loosen gates, or add a functional-name exclusion.

A separate execution audit finds that the existing nonlocal density-feature
entry points hardcode StrictAudit. The proposed census flag alone therefore
overstates FP32 products for that chain. Cancel the remaining WB pilot job1678
and retain its baseline and partial candidate; it cannot qualify an XC speed
claim. Propagate the explicit physical precision phase through the shared
density-feature entry points, retaining their strict default for independent
consumers. The next production identity must be frozen and retested; v21-r2
is not a passing endpoint or performance candidate.

## V22 representability correction under qualification

A fixed physical hcore-density diagnostic reproduces error3 with FP32
products and error0 with strict products. Independent CPU emulation finds
negative rho/tau as small as approximately 1e-133/1e-132, plus thousands of
nonzero strict tails lost by unscaled FP32 products. The fix is not a density
floor, clipping, relaxed point domain, functional whitelist or global strict
recovery.

Give each FP32 point/jet column one max-absolute binary exponent. Normalize
AO operands before their FP32 dot, then restore the exact binary scale into
FP64 work storage. Dot products and inner sums remain FP32; only exponent
selection/storage rescaling uses FP64. No extra AO panel or device allocation
is introduced, and the FP64 template compiles out this branch. Preserve the
strict point algebra/totals and explicit density-feature execution phase.
This is a new candidate requiring fresh native, physical-tail, complete E/F
and performance qualification; no gain is claimed from rejected v21-r2.

## Qualified v22 evidence and exact-PR integration boundary

The immutable v22 source is
`10544b5bfa013ca944185750b194ba2d92115ee6645a1cb1d24c429f57c86bbe`;
its core is
`b7736881e945aff0e0744a21bb640a45f0ff09a90cbc07647c37a026cebea0c2`.
Native job3298 passes the complete independent E/V matrix population and20
generated NumPy cases, both under memcheck with zero errors. Numerical jobs3300
and3302 qualify four physical-tail diagnostics, all32 small-family strict/Fast
returns and six PBE/PBE0/WB97M-V96 returns. Job3300's first96 command failed
before native execution because a worker was not staged; its exit2 remains
retained, and job3302 is a separately identified continuation, not a hidden
replacement. Every accepted Fast return has one solve, zero retries, both
mixed XC matrix signatures and strict point algebra. Node2 timings provide
no speed evidence because of the retained NVML mismatch.

On node5, job1680 completes all18 predeclared endpoints with one verified
CUDA/NVML RTX5090 UUID, alternating baseline/candidate ordering and three
pairs per method. Complete preparation/E+F/export/teardown timing includes
the transparent force-work wrapper; reference timings are never used.

| 96-atom method | Previous Fast median, s | v22 Fast median, s | v22 strict median, s | Median paired previous/new ratio |
| --- | ---: | ---: | ---: | ---: |
| PBE | 60.0926293881 | 54.8954047110 | 73.2874922600 | 1.0946750407 |
| PBE0 | 60.9511062480 | 55.6874109140 | 75.8472375630 | 1.0934261103 |

Both methods improve in all three pairs with unchanged old/new Fast Fock
counts. All18 endpoints satisfy independent E/F gates and retry0. Worst
candidate PBE errors are2.19e-9 Eh /7.38e-6 Eh/Bohr /2.19e-6 Eh/Bohr;
PBE0 errors are2.29e-8 /1.06e-5 /3.96e-6 in the same E/maxF/RMS units.
Conservative force-work comparisons retain source/artifact provenance
differences rather than silently normalizing them away. These are historical
v19-to-v22 incremental Fast results, not latest-master or v26 timings.

The separate PR worktree starts from Boys head29a181c40. Its14 owned files
first byte-match v22; subsequent C++ formatting changes no mathematics.
Its new production identity is
`e1a593231158cbdb9385c688b7c379d3c2277768572045064e130484e7bd30ba`.
Exact-PR native job3307 passes the complete matrix CLI and all20 NumPy cases,
both memcheck0. Host validation passes35 cases with20 explicitly opt-in CUDA
skips; the compiler dependency audit passes565 modules. The exact-PR core
`8806b070cb6caaf7c921823a133469c2d2751b9d2847e1809b1a70dde2fc427a`
authenticates its compiled source getter and all six official force manifests.
Fresh job3308 qualifies all32 small-family E/F returns and four physical-tail
diagnostics; its six96 returns remain in progress. Neither formatting
equivalence nor native qualification substitutes for the complete integration.
No experimental FMA candidate is included in this PR implementation.

The separate clean WB96 pilot retains a first-use regression: previous Fast
665.5981824s versus v22 Fast692.2173943s, ratio0.961545x, with identical21 Focks
and no retries. Candidate force preparation is58.7746s versus6.3006s,52.4740s
higher; the full endpoint, not a component-subtracted reconstruction, is the
performance observation. Do not call this a cache-state-matched cold-source
experiment or hide the slower sample. A new three-pair existing-cache follow-up
is declared independently, does not reuse the pilot as repeat1, and requires
matched complete endpoints. Its results and the six96 PR integration gates
remain unresolved before promotion.
