# Decision: stable B88 AD and exact-input spin-response qualification

Status: implemented; scoped synthetic CPU qualification, not public ECP admission
Date: 2026-11-11

## Problem

The [retained synthetic UKS/ECP candidate](../../proposed/2026-11-11-analytic-ecp-uks-spin-response-gate.md)
passes its complete physical coordinate matrix but fails an arbitrary-AO
comparison against an independently converged Libxc state. Those are not
interchangeable derivative contracts. At an orbital node, minority density
`7.4225911645623816e-9` receives an unscaled AO tangent of about `-0.02077`.
Small, acceptable ground-state density differences then produce large kernel
differences; evaluating Libxc at the identical native Cartesian fields still
has a separate numerical discrepancy. Neither SCF convergence nor a molecular
density tolerance bounds this arbitrary-AO comparison.

There is also a genuine native numerical problem. Differentiating the original
B88 enhancement separately through `rho^(4/3)` and
`y = sigma*rho^(-8/3)` cancels large terms. An 80-digit independently
differentiated original B88 primal exposes relative response errors above
`3e-13`. A complete exact-input symmetric-AO comparison against XCFun has a
maximum absolute error `2.248998498544097e-8`, exceeding the original `2e-9`
gate. Changing only the independent oracle would not fix the native error.

## Numerical correction

Use the same B88 spin energy, algebraically written before differentiation as

```text
d = rho^(4/3)
g = sqrt(sigma)
e = -Cx*d - beta*sigma / (d + 6*beta*g*asinh(g/d))
```

This cancels the inverse density powers before AD, rather than subtracting
their large differentiated contributions. Retain the existing analytic
`x*asinh(x)` series for `y <= 1e-8` so the zero-gradient second derivative is
finite. The same lazy empty-spin and total-vacuum branches remain. There is
no density/tangent clipping, displaced production calculation, new precision
mode, reference substitution or independently handwritten production Hessian.
The canonical compiler primal continues to generate native first derivatives
and stationary second derivatives.

## Explicit reference contract

Keep the original molecular `1e-5` gates and the `2e-9` spin-response gate, but
apply each to mathematically matching inputs:

- All 42 unit symmetric-AO input columns, both output-spin matrices, plus the
  two original random spin tangents: independent XCFun B88/VWN3/LYP/Slater
  kernel at the **identical native Cartesian fields**, with independent
  PySCF AO evaluation and J/K integrals. Neither SCF is seeded or modified.
- All 13 occupied/virtual spin-input directions, projected into both output
  OV blocks: the original independently converged PySCF/Libxc response.
- Complete raw 9x9 coordinate Hessian, unit-Na-x and mixed HVPs: independent
  unseeded moving-grid gradients at all three original displacement steps.
- Keep the old unconstrained, independently converged Libxc AO discrepancy
  as a named diagnostic and retain the original failed process/recovery.
  It is not retroactively passed or erased.

The exact-input AO check tests the complete derivative map, not only its
physical projection. The independent physical-OV and molecular checks still
test state matching, implicit response and moving geometry. Supplying a
kernel's primal fields to an independent derivative oracle is not supplying
that density as an SCF guess, or using the oracle in production.

## Evidence

- New independent high-precision B88 regression: old expression gives two
  failures/two passes; rearranged expression gives four passes.
- Frozen-field offline check: all 42 unit symmetric-AO columns and both
  output spins pass `2e-9`, maximum `3.128661774098873e-10`. This is a scoped
  diagnostic, not complete molecular/public acceptance.
- Native CPU rebuild exits zero in 29.25 seconds. Verified sccache 0.16.0
  reports 15 requests, one hit, fourteen misses and zero errors. Library SHA256
  `afb07c88c86bcfa2959cfa35bfcc138266c0d6fed8fe5abe16cac02b10ac0ec9`.
- The new complete molecular process freezes its source/library identities
  and unchanged gates before execution in
  `build/qualification/uks-analytic-ecp-20261111/stable-b88-complete-matrix-manifest.json`.
  It actually exits zero: five tests pass in 217.91 seconds. All 42 complete AO
  columns pass the same `2e-9` gate, maximum `9.240466170012951e-10`, including
  the independently evaluated AO/J/K pullbacks. Both original random spin
  probes pass at exact inputs; maximum error `1.0913936421275139e-10`.
- All 13 physical OV inputs / 26 output blocks pass `2e-9`, maximum
  `3.401153115184142e-10`. Complete raw 9x9 matrix errors at the original
  `0.003`, `0.001`, `0.0003` steps are respectively `4.838986592203831e-6`,
  `5.372590071162087e-7`, `5.1598643191486104e-8`; both fresh HVPs, raw
  symmetry/translation, components and response residuals also pass.
- The unconditioned beta AO comparison remains `0.0015167409901550855`,
  explicitly retained as a failed historical oracle contract/diagnostic.
  The new acceptance does not reclassify the old failed process as passed.
- Source/resource/fallback and native B3LYP point regressions: 171 pass in
  3.83 seconds. Host-compiled equal-spin/native CUDA-source arithmetic
  controls: 19 pass in 11.58 seconds; these are not real-device evidence.
  Compiler dependency audit checks 604 modules with zero errors.
- The independently bounded whole-molecule regression cohorts also actually
  exit zero: six UKS tests (448.95 seconds), thirteen RKS/curvature tests
  (1339.12 seconds), nine canonical-ECP force tests (144.23 seconds), and
  four UKS/D3 total/public-composition tests (155.51 seconds). The original
  combined invocation reached its 1800-second outer limit after twenty test
  dots and produced no completed XML; that timeout remains retained and is
  not relabeled as a pass. Splitting the invocation changes neither source
  files nor numerical gates.

The native matrix/two-HVP scope takes 204.8955953 seconds, including source
checks and concurrent CPU qualification. It is not an isolated performance
claim. The native library was compiled before commit; the cohort is not an
installed wheel/public endpoint or canonical def2 qualification. Raw source,
library identities, XML properties, independent matrices and the scoped
receipt are retained alongside the manifest. The former failed-candidate
archive is not changed.

The immutable installed public Ni process uses its separately frozen older
source/library. This correction does not restamp that cohort or authorize
editing its prefix, driver or inputs. Public ECP and PBE0-UKS admission remain
closed. Canonical def2 total matrices, mixed Ni/I/Cs, scale and TS/frequency
qualification are still separate unresolved work.

## Subsequent canonical CsH RKS total matrix

The same library/source cohort additionally completes a low-level canonical
real-spherical def2-SVP / Cs def2-ECP46 RKS check on the native default grid:
29 AOs, 33,516 points, five doubly occupied orbitals. It computes every raw
6x6 electronic Hessian column and two fresh HVPs, then adds uncut two-body
D3(BJ) exactly once with elemental Cs identity. All original raw invariants,
response and component gates pass. Independent unseeded moving-grid gradients
give total matrix errors `7.063509033511384e-7`, `7.3622138857532e-8`,
`1.0704201497402366e-8` at the same three steps. The D3-only matrix errors are
at most `1.507828555583231e-11`, below its frozen `2e-9` gate.

The actual process exits zero in 1465.9052 seconds including all independent
gradients, concurrent CPU work and JIT; this is not an isolated performance
claim. Raw native/independent matrices, individual displaced gradients,
source identities and the scoped receipt are retained in
`build/qualification/canonical-cs-rks-total-budgeted-20261111/`.

Two preceding driver failures remain failed: one accessed a UKS-only layout
attribute on an RKS owner; the next omitted the explicit matrix output budget
required by the D3 composition contract. They are harness errors, not passed
total calculations or a reason to modify production dispatch. The accepted
driver retains the electronic raw matrix before later composition/validation.
This extends only low-level CsH RKS evidence; it is not a public/installed,
canonical UKS, I, mixed Ni or scale qualification.

## Rejected alternatives

Do not relax the absolute gate, scale the AO tangent, drop minority-spin
coupling, compare only OV projections, clip density tails, or replace native
responses with reference derivatives. Do not call the old failed test passed.
Revisit the exact-input oracle if its independent high-precision/component
controls cease to agree or a new numerical domain is admitted.

## Verified scoped recovery

The local member/hash-verified recovery
`build/qualification/uks-analytic-ecp-20261111/stable-b88-scoped-recovery.tar.gz`
contains 738 members / 73,801,073 bytes, SHA256
`342d58b9a233008ab345ca489ac52845ee1444b4ec6e946483e2232e0db55f10`.
Its manifest SHA256 is
`8840557fbb844a7618ffa6da4ca63b50008e829d169fcbcf36043d7278720b3e`.
It preserves the source/library cohort, complete synthetic and canonical CsH
RKS raw/reference matrices, JIT artifacts, all passed regression XMLs, the
original combined-suite timeout, and both failed canonical driver attempts.
Every member and original source was rehashed after writing. The archive
predates this footer and the historical proposal's lifecycle-label clarification;
it is not restamped. It explicitly excludes live installed Ni and canonical
CsH UKS calculations and does not confer public/installed or full ECP-domain
qualification. The earlier failed-candidate archive remains unchanged.
