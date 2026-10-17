# Decision: retain generated Direct-force recurrence and density reuse as opt-ins

Status: implemented opt-in; integration/default-promotion gates remain open
Date: 2026-10-17

The later [default-promotion decision](2026-10-17-direct-force-default-promotion.md)
supersedes this note's opt-in-only policy after testing the unset selector. The
source-matched measurements and rejected experiments here remain historical;
promotion does not turn them into measurements of unmodified current master.

## Decision and source boundary

Retain the order-six relative-response recurrence consumer in PR #2587 and the
order-six/seven density-first Hermite contraction in stacked PR #2588. They
reduce scientific work, not just candidate enumeration. Keep explicit selection
and the original bounded component fallbacks; do not enable a default, merge a
PR, or replace homepage results from these trials.

This supersedes the pending endpoint gates in the earlier
[order-six trial](../../proposed/2026-10-17-direct-order-six-shell-relative.md) and
[density-first trial](../../proposed/2026-10-17-density-first-direct-force-contraction.md).
Their preliminary failures and operator-only measurements remain historical.

The frozen source composes master `a6ac79757` with the selected scientific force
producer from #2558. It excludes the old cold-planning stack, occupied-XC #2577
and guarded FP32. Compilation used non-Git exports, subsequently authenticated
byte-for-byte against immutable reconstructions; do not call them unmodified
master or clean Git builds. The density native identity is
`d236a780a6402fc4d526e347b60d70ff3dbc2aad5e7c28e2065253ce8d162b74`.
Later test, typing and formatting changes retain all fourteen generated fixture
headers byte-for-byte, without substituting their newer identity for the binary.

## Complete endpoint qualification

Finite Slurm allocations preserve assigned visibility, strict FP64, exact Direct
J/K, canonical def2-mTZVPP, spherical AOs, moving unpruned 48 x 16 x 32 grids,
original screening/convergence and the 1800-second population cap. Populations
include cold, five warm, moved and five moved-warm calls. Warm figures are
complete-endpoint medians, not kernel sums.

Node4 RTX 5090, Slurm1109, whole change `spatial-reuse` -> `relative-density`,
96 atoms / 1088 AOs, seconds:

| Phase | Baseline | Candidate |
| --- | ---: | ---: |
| Cold | 256.959771 | 244.429102 |
| Warm | 43.195500 | 30.409994 |
| Changed geometry | 157.405227 | 144.724426 |
| Changed-geometry warm | 43.145708 | 30.422604 |

Forward warm gains are 29.60%/29.49%; the separately declared reverse repeat
gives 29.68%/29.68%. Both arms and both orders actually use 24 iterations/24
Focks cold, 13/13 moved and 1/1 every warm call. All eight primary populations
and two reverse populations complete; all 120 returned results pass independent
geometry-specific 1e-8 Eh / 1e-7 Eh/Bohr gates. Across the primary populations,
maximum errors are 2.502e-11 Eh / 6.308e-11 Eh/Bohr. R2SCAN-3c48, PBE048 and
HF48 controls have no >5% regression. The old PBE0 cold regression is not
reproduced in this cohort. PBE0/HF retain their existing def2-SVP control basis.
Across all 120 main/reverse calls, maximum errors are 2.502e-11 Eh /
6.857e-11 Eh/Bohr; no population is incomplete.

The independent order-six-only RTX 5090/node5 cohort reduces warm
43.320766 -> 35.680117 s and moved-warm 43.371734 -> 35.719521 s, reproduced
in reverse order. Do not add these gains to the whole-change percentages.
Its reverse cold actually takes 25/25 -> 24/24; secondary incremental density
moved takes 14/14 -> 13/13. Neither trajectory difference is an isolated kernel
gain. Clean diagnostic density cold instead takes 24/24 -> 25/25, and remains
in the evidence rather than being normalized away.

## Actual work, clocks and residual bottlenecks

Source-matched counted warm device times for order six/seven/eight are
13.808697/2.000894/0.453909 -> 2.612973/0.492687/0.454147 s. Order-six entered
primitive-pair products fall 2,400,504,912 -> 8,666,248 with 4,101,784 admitted
shells unchanged. Density factorization additionally executes 1,882,836,519
density terms, 3,153,588,800 Hermite joins and 8,455,560 Coulomb preparations.
Unknown aggregate recurrence coverage remains null, not an invented FLOP count.

Order seven retains 1,250,544 primitive-pair products. Its 754,811,008 old AO
contractions become 4,136,704 fallback contractions **plus** 453,127,296 density
terms and 870,721,600 Hermite joins. Reporting only fallback counts hides the
new work. Nonzero exchange, seeds, partial masks and nonfinite reordered inputs
decline before publication and use the original component fallback. No molecule,
functional, AO-count or GPU-model dispatch is introduced.

Clean candidate warm device scopes include low orders <=5 together 4.614272 s,
order six 2.511868 s, order seven 0.480572 s, order eight 0.453249 s, SCF exact J
4.237880 s, XC/AO/grid 3.962804 s, one-electron response 2.272368 s and other
stationary/grid response 9.319335 s. Low <=5 is not isolated order five.
The latter owner decomposes without overlap: geometry point 2.322134 s,
geometry cooperative 1.740051 s, geometry reduction 0.096940 s, center validation
0.025681 s, phased Becke 2.490280 s, AO radial 0.509711 s, CUTLASS contractions
1.934119 s and other 0.200419 s. Do not call all 9.319335 s Becke response.

Complete exclusive Python wall ledgers, exclusive NVTX host/device windows and
kernel device sums are different reconciled views, not additive categories.
Preparation, synchronization and final-state validation stay in the endpoint.
Counter-enabled profiles are diagnostic only.

## Same-device peer and next boundary

Fresh node4/Slurm1109 GPU4PySCF1.8.1 / PySCF2.14.0 / cuTENSOR2.2.0 uses the
same GPU UUID and unchanged protocol. Complete seconds are
142.375318 / 28.830030 / 120.502940 / 26.643178 for cold / warm / moved /
moved-warm. All twelve calls independently pass the pinned geometry oracle;
maximum errors are 1.819e-12 Eh / 4.674e-12 Eh/Bohr. Actual iterations/Focks are
54/55 cold, 45/46 moved, 4--5/5--6 warm and 3--4/4--5 moved-warm. Preserve these
different solver trajectories without normalization.

The retained candidate is still 5.48% slower warm, 14.19% slower moved-warm,
71.68% slower cold and 20.10% slower moved. It has not reached complete-endpoint
parity. Historical 26.59 s is not this fresh warm median. One peer population
does not establish a reproducibility envelope.

Next cold work should reconcile #2577's independently qualified occupied-frame
XC consumer before duplicating it. It reduces SCF density-feature work, not warm
stationary forces. Remaining force opportunities include low <=5, one-electron
response and separately measured grid-response components. Reusing accepted
SCF point-model results for stationary geometry points needs exact final-density/
grid identity, ownership/lifetime checks and bounded fallbacks; neither PR does
this. The existing cooperative reducer already uses warp shuffles and two CTA
barriers, not an eight-barrier baseline.

## Evidence and limits

Independent raw-AD/host-orbit fixtures, sixteen linked Libcint cases, both spins,
signed/seeded sources, both representations, partial masks, coincident atoms and
nonfinite fallbacks pass. Generated native memcheck/synccheck/racecheck pass with
zero errors/hazards. Full linked memcheck/synccheck pass; unfiltered linked
racecheck exceeds 420 seconds. Only the separately declared two-launch bounded
linked racecheck passes. The charged UKS dimer's two pre-derivative SCF failures
remain negative evidence, not successful force tests.

Raw exports, returned force arrays, profiler databases, convergence, compiler-
cache/resource and scheduler receipts remain ignored under
`.artifacts/direct-parity-20261017/`. PR bodies retain lossless scalar receipts,
not offline force-array replays. Historical packet recovery follows the separate
evidence-retention note. Do not bypass the evidence budget or publish a Release.

Both PRs stay default-off. Density fixtures use 31,720 shared bytes; register and
stack receipts are instantiation-specific, not measured occupancy. Current-master
integration, formal peak-allocation/compile-cost envelopes and default promotion
remain separate qualification gates.

Three further recurrence/CSE trials are documented in the
[rejected-follow-ups decision](../../rejected/2026-10-17-direct-density-followup-trials.md).
Their scientific changes are not included in the retained implementation.
