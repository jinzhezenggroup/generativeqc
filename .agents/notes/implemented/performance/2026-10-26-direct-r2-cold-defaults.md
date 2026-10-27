# Decision: cooperative setup and bounded pure-J proposal work

Status: implemented
Date: 2026-10-26

## Problem

After the preceding cold HF/DF improvements, Direct r²SCAN-3c96 remained slower
than GPU4PySCF. Setup repeated nuclear and symmetric Schwarz work. More
significantly, the existing finite pure-J proposal epoch requested a `1e-9`
screening allowance while the MD engine still limited every invocation to
`1e-12`. The proposal facade therefore left most potential source omissions
unused even though complete physical closure already used the requested target.

## Decision

Use eight cooperating lanes per AO pair for one-electron values on qualified
CUDA providers. Partition only the nuclear source loop, evaluate overlap and
kinetic channels on lane zero, and reduce projected channels in FP64. Reuse
the existing scalar compiler IR. Explicit thread, shell-warp and 32-lane
schedules remain available; providers without the subgroup capability retain
the thread schedule.

Compute one canonical public-AO Schwarz triangle and publish both symmetric
entries, including permuted AO order. Packed through-d Schwarz uses 32 threads
per block; larger angular frames retain the previous bounded route. The
materialized dddd force owner defaults to its existing class-major topology,
with unchanged cache/topology/domain guards and generic bounded fallback.

For the restricted primitive pure-J proposal facade only, copy the MD view
and cap its omission allowance at `min(proposal_screening, 1e-9)`. The
cumulative, directed-rounding proof is unchanged; it receives the explicit
proposal allowance. Never mutate the prepared physical view. Hybrid,
unrestricted and nonprimitive invocations keep their previous cap. Full-density
closure, graph capture and warm replay retain `min(target_screening, 1e-12)`.
The existing limit of eight provisional deltas, convergence settings and full
physical closing energy pair remain unchanged. The existing coarse-screening
off control retains the tighter proposal route.

## Evidence and limits

The source-bound capsule at `benchmarks/results/cold-r2-defaults-20261026/`
retains clean alternating default endpoint populations, semantic work,
independent observable gates, larger/displaced geometry checks and rejected
experiments. Direct r²SCAN-3c96 cold takes ten Fock builds with both proposal
caps; the improvement removes primitive source work rather than hiding
iterations or skipping complete analytic forces.

The default v2 three-round cold median is 31.3617 s against GPU4PySCF 33.1508 s;
the ranges are [31.3293, 31.5113] and [32.5021, 33.2414] s. The preceding v1
combination remained slightly slower: 32.5703 versus 32.4400 s. All v2 main
r²SCAN populations pass unchanged common gates. Two earlier cap experiments
also showed 31.38–31.45 s versus 32.55–32.59 s. These are sampled water-cluster
observations, not a universal method/hardware performance guarantee.

Signed-density oracle checks cover Cartesian/spherical pure and hybrid plans,
restored physical actions, and graph capture immediately after a provisional
action. The host allowance proof covers zero, tighter and coarser screening.
Separate work-counter execution is intrusive and supplies no headline timing.
It reduces cold primitive roots by 17.74% and contraction summands by 17.54%,
while full-density closing and warm counters remain exactly equal to v1.
The measured archive, extracted source and actual compiler inputs are audited;
the frozen dirty source is not mislabeled as the parent commit's exact bytes.

Inherited DF HF96 fitting failures and the stronger default-HF force diagnostic
failure remain visible in their original evidence. Neither this change nor the
README refresh relaxes those gates.

## Rejected alternatives

Reducing MD cursor workers from 4096 to 512/1024/2048 did not improve complete
cold endpoints. The 512-point SCF tile and XC matrix tile 8 did not provide a
stable cold gain and regressed warm behavior. Split direct linking was slower;
a complete no-RDC build timed out after 60 minutes and provides no valid timing.
Keep these measurements and the timeout rather than promoting promising
microbenchmarks. The research-only worker and cap environment controls are
absent from production.

## Revisit when

Requalify if the primitive omission certificate, finite proposal controller,
full-density publication proof, subgroup capability or materialized force
ownership changes. Require complete cold/warm/moved endpoints, signed-density
oracles, graph-lifetime checks, bounded fallbacks and larger sizes before
broadening admission. Do not increase the cap merely to win a benchmark.
