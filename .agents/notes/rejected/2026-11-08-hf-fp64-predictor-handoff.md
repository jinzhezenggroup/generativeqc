# Decision: reject keeping the complete screened-FP64 DIIS ring at handoff

Status: rejected; experiment removed, no default or opt-in promotion
Date: 2026-11-08

## Problem

HF96 remains the main complete-cold gap. Historical strict-screened solves take
eight provisional and seven strict iterations; repeated exact K, not diagonal
solves, dominates. The handoff currently discards every DIIS direction despite
already using FP64 on both sides. The occupied-FP64 DFT handoff provides a useful
separate precedent, not HF qualification.

The older384-AO screened-proposal screen Slurm16823 passes the independent
HF48 force diagnostic but increases complete5090 cold from about6.35 to6.71s.
Its fifteen target/provisional Focks versus ten ordinary Focks do not justify
default expansion. That source9e29/ELF0bff experiment remains distinct from
current capability-based source and the present history hypothesis.

## Tested hypothesis and boundaries

For an otherwise admitted same-precision screened-FP64 cold owner, keep the
existing bounded DIIS ring as provisional predictors across handoff. Always
reset convergence, target iteration accounting and the physical energy baseline.
Every strict iteration freshly computes the requested physical Fock, energy
and residual; no provisional history entry can certify convergence or forces.
Chronological strict insertions overwrite every old predictor within the caller's
existing history capacity. No new history storage, operator, GPU-identity gate,
production oracle or increased iteration allowance is introduced.

During the experiment only existing explicit cold-screening1 retained predictors;
automatic eligible owners keep their existing full restart. This allows a
same-image control/candidate at HF96 without adding a new public switch. Warm,
mixed, UHF, physical-reference, seeded and refused owners retain their existing
policies. The earlier explicit384-AO diagnostic is separately rejected and removed
from the working policy; it is not a condition for solving the HF96 gap. The frozen
source682561e9 screen still contains that earlier admission for provenance,
but its HF96 unset/1 comparison never enters the rejected384-AO-only domain.

This tested an alternative to the complete history restart in
`../implemented/performance/2026-11-07-cold-hf-strict-screened-tail.md`.
The existing restart remains production behavior. Its strict physical target,
finite shared allowance and no-provisional-verdict requirements remain mandatory.
Retaining mixed-precision histories or treating temporary operator values as
target measurements was outside the experiment.

## Predeclared acceptance

Use frozen source/binary and finite actual A100/5090 allocations. Measure complete
cold E, analytic forces and host export plus each own-density frozen warm; report
provisional/strict/initial-guess work separately. Preserve every failed condition.
Require unchanged independent E1e-8/Fmax1e-7 and every paired warm within3%;
require a credible complete-cold benefit beyond the existing2%/noise floor.
Check displaced/restored, checkpoint, short allowance and refusal behavior
before promoting any automatic request. No kernel-only win, old-source timing
or looser force contract substitutes for these gates.

## Measured outcome and decision

Frozen source682561e9, complete ELFc368db4b, Slurm16828 ran four fresh-process
HF96 ABBA endpoints on RTX5090. Controls took16.920584/17.012708s; candidates
took18.175454/18.223213s, a7.265% mean complete-cold regression. The same eight
screened iterations were followed by eight strict target iterations rather than
seven. Initial-guess work remained separately reported. Every independent
E1e-8/Fmax1e-7 check and every paired warm3% guard passed, so this is a genuine
cold-work loss rather than a hidden reference or warm-speed tradeoff.

Reject this complete-ring retention policy and remove its implementation,
experimental switch behavior and temporary test files. Preserve the
frozen source patch, actual kernel-state test16826 and full endpoint observations
in the evidence capsule instead. The test's native CUDA compilation targets were
build provenance, not a production dispatch whitelist.

This does not prove that every possible transfer of solver information is bad.
It specifically disproves this candidate as a profitable explanation/solution
for the observed strict tail. Partial histories, target-recomputed residuals or
earlier provisional handoff would be different experiments requiring fresh
endpoint qualification; do not simply rerun this unchanged proposal.

The A100 counterpart LSF229249 confirms the rejection: controls took
14.584671/14.665497s and candidates15.981274/15.998760s, a9.333% mean cold
regression, again seven versus eight strict target iterations. Every independent
E/F check and paired warm3% guard passed. Neither GPU identity nor an opt-in can
turn this result into automatic promotion.

Build16825 completed successfully, used the verified owned compiler cache and
restored the borrowed source/runtime exactly. CPU-only delta production16827
first failed because zstd refused a symlink input; the resolved-file retry fully
reconstructed the exact target. It did not repeat a GPU solve. A100 setup229248
and both real-GPU populations completed once; the owned upload receiver was
audited by UID/cwd/executable/argv and stopped only after both uploads arrived.
All33 original adaptive driver change lines and the other12 dirty paths remain
unchanged and excluded from the experiment and PR.

## References

Issue2976; draft PR3048; local immutable experiments under
`.artifacts/hf-fp64-predictor-handoff-20261108/`; lossless published capsule at
`benchmarks/results/hf-fp64-predictor-rejection-20261108/`.

The distinct four-step earlier-handoff experiment is also rejected on both
cards; see `2026-11-08-hf-earlier-screened-handoff.md`. Its source and observations
remain separate. All raw records and complete compiler stdout now share lossless
semantic compression across these related experiments; per-file identities and
the historical numerical verdicts are unchanged.
