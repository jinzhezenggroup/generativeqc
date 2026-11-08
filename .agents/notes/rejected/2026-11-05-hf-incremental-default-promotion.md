# Decision: defer default HF incremental J/K after the 96-atom check

Status: rejected default promotion; convergence/composition repairs retained
Date: 2026-11-05

## Problem

The user conditioned default promotion on a genuine **96-atom** improvement,
not the previous 96-AO/12-atom fixture. A fresh independent full-density
GPU4PySCF oracle qualifies original and displaced energy/forces for a
96-atom/768-AO spherical def2-SVP cluster. The repaired incremental algorithm
is scientifically acceptable, but that alone does not establish a sufficiently
large cold-start performance improvement.

## Decision

Keep HF incremental Direct J/K experimental and opt-in. Retain the
[convergence/composition repair](../implemented/numerics/2026-11-05-incremental-hf-convergence-composition.md)
and [early full-energy refinement](../implemented/performance/2026-11-05-incremental-hf-early-full-refinement.md).
Do not relax numerical acceptance or the performance promotion threshold to
turn a small positive median into a default-policy win.

The tested automatic candidate admitted singleton exact CUDA RHF, explicit
FP64, spherical all-electron s/p/d shells, 12–96 atoms, 96–768 public AOs,
screening `1e-12`, and no physical-reference export. Explicit HF cold-screening
experiments suppressed automatic admission. Its actual unset-selector path
passed independent E/F and native convergence tests, not just an explicit-on
surrogate. The candidate method predicate and automatic default are removed
from the delivered implementation because the performance condition is unmet.

## Evidence and interpretation

Slurm `15983` measured explicit OFF/ON on the repaired pre-default source;
`15996` tested the actual unset/default candidate against explicit OFF after
rebasing to the merged DF-KS repair. Each uses four AB/BA/BA/AB sequences, with
separate native frozen cold densities and unchanged independent E/F gates.
Four measurements per side are descriptive, below the shared protocol's
five-per-side minimum for formal performance promotion.

The default candidate source identity is
`6122b317aad07cb06d44e159414ce8fcd24a77c1f587fb9107e37c205275c2d3`;
library SHA256 is
`1179d53359d69783477a1f63116346f7f3b323dd8006903ed834d0eb1fc98a43`.
Its 96-atom Standard cold medians are `16.751647 / 16.587898 s`, about **0.98%**
lower latency. Frozen warm is `2.872909 / 2.875717 s`, effectively unchanged;
moved is `14.056375 / 11.508223 s`, about **18.1%** lower latency. The earlier
explicit-on experiment found about 1.1% lower cold latency, consistent with a
small effect rather than a robustly large improvement.

Standard cold is not uniformly faster even in the bounded candidate domain:
the 96-AO and 384-AO default-candidate medians regress by about 0.51% and 1.31%.
Strict and moved endpoints can benefit materially; do not substitute those
populations for the Standard cold-start question after seeing the results.
The shared promotion rule requires improvement exceeding 2% and twice the sum
of relative MADs, with resource/compilation and independent solver evidence.
The 96-atom Standard cold median fails the 2% floor even before those additional
requirements. More repeats cannot make that same effect size pass the floor.

An initial default qualification job, `15995`, passed all three native suites
and the four live CPU-reference composition cases, then stopped because the
host-probe launcher expected `ccache` and was not given the available `sccache`.
`15996` fixes the runner environment and passes all 56 focused host tests and
both memcheck suites. This setup failure is retained separately, never counted
as a timed sample or a numerical failure.

Cold denotes calculator construction, preparation and complete host E/F;
common imports/library metadata and tiny-water bootstrap are outside timing.
Cleanup is separate. Both warm and moved independently replay each arm's own
frozen cold density. Resident/streamed quartet census is unavailable, not zero.
No CPU/PySCF/reference density enters production. GPU work always uses finite
Slurm allocations with assigned device visibility preserved.

## Delivered opt-in qualification

After removing the automatic predicate, Slurm `16001` qualifies the delivered
opt-in source separately with six independent sequences per side at all four
sizes and both profiles. All 288 complete timed E/F points pass, along with
three native suites, four live CPU-reference composition cases, 61 host tests
and two zero-error memcheck suites. Native identity is
`8cecc4e053d209fdcf37d9f7323757f2885018aa899590fc20235b94d64d7e18`;
library SHA256 is
`67bf62291385eccecc5185bd4364d458fdfecbfb6dd69135673bd2d18aa84e47`.

96-atom Standard cold is `16.764247 / 16.610585 s` OFF/ON, about 0.92% lower
latency; moved is `14.062550 / 11.513996 s`, about 18.12% lower. Strict cold
is `23.679828 / 21.332947 s` and moved `22.490326 / 16.186528 s`. Frozen warm
is effectively unchanged. The larger sample count confirms the small Standard
cold effect; it does not waive the >2% floor or missing full native histories,
independent final-native-density residuals and measured allocator peaks.

The [compact reviewed publication](../../../benchmarks/results/hf-incremental-composition-20261105/README.md)
retains final/candidate full-precision samples, independent references, both
losslessly encoded source reconstruction patches, receipt hashes and an explicit rejected
performance decision. No external archive or release is required.

## Revisit when

A candidate improves the actual Standard cold endpoint beyond the shared
performance floor on the admitted cohort, with complete independent scientific,
resource and compilation qualification. Keep strict and geometry-update results
visible without advertising a universal default speedup. DFT has a separate
guarded cold default and canonical/proposal owner; examine it independently.

## References

- `docs/maintainer/validation.md`, shared results and performance protocol
- `docs/user/batched_hf.md`
- `benchmarks/hf_incremental_direct_jk.py`
- Ignored local evidence: `.artifacts/delta-jk-20261105/`
