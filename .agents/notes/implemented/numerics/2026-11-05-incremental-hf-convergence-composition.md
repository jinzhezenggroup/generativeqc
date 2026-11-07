# Decision: compose incremental HF with ordinary convergence and final-state reuse

Status: implemented; incremental J/K remains opt-in
Date: 2026-11-05

Later qualification: [96-atom default-promotion decision](../../rejected/2026-11-05-hf-incremental-default-promotion.md)
retains opt-in status after the cold-start effect fails the performance floor.
The [source-matched final publication](../../../../benchmarks/results/hf-incremental-composition-20261105/README.md)
retains independent E/F requalification of the delivered repair.

Follow-up: [overlap full-energy refinement with the SCF tail](../performance/2026-11-05-incremental-hf-early-full-refinement.md)
supersedes the transition timing below, not its acceptance or finalization
contract. The measurements in this note remain the pre-follow-up snapshot.

## Problem

The [default-toggle diagnostic](../../rejected/2026-11-05-hf-incremental-jk-default-toggle.md)
compared identical public E/D controls but different internal policies. Turning
on incremental J/K disqualified ordinary all-electron restricted FP64 Standard
stationarity, hardcoded a strict physical gate before full-energy refinement,
disabled the bounded ordinary resident source, and disallowed converged-state
reuse after actual delta work. It was therefore not a comparison of just full
versus density-difference contraction. The historical measurements remain valid
for their source, not for this repaired implementation.

Exact J/K is linear in density, but density-weighted screening and floating-point
reductions are not bitwise-linear. A repair must unify scientific acceptance
without assuming an identical SCF trajectory or publishing a reconstructed
operator as an independently verified physical Fock.

## Decision

- Owner qualification no longer depends on incremental J/K. Ordinary restricted
  all-electron FP64 HF uses the same capped Standard stationarity limit with
  either schedule. Explicit tight controls, UHF, ECP, mixed precision and
  detached physical-reference exports keep their existing qualification limits.
- Full-energy refinement consults that same owner-qualified stationarity limit,
  instead of a separate hardcoded strict gate. After any actual delta work,
  convergence first enters sticky full-only refinement and requires two fresh
  full-density energies. Discard a reconstructed current energy; retain a current
  full evaluation as the first usable baseline without comparing it to an older
  delta energy. A stationary no-delta full step retains ordinary convergence;
  if energy alone requires another step, keep full-only rather than start deltas.
  This also
  covers unscreened rolling deltas, so retained publication never depends on
  assuming exact cancellation of floating-point reconstruction error.
- The convergence kernel retains the certified current density, not its next
  proposal, just as for ordinary direct HF. The tracked determinant orbital
  frame, energy, physical Fock, Pulay weight and returned restart density stay
  on that generation. The warm-only density-restoration workaround and its
  extra host census/fence are unnecessary and removed.
- The existing phase-local resident source may consume full or delta inputs
  for ordinary HF under the unchanged admission/budget policy. Its phase graphs
  still drain before the source is destroyed. Resident contractions do not
  populate streamed quartet counters; mark that census unavailable rather than
  manufacture zero work. Full/delta Fock counts remain available.

## Invariants and boundaries

- Keep fixed screened full-density anchors, requested refresh cadence, finite
  gates, energy roundoff allowance and the original total iteration bound.
  Full-only refinement consumes that bound and is never itself convergence.
- Keep physical-Fock residual and determinant/orbital-frame proof. Unproven
  external or changed-geometry seeds and the explicit final-rebuild diagnostic
  retain the canonical physical rebuild/residual fallback. Detached references
  retain their stronger independent canonical reconstruction proof and bounded
  fallback. Do not restore density from a stale full anchor after later deltas.
- Ordinary OFF execution is unchanged. Do not obtain a speedup by slowing OFF,
  changing caller tolerances, borrowing an oracle seed or timing only kernels.
- Resident selection is optional, resource-bounded and scoped to one execution;
  tiny/oversize, unsupported or capacity-refused sources recompute directly.
  Mixed arithmetic and the separately opted-in provisional cold-screening
  experiment remain uncomposed; this repair does not widen those numerical
  qualification domains. KS policies and the HF default switch are unchanged.

## Validation

Native GPU coverage exercises Standard and strict full-energy transitions,
restricted and unrestricted residuals, both density-retention modes, nonfinite
and inactive items, accidentally passing delta energies, sticky interval-zero
refinement, and two full evaluations before acceptance. Independent two-center
PySCF E/F checks cover cold, repeated resident warm, external and changed-geometry
states, refresh intervals 0/1/8, unscreened rolling anchors, source-capacity
refusal, and explicit final-rebuild overrides.

The new Python regression checks independent live full-density CPU PySCF
original/displaced references at 96 AOs. It exercises both full and incremental
schedules, Standard and strict controls, and resident-disabled/automatic sources
through complete cold, frozen warm and moved energy/force returns. It does not
feed reference density into native SCF or require identical iteration counts.

Source-matched qualification uses Release CUDA 12.9.1, official sm120, AOT on,
fast compile off, and verified sccache 0.16.0 on n1/node1. The isolated owned
build reuses the previous cache without clearing it. The original diagnostic
binary is preserved separately before rebuilding.

- Native source identity:
  `be3b91e2ce7238febae1d27dfc56542ffe333bdc7b25736e6067a42e25d11041`.
- Library SHA256:
  `cc9214f9414f7080f90e94e90f801ee0e4482d1e14a880d03eb432a6dcb4e98c`.
- Base revision: `7618eb29e696daab073c027728e88a9c78a8110c`, plus this patch.
- Slurm job `15963`: two native suites and four live-reference composition
  cases pass. The focused host fixed-anchor/default-control suite passes 43 tests.
- Final-source Slurm job `15964`: both native suites pass Compute Sanitizer
  memcheck with zero errors, and the focused 43-test host suite passes again.
  Assigned GPU visibility remains `3`; no GPU work bypasses Slurm.

Raw receipts and the unchanged matched-endpoint harness are retained under
ignored `.artifacts/delta-jk-20261105/` and the owned n1 directory
`/data/jzzeng/qc888-delta-jk-20261105/`. The failed first GPU test attempt is
retained: its numerical checks pass, but the new test incorrectly assumed a
two-AO source bypassed the hard 64-AO resident admission floor. Correct the
test to require bounded fallback; do not weaken production resource admission.
The initial local host run lacks a native library (25 tests pass, 18 setup
errors); the source-matched remote rerun passes all 43. These are not successful
GPU endpoint samples and are not included in performance populations.

An intermediate repair (`c788930f...`, job `15961`) passes all numerical
checks but exposes a strict 96-AO warm regression: a resident-source cold
baseline needs an ordinary second full evaluation on warm replay. Starting a
delta for that energy-only tail then forces two more full checks. The final
controller keeps a stationary full step's usable baseline and stays full-only;
native coverage independently tests both no-delta energy-only tails and fresh
full evaluations after previous delta work. The intermediate binary, timings,
and clean memcheck receipts (`15962`) are preserved, not relabelled as final.

## Complete endpoint measurements

Final job `15963` repeats the original matched RHF/def2-SVP spherical FP64
96/192/384-AO campaign. Each arm has four independent sequences, ordered
AB/BA/BA/AB. Cold includes Calculator construction, preparation and complete
host energy/analytic-force return. Common imports/tiny-water bootstrap are
excluded; cleanup is recorded separately. Warm and moved independently replay
each sequence's own frozen post-cold density. Screening remains `1e-12`, refresh
interval eight, and no oracle density enters production.

Median seconds, OFF / incremental ON:

| AOs | Standard cold | Standard warm | Standard moved | Strict cold | Strict warm | Strict moved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.361068 / 0.372844 | 0.117167 / 0.117355 | 0.482537 / 0.509810 | 0.443315 / 0.462324 | 0.161640 / 0.161401 | 0.761330 / 0.781329 |
| 192 | 1.529622 / 1.633152 | 0.252401 / 0.250760 | 1.301683 / 1.250906 | 2.326895 / 2.152737 | 0.251908 / 0.251334 | 1.990479 / 1.708467 |
| 384 | 5.040996 / 5.389857 | 0.733186 / 0.731407 | 3.882105 / 3.524167 | 7.471498 / 6.774284 | 0.735829 / 0.734483 | 6.376679 / 4.754485 |

Standard is `E=1e-9`, `D=1e-5`; strict is `E=1e-12`, `D=1e-9`.
At 384 AOs, repaired Standard ON cold/moved is 24.9%/28.0% faster than the
original ON implementation, but cold remains 6.9% slower than matched OFF.
Moved improves 9.2% against OFF. Strict ON cold/moved improves 9.3%/25.4%
against OFF. The 192-AO Standard moved improvement is 3.9%; small cold/moved
endpoints still lose. These are diagnostic scope-limited results, not uniform
speedups or default-promotion authority.

All new warm endpoints are effectively at matched OFF latency. Strict 96-AO
ON now takes two full steps, like OFF, rather than the original ON's one-step
0.116678-second warm branch. Its 38% increase relative to that old ON warm
branch is retained as a tradeoff of restoring cold resident scheduling, not
hidden by comparing only to OFF. In return, strict 96-AO ON cold falls from
1.031973 to 0.462324 seconds. Warm samples without deltas are not evidence of
delta-contraction work savings.

Semantic target work at 384 AOs:

| Controls/phase | OFF SCF full | ON SCF full + delta | Final full OFF / ON |
| --- | ---: | ---: | ---: |
| Standard cold | 10 | 3 + 8 | 0 / 0 |
| Standard moved | 7 | 3 + 5 | 2 / 2 |
| Strict cold | 16 | 4 + 14 | 0 / 0 |
| Strict moved | 14 | 4 + 12 | 1 / 1 |

Both cold arms additionally perform seven native preliminary-guess Fock
builds. Every formal target has one attempt, zero discarded target work and
no warm fallback. Retention removes the old two-build post-SCF penalty for
qualified cold incremental states; moved states keep the same bounded final
fallback as OFF. Remaining iteration differences reflect full confirmation
and floating-point/screening trajectories, not a stricter incremental-only
acceptance ceiling. Streamed quartet counts at 192/384 AOs and resident
96-AO cold counts are unavailable; do not infer integral-work savings from
zero census fields or Fock application counts.

All **144 final formal E/F endpoints** pass the same predeclared independent
tight CPU PySCF references from
`benchmarks/results/incremental-hf-full-energy-refinement-20261015/validation.json.xz`.
Standard gates are `2.5e-6 Eh/atom`, maximum force `2.5e-5 Eh/Bohr` and RMS
force `7.5e-6 Eh/Bohr`; observed maxima are `3.36514e-11 Eh` absolute,
`6.33211e-7 Eh/Bohr` maximum force and `2.66119e-7 Eh/Bohr` RMS force.
Strict gates are `1e-8 Eh` absolute and `1e-7 Eh/Bohr` maximum/RMS force;
maxima are `9.09495e-12 Eh`, `1.75886e-10 Eh/Bohr` and `7.14737e-11 Eh/Bohr`.
The archive hash, exact inputs, every returned force, library identity and
assigned GPU visibility (`3`) are retained in the reports. Matched OFF medians
stay within roughly 0.6% of the old OFF measurements except the 96-AO strict
cold variation (-2.0%); there is no OFF slowdown creating the large ON gains.

## References

- Supersedes the strict incremental-only admission and redundant finalization
  boundaries diagnosed in the [negative default-toggle experiment](../../rejected/2026-11-05-hf-incremental-jk-default-toggle.md).
- Extends [full-energy refinement](2026-10-15-incremental-hf-full-energy-refinement.md)
  and replaces the narrow [no-delta warm workaround](../performance/2026-10-15-incremental-full-only-warm-finalization.md)
  with ordinary same-generation convergence retention after full confirmation.
- Current contract: [direct CUDA HF finalization](../../../../docs/developer/hf_force_finalization.md).
