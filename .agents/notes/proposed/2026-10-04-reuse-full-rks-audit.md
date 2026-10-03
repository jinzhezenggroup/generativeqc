# Proposal: reuse an already full-density RKS final audit

Status: proposed; optional incremental-KS qualification, not default promotion
Date: 2026-10-04

## Problem and retained negative result

The initial #1803 prototype always entered an additional full-density closure.
The frozen source 8269ab44/library0455a330 48-atom campaign passes all independent
E/F comparisons, but OFF warm calls take 17.82–17.87 s with one build. ON warm
calls take 143.253, 22.395, 22.421, 148.156 and 22.387 s. The first and fourth
ON calls explicitly report `warm_start_fallback=true`; returned diagnostics
describe only the successful cold retry, not all endpoint builds. Those
failures and costs must not disappear from the qualification evidence.

Even an ON warm call without retry executes one full anchor build plus one
full closure build. That duplicates expensive full-range J/K and XC at a state
whose physical full-density residual has already passed the ordinary RKS gate.
The relation between the extra closure and the retry failures still needs a
separate diagnosis; removing one cannot be claimed to cure all retries.

## Decision under qualification

For all-electron RKS only, reuse a converged iteration whose observed device
prepare flag says it built full density. It has current physical J/K, freshly
evaluated nonlinear XC, energy-change/density-change and RMS/maximum residual
checks at the same accepted density. Final-state export still diagonalizes the
unshifted physical Fock and validates canonicality and stationary weights.
The existing FinalAudit event describes those checks without inventing another
Fock build. The actual build remains an anchor/refresh count, not a post-SCF one.

A delta-built converged RKS state still enters bounded full-density closure.
UKS and ECP retain their independent proposal-correction requirement regardless
of the full/delta flag. No threshold, cache lifetime, failed-attempt isolation,
geometry identity, or nonconverged publication rule changes.

This supersedes only the unconditional-RKS-closure invariant of
`2026-10-04-shared-incremental-ks-jk.md`; its frozen observations remain historical.
It does not establish integral-work reductions for delta builds.

## Validation gates

Host decision tests must cover full/delta, RKS/UKS and ECP, including OFF.
Native independent CPU components and unshifted final-state checks must pass.
Public independent spherical def2-SVP E/F and reconverged FD remain required,
followed by every 48/96-atom endpoint and interleaved causal timing controls.
No runtime measurement is claimed for this follow-up until those tests run.

The shared linear-kernel test adds a standalone `--linear-kernels-only` mode
and cross-block/tail sizes to isolate sanitizer coverage. This cannot erase the
retained whole-HF synccheck failure in density-bound reduction, reproduced on
unmodified master. Full-suite and isolated results must remain separately named.

## First source-scoped qualification

Source `fe6aea98cf343d34e338cf0d039a2de219fb3f013a63f74448a17d24e3860527`,
library `13e306950db6e53906bf0f736c784354a0bf1407bf29568716ef90a91e038068`:
n1 job 5669 passes both native incremental-HF/complete-CUDA-KS suites, including
independent CPU components and unshifted final-state validation. The isolated
linear-kernel memcheck/initcheck/synccheck/racecheck all pass with cross-block
and inactive/tail controls. This does not qualify the unchanged density-bounds
kernel whose whole-HF synccheck failure remains open.

All seven public controls pass: three shape/overflow cases, and OFF/ON spherical
def2-SVP water RKS and water-cation UKS complete E/F, warm and two-step reconverged
FD. The RKS warm control performs exactly one full build, zero delta builds and
zero extra closure builds, while retaining one final residual audit. UKS still
requires its separate closure. Nineteen focused host checks and changed-file
pre-commit pass. The ccache build runs on n5 with explicit CXX/CUDA launchers;
the library transfers directly to the separate n1 measurement tree.

Jobs 5670/5671 run fresh-reference/OFF/ON 48/96-atom complete campaigns, and job
5672 measures H2CO independently. Their receipts additionally retain returned
histories, explicit warm-retry flags and cumulative transport observations to
distinguish returned-attempt counts from discarded work. These campaigns are
still incomplete at this checkpoint; no large-system improvement is claimed.

Master has advanced to 79418329e. These source-scoped runs retain the original
837c2a51c base; they do not automatically qualify that newer integration.
