# Decision: reciprocal uniform MD-J radial source reuse

Status: implemented
Date: 2026-10-10

## Problem

The master PBE0 96-atom exact-direct cold profile attributes 41.59% of native
SCF device time to MD-J, 32.23% to XC/AO and 23.20% to independent K. The
uniform oriented J consumer rebuilds the same native Coulomb auxiliary for
both density directions. Its source repetition, not a dddd-only K kernel or
an absent compact-XC enable flag, is the first complete-endpoint opportunity.
The [investigation](../../proposed/2026-10-10-master-pbe0-96-cold-roadmap.md)
retains the separate clean/intrusive definitions and subsequent roadmap.

## Decision and invariants

The integral compiler emits reciprocal uniform contraction for unordered
pair-angular classes through two. Both directions use the existing native
`fill_coulomb` result once; parity under reversed displacement determines the
reverse contraction sign. Each output direction independently retains the
incumbent minimum-Schwarz and upward-rounded density contribution admission.
The diagonal primitive contributes once. Neither consumer acquires the other
direction's screening authority.

An 8x8 primitive tile uses 64 source lanes, shared tile reductions and at most
4096 persistent workers on the owner's existing stream cursor. Angular-two
has 80 publication slots, so publication must stride by the block width;
simply changing an angular guard truncates valid output. No new production
queue, tensor, device allocation or CPU/reference computation is introduced.
Higher angular uniform work, residual work, native generated K, XC and forces
remain independent and unchanged.

Unset or `GENERATIVEQC_MD_J_RECIPROCAL=1` selects this qualified schedule;
`0` retains the oriented incumbent. The native owner freezes policy before
replay, and invalid active values fail. The existing 128-MiB allowance,
optional-allocation rollback, normal-J and public-ledger fallback remain.
Sharing changes floating-point accumulation order, not radial mathematics;
global publication order is not deterministic. SCF histories must therefore
remain part of endpoint evidence rather than be forced to equal iteration
counts with a different DIIS policy or normalization.

Optional work instrumentation uses compile-time counted/uncounted kernels
and a 1200-byte owner-local allocation charged within the same cap. Uniform
radial evaluations, density directions and Hermite summands are distinct
counts. Residual candidate tests, admitted shell tasks and primitive radial
evaluations are distinct counts. Counter fences/downloads are diagnostic and
must not enter clean timing populations.

## Rejected alternatives and evidence trap

The angular-zero/one-only prototype improved fixed-density J but regressed
complete cold by 4.16% (Slurm 7003); it failed the robust gain gate and was not
promoted. The expanded through-two candidate addresses materially more work,
not a looser convergence or screening contract.

Public host `FockPlan.evaluate` executes the retained compatibility evaluator
even when preparation reports MD admission. The original 14 tests therefore
did not qualify the resident kernels. A test-only adapter now calls the actual
native-KS prepared device seam without adding a production ABI. The corrected
tests exercise the actual schedule and an independent libcint oracle. A failed
large diagnostic lacked optional MD allowance; its explicit 1-GiB test plan
now admits the owner while MD retains its own 128-MiB cap. Retain failed harness
runs 6992, 7008, 7010 and 7015 instead of silently replacing their outcomes.

## Qualification and consequences

Final-source Slurm 7045 on master `82c166cac` passes all 16 resident tests,
memcheck with zero errors and racecheck with zero errors/warnings/hazards.
Complete cold medians are 94.944027/89.159602 s, a 6.09% gain above the 2%
robust-MAD gate. Candidate Focks are 22/17/17; the slower 22-Fock trajectory is
retained. Supporting 3/48/96-atom cold/warm/moved/moved-warm populations pass
independent complete E/F gates and their non-regression envelopes. Integration
on master `8eaaa66b3` changes only CI/diagnostic tooling, not measured numerical
source. Post-integration host qualification passes 126 focused and 1337
additional tests.

Same-density Slurm 7024 reduces actual uniform radial evaluations 48.02%,
retains every density direction/Hermite summand and all residual work, and
reduces resident J event time 20.22%. This supports causality but does not
substitute for the complete endpoint gate. All retained arrays, histories,
source/binary receipts, per-sample errors and statistical descriptors are in
[the publication](../../../../benchmarks/results/md-j-reciprocal-cold-20261010/README.md).

## Revisit when

Extend angular coverage only with corresponding publication-capacity,
independent raw-J, sanitizer, work and complete-endpoint gates. Investigate
deterministic publication separately if tail/iteration variability becomes
an acceptance concern. Preserve the incumbent for resource/ownership misses.
P1 actual-extent XC, P2 cold order-two force discovery/setup and P3 broad-class
K are separate changes and require independent endpoint qualification/PRs.
