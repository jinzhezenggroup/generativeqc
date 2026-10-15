# Proposal: consumer-owned bounded XC force powers

Status: proposed; generic GPU and endpoint qualification pending
Date: 2026-10-15

## Problem

PR #2506 moved #2505's M06-2X/RKS-only performance exception into a generated
method/code/spin manifest. This relocates specialization rather than removing
it. The generic scalar IR already has a bounded, shared integer-power pass;
curated semilocal generators apply it after AD, but imported split hybrids did
not. The historical global M06 DF96 warm regression adds a fourth Fock build
instead of three. It cannot be repaired by a method whitelist, hidden solver
work, different initial densities, or relaxed scientific tolerances.

## Proposed decision

Keep the consumer boundary, remove the method boundary. The FP64 stationary
force consumer selects bounded power expansion for every imported program and
both spins, using the existing IR pass after independent polarized AD and
before equal-spin binding. No functional name/code/spin participates in the
schedule selector. Remove the schedule manifest/generated-registry plumbing.
Curated semilocal programs already use this same bounded pass and do not change.

SCF point generation and float scalar ordering stay unchanged. A stationary
derivative schedule is not the self-consistency map: rewriting the latter can
perturb a convergence boundary enough to add a whole expensive Fock build.
This consumer distinction is generic, not a benchmark- or molecule-specific
exception. No #2486 AO/dddd/page/tile negative experiment is repeated.

## Required gates

- Exact source preservation for every registered SCF point and float graph;
  exact preservation of #2505's already-qualified M06 RKS force source.
- Independent Libxc Cartesian partials, displaced-energy checks, tau/2 ABI,
  and actual RKS/UKS force emission for both registered imported hybrids.
- Real-GPU scalar acceptance of both actual production spin consumers through
  finite Slurm, including existing independent boundary/oracle gates.
- Paired timings and semantic work: distinguish scalar-kernel measurements
  from complete E+F endpoints, and retain every sample and negative verdict.
- Revisit the DF96 extra-build population without re-enabling the known-bad
  global SCF rewrite. Never credit incidental baseline iteration variability
  as a solver improvement or claim a universal hardware speed guarantee.

## Evidence and history

The policy-only admission refactor remains historical rationale in
`.agents/notes/implemented/architecture/2026-10-15-xc-scalar-schedule-qualification.md`.
The original negative experiments and #2505's 252 native observations remain
unchanged and recoverable through their existing checksummed Git snapshot.
New evidence and the final admission decision will be appended after gates run.

Initial host acceptance: 64 tests pass with the pinned Libxc 7 environment,
including actual Cartesian force source for both imported methods and spins,
unchanged independent partial/finite-difference gates, and scalar ABI probes
bound to both production spin consumers. No real GPU result is inferred from
these host tests. The initial probe harness failed compilation because two
namespaces have distinct C++ result types; a generic output sink fixes that
harness error without changing generated scientific code or tolerances.
