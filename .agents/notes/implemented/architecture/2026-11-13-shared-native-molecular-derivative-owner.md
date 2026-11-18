# Decision: qualify the actual native molecular derivative owner

Status: implemented
Date: 2026-11-13

## Problem

The molecular ECP constructor fix preserved potentials and effective ionic
charges, but constructor coverage alone did not exercise the CLI corrector's
SCF/force batches, prepared Hessian consumption, accepted-root tracking or
failure recovery. A separately implemented qualification provider, or locally
rewriting the public ECP guard, would not establish these ownership contracts.
Energy agreement alone also cannot establish that two UKS calculations selected
the same occupied spin subspaces.

## Decision

Move the existing numerical implementation into the internal
`cli::NativeMolecularEvaluator`. The CLI adapter derives from it and retains
public admission, progress/cancellation and callback/error policy. Internal
qualification invokes this exact numerical owner. Neither Python nor CLI public
molecular ECP saddle admission changes.

The owner copies/moves complete basis data, including ECP core/potential arrays.
It borrows physical options and the linked program for its lifetime. Each trial
owns its system and cold SCF/force batch; its Hessian consumes that same batch.
The context outlives all frames. Only explicit accepted-geometry observations
replace the root reference; failed or rejected trials cannot replace it.
Resource controls may change between, never during, synchronous calls.

Expose a budgeted detached orbital snapshot at an exactly matching current or
accepted geometry. This exports the actual stored native coefficient matrix and
spin/occupation shape, not a reconstructed determinant or another solve.
External independent state matching remains qualification-only and never
supplies production gradients, curvature or optimizer decisions.

## Rejected alternatives

- A duplicate ECP-only molecular provider would leave the actual CLI numerical
  path unqualified and create a second owner to maintain.
- Rewriting admission guards for qualification would conflate diagnostic
  execution with publicly accepted real-element chemistry.
- Inferring a matched UKS root from total energy or electron count would miss
  changes of occupied subspace. Missing native orbitals cannot be reconstructed
  afterward from an independent solver and called a native state.
- Retaining a shared unbounded geometry/orbital cache would complicate lifetime
  and resource guarantees. Keep only current and accepted frames, with explicit
  root and detached-output bounds.

## Invariants

- Production derivatives use no oracle, displaced production SCF, external
  optimizer or runtime compilation.
- A Hessian requires its exact prepared energy/gradient geometry.
- Root and derivative failures do not publish partial numerical outputs.
- Snapshot mutation is detached from current and accepted frames; exporting an
  accepted state must not return a different current trial's orbitals.
- Minimal ECP component parity does not grant I/Cs saddle, Ni TS or scale
  qualification. Existing chemistry retains its original producer identity.

## Evidence

`tests/native/test_cli_molecular_evaluator.cpp` executes RKS NaH and UKS NaH2
minimal-ECP B3LYP-D3(BJ) component contracts against explicit native batches.
It covers changed geometry, copied core/potential ownership, accepted root
retention after a rejected trial and remote-root failure, root/ERI budget
refusal and recovery, exact geometry/output binding, both spin layouts,
detached snapshot mutation and snapshot-budget refusal. These fixtures are not
independent real-element chemistry oracles.

The snapshot-enabled CPU build passes seven native/API/CLI contract groups and
106 targeted public/native/saddle/source-identity Python regressions. Logs,
actual exit codes, source/build binding and preserved initial failed build
attempts are under
`build/qualification/host-ecp-molecular-consumer-20261113`. The original real
I-H-H diagnostic was frozen before orbital snapshots were added; it must not be
relabelled as exporting a matched native determinant.

The independent orbital comparison helper additionally exercises a nonidentity
AO overlap metric, orbital phase and occupied-subspace rotation invariance,
wrong beta/RKS roots, scaled nonorthonormal coefficients, and mismatched
geometry, population, spin, AO shape and nonfinite exports. Its 17 non-nightly
contracts pass; the combined targeted Python regression passes 123 cases with
five expensive chemistry cases explicitly deselected. The snapshot-enabled
qualifier records the actual native orbitals and every returned raw Hessian;
its independent validator refuses the older unsnapshotted producer before
starting any oracle SCF. Compiling this qualifier is not chemistry acceptance.

## Consequences

Component qualification now tests the same numerical implementation as the
CLI, without widening public admission. The thin adapter still needs its own
CLI contract tests. Real-element stationary mode/state qualification and
complete-endpoint work/wall/memory qualification remain separate gates.

## Revisit when

Actual I/Cs molecular saddle consumers pass independent same-level state,
multi-step raw-curvature and mode acceptance. Only then consider a separate
public admission change; do not infer it from this ownership extraction.

## References

- [Molecular ECP construction](2026-11-13-molecular-ecp-frame-binding.md).
- [Current Hessian contract](../../../../docs/developer/dft_hessian.md#molecular-corrector-frame-ownership).
- Issues #2844--#2848.
