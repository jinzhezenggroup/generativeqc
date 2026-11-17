# Decision: energy-only KS owners do not promise derivative export

Status: implemented
Date: 2026-11-17

## Problem

The explicit export-certificate repair correctly closes the independently
observed force regressions, but requiring that certificate for every CPU KS
energy call conflates two consumers. The ordinary OH LDA energy fixture reaches
a physical commutator of about 2.4e-15 while the unshifted Aufbau reconstruction
differs by an RMS of 1/6. Increasing its iteration budget from 150 to 240 does
not resolve that occupation-root distinction. Neither a larger budget nor a
guessed degeneracy exemption is the appropriate energy contract.

## Decision

CPU DFT prepared batches retain their certified derivative-export contract by
default. An explicit `ks_export=False` request selects an energy-only owner,
using `GENERATIVEQC_BATCH_DISABLE_KS_EXPORT`. That owner neither retains nor
publishes a stationary derivative snapshot, advertises only energy, and rejects
forces. The ordinary SCF gates and occupation-stabilization algorithm are
unchanged. Explicit CPU DFT `singlepoint(properties=("energy",))` calls select
this mode; electronic properties and derivative consumers retain certification.

This narrows the consumer selection in the earlier
[export-certification decision](2026-11-17-explicit-ks-export-certification.md).
It does not supersede the physical export checks or the independent force gates.

## Invariants

- No weakening of SCF, projector, derivative or independent-reference tolerances.
- No snapshot, force, Hessian or response admission from an energy-only owner.
- No hidden solve, larger iteration budget or reference-engine production path.
- Existing prepared-batch callers keep their derivative-capable default.
- Small numerical regressions do not qualify large-molecule HVPs, Ni reaction
  saddles or arbitrary-angle continuum rotational invariance.

## Evidence

The bounded reproduction and source-specific closing tests are retained under
`/data/jzzeng/pr3083-fix-20261117/`. The older source's 149 native and 45 focused
passes remain attached to that older source, not relabeled as this change.

The final source-matched CPU library passes 285 selected Python cases and 17
subtests (121.03 seconds), including all 18 independent LDA/PBE SCF controls,
the public native Hessian diagnostics and opt-out rejection controls, both
previously failing independent export/force cases, compiled grid contraction
oracles and the affected build/CI/batch contracts. The CPU-only selection
deselects 32 cases; they are not counted as passes. Four rebuilt native groups
covering DFT API, UKS state, final-state validation and host Hessians pass
(21.09 seconds).

Separate exploratory combined and matched-curvature runs hit 360-second and
180-second caps without complete summaries. The former also records an
intermediate failure before the public native assembly-timing repair. Neither
run is acceptance, and neither is relabeled after the final identity relink.
Normal required CI remains authoritative for the broader suite. Practical-scale
and reaction campaigns remain separate product follow-ups, not a reason to
force additional chemistry runs for these interface and CI repairs.
