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

Repair commit `0ac84aca7`'s source-matched CPU library passes 285 selected Python
cases and 17 subtests (121.03 seconds), including all 18 independent LDA/PBE SCF controls,
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

## Upstream integration follow-up

The repair was subsequently integrated with `master@f73be1630`, retaining the
wheel snapshot retention update and the separate CUDA AOT packaging/execution
policy from #3094 and #3106. The only merge conflict was the derived AOT audit's
manifest/source digests. Recompute them from the combined tree rather than
choosing either branch's stale identity. No CPU numerical implementation changed
in this merge; the native build relinks its updated source identity.

The source-sealed default decisions and measured evidence receipt inventory
remain byte-for-byte equal to upstream. The offline policy audit retains every
default, four historical failed endpoint groups and historical source statuses;
it makes no new device or endpoint claim. All 329 selected AOT/profile/default
policy tests pass. The post-integration CPU qualification uses separate
`upstream-*` receipts, not a relabeling of the repair commit's earlier results.
The merged source-matched CPU build repeats all 285 selected Python passes and
17 subtests (121.00 seconds; 32 deselected), and all four native groups pass again
(21.10 seconds). No larger chemistry campaign is added to this closing scope.
