# Decision: certify retained curated CPU KS projectors before publication

Status: implemented
Date: 2026-11-17

## Problem

The source-matched PR #3083 integration at `5dfdee202` passes the bounded native
CTest suite but fails the existing independent CsH(-) B3LYP UKS energy/force
gate. Its energy-only endpoint reports convergence after 21 iterations; its
force endpoint rejects the same final state as a numerical failure during
physical-reference export. The original source identity is
`36779c6d8702f4c17c1614755ebcf283a07c1a2eea43b576e971a4f7b57fea9b`.

Both CPU KS drivers already have a bounded physical-Fock closure and may resume
DIIS from a rejected closure within the original SCF iteration budget. However,
the final-Fock projector check runs only for automatic point programs. Curated
owners retaining a derivative state can therefore stop on their RMS density and
commutator gates without satisfying the maximum-entry canonical-density gate
required by the existing export validator.

## Decision

The consumer-selection rule in this initial decision is superseded by
[explicit export certification](2026-11-17-explicit-ks-export-certification.md).
The bounded closure and physical acceptance thresholds remain unchanged.

Run the existing final-Fock projector check for retained curated states too,
for both RKS and UKS. Retained states use the owner's physical residual tolerance
for both RMS and maximum-entry density reconstruction, matching the stationarity
budget supplied to final-state export. Non-retained automatic programs keep
their existing RMS tolerance while checking maximum-entry drift against that
same tolerance rather than an unrelated fixed ceiling.

A rejected projector returns through the existing bounded correction/resumed
DIIS machinery. Do not bypass the final-state validator, increase the original
SCF iteration cap, accept an unsuccessful result or call an independent solver
from production. No functional expression, integral, grid, force stencil or
scientific acceptance threshold is changed.

## Evidence and boundary

The existing `test_global_hybrid_ecp_forces.py` regression compares the public
endpoint with independent PySCF ECP integrals, occupied states, analytic forces
and independently reconverged displaced energies. Its CsH(-) UKS case reproduces
the original failure and passes after retained curated closure is enabled;
broader current-source qualification is recorded separately under
`/data/jzzeng/pr3083-ready-20261117/`.

The first 58-module run is stopped after exposing this real failure and is not
counted as passed. Original CTest/provenance logs and failed numerical receipts
are retained. Small-system state/force acceptance does not qualify a 25--40-atom
native Hessian/HVP, a representative Ni saddle or continuum rotational
invariance; those separate PR gates remain open.

## Degenerate UKS occupation boundary

The initial guard below is superseded by
[explicit export certification](2026-11-17-explicit-ks-export-certification.md).
It passes all 149 native CTests at `199084bb3` but fails the independent Cs and
mixed RSH/DF force controls (43 passed, two failed). Do not restore the blanket
stabilized-occupation exemption for production export consumers.

Full native CI on the subsequently integrated source exposes the retained
LDA/PBE OH state regression in `generativeqc_uks_state_tests` on both GCC and
Clang. The existing curated occupation stabilizer preserves a physically
stationary integer-occupation projector at a degenerate frontier. An extra raw
unshifted eigensolver ordering can choose another occupied subspace and must
not turn that already-qualified SCF root into exhausted-budget nonconvergence.

Retain the original curated stabilized-occupation physical closure rather than
adding the new raw-projector check to that path. Ordinary retained curated
states still receive the new RMS/max-entry export-projector gate. Automatic
point-program policy is unchanged. Unshifted energy, per-spin commutators,
density, electron trace, idempotency and iteration-budget checks remain intact;
the derivative-state exporter independently certifies every public request.
The existing native OH retained/unretained and exhausted-budget tests remain
required; none of their numerical acceptance checks is removed or relaxed.

## Revisit when

The SCF owner and final-state export share a single typed canonical-projector
acceptance contract. Do not replace executable scientific gates with assertions
about the spelling or call count of this closure.
