# Decision: align default restricted CUDA KS stopping with GPU4PySCF

Status: implemented
Date: 2026-11-11

## Problem

The force-stationarity repair in #3060 made a default full-range force request
use a `1e-9` AO-commutator ceiling. Energy-only callers had a different allowance.
Neither that AO maximum nor a spin-RMS is GPU4PySCF's global MO-gradient norm.
Treating stringent force repeatability as an implicit default SCF contract
increased cold physical-Fock work and made a default-settings timing comparison
unequal. Kernel tuning does not address that semantic mismatch.

This supersedes the *default force stopping policy*, not the physical-operator,
canonical-proof, UKS proposal-metric or reference-identity repairs recorded in
[the original gate completion](2026-11-10-original-ks-gate-completion.md).

## Decision

Admitted Standard all-electron FP64 restricted CUDA full-range KS owners use
the existing nonlocal owner's physical energy/MO-gradient candidate policy:
`abs(delta E) < energy_tolerance` AND
`norm(2 C_vir^T F[D] C_occ) < sqrt(energy_tolerance)`.
Energy-only and force callers select the same policy. The complete composition,
default grid, resident validation resources, full-target energy history and
existing precision/spin/provider admission still matter. This is not a blanket
promotion of custom models, UKS, ECP or approximate Focks.

Evaluate the physical norm as `norm(X^T(FDS-SDF)X)/sqrt(2)`, with `X^T S X=I`,
not by relabeling an AO diagnostic. Neither strict-only canonical early probes
nor the optional AO early-probe path may bypass a failed orbital-gradient gate.
Successful candidates still need the complete canonical publication proof,
including the existing Standard `1e-5` stationarity/density ceilings and unchanged
trace, idempotency, eigenframe and token checks. Rejected proofs resume SCF.

Either tighter public E/D control retains the strict path. An explicit
`GENERATIVEQC_CUDA_KS_CONVERGENCE=strict` also preserves the historical policy at
unchanged E/D controls. It is owner-snapshotted and bound into CUDA resource and
checkpoint identity; unknown values fail closed. Full-range physical-reference
exports retain their strict admission. GPU4PySCF is not modified or retuned.

## Independent contracts

The original-grid water96 rejection/restore/replay/moved/fresh regression retains
its original E/D controls, geometry, orbital/auxiliary basis, memory cap and
`1e-7` force gate, with the historical stopping policy now **explicitly** selected.
Its tighter-SCF-only arm remains separate. Do not silently replace this stringent
task with the application-budget default qualification, or describe matched
default SCF thresholds as a `1e-7` observable guarantee.

The stock installed GPU4PySCF 1.9.0 source has `conv_tol=1e-9`, an unresolved
gradient tolerance of `sqrt(conv_tol)`, and `level_shift=None`. Its extra cycle
is conditional on a non-None level shift; a PySCF CPU extra-cycle rule must not
be assumed for this GPU implementation. Stock `scf/hf.py` SHA256 is
`c4caa754089e22927377408693602bd3d65574597f9817079ebb1e9e7a19f8c6`.

An untimed original-geometry, same-orbital/same-auxiliary PBE DF stock-default
observation gives replay force difference `5.154038422716667e-7` and moved/fresh
difference `8.143674081750873e-7` Eh/Bohr. SCF/grid settings are unchanged;
analytic forces use the common full-grid-response contract. Two preceding
environment-prerequisite failures are retained, not numerical observations.
Receipts: ignored `.artifacts/ks-tail-20261111/stock-repeatability-*` and matching
`n1:/data/jzzeng/ks-tail-20261111/stock-repeatability-run3`.

## Invariants and validation

- Never use GPU4PySCF or a CPU oracle inside a production native solve.
- Do not change independent E/F gates, default-grid failures or fitting errors
  to manufacture timing wins.
- Host probes execute admission, exact norm normalization, NaN/boundary/history
  behavior, strict selection and resource/checkpoint policy identity.
- CUDA default and explicit-strict E/F tests compare cold/frozen-warm and moved
  geometries against independent tightly converged same-model references.
- README claims require a fresh complete endpoint cohort with actual Fock work;
  the earlier combined cohort is not evidence for the changed default.

## Revisit when

New spin/grid/model domains have independent endpoint qualification, or a
separate explicit force-accuracy contract can establish stringent observable
bounds without silently tightening ordinary default SCF requests.
