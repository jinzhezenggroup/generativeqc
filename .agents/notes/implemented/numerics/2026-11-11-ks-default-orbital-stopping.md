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

## Final-source evidence

Measured native source `886556ce5504dffa10679e624c963e795eb12e64` has input
identity `01e5b8baaa5bdb377a60689f184f5b16eaab0e1ba93a3f3a058e51db9900eba8`
and ELF SHA256 `ff0c1473889c8781bd005af0f7b3e6a06571cfc9c1a9af1e9ed980fe86f61555`.
The final production image passes 149 host/resource/checkpoint guards (17
explicitly skipped optional cases), 26 independent complete E/F tests, and both
original-grid water96 strict-policy arms. The latter retain their original
`1e-9 / 1e-5` and `1e-12 / 1e-10` public controls; moved/fresh maximum force
differences are `3.3991831571711373e-10` and `1.404210081545898e-11` Eh/Bohr,
respectively, below the unchanged `1e-7` gate. All rejection/replay/move checks
remain active. Eight additional default/strict observations validate separate
receipt destinations, preventing strict controls from overwriting default
evidence; this changes no scientific gate or benchmark sample.

The complete fresh README cohort has 128 processes / 512 endpoints, no retries,
63/64 cold wins and 64/64 warm wins. Both engines still independently qualify
58/64 pairs under unchanged observable gates; all 58 qualified pairs are faster
in both phases. The one cold DF HF96 loss and all six independent fitting/grid
failures remain plotted and disclosed. Native-only timing processes reject
reference-engine imports. Full-vector and scalar replay agree in normal and
optimized Python. The source-matched receipts live under ignored
`.artifacts/readme-rtx5090-aligned-20261111/` and the matching node1 tree.

## Rejected alternatives

Do not substitute unrelated solver/kernel tuning for the stopping-policy fix.
An exploratory extension of the UKS orthogonal Pulay metric to full-range RKS
saved one B3LYP96 iteration but added an iteration for PBE0 and r2SCAN-3c.
Two-sided DIIS Gram equilibration passed host algebra probes but did not improve
complete endpoints; intrusive captures showed the incumbent retained all eight
histories, disproving the suspected history-retirement cause. Larger DIIS
histories and smaller XC tiles gave isolated modest gains, not a resolution of
the unequal force contract. None is promoted. A separate low-FP64 GEMM admission
idea was canceled before its build and has no performance evidence. Receipts
remain under ignored `.artifacts/ks-tail-20261111/` and its matching node1 tree.

## Revisit when

New spin/grid/model domains have independent endpoint qualification, or a
separate explicit force-accuracy contract can establish stringent observable
bounds without silently tightening ordinary default SCF requests.
