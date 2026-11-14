# Decision: equilibrate only the independent Newton augmented subspace

Status: implemented for explicit isolated reference use; scale qualification pending
Date: 2026-11-14

## Problem

The new 25-atom `[Ni(NH3)6]2+` triplet reference first failed its explicitly
bounded 240-cycle independent DIIS request. A separate cold PySCF Newton
request, with the same Hamiltonian, default quadrature and physical convergence
gates, then stalled. Its reported orbital-gradient norm was
9.715796233158906e-7, far above the required 1e-11. Fifty macro observations
were retained; many successive actual coefficient arrays were bit-identical.

That reference alone was terminated explicitly as nonconvergent, exit143,
after the recorded identical-state futility check. Native requests were not
cancelled. No convergence flag or numerical acceptance threshold was changed.
The retained states and termination decision are not a successful reference.

PySCF's small augmented-Hessian overlap combines an affine unit coordinate and
unnormalized orbital trial vectors. Its absolute `safe_eigh` cutoff can remove
independent trial directions solely because their norms become small. Merely
running more unchanged macro iterations cannot recover such a discarded step.
This is related to, but distinct from, the error-Gram issue described in
`2026-11-14-independent-diis-common-gram-normalization.md`.

## Decision

Add an explicit reference-only context in `tests/python/independent_newton.py`.
For the small projected Hamiltonian `H` and metric `S`, form the diagonal
congruence `D = diag(S)^(-1/2)`, leaving the affine coordinate unchanged.
Solve `(D H D, D S D)` with the existing upstream selector and cutoff. Scale
the reconstruction vectors consistently and map the returned generalized
eigenvector back through `D`, so the caller's original physical Hessian actions
and residual reconstruction remain unchanged.

The transformation changes the units of the projected basis, not the physical
augmented eigenproblem. Its linear-dependence criterion now acts on a
dimensionless correlation metric. Genuine dependent directions remain
dependent; indefinite, nonfinite, inconsistent-affine or invalid-dimensional
inputs fail explicitly. The bound is 129 projected coordinates, independently
of the much larger physical orbital response dimension.

The context temporarily binds the upstream projected-step routine and restores
it on both success and exceptions. It is only for a separate, serialized
independent reference process. It must not be imported by production paths or
used concurrently with unrelated SCFs in one Python process. Installed upstream
files, physical Fock/Hessian actions, SCF flags and physical acceptance gates
are not modified.

## Rejected alternatives

- Lowering the physical orbital-gradient gate would accept the actual failure.
- Lowering every absolute overlap cutoff without equilibration could retain
  numerically dependent directions and would not separate scale from rank.
- Normalizing the physical trial vectors before their Hessian actions would
  change the perturbations seen by a vendor numerical provider. The chosen
  congruence acts only after those actions, in the small projected problem.
- Reusing a failed or native density as an unreported independent initial state
  would weaken the cold-reference contract.

## Evidence and boundaries

Fourteen executable controls pass. They compare the physical augmented
eigenstep against an independent dense eigensolve under trial scales spanning
1e-12 to1e12, retain true rank deficiency, reject invalid inputs, restore the
vendor binding after injected failure, and run real cold RKS/UKS molecular
references to their unchanged independently recomputed 1e-11 physical gate.
Ruff and format checks pass. These small fixtures are algebra/reference controls,
not Ni, ECP, transition-state or practical-scale acceptance.
The combined independent DIIS/Newton suite has 41 passing tests, including
failure certification and restoration of the normal reference behavior.

A new cold 25-atom reference explicitly selects this context and freezes its
source, the existing 1e-14 relative subspace cutoff, 1e-13 energy and 1e-11
physical orbital-gradient thresholds before execution. Its predecessor's work
and failure remain visible; no saved state seeds it. Its actual large-molecule
outcome must be audited separately before any native state/gradient or HVP
qualification claim.

Evidence is under
`build/qualification/host-ni-hexaammine-scale-20261114/`, including
`reference-newton-stall-termination.json`, `newton-unit-v2.log` and
`reference-newton-equilibrated-preparation.json`. The new request is not a
production optimizer or a substitute for the remaining complete native gates
of #2844--#2848.

## Follow-up: null inherited trials and inner tolerance

The first equilibrated cold request actually failed after eleven macro
observations; its last reported gradient was 5.21933484309074e-7. A separately
identified warm diagnostic from the authenticated last callback state retained
the failing projected problem: a zero 2x2 Hamiltonian, metric `diag(1,0)` and an
exactly zero inherited trial of length13930. This is not an independent small
direction that equilibration can rescue. It must not be divided by zero or
silently declared a converged physical state.

The reference context now also scopes the upstream Davidson binding. Only an
exactly zero initial trial restarts from a finite, nonzero copy of the unchanged
current residual. A missing, nonfinite or zero residual still fails. Nonzero
initial trials, physical Hessian actions, preconditioning, iteration limits and
SCF/step acceptance remain upstream-owned; both bindings restore on failure.
Seventeen helper controls pass, including a dense independent augmented step,
its physical residual, zero-seed failure cases and binding restoration.

The bounded warm diagnostic no longer throws the null-metric error, but it
stalls at an actual gradient2.7004311727271134e-7 after five macro cycles. Its
exit0 is diagnostic completion, **not** physical convergence or acceptance.
The upstream inner residual target is the square root of `ah_conv_tol`; the
previous1e-12 setting therefore permits an approximately1e-6 inner residual.
A distinct cold request tightens that algorithmic tolerance to1e-24, with the
same1e-11 physical gate, no state seeding and a finite3600-second cap. Its
outcome is pending and must not be inferred from these algebra controls.

Failure retention must use actual callback states. An interrupted PySCF Newton
object can retain initialization `mo_coeff` and `e_tot=0` even after many real
macro steps; the first `failed-state.npz` contains those attributes, not the
latest callback state. The harness-misbound diagnostic and its explicit
termination/launcher error are retained. The corrected diagnostic uses the
source-hashed `macro-011.npz`. The new cold driver records callback states and
projected histories, and does not relabel stale solver attributes as current.

Sources, matrices, receipts and failures are in the same cohort under
`artifacts/projection-diagnostic-v2`, `artifacts/restart-diagnostic`,
`artifacts/equilibrated-v1-sources` and `artifacts/independent-newton-restart`.
The original failures and their executing helper bytes remain unchanged.

The tightened-inner-tolerance **cold** request subsequently completed, exit0:
19 macro observations,172 projected eigensolves,1581.3014678927138 seconds,
peak2852840 KiB. A fresh returned-state physical orbital gradient is
3.3260817238274695e-12, below the unchanged1e-11 gate. Its default-grid AO
envelope audit and complete independent electronic+D3 gradient also return.
This accepts the independent central reference computation only; native state
matching and practical native derivatives are still unaccepted. Sixteen new
cold displaced references are now separately requested for the originally frozen
two directions, two step sizes and fourth-order stencils, with no inter-state
seeding and at most four independent child processes. Their results remain
pending. A separate25-atom Ni/I28-ECP force case also explicitly selects the
same reference context and physical gates; it is not covered by the all-electron
central result.
