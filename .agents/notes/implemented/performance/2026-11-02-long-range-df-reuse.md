# Decision: retain the radial identity while reusing long-range DF work

Status: implemented
Date: 2026-11-02

## Problem

The README WB97M-V auxiliary basis contains g shells. The fitted LR CUDA source
rejected these even though the compiler already owned the required g value and
response algebra. After admitting that algebra, large grids incorrectly tried
to prepare an exact four-center cache merely because a separate LR provider
existed. Small-grid tests did not trigger the automatic cache policy.

Once that mismatch was removed, fitted LR still regenerated large raw panels
for dense K. A100 source `4ad0230efc5f9433c4e8939e7ee301284853930c0384dd5f99e28a98449657e6`
spent about 112 seconds per LR K application on the 96-atom README input.
The full-range owner already retained packed values and used occupied RI-K.
The LR owner lacked both its own occupation reservation and the fast prepared
raw-value path. Merely permitting an occupied contraction did not fix source
regeneration.

## Decision

- Dispatch auxiliary g through the existing omega-aware compiler algebra.
- Require an exact exchange approximation before preparing the exact ERI cache.
- Let the independent fitted LR provider consume the method's proved occupied
  coefficients, with stream events ordering their lifetime. Each radial owner
  retains its own metric and transformed factors.
- Freeze omega in the prepared Hermite raw owner. Extend the shared radial
  emitter using the erf substitution, `s=omega/hypot(omega,sqrt(rho))`, effective
  exponent `rho*s*s`, and prefactor `s`. Coulomb and LR share the recurrence;
  the Direct emitter's source and arithmetic remain unchanged.
- Reserve the restricted method's own projection capacity for both radial
  owners. If complete U cannot fit but packed B can, retain B and use the
  existing bounded projection panels. Preserve the requested-rank response
  floor, explicit caps, and absolute live-memory headroom. The remaining value
  allowance can admit optional prepared E transforms. Otherwise keep streaming.
- Cache the frozen VV10 AO potential during inner iterations. A separate NLC
  grid lends its existing immutable matrix; a shared grid uses optional FP64
  storage. Allocation refusal preserves reassembly. Only full target operators
  can certify the final energy, residual and forces.

## Independent reference representation

An explicitly spherical auxiliary BasisSet remains spherical when orbital AOs
are Cartesian. The original mixed-representation test oracle inherited the
orbital representation for its auxiliary molecule and compared different fitted
models. The corrected oracle starts from libcint Cartesian integrals and applies
independent PySCF transformations to each public basis. Native factors never
enter that reference. One diagnosed water case changed its apparent energy
discrepancy from 2.52e-6 Eh to about 1.1e-12 Eh without changing an error gate.

## Invariants and rejected alternatives

LR factors cannot reuse the Coulomb metric. Exact LR derivatives cannot
differentiate a fitted LR Hamiltonian. Exports and rebases lacking a proved
occupied factor retain dense execution. An environment change after preparation
cannot alter the Hamiltonian. Prepared E is optional and all failed admissions
must leave a valid original source. No GPU product name determines these choices.

Keeping only the occupied contraction while streaming the same expensive
integrals was measured and rejected as insufficient. Requiring complete U before
retaining any B was likewise insufficient on devices with room for two B owners
but only one complete U. Lowering accuracy or removing the final full VV10 target
was not used to resolve the performance problem.

## Evidence

Independent gates include all Cartesian/spherical f/g LR values and center
responses, 360 raw Hermite Coulomb/LR cases, restricted/unrestricted occupied K
against dense CUDA and independent CPU DF, and complete energy/force/geometry
replay with the actual README auxiliary basis. Host resource tests cover the
complete-U, B-only and streamed alternatives and explicit-cap exclusions.

Incomplete and cancelled diagnostic runs remain failures, not endpoint latency
samples. A speedup claim still requires complete source-matched cold/warm
measurements and independent observable gates on the target device.

## Revisit when

The paired radial owners acquire a shared lifetime-aware scratch allocator, or
a source-driven streamed contraction can avoid repeated raw integral work under
caps too small to retain either B. Such changes must retain independent radial
identity and explicit resource refusal.

## Truncated LR response follow-up

The first complete A100 fitted attempt (`224548`, source
`9d0be57afa5ba55df766eb0320e47706e886b64aef8558ce7a6e7774729f8e16`)
converged in 28 Fock builds but failed in LR force response. The metric was
rank-truncated; its dense inverse-root adjoint demanded a complete
`Naux * Nao^2` tensor, larger than the retained response allowance. Thus this
attempt has no complete endpoint timing, despite a converged SCF.

The restricted response now borrows only the authenticated KS factor
`D = 2 C C^T` across radial providers. It forms raw `P_Q = C^T A_Q C` and
`Z = P M^(-1/2)`, and differentiates Z with the existing inverse-root Frechet
map. Raw P remains live until its metric seed is complete. This preserves
retained/discarded subspace motion, which cannot be recovered from fitted B
alone. Only bounded derivative output panels scale with `Nao^2`; the two
retained tensors scale with `Naux * Nocc^2`. Full-rank response keeps its
existing factor-first path, and unsupported factors keep the original route.
The response synchronizes the factor's producer before using another radial
provider's stream and drains before returning the borrowed lifetime.

Minimum B/U admission also left insufficient spare bytes for optional Hermite
E on A100. Its optional source now receives a ceiling bounded by the existing
shape-only resident capacity and observed free memory minus absolute headroom,
while preserving the selected response allowance. Only an actual successful
source allocation can increase the charged value allowance; explicit caps and
probe-failure budgets cannot grow. B/U geometry is not replanned.

A standalone candidate response object passed 201 independent 5090 adjoint
cases in Slurm job 15252, including 12 truncated-metric cases with positive and
negative exchange coefficients, ragged panels, and rotating retained/discarded
subspaces. The reference contracts full AO densities independently; a
finite-difference eigenvector rotation separately checks subspace motion.
This is numerical kernel evidence, not complete endpoint qualification.
