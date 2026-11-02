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
