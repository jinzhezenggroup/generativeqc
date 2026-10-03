# Experiment: compensated production CPU energy traces

Status: proposed; locally probed, not endpoint-qualified or promoted
Date: 2026-10-03
Base: a7c118aa3076dbfdc13e47ed2d1056c3b0fe6e80

## Problem

The frozen PBE96 CPU endpoint fails final delta-E 1.3073986337985843e-12 at the
unchanged 1e-12 tolerance after 25 iterations and two final physical builds.
Density and physical-residual gates pass at roughly 1e-14. The baseline and
candidate histories are bitwise equal. Final-vs-iteration-25 changes are
one-electron -6.82e-13, Hartree -7.39e-13 and XC +8.53e-14. This makes trace
rounding plausible, but component histories alone cannot prove the cause.

## Shared owner and bounded delta

The shared host primitive tensor::CpuCompensatedSum implements streaming FP64
Neumaier accumulation, following existing CUDA diagnostic signed-trace policy.
Its volatile term boundary ensures the identical rounded caller product enters
both the leading sum and the correction, including when FMA is enabled. It
requires ordinary IEEE semantics without unsafe reassociation. It allocates no
heap storage. Nonfinite/overflowing leading sums remain nonfinite and existing
caller validation/exception ownership is retained.

cpu_compensated_dot uses that owner for the RKS one-electron energy only. The
independent scalar reference::dot in src/scf/reference/linalg.cpp is unchanged,
as are density construction, residuals, electron counts, eigensolvers and DIIS.
contract_fock_energy_components uses one accumulator per Coulomb/exchange
component, retaining term expression, scaling and original spin/element order.
Fock/potential construction and source work are unchanged. XC accumulation is
not modified in this worktree. All physical and final-state tolerances remain
unchanged. This is explicitly a precision experiment, not a timing optimization
or a claim that the endpoint stall has been fixed.

The shared Fock reduction is also consumed by UKS, the fixed-density Fock C API,
and contract_exact_direct_energy_derivative. Their reduction results therefore
change and must receive independent derivative/force qualification. UKS's
one-electron trace remains unchanged in this narrow experiment.

## Local evidence and its limits

- Optimized unit controls cover signed cancellation, extreme magnitudes,
  subnormals, infinity/NaN and overflow, invalid dot extents, and separately
  rounded products under FMA. Existing native Fock provider tests pass with
  added cancellation, indefinite-density, nonfinite and wider-sum controls.
  The same new Fock test relinked to baseline fails its unit-remainder gate.
- At the saved converged independent PySCF PBE96 density, using its exact BSE
  input and newly evaluated one-electron integrals only, naive scalar dot
  differs from a 420-digit Decimal sum of identical FP64 products by
  1.79540724562235e-12. Compensated error is 2.35821579235037e-14 and equals the
  correctly rounded Decimal/math.fsum result. No new SCF or J/K work was done.
  This is a realistic matrix control, not the unavailable late native density.
- Synthetic 96-AO signed, scaled UKS streams give J error
  2.10091568048338e-11 -> 9.07786640564790e-14 and K error
  9.33206689274124e-12 -> 2.37119875011958e-13. Both compensated outputs are
  correctly rounded for the exact product terms. A long-double control agrees.
  These arbitrary matrices test the reduction contract, not a physical model.
- Only small ccache source/probe builds and source boundary checks ran, borrowing
  the frozen generated files/support library read-only. No complete native
  package, SCF, force endpoint, or performance qualification is claimed.

## Independent acceptance plan

1. Freeze source/binary provenance and run full native/package, Fock API,
   restricted/unrestricted/range/scaled derivative and final-state test suites.
2. Diagnostic-only capture of exact native D, Hcore, J/K terms, XC weighted
   terms and energies at every relevant late/final build of the unchanged PBE96
   case must isolate the source of drift. Compare scalar/reference, compensated,
   long-double and high-precision controls over identical rounded products;
   separately compare wider products. Preserve source term order and all checks.
3. Qualify trace-only versus frozen baseline and separately attributable XC-only
   and combined variants. Require native convergence, unchanged strict density,
   energy and residual gates, independent PySCF endpoint energy, electron count,
   overlap/projector checks, and successful final-state validation. No new guess,
   grid, tolerance, max-cycle budget or unchecked final-state substitution.
4. Qualify forces via existing analytic/oracle and finite-difference checks at
   multiple steps, including signed response densities and shared derivative
   consumers. Include PBE0/RSH, UKS, fixed-density Fock API and zero/negative/
   nonfinite edge cases for each affected owner.
5. Only then measure all-repeat cold/warm/changed-geometry/larger endpoints with
   actual source/Fock/iteration counts, without discarding failed or slower
   samples. A compensated sum's microsecond cost cannot justify a performance
   claim by itself.

If fixed-density high-precision controls do not identify reduction error as the
cause, reject the stall-fix hypothesis and investigate point/products/density or
Fock changes rather than expanding tolerances or modifying the reference oracle.
