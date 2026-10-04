# Experiment: compensated production CPU energy traces

Status: proposed; initial probe rationale followed by endpoint qualification below
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

## Subsequent endpoint qualification (2026-10-03)

The earlier sections record the initial hypothesis and acceptance plan. The
following fresh results qualify the trace-only source at commit
7cef255edc89ff6903694b3b760d0fef1fb58b81, tree
df629edabf7ee4f33f442508940c001cb03dafba.
No XC-only or deferred-mirror change is included in this source.

- Original strict PBE96 now converges in 17 SCF iterations plus two final builds:
  final energy change 3.410605131648481e-13, density change
  1.5248737028459072e-11 and physical residual 2.603467565262699e-11.
  Independent PySCF energy error is 1.4210854715202004e-12 Eh.
- The actual 1.01-scaled 96-AO tetramer converges in 18 iterations plus two final
  builds, with energy change 3.979039320256561e-13 and independent energy error
  3.353761712787673e-12 Eh. Its exported state has 39.99999999999995 electrons
  and maximum overlap-orthogonality error 8.143824355356254e-14.
- Both retain the original core guess, basis, resolved 331776-point grid,
  maximum 150 iterations, energy tolerance 1e-12 and density/residual limits
  1e-10. The original failing calculation is retained, not relabelled as a
  converged or performance baseline.
- All 17 shared intermediate histories have identical density changes,
  residuals, electron counts, occupation flags and XC energy values. At
  iteration 17, the old energy change is 2.4442670110147446e-12 and the new value
  is 1.1368683772161603e-13. This isolates the observed stopping difference to
  changed scalar traces along an otherwise equal observed trajectory. It does
  not supply the uncollected late matrix/product capture proposed above.
- Full native tests pass (67 suites), as do 60 independent DFT energy,
  analytic-gradient and reconverged finite-difference tests (20 CUDA cases
  deselected), 42 shared Fock API tests (two CUDA-only skips), six signed-density
  restricted/unrestricted fixed-density cases, and PBE0 RKS/UKS cold, warm and
  changed-geometry analytic/finite-difference probes.
- Five supported WB97M-V native cases pass independently reconverged PySCF
  energy/Fock/density checks; seven public WB97M-V state and multistep force
  finite-difference tests pass. Independent WB97M-V analytic-oracle forces and
  large PBE96 forces/finite differences remain unqualified.
- Failed local attempts remain recorded: the original two-electron-versus-total
  finite-difference harness scope mismatch, JSON/grid representation and report
  serialization errors, and unsupported RSH-lowerer preflight attempts. Only
  the harness was corrected; no acceptance threshold or production source was
  changed to make those reruns pass. The nonexistent temporary-file listing is
  corrected by a separate erratum while original receipts remain unchanged.

A later source-identical production integration additionally passes the 67
native suites and the same 60-test independent DFT suite. These are numerical
robustness results, not complete performance/default-promotion qualification.
Full raw matrices/logs are local; published hashes cannot recover absent files.

## Current-main publication integration

The compact retained publication is
`benchmarks/results/cpu-trace-precision-20261003`. It contains selected numerical
summaries, every attempt status, exact replay scripts and local-original checksum
anchors, not lossless raw receipts. Independent review inspected all 69 original
manifest members before approval. Large force/FD, late native matrix capture and
independent WB97M-V analytic-force qualification remain absent.

Public source alias `e4cc2787c02959ea287eb81e4541d009fe860f8c` has the original
complete measured tree; it is not a new measurement. A fresh alias build reproduced
the observed original library hash. Current-main integration and replay preflight
have separately bound receipts in `integration.json`; the current integration is
not substituted for the original binary. The top-level maintained staging helper
admits only the measured commit or that exact-tree alias and validates the oracle
scientific values; embedded historical scripts remain unchanged.
