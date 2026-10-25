# Decision: bound automatic CUDA KS admission at the empty-spin work boundary

Status: implemented
Date: 2026-10-25

## Problem

Automatic CUDA KS previously admitted empty-spin UKS while its point provider
exports interior Libxc derivatives evaluated at floored/clamped work inputs.
Those coefficients do not differentiate the original-density-weighted composite
energy under a physical spin birth. Raw Libxc parity, occupied-space stationarity
and a converged SCF flag cannot certify the missing full-Fock derivative.

Let R be the original total density, W the projected work inputs and N their
total density. The reported energy is E(R)=R*epsilon(W), whereas the raw density
coefficient differentiates N*epsilon(W) with respect to a work coordinate.
At an empty spin, choose a density-only PSD birth with zero AO spatial
derivatives. For 0<h<the density floor every work input stays fixed, hence
E'(0+)=epsilon(W)=E(0)/R. The raw coefficient is a different derivative.

The executable TPSS control at rho_active=1e-10 gives E'(0+) about
-0.00021366864422640795 and raw vrho_empty about -0.0007225004260050542.
It tests both spin orientations and three strictly subfloor steps against
the analytic identity and independent pinned Libxc energies, with the unchanged
1e-10 slope gate. This is an exact local branch calculation, not extrapolation
of the unconverged molecular birth stencils. Their retained inputs and failed
historical measurements remain in the earlier rejected note.

## Decision

Preserve the existing raw-work scientific contract. Automatic CUDA UKS now
requires nonzero occupation in both spin channels. Public batch preparation
returns NOT_IMPLEMENTED with an explicit empty-spin physical-birth diagnostic.
Direct native owner construction and detached final-state validation enforce
the same condition. Check each molecule before constructing the execution owner.

This is an admission repair, not a numerical repair or qualification of
empty-spin physical birth. The requested empty-spin endpoint is explicitly
unsupported. CPU KS, curated CUDA KS and fixed-density automatic XC retain their
existing contracts. The restriction covers all automatic registrations because
the compiler/provider contract supplies no independently qualified empty-spin
continuation capability; successful raw coefficient parity is not such a
capability. It is not a functional-name whitelist or an occupied-space projection.

The supported RKS/nonempty-UKS lifecycle gates, full Fock comparisons, density
directions, precision gates and AOT default remain unchanged. No coefficient,
floor, energy, equation identity or public ABI is altered. Old detached snapshots
for the now-unsupported CUDA product cannot bypass the live admission policy.

## Rejected alternatives

- Replace raw potentials by Libxc raw coefficients: the independent counterexample
  agrees with Libxc and still violates the composite derivative identity.
- Differentiate floors in the existing point domain: that silently changes the
  established work contract and CPU/CUDA scientific identities. A separately named
  energy-derivative domain needs its own design, consistent coefficients and
  independent boundary/endpoint validation before it can admit this product.
- Keep the endpoint and omit beta Fock entries or relax directional gates: this
  would claim a complete physical state without proving it.
- Declare short molecular secants to be the limiting derivative: retained
  floor/ferromagnetic branch transitions and cancellation prohibit that inference.

## Validation and reproduction

`test_empty_spin_raw_work_is_not_the_composite_energy_derivative` in
`tests/python/test_libxc_physical_kinetic_predicate.py` is the independent host
control. Public negative tests in `test_libxc_automatic_cuda_ks.py` cover H2+
and fully polarized H2, eleven installed providers and both selector/MethodIR
forms. The four TPSS cases fail against the previous library by successfully
preparing an unsupported owner; the red log remains retained.

`generativeqc_ks_cuda_tests --automatic-empty-spin` checks direct construction.
`generativeqc_ks_final_state_tests` checks analytic nonorthogonal detached states
with either spin empty and both nonempty, and preserves CPU acceptance. The
optional CUDA arm runs when the static TPSS provider is installed.

Build with the same eleven-provider AOT selection as the existing public
lifecycle suite. Run real-GPU controls only in finite Slurm allocations on
node1/main with `--gres=gpu:5090:1`, preserving assigned device visibility.
Raw build, red/green and endpoint receipts remain in ignored
`.artifacts/repair-20261025/remote/` and the matching node1 source directory's
`evidence/empty-spin-*` and `evidence/boundary-admission-lifecycle/`.

Final source validation uses Release sm120, CUDA 12.9.1 and verified sccache
0.16.0. Slurm 14099 (assigned device 1) passes direct native rejection, the
complete CUDA KS runner and all 91 public automatic tests: 44 empty-spin
rejections, 44 supported lifecycle cases and three existing policy rejections.
The 44 lifecycle receipts retain 132 cold/warm/moved rows with independent
PySCF 2.14.0/Libxc 7.0.0 references. Maximum energy/fixed-D energy/density/full
Fock/directional errors are 2.51e-14/2.49e-14/1.08e-8/3.15e-14/9.43e-11.
Library SHA256 is
`887f2ae0a06687dbb521f86e9e4af55b5bfda91c2076ade4596d99c7cd97e7eb`.
Slurm 14100 (assigned device 2) passes direct-construction leak checking and
eight TPSS public cases under Compute Sanitizer memcheck: zero errors and zero
leaked allocations. The same library passes all 26 public CPU lifecycle/budget
tests and native final-state tests. Focused host checks pass 69 tests, with one
existing environment-dependent skip; changed-file pre-commit passes. These
are correctness/admission results, not a performance or empty-spin continuation
claim.

## Revisit when

A separately specified physical-birth derivative contract has independent
same-input AO/energy evidence, including occupied, virtual and mixed births and
their floor/ferromagnetic limits. Restoring admission requires that contract
and complete endpoint qualification, not removal of the guard alone.

## References

- PR #2276.
- [Retained negative evidence](../../rejected/2026-10-25-tpss-empty-spin-raw-work-clearance.md).
- [Current CUDA XC/KS contract](../../../../docs/developer/xc_native_cuda.md).
