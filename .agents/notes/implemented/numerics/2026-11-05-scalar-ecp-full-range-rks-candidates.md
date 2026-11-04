# Decision: reuse scalar-ECP sources for low-level full-range RKS qualification

Status: implemented
Date: 2026-11-05

## Problem and decision

Issue #2847 needs the same physical B3LYP/def2-ECP Hamiltonian in nuclear
response and curvature that #2846 qualifies for first derivatives. Existing
WB97M-V scalar-ECP response already owns the effective ionic charges, residual
ECP matrix, all-center frozen perturbation and local/nonlocal second-order
sources. A B3LYP-specific second ECP equation would create competing scientific
owners without addressing the missing method/domain qualification.

Allow a CPU/direct/full-range RKS qualification candidate to bind either an
all-electron or scalar-semilocal-ECP native Hamiltonian. Keep the existing
current-owner state proof, exact J/K provider proof and declared primitive
inventory. The same `ECPSecondOrder` binds its matrix into the Fock reference,
its first nuclear perturbation into the shared CPKS RHS, and both residual
channels into the common stationary HVP assembly. Effective charges are applied
through the existing physical-center mapping, not a second nuclear model.

Retain the RKS restriction in this slice. UKS qualification is a separate
dependency (#2844), and this change does not silently duplicate that branch's
admission change. CPU density fitting, CUDA, spin-orbit and unqualified angular
domains do not enter through this predicate. Full-range candidates still do not
inherit the canonical WB97M-V public capability registry.

## Resource and numerical invariants

The existing grid owner reserves its persistent storage before handing the
remaining explicit byte budget to ECP radial-layer jets. Keep this bounded
failure path and the unchanged two-grid ECP HVP convergence gate. A successful
SCF or matrix does not waive derivative quadrature admission. Do not replace
an admission failure with reference work, displaced SCFs, symmetrization or
inferred Hessian columns.

Independent small-fixture qualification must match the canonical ECP, spherical
space, MethodIR, charge/spin, discretized moving grid and converged state.
Freeze the independent normalized HVP threshold at `1e-5 Eh/Bohr^2`, raw
translation/symmetry and native action parity at `2e-7`, and response residual
at `1e-9`, without relaxing the source quadrature or SCF controls. Complete
endpoint timings include native preparation, SCF, owner construction and the
requested action; standalone assembly times are not a production speed claim.

## Independent grid-oracle arithmetic

Retain the initial failed HVP receipt: native assembly and point-state checks
passed, but a raw quadrature-weight equality assertion failed before the HVP
oracle ran. Rational far-field volumes reach about `1.5e6`; equivalent native
and PySCF Becke arithmetic can differ by `7.66e-8` there while the AO density
vanishes. Comparing those bare volumes is not a physical integration gate.

Instead bound the partition difference in the AO/first-jet quadratic space:
the maximum element of `A.T @ (abs(delta_weights)[:, None] * A)` must be at most
`1e-12` times the corresponding reference Gram scale (with a unit floor),
where `A` bounds the absolute AO and first-derivative jets. Reconverge the
independent SCF on its own partition arithmetic before its analytic moving-grid
gradient, requiring energy drift at most `1e-10` and density drift at most
`2e-8`. This fixes a mismatched primal/gradient oracle, not the native grid,
physical force/HVP tolerance, ECP quadrature gate, or production implementation.

## Evidence

Canonical neutral singlet CsH at the unchanged #2846 geometry, physical
Gaussian/VWN-RPA B3LYP, real-spherical def2-SVP and scalar def2-ECP46 passes
one native ECP-center HVP on the explicit `16x6x12`-per-atom grid. Direction
is `((.02, -.03, .1), (0, 0, 0))`; energy and density SCF tolerances remain
`1e-13` and `1e-11`, with at most 240 iterations.

- Independent reconverged analytic-gradient differences at steps `3e-3`,
  `1e-3`, `3e-4` give normalized HVP errors `7.93351e-8`, `6.55499e-9`,
  `1.97629e-9`, all below the frozen `1e-5` gate.
- All ten sources, including both residual ECP channels, are present. The
  response uses one solve / 13 iterations, residual `1.55937e-13`, with zero
  production displaced SCFs. Public second-order capability remains false.
- Complete native preparation, SCF, owner construction and HVP take 281.344 s;
  assembly alone takes 260.560 s. ECP work observation reports 736 radial
  layers / 2,590,784 AO points. The whole independent test passes in 285.30 s,
  maximum process RSS 641,424 KiB, including the reference. Source checksums
  match before/after the finite 1800-second run, whose whole exit status is zero.
- Existing full-range RKS and WB97M-V scalar-ECP RKS/UKS/DF Hessian regressions:
  17 passed in 288.86 s, CPU only. No real-GPU execution occurred.
- A separate complete *reference-only* raw `6x6` matrix at each step `1e-3`,
  `5e-4`, `3e-4` takes 13.859 s. Unsymmetrized asymmetry is at most
  `2.751e-8` and translation residual at most `2.763e-8`; its action agrees
  with the native direction within normalized errors `8.30138e-7`,
  `2.06985e-7`, `8.22575e-8`. Reference completeness does not imply a complete
  native matrix or an independent native bilinear qualification.

Receipts are retained locally under `/tmp/gqc-2843-2849/` as
`def2-ecp-hvp-v1.*` (failed), `def2-ecp-hvp-v2.*` (accepted),
`def2-ecp-hessian-regression-v1.*`, and
`def2-cs-independent-raw-hessian.npz`. Native library SHA-256 is
`e3a29ef62d72b70a2251721321a7dbf7484e2b3649f636d0cdc10aac016b2dc6`;
PySCF 2.14.0 is validation-only. These are source-tree, tiny electronic-only
receipts, not installed-artifact or production-scale claims.

## Boundaries

A subsequent unchanged native common-assembler experiment at commit
`82efe3c390e1a9b9c63cf54ab0bc4dfa32be112b` has **failed**, not merely remained
unrun. Its separate ECP-center HVP passed again, but the first raw unit
Cartesian column triggered `ECP Hessian quadrature convergence gate failed`
inside the retained coarse/fine ECP source check. No native raw column or
complete matrix was accepted; the second native action/bilinear stage was not
reached. Whole exit status is 1 after 398.74 s, peak process RSS 546,760 KiB.
The finite 3600-second guard was not reached. Retain
`def2-cs-native-raw-qualification.log` and the exact script
`qualify-def2-cs-native-raw.py`; do not infer a matrix from the passing direction
or the reference-only complete matrices. The fixed absolute ECP source gate
is not changed, unit columns are not rescaled to bypass it, and no posthoc
symmetrization or reference fallback is introduced.

This is low-level electronic-only candidate admission, not completion of #2847.
IH molecular HVPs, every native raw Hessian column, independent bilinear
and rotation checks, failure/lifetime/resource gates, mixed Ni/ECP and UKS
qualification remain individually necessary. #2846 retains a failing mixed
Ni/I/H quadrature fixture and native displaced IH(-) SCF failures; no second-order
claim bypasses them. Default-grid domains, installed public endpoints, total D3
curvature, stationary modes and the 25–40-atom workflow remain separate gates.

## References

- Issues #2847, #2846, #2844 and the #2224/#2843 shared prerequisites.
- `python/generativeqc/composite_rks_hessian.py`.
- `python/generativeqc/ecp_hessian.py` (reused without new ECP mathematics).
- `tests/python/test_global_hybrid_ecp_hessian.py`.
- `2026-11-05-def2-ecp-global-hybrid-force-gates.md` for retained inputs and failures.
