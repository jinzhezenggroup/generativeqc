# Decision: form VWN spin fractions directly from spin densities

Status: implemented
Date: 2026-10-04

## Problem

The original full native CUDA DFT regression exposed a B3LYP empty-spin
potential mismatch while qualifying the resident active-AO stack (#1774).
Changing the tile size did not account for it. A private diagnostic on n1
Slurm 5623 captured actual AO features and point coefficients for polarized
H2/STO-3G on 49152 points and replayed those inputs through the CPU evaluator.
GPU coefficients assembled on the host reproduced the device potential;
CPU point evaluation of identical GPU features also agreed. The discrepancy
was amplification of different feature rounding, not a matrix assembly error.

For majority density 0.010594746189871745 and exactly zero minority density,
changing only the majority density by one ulp changed the B3LYP minority
potential by about 7.8e-8. The pinned Maple VWN interpolation raises
`1 +/- zeta` to a fractional power, where `zeta=(rho_a-rho_b)/(rho_a+rho_b)`.
Finite arithmetic in `rho * (1/rho)` can create a tiny minority fraction at
an exactly empty spin channel. Its cube root amplifies that rounding in the
first derivative. This affects both VWN and VWN-RPA.

## Decision

After importing the pinned Maple energy, replace its `1+zeta` and `1-zeta`
subexpressions with `2*rho_a/(rho_a+rho_b)` and `2*rho_b/(rho_a+rho_b)` before
automatic differentiation. The formulas are algebraically identical on the
positive-total-density domain. The empty channel now contributes an exact
zero fractional-power base. The adapter requires both original nodes to be
present and fails explicitly if the pinned import changes that structure.

Only polarized VWN lowering changes. Unpolarized lowering, source Maple bytes,
density-domain admission and numerical tolerances remain unchanged. The
existing adapter SHA-256 provenance deliberately changes with this lowering.
The second derivative at an exactly empty spin channel need not be finite;
this change does not broaden the public interior-domain derivative contract.

## Rejected alternatives

- Relaxing the integrated CPU/CUDA potential gate would hide unstable point
  arithmetic and would not protect nearly empty spin channels.
- Clamping zeta or imposing a new minority-density floor changes the model
  or introduces a derivative boundary. Direct fractions need neither.
- Changing tile sizes cannot repair a point formula whose derivative jumps
  under a one-ulp input perturbation.

## Evidence

`tests/python/test_vwn_spin_boundary.py` independently evaluates the original
VWN/VWN-RPA closed forms with 90-digit mpmath arithmetic. It checks energy and
both first derivatives for both spin orientations, majority densities
1e-12 / 0.010594746189871745 / 1e4 and minority ratios
0 / 1e-30 / 1e-18 / 1e-12. Derivatives at zero are one-sided. Acceptance is
rtol 5e-12 and atol 5e-14; a one-ulp majority perturbation must change the
minority potential by less than 5e-13. The same oracle checks interpreted
graphs, optimized emitted C++ and opt-in real-device CUDA arithmetic.

An immutable base-adapter negative control produces 80 failures and 24 passes
in the new graph/C++ cases. The corrected host run passes 119 cases including
the existing VWN interior Hessian, provenance and point-dispatch checks;
52 CUDA cases skip outside an explicit Slurm allocation. On n1 Slurm 5625,
all 168 point/VWN cases pass, including those 52 CUDA cases. The original
complete native DFT CUDA executable also passes without weakening its B3LYP
potential comparison. Both C++ and CUDA compilation use ccache.
The same finite GPU job also passes seven independent complete WB97M-V
energy/analytic-force, displaced-energy and stale-snapshot cases with joint
SCF/force AO maps and the indexed force schedule (66 successful native calls,
467 actual XC submissions). These are qualification runs, not endpoint timing.

Native integration qualification uses master 9c54107ca plus the resident AO
and indexed-force composition at 5ac08497e, with this adapter patch. It is
not a standalone-master endpoint benchmark. The retained local artifact is
`.artifacts/vwn-spin-boundary-20261004` in the integration checkout:

- Source identity: `ebdf07921464440e085b2925a1bd061ba9090788485d1cab01e3cada7122814c`
- Library SHA-256: `5a1b86cef1c0e978118d3024dc36861ebe8cd326dee8a6fe944a6b34000a5461`
- Full native test SHA-256: `f43506e0489e76852f63c1f3735e1acbe0de8cfe0681b2360a2994396b4734f7`

This repairs a shared-stack correctness gate; it makes no B3LYP performance
claim and does not change the WB97M-V formula or its benchmark attribution.

## Revisit when

Re-audit the explicit substitution if upstream VWN interpolation changes,
or if a consumer extends derivative admission to the spin boundary. Preserve
the independent closed-form and one-ulp tests when changing emission order.
