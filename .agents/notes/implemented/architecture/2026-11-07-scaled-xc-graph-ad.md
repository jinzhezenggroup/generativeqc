# Decision: compiler-owned arithmetic and physical scaled XC responses

Status: implemented; full endpoint performance qualification pending
Date: 2026-11-07

For the subsequent ownership census and source-migration closure, see
`2026-11-08-derivative-retirement-closure.md`. Earlier qualification states
below describe their original experiments, not the final campaign.

## Problem

The scaled PW92/PBE point contract had handwritten first-direction arithmetic,
nested response arithmetic, physical exchange potentials, and a special tail
quotient JVP. Simply replacing this contract with an ordinary Libxc work domain
would lose its empty-spin/C2, subnormal-gradient and energy-underflow semantics.
Moving those derivative formulas to string templates would not retire their
independent mathematical maintenance.

## Decision

`xc.point_ad` uses the existing scalar Graph to differentiate primal operations
and normalized physical exchange energies. A packaged generated header supplies
the existing first-channel and nested directional containers. Native consumers
retain fixed numerical scales, primal correlation composition, boundary branches,
domain admission, direction seeding and publication. Both checkout and installed
JIT identities include the generated header, including stationary asset closures.

The quotient JVP is obtained by implicitly differentiating `y*right-left=0` and
solving its scalar residual, rather than explicitly differentiating an inverse
square. Subnormal reduced-gradient directions use the same primal expressed in
fixed local perturbation coordinates, with separately scaled tangent inputs.
Neither operation introduces a density floor. Powers are explicit compiler
specializations for the finite set used by the primal, not a dynamic native
derivative rule or backward-compatibility facade.

Exchange is differentiated in dimensionless physical coordinates. Its density
coordinate is a logarithmic perturbation about the fixed reference density;
gradient coordinates use rho^(4/3) or the reference gradient norm. Restore
density potentials with rho^(1/3), independently of whether the energy's
rho^(4/3) is representable. Differentiate an algebraically equivalent reciprocal
enhancement to avoid cancellation in quotient-rule partials, then lower the
shared primal back to the original branch's stable value expression. The
second-order consumer seeds the generated first-channel arithmetic through
these same automatically derived physical potentials, without storing a point
Hessian or adding another XC model.

## Validation and limits

- The generated (not fingerprint-pinned) header is compiled against 30 RKS and
  48 UKS independent 450-digit unscaled-energy response fixtures, retaining the
  pre-existing relative plus cancellation-magnitude roundoff gates.
- The packaged header executes all 97 independent energy/potential domain
  fixtures, including minority-spin, exact vacuum and extreme tails.
- Weighted LDA/PBE interior responses retain independent Libxc finite-difference
  gates; FP32/strict point tests retain independent Libxc and tail policy gates.
- Host Graph/classical/point focused tests pass (102 plus CUDA opt-in skips);
  the expanded emitted/package response and physical point tier passes 28 tests.
- Real sm120 device point and mixed-point qualification is run separately on n1
  through finite Slurm allocations. Compilation or point qualification alone
  does not establish molecular SCF/force no-regression acceptance.

The independent oracle tables and their generators are unchanged. Remaining
repository-wide derivative owners and complete method endpoint gates are not
retired by this change. See `2026-11-07-classical-scalar-ad.md` for the broader
migration and earlier correction-endpoint regression diagnosis.
