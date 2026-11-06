# Decision: admit qualified full-range RKS graphs through the public shared owner

Status: implemented
Date: 2026-11-05

## Problem

The full-range composite work retained a separate public-admission gate while
its independent molecular and installed spherical-Ni evidence was incomplete.
Merely changing a PR from Draft to Ready or passing generic CI does not complete
those scientific gates. Conversely, a low-level result must not claim to be a
public Calculator endpoint simply because its MethodIR is mathematically eligible.

## Decision

Keep the native mixed J/K/XC response, generated source weights, spherical
pullbacks and generic stationary executor unchanged. Add a separate public
admission predicate for the independently tested PBE0 and Gaussian/VWN-RPA
B3LYP semilocal compositions with one full-range coefficient in `(0, 1]`.
Compare semantic payloads, not user-provided method names or identifiers.
The Calculator also requires CPU, direct all-electron restricted FP64,
Cartesian/spherical AOs through f, and no dispersion or other extra primitive.
WB97M-V retains its distinct existing admission predicate and backend domain.

Only the actual public wrapper adds `public_calculator_endpoint=True`, native
SCF energy/iterations/Fock-build/backend receipts and complete endpoint time.
The underlying owner always reports a low-level entry point. A receipt therefore
describes what ran, not what the same method could theoretically expose.

## Qualification and invariants

- Keep the independent three-step energy-matrix and analytic-gradient/component
  checks for PBE0, B3LYP and 37/100 exchange, including asymmetric HeH+.
- The opt-in installed Ni producer must assemble every actual raw column.
  Do not fill a missing column from translation, symmetry or a previous HVP.
  Compare it with the independently converged spherical def2-SVP references;
  retain the earlier partial-matrix timeouts and failed reference guesses.
- Keep the frozen `1e-5` independent matrix/action/component gate, `2e-7`
  native raw invariance/H@v/bilinear gate and `1e-9` response residual gate.
- The default-grid reference requires Treutler's Bragg-radius partition
  adjustment. Reusing the explicit equal-radius fixture's reference prescription
  gives a different Hamiltonian. Reject that mismatch rather than loosening
  density/energy or derivative gates.
- Preserve all 59872 default Ni quadrature points. An insufficient AO/grid lease
  must fail explicitly; an explicitly larger lease is not a whole-process cap.
  Measure full public calls and simultaneous process-tree RSS separately.
- Rotation covariance on the explicit angular quadrature uses a pi/6 z rotation
  that preserves the exact finite point set. This is not a claim of arbitrary
  grid-orientation invariance or rotational zero modes at a nonstationary geometry.
- The 25–40-atom installed Ni production-scale gate remains explicitly open under
  #2843. This slice does not claim UKS, D3, ECP, TS optimization or frequencies.
  Through-f Hessian-source qualification also does not promote a separate public
  first-order force consumer beyond its independently admitted angular domain.

## Rejected alternatives

Do not select a simpler semilocal carrier as the physical method, combine
separately converged HF and semilocal Hessians, call PySCF or displaced SCFs in
production, or treat a green general CI run as scientific closure. Do not use
memory boundedness or warm compiler artifacts as a cold-endpoint speedup claim.

## References

- Supersedes the public-admission deferral in
  [the initial full-range note](2026-11-04-full-range-composite-second-order.md)
  and [the coefficient-provenance note](2026-11-04-cpu-hybrid-coefficient-provenance.md).
- `tools/qualify_global_hybrid_ni.py` and
  `tools/qualify_global_hybrid_ni_grid.py` preserve executable acceptance protocols.
- `docs/user/hessians.md` and `docs/developer/dft_hessian.md` own the current API contract.
- Issues #2224 and #2843; PR #2977.
