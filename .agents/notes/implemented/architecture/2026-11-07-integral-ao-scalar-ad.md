# Decision: automatic scalar channels and Gaussian axis jets

Status: implemented; complete endpoint qualification in progress
Date: 2026-11-07

For the subsequent ownership census and source-migration closure, see
`2026-11-08-derivative-retirement-closure.md`. Earlier qualification states
below describe their original experiments, not the final campaign.

## Problem

Unrolled integral derivatives did not retire the dynamic CPU ERI/DF interpreter
or CUDA generic fallback. Their Jet/Dual/Dual3 arithmetic and Boys seeds still
maintained separate chain rules. CPU AO grids also differentiated coefficients
by hand. Moving expressions into string templates merely relocates authority.
r2SCAN's tail had three handwritten derivative/composition copies as well.

## Decision

`integral.scalar_ad` differentiates scalar primals using the existing Graph.
Implicit residuals provide quotient and square-root schedules. Channel-invariant
frontiers are constructed once per operation, outside the CPU channel loop and
shared by three CUDA directions. Native storage, recurrence traversal, precision
wrappers and linear contraction/scatter remain. The CPU recurrence is still
structurally independent of the unrolled Hermite DAG, but is no longer an
independent handwritten chain-rule implementation. Independent complex-step,
quadrature and molecular oracles remain necessary.

Radial seeds differentiate `exp(-T*u*u)`, then integrate the resulting Gaussian
monomial to the shifted value moment. Full/LR/SR boundaries are fixed with
respect to T. This does not differentiate an approximate value evaluator or
restrict omega.

`dft.ao_axis` differentiates the complete Cartesian Gaussian monomial. CPU
lowering collects powers of x and emits bounded coefficient stages. Every
coefficient, including zeros, is retained operationally: extreme finite alpha
can overflow `2*alpha`, and `Inf*0` must still reject an AO jet. CUDA removes
the common Gaussian from the AD result and lowers the polynomial. Neither
schedule evaluates another exponential or divides by an underflowed Gaussian.
The CPU seven-coefficient internal bound, public through-f/order-three domain,
Horner order, primitive summation, selected columns and one radial factor per
primitive/AO/point remain.

`xc.point_ad` owns r2SCAN's C2 tail and its energy composition. The reflected
quintic primal avoids cancellation of order-one derivative terms near the upper
endpoint. Native density admission and thresholds are unchanged. Host, grid
CUDA and stationary geometry CUDA call the same generated helper. The two
density channels differentiate the continuation; other features hold it fixed.

Assets are checked deterministically and packaged. The range CPU JIT closure
includes the new scalar header. The neutral Graph/emitter boundary gains no
SCF policy. Generated source remains conservatively scientific in the CUDA
ledger; changing a count is not repository-wide derivative retirement.

## Evidence

- Host native Dual3 arithmetic matches primal-only complex-step differentiation.
  Independent adaptive quadrature checks moment directions for orders 0/1/4/8,
  T=0/1e-10/0.3/8/120 and all three radial intervals.
- r2SCAN composition matches a 130-digit incomplete-beta primal at adjacent
  floating-point endpoints, midpoint and interior/tail points, for all seven
  physical feature directions, not merely the switch slope.
- The combined scalar/AO host tier passed 148 checks with 10 device opt-in
  skips. Pinned independent recurrence, long-double Leibniz, selected/dense
  evaluator, failure and extreme IEEE gates remain. Historical bitwise AO
  results also survived; no new production-source fingerprint gate is added.
- Freshly emitted integral/XC arithmetic plus scaled-XC, GFN2 origin and actual
  nuclear CUDA TU gates passed 63 checks in a finite 15-minute `srun` on node1,
  main/gpu:5090. Device visibility was not overridden.
- Full-build qualification found a generated temporary-name collision in the
  tail helper and an omitted generated include directory for standalone
  internal-header tests. Both were fixed at their generation/CMake boundaries.
  The complete CPU Release build, all native test executables and default
  stationary derivative AOT then passed.

Ignored receipts live in `.artifacts/classical-derivatives/` and on n1 under
`/home/jzzeng/generativeqc-qualification/classical-ad-20261107-b40b57be5`.
The GPU snapshot is `candidate-axis`, with `axis-gpu-gates.log`. The complete
local build is `scalar-ao-full-build-v3.log`. Workers use verified sccache
0.16.0; neither the cache nor another process's server is stopped or cleared.

## Rejected alternatives

- Keep hand chain rules in fallback because most requests use AOT: fallback
  is production and retains arbitrary angular/range domains.
- Fully simplify CPU AO polynomials: deleting nominal zero operations changes
  extreme finite-input failure behavior.
- Differentiate a whole Gaussian anew per AO jet: repeats radial work and risks
  dividing by an underflowed Gaussian.
- Treat point tests as SCF/force performance evidence: full endpoint, iteration
  and semantic work gates remain separate requirements.

## Remaining scope

This is not the full-retirement PR. VV10/rVV10 still maintains pair/local-scale
partials and row composition manually, including native fallback compositions.
Audit its primal authorities, preconditioned rVV10 and exceptional domains.
ECP quadrature also emits a supplied Legendre slope for Newton iteration;
differentiate its bounded primal recurrence instead. Molecular CPU/CUDA
performance, GFN2 endpoint evidence, method ownership and bounded fallback
checks remain necessary before claiming repository-wide retirement.
The conservative AO-region envelope also retains a derivative-coefficient
recurrence in `dft.envelope_cuda`; migrate it without weakening directed
rounding or bound coverage, rather than confusing it with evaluated AO jets.

## Revisit when

The nonlocal, ECP/envelope and further quadrature source migrations are
superseded by `2026-11-08-nonlocal-quadrature-ad.md`. Complete endpoint gates
remain mandatory; this is not a full-retirement acceptance claim.

Promote another recurrence or AO lowering only with independent numerical,
exceptional-domain and full-endpoint no-regression evidence. Do not remove
retained zero operations merely because symbolic algebra calls them zero.
