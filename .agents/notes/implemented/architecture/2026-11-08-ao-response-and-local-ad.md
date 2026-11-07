# Decision: derive AO responses and remaining local potentials from primals

Status: implemented source migration; final qualification in progress
Date: 2026-11-08

For the subsequent ownership census and source-migration closure, see
`2026-11-08-derivative-retirement-closure.md`. Earlier qualification states
below describe their original experiments, not the final campaign.

## Problem

The scalar/native migration left AO-density geometry product rules, sigma
responses, matrix assembly JVPs and several compiler-owned but manually supplied
local derivatives. A generated header and an AD witness do not establish that
the production output itself came from AD. The final census also found native
D4 CUDA Gaussian/normalization slopes distinct from the migrated CPU path.

## Decision

- `dft.density_bilinear_ad` differentiates independent AO Taylor polynomials.
  Exact monomial collection lowers those results onto retained D*AO products
  and atom-owned columns. Symmetric D can be indefinite; no occupied-orbital
  factorization, positivity assumption or additional BLAS product is needed.
  Equal coefficients scale a completed sum to preserve finite cancellation.
- First gradients and higher responses compile cached, straight-line NumPy
  panel schedules. There is no per-point/per-tile metadata interpreter or new
  scalar algebra. Sigma responses and pullbacks differentiate actual products.
- `dft.ao_response_ad` differentiates left*D*right, with independent AO geometry
  and right-density directions. Spatial differentiation of the same primal
  supplies gradient-feature responses. Only complete legs may be exchanged
  using the accepted real symmetric-density contract.
- Matrix JVPs differentiate the scalar integrand, then lower its terms to the
  existing bounded GEMMs. Measure JVPs remain retained intermediates rather
  than distributing weight/coefficient motion into extra GEMMs. Fixed branch,
  shape, finiteness, kinetic-half and final symmetrization contracts remain.
- GFN Cartesian radial projection differentiates the norm before binding its
  cached reciprocal; native scaling order is retained. COSX ESP/product JVPs,
  molecular AO cotangents and H0 AO seeds use actual TensorIR AD, not formulas
  moved into compiler strings.
- `method.scalar_graph_ad` converts the actual admitted scalar FP64 TensorIR
  energy to Graph. Exact polynomial AD supplies onsite cubic potentials and
  symmetric-spin streaming coefficients. The schedule may order factors, not
  supply their coefficient/exponent. Altered energy monomials fail closed if
  they no longer fit the qualified schedule. Shared onsite lowering covers
  GFN1 as well as GFN2.
- D4 CUDA reference-weight Gaussian slopes and normalization JVPs now call the
  same generated scalar programs as CPU. Native underflow fallback, charge
  cotangent contractions, reductions and scatter are unchanged.

## Boundaries

The final native census additionally migrated CPU Becke pair coordinates,
norm partials, smoothing/radius adjustment, logarithmic partials and normalized
log-weight responses. The native owner retains clamping, zero/coincident-norm
admission, sparse incidence and reductions, not independently supplied slopes.
Classical D3/D4/gCP and GFN2-D4 cached radial projections now use AD of the
Cartesian norm. Exact monomial recognition cancels a Cartesian factor for
scatter coefficients before execution, including zero components; the explicit
projection retains multiply-then-divide order.

Runtime owners retain linear contractions of supplied partials/cotangents,
fixed incidence/scatter maps, solver iteration and bounded implicit solves.
These are execution of derivative data, not independently supplied nonlinear
derivative equations. The shared AD engine necessarily owns primitive rules;
special-function derivatives retain independently qualified mathematical
primitives. Explicit independent reference/oracle implementations remain tests
or declared diagnostic consumers, never hidden production dependencies.

## Rejected alternatives

- Interpreting sparse AD metadata on each feature tile measured 5–16% overhead
  and is rejected. A generic static schedule without bulk gradient publication
  still regressed small tiles. Bulk generated panel writes eliminate that
  overhead without changing retained products or storage complexity.
- Replacing symmetric D by orbital factors would reject indefinite diagnostic
  densities and change the public contract.
- Expanding moving measures across matrix products adds GEMMs. Factoring the
  differentiated measure first retains the prior work bound.
- Keeping a manually supplied potential with an unrelated AD witness repeats
  mathematical ownership and is not retirement.

## Evidence

Ignored receipts are under `.artifacts/classical-derivatives/`.

- AO-density spatial/center derivatives cover all Cartesian point/center
  direction pairs against 70-digit primal-only Gaussian densities, including
  indefinite D, ownership subsets and large finite Hessian cancellation.
- `ao-response-ad-oracles.log`: 110 passes, including native displaced AO
  collocation, mixed XC response, and 70-digit primal-only matrix integrands.
- `cosx-h0-ad.log`: 25 passes, one unavailable-toolchain skip.
- `remaining-gfn2-ad.log`: 26 passes, one unavailable-toolchain skip; retains
  scaled extreme-range ES3 and spin FMA/ragged ownership gates.
- `scalar-cosx-import-oracles.log`: 43 passes; high-precision primal-only COSX
  summands, changed-energy coefficient checks, and bare/import vendor gates.
- Pinned balanced complete array-feature requests measured candidate/master
  ratios 0.96484, 0.98419, 0.98550 and 0.99639 for (AO,points)=(4,16),
  (16,256), (32,256), (64,512), respectively. These are diagnostic array
  endpoints, not SCF/force performance claims. See `profile_density_arrays.py`
  and `density-bulk-ad-profile.json`.
- The first full GPU rerun passed 1624 tests but failed five CPU-reference
  consumers because the isolated remote environment lacked the declared LP64
  provider. Installing the same scipy-openblas32 0.3.34.0.0 as the local
  environment repairs this prerequisite without modifying system Python.
  Failure receipts remain in `gpu-response-v4/`.

## Remaining acceptance

Final source-matched GPU scientific/reference tests and balanced cold/warm/
changed complete lifecycles, CPU method qualification, ownership/generation
checks and a completed census are required before the full retirement PR.
Small positive timing differences must be investigated, not offset by wins in
other workloads. Real GPU execution stays under finite Slurm on another node,
never n3 or overridden CUDA visibility. No release publication is authorized.

## Late qualification findings

- `native-becke-ad.log`: 77 passes; direct native weight-energy differences
  cover grid versions 1–4, one/three/five smoothing steps and translation
  invariance. `latest-local-oracles.log` subsequently passed 134 checks,
  including the cached-norm primal-only oracle and vendor provenance.
- `final-flat-ctest.log`: all 122 host tests pass after these changes.
- Slurm n1 job 16751 passed 1637 scientific tests and actual native D4/ragged,
  D3, EEQ and gCP gates. The independent tblite/xTB and libcint gates remain.
  Later source changes require their own final source-matched GPU rerun.
- The complete COSX diagnostic fails identically on baseline and candidate:
  `native/Python grid plan mismatch`. Existing AO-panel storage accounting
  predates this migration. This experimental diagnostic is not an accepted
  complete force endpoint; its guard was not weakened. Generated COSX scalar
  CUDA summands pass the independent primal-only oracle separately.
- Quiet n5 Slurm job 2014 completed randomized balanced GFN2 and nonlocal
  lifecycles. GFN2 used 64 pairs; small nonlocal used 128 and large nonlocal
  used 32. The n1 binaries require glibc 2.39 while n5 has 2.35, so both were
  executed with the same explicitly private n1 loader/runtime, with hashes
  retained. No system runtime was modified. Failed manifest/runtime setup
  receipts remain. Small positive medians have intervals crossing parity;
  they are not evidence for a speedup or a zero-overhead universal claim.
- Complete CPKS exposed a genuine 4–9% regression in initial nested-Jet
  emission. Scalar sign cleanup, power-exponent alignment and host noinline
  experiments did not resolve it; noinline was worse and is not retained.
  This is not offset by improvements in other workloads. The emitter now
  differentiates a single scalar DAG for value, first, directional and mixed
  lanes, hoisting shared nonlinear work before the channel loop. Implicit
  mixed quotient residuals are reconstructed from the actual differentiated
  first-response numerator/denominator, not supplied derivative equations.
  The 450-digit tail/Libxc point suite passed 27 checks. Diagnostic point
  batches improved substantially, but complete CPKS acceptance is pending.
