# Decision: bounded host AO shell-factor reuse

Status: implemented
Date: 2026-11-13

## Problem

The independently stationary I–CH3–I saddle reference exposed a native first
energy/gradient timeout before Hessian work. A separate, short source-bound
diagnostic places batch preparation at 4.57 seconds and then enters SCF/forces.
Three owned-child debugger samples find one restricted XC density-feature
contraction and two AO evaluations in that SCF chain. These are not sampled
percentages, a converged state, or an attribution of the entire timeout.

Earlier AO evaluation reused a radial factor across jets of one component but
repeated exponentials and axis polynomials for other components of the same
shell. Raising the optional full-grid AO cache cap would not solve the bounded
fallback, UKS, or larger-grid paths.

## Decision

Group adjacent **selected** records with identical atom and primitive-range
keys. Cap each group at ten components, sufficient for Cartesian f, and reuse
displacement, primitive exponential and requested one-axis polynomials inside
the group. Traverse primitives, expansion terms and axes in their original order
for every component/jet. Do not contract radial moments or reassociate products.
Both RKS and UKS consumers inherit this shared evaluator; no new resident cache,
resource admission, external numerical provider, or backend policy is needed.

The small stack workspace depends only on the through-f/order-three contract,
not molecule or grid size. Larger same-key record runs are chunked. Sparse maps
remain sorted/unique and are applied before scientific evaluation; missing shell
components are never evaluated. Exactly underflowed radial factors skip all
far-tail polynomials, preserving the finite-result gate.

## Observability

The native molecular owner separately streams system preparation, named-batch
preparation and SCF/force phase starts and terminal durations. Completed work
counters are not estimates of in-flight work; a last phase-start timestamp is
not an endpoint duration. Failure publishes no energy/gradient state. This
diagnostic does not change SCF tolerances, root tracking or saddle admission.

The optional existing append-only `GENERATIVEQC_DF_PROGRESS_TRACE` journal now
also brackets CPU RKS/UKS SCF and physical evaluations. It records basis/grid
sizes, completed iteration/Fock-build observations and explicit terminal counts
and convergence, including exhausted UKS and incremental RKS returns. It does
not equate a returned physical evaluation with convergence. In preliminary RKS
dispatch, the actual preliminary/target SCF calls own their observations rather
than counting the dispatch wrapper as another SCF solve.

## Evidence and limits

- Two hundred compiled AO checks pass, including independent long-double
  Leibniz jets through order three, signed expanded f records, partial slices,
  sparse shell holes, different contraction ranges, empty selections, far-tail
  underflow, nonfinite rejection and canaries. The existing full-constructor
  comparison also passes both Cartesian/spherical f at O2/O3 and strict FP.
- The old source-spelling tests for radial-loop placement are replaced by
  executable numerical/selection/error checks, not new internal call counts.
- The phase diagnostic is retained separately under
  `build/qualification/host-native-phase-trace-20261113`; its source/core identity
  and intentionally killed child must not be relabeled as endpoint success.
- These local tests alone do not qualify complete endpoint speed, stationary
  heavy-element saddles, 25–40-atom scaling, or requested-level Ni transition
  states. Public ECP saddle gates remain closed pending those independent gates.
- The changed-AO cold electronic attempt remains unsuccessful: exit 124 at
  420.02 seconds, 364832 KiB RSS, preparation 4.556 seconds, no returned state or
  raw Hessian. Completed SCF work is unknown for this run. Its seal is
  `4ea3a10cd768dc54bf2054cb2afa86c38fbf42f7066772d5138d1a17f83c204c` under
  `host-shell-shared-ao-20261113`. Its 283 Python cases/nine native groups and
  isolated AO schedule measurements are not an endpoint speedup claim.

## Rejected alternatives / revisit when

Do not tune ERI preparation based solely on the old timeout; measured preparation
is already distinct from SCF. Do not increase full-grid cache limits to hide an
unbounded fallback or optimize only RKS. Consider further XC contraction/solver
work only with source-bound iteration/phase evidence and independent scientific
acceptance, retaining complete-endpoint timing and semantic work counts.
