# Decision: Keep fitted long-range Hessian sources separate from full-range DF

Status: implemented
Date: 2026-11-20

The two- and three-center AOT geometries do not change with radial selection.
Only the Boys boundary binding changes. The expanded generator supplies the
Gaussian reduced exponent and fixed omega to the same bounded interval-moment
leaf as the factorized schedule. Fixed exponents and omega keep the usual
nuclear `dF_n/dT = -F_(n+1)` rule. Component selectors and their scientific
identities include the radial request, preserving existing full-range keys.

The raw owner's normalized basis lease carries the actual source omega. It
must match the requested closure exactly. A complete full-range closure cannot
rescue a missing long-range leaf, even if topology and normalization agree.
The fitted AD component selects an exchange-only graph: deriving it as
hybrid minus Coulomb would perform unnecessary work and subtract unrelated
large terms. It differentiates the physical inverse, not its whitening gauge.

Before enabling the stationary endpoint, bind the second prepared provider
under the accepted SCF token, with its own metric, raw values, first/second
derivatives and budgets. The frozen Fock RHS, every induced CPKS action, and
the moving source weights must use that same provider. An exact LR replacement
or sharing the primary metric would differentiate a different Hamiltonian.
The public endpoint remains fail-closed for fitted LR until end-to-end
reconverged RKS/UKS oracles and resource/endpoint work gates pass.

Evidence: 53 host AOT/factorized/package tests, 16 host DF response tests, and
47 raw source plus 30 response tests on Slurm node4. Six actual-KS-owner tests
were skipped because their separate probe was not configured. The independent comparisons
use libcint values and displaced analytic gradients for the raw source and
expanded fitted ERIs for response. GPU raw probes exercise both ranges in
one linked inventory, including a rejected range mismatch and the real
spherical f/g projections. Logs: `build/hessian-gpu-sm120/lrdf-host.log`,
`build/hessian-gpu-sm120/lrdf-response-host.log` and
`build/hessian-gpu-sm120/lrdf-complete-closure-node4.log` and
`build/hessian-gpu-sm120/lrdf-probes-node4.log`.
