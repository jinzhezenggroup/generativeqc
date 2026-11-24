# Decision: Apply CPU KS Hessians to direction panels on the live native state

Status: implemented
Date: 2026-11-24

The CPU batch action accepts complete atom-major Cartesian directions in a
direction-major panel. It uses the same prepared-state token, response solver
and physical Hessian-column assembly as the scalar action. Concurrent RHS
actions, when admitted, are independent solves, not a block Krylov space or
part of the recycled dense schedule. The width-one fallback retains the
same physical equations. Repeated and dependent directions are legal inputs.

Keep the full input snapshot and the complete candidate charged to the output
and aggregate reservations. Neither an earlier successful group nor a partial
receipt may escape after a later failure. Coordinate-only integral retention
stays on the dense coordinate path; external actions do not assume unit
vectors or construct a dense matrix to obtain the action. Work limits include
all requested directions, including group tails and additive corrections.

Validation uses the prior scalar C/SDK actions, independent reconverged
moving gradients, exact/short output limits, and a streamed failure after a
converged zero first direction. The Python panel façade composes the native
pre-SCF byte floor and its marshalling storage with the primal request. The
direction count is part of that resource plan's topology. The installed CLI
uses the same count-based native query for `resources` and `hvp`, separately
charging its packed input buffer and excluding JSON conversion overhead.
This does not admit CUDA or establish complete #932/#2227 coverage.
