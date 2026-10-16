# Decision: public CUDA second order uses the same native owner as the CLI

Status: implemented; native frontend integration qualified
Date: 2026-10-16

## Problem

Moving the C++ CLI off a Python worker did not migrate `Calculator.hessian*`:
the public Python CUDA route still constructed snapshots, AO providers, Python
response schedules and runtime-generated derivative kernels. A second production
numerical owner would preserve the very dependency the migration is removing.

## Decision

`_native_hessian` is only a frontend for a synchronous private native C bridge.
It checks capacities/shapes and the linked-package capability **before** SCF,
prepares/executes an ordinary native energy batch, then calls the same
`CudaKsWb97mvHessian` used by the CLI. Under the context mutex, the bridge obtains
the live final-state token and producer stream directly, without snapshot export.
Native resources own upload/output lifetimes and drain failed downloads before
host targets die. Only complete converged terminal panels and scalar receipts
cross the boundary. Caller output and receipt remain untouched on failure.

The source-specific AOT definition prevents CUDA-enabled/AOT-disabled libraries
from referencing unavailable Hessian owners. Such builds expose capability zero
and a fail-closed stub. It does not globally alter every source's build flags.
CPU/development implementations remain available but are not CUDA fallbacks.

Native results expose immutable aggregate `value`/`matrix` and detached work
diagnostics. Intermediate source splits stay resident instead of being exported
just to reproduce legacy debug records. The public methods already promise the
aggregate field, not the CPU/development result class.

The default strategy is now `None`, resolved by backend. Native CUDA accepts
only scalar columns, not an explicit `recycled` request masquerading as actual
recycling. CPU defaults still resolve to the existing `recycled` strategy.
Supported GMRES controls are forwarded exactly. Nondefault legacy heuristic
controls are refused explicitly; native true-residual convergence is authoritative.
Global ResourceBudget planning of the response arenas is not implemented:
such requests fail before SCF rather than silently charging only SCF storage.
Endpoint-specific limits remain explicit. Result identities bind model,
geometry, charge/spin, shape, direction and values, not merely output bytes.

## Source-only auxiliary role

The C++ CLI already admits g auxiliary values, but the Python/generic C system
gate correctly retained orbital-through-f limits. Do not globally widen it or
reuse a correlation-only handle under a different physical consumer. A distinct
KS auxiliary role admits values-only native CUDA DF preparation, cannot serve
as an orbital or generic HF auxiliary, and does not authorize derivatives.
Python selects this scope only for canonical WB97M-V values; first-derivative
preflight stays independent. Hessian admission still reauthenticates the actual
model, DF source and linked f/g closure.

## Rejected alternatives

- Running the CLI as a Python subprocess: duplicates SCF/serialization and does
  not give Python callers a native prepared-state lifetime.
- Constructing `NativeKsSnapshot` first: exports D/W and other numeric panels,
  defeating the resident accepted-state handoff.
- Silently ignoring requested recycling, custom solver controls or byte limits.
- Returning partial results after a native nonconvergence or asynchronous failure.
- Advertising CuPy/compiler extras as production CUDA dependencies after cutover.

## Evidence

Host admission/dispatch gates reject legacy imports/numerical owners and missing
AOT packages before SCF, preserve supported options, detach results/directions,
and refuse unsupported strategies. Native API input tests cover source-only roles
and unchanged publication on rejection. Real-GPU integration extends the actual
CLI oracle tests with public Python HVP/full-matrix calls, forbidding *all*
subprocess creation during the numerical endpoint. Slurm job 3223 (node2/pro6000,
finite 15 minutes, PTX JIT disabled) passed all 25 high-angular signed-contraction/
late-invalid-record controls and all three actual CLI plus public frontend cases
in 29.81 s and 179.12 s respectively. These cover a complete direct RKS matrix/HVP
and UKS/DF/ECP with separate spherical d and g auxiliaries. The g case uses the
independently reconverged values-only molecular oracle, not a second expanded
derivative implementation. All subprocess creation was forbidden inside public
frontend numerical calls. The corresponding library SHA256 is
`29cf2e51a26d0044b1947e5519c0e55ec50e38090319e6d7472585c496964eb9`;
the linked factor inventory hash is recorded in the factorized-closure note.

Earlier job 3220 passed direct RKS and UKS/DF/ECP+d but failed g before native
execution at the generic auxiliary-through-f preflight. That was not a missing
derivative kernel; the distinct values-only KS handle and scoped preflight fix
this ownership mismatch. Neither the generic orbital gate nor first derivatives
were globally widened.

## Revisit when

Native multi-RHS/recycling or optional source-component publication is implemented
with independently bounded resident storage and complete convergence/lifetime gates.
