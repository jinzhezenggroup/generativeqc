# Decision: selected-argument fixed-mask public VJP

Status: implemented
Date: 2026-10-18

## Problem

P0 issue #2137 requests real branch JVP/VJP for where. The canonical generated
fixed-external-mask VJP exists, but public vjp marks all positional arguments as
differentiable and cannot represent an external Boolean selector.

## Decision

Add explicit argnums selection to public vjp and retain its tuple result. The
default None selects all supplied positional primals, preserving existing calls.
Negative positions resolve against supplied primals, not omitted defaults. Add
allow_external_mask_where=False; its explicit true path delegates to canonical
transpose_program and admits only generic finite FP64 reference/materializing
CPU execution. No selector adjoint, implicit dtype filtering or derivative rule
is added to the public runtime.

Snapshot every primal, including the mask and non-selected real inputs. Return
selected cotangents in caller order while validating every requested output seed.
The canonical VJP retains eager primal/forward domains and freezes generated
execution feeds as non-differentiable, including disconnected/empty gradients.

## Rejected alternatives

Automatically skipping Boolean primals would silently change the argument/result
contract. Requiring fake numeric mask tangents would weaken Boolean ownership.
Changing pullbacks to return a scalar result for integer argnums would diverge
from their existing tuple ABI. Public JVP needs its own selected-seed contract.

## Integration boundary

The scalar-gradient opt-in in #2603 is complementary and remains unchanged.
This change extends only the standalone pullback interface; it does not alter
its derivative algebra or re-enable the deferred public JVP/higher-order paths.
The current contracts include both interfaces after normal master integration.

## Evidence

Independent NumPy selection/scatter adjoints, central finite differences and dot
identities qualify multi-output contractions, broadcast and repeated gather.
Reference/native tests also exercise snapshot mutation, empty/disconnected
outputs and inputs, all cotangent validation, domain rejection, budgets, owner
lifetime, compiler failure and default compatibility. Native cases forbid the
reference interpreter and reference AD entry points. Numerical acceptance uses
rtol/atol 2e-13 for analytic contractions and 2e-8 for finite differences.

## Revisit when

Broader dtype/scientific/backend or higher-order domains are independently
qualified. Parent issues #2137 and #2111 remain broader than this public VJP slice.
