# Decision: Keep budgeted KS AO maps on the dense incumbent

Status: implemented
Date: 2026-10-07

## Problem

The public KS numeric inventory reserves dense XC, retained fleet owners and
later force storage. Optional geometry-owned local-AO maps append worst-case
per-tile indices to the retained dense arena. They have no separately declared
allowance in that inventory. Early allocations can consume later owners' or
force capacity even when every allocation passes the common ledger.

After the separate fifth-grid-array inventory correction, H2/STO-3G PBE with
49,152 points and tile 256 reserves 43,472 private dense XC bytes. Default and
explicit local-map requests instead retain 46,544 bytes, an additional 3,072
bytes per owner. In the real public 2,048-owner plan, the 4,665,708,156-byte
numeric ledger admits those owners but then rejects a 512-MiB allocation from
the declared force reservation. Dense preparation leaves that reservation intact.
This is a reservation-contract proof, not a molecular force or GPU OOM receipt.

## Decision

The existing shared native `admit_ao` boundary fails closed whenever the native
device ledger is active. That ledger has no optional-map allowance. The owner
retains dense execution and its existing work diagnostics distinguish requested
maps from selected maps. Public plan decisions explain the budget fallback.

This includes unlimited public `ResourceBudget()` and explicit
`GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`. Capability and invalid-control validation still
precede admission. Ordinary unbudgeted default/explicit requests and `=0` retain
their existing behavior. No resource estimate, budget, ABI, scientific kernel,
force tiling, CPU policy or matrix-provider reservation is changed. The existing
dense layout continues to own its arithmetic capability decisions; there is no
new mixed-precision or matrix-cache policy.

## Rejected alternatives

Current free ledger bytes are not an optional allowance: later owners and force
storage have not necessarily allocated yet. Counting every optional map as
mandatory would reject dense-fitting budgets. A separate optional-map ABI or
planner candidate is beyond this bounded correctness repair.

## Evidence and limits

`test_ks_active_ao_resources.py` compiles the actual native request/admission,
resource arithmetic and dense fallback diagnostic blocks using the shared
native-source fixture. Its real public planner and C-API ledger checks cover
RKS/UKS, unlimited/exact caps, 1/2,048 owners and default/0/1 controls. Three
bind/unbind/rebuild rounds keep a 512-MiB force allocation live across subsequent
owner replacements, retain charges between observation scopes and release all
charges after teardown. Unbound controls recover ordinary map admission even
while the ledger object still exists. Invalid explicit modes and incompatible
explicit requests remain rejected. Removing the ledger guard reproduces 16
failures, including force-reservation allocation rejection for 2,048 owners;
unconditional disable and an inverted ledger guard also each fail 16 controls.
The native admission matrix checks that dense fallback preserves the established
component-wise precision capabilities instead of imposing a new schedule veto.

Host CUDA allocation doubles do not execute AO discovery, scientific kernels or
molecular forces. No new GPU or performance qualification is claimed, and the
separate current-source default-batching slowdown remains unresolved.

## Revisit when

A public plan separately reserves optional-map residency over all owner and force
lifetimes and conveys that allowance to native preparation without rejecting
budgets that fit the dense incumbent.

## References

- PR #2089
- `2026-10-07-xc-point-batch-public-budget-fallback.md` in the performance notes
- `docs/developer/xc_native_cuda.md`
