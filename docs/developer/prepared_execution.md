# Prepared execution lifecycle

Built-in method owners share the host-control gate in
`src/runtime/compiled_execution_region.hpp`. `PreparedCalculation::execute` and
`PreparedBatch::execute` enter that gate before invoking the original native
implementation. CPU execution is synchronous; CUDA owners keep their streams,
arenas, handles, queue/capture policy and completion fences. The gate performs
no scientific work, compilation, transfer or synchronization.

The existing Python `PreparedExecutionLease` supplies the same vocabulary for
artifact-qualified stationary CUDA and generated D3 execution. It keeps the
existing prepared-execution schema and artifact bindings; it does not create a
second artifact cache, allocator, solver, scientific generation or registry.

## States and publication

| State | Legal operations |
| --- | --- |
| Unbound | Install/bind an already prepared owner, or close |
| Bound | Begin execution, rebind, reset/invalidate, or close |
| Running | Complete or fail; rebinding, reset, close and reentrant execution are rejected |
| Warm | Begin a compatible transaction, reset/rebind, or close |
| Failed | Revoke exports and perform owner-specific reset/refresh; an unchanged binding cannot erase failure |
| Closed | Idempotent close only; execution and binding are terminally rejected |

“Warm” means a successful endpoint transaction, **not** proof that all prepared
storage was reused. Actual source, geometry and provider reuse remains measured
by the original owners. A completed ragged dispatch also does not imply every
item succeeded: each item's status and scientific publication gate remain
authoritative. Only fleet-level exceptions fail the fleet gate; neighboring
last-good state survives isolated item failure.

The ABI publication guard revokes native exports on whole-transaction argument
or copy-out failure, including rejection before the numerical endpoint is entered.
Per-item output-buffer rejection remains a separate existing boundary: a completed
fleet can return an invalid-argument item while retaining that item's newly solved
state and diagnostics. This gate does not yet revoke that item independently;
callers must check its result status before consuming it, as required by the
{ref}`existing diagnostic contract <python-results-values>`.
Completing per-item publication/borrow lifetimes remains separate work.

A nonconverged return remains an explicitly failed attempt with its completed
diagnostics; it is not a successful solve. Exceptions cannot publish a successful
transaction. Recovery invokes the original owner's reset before resubmission.
Reset retains the original binding if allocation fails, so a later attempt can
recover. Destruction continues to release the original owner and its resources,
not a parallel owner.

## Scientific binding and placement

The gate's native owner/target binding is **not** a scientific identity, #929
generation, consumption fence or borrowed-state lease. The immutable method
controls, exact normalized source comparisons in `PreparedFockPlan::matches`,
native HF/KS solve tokens and `NativeKsSnapshot.check_current` remain the proof.
Equal tensor extents, pointer equality or a successful earlier invocation cannot
renew those proofs. ProgramIR/TensorIR and SolverRegion retain their existing
logical-storage and scientific ownership.

A CPU/CUDA switch constructs a separately qualified native owner and rebinds
executable, placement, buffers and context. Semantic model identity can survive
that switch; device pointers and capture state cannot. CUDA capability refusal
still propagates rather than silently running CPU scientific work labeled CUDA.
Python artifact consumers retain their #459 artifact and guard identities.
Stationary CUDA validates the original source/snapshot and refreshes failed or
changed-geometry storage before entering its lease.

Owned arrays, borrowed sources, views and pinned staging retain their existing
typed interfaces. The shared gate is deliberately not a replacement tensor
interface: only the actual owner can establish allocation identity, consume a
borrowed buffer and fence its lifetime. The gate itself never emits source,
compiles or downloads dependencies. Native AOT preparation/execution remains
unchanged. CPU-only builds do not include CUDA headers or acquire a driver
through this interface.

Native-admitted KS forces use the source-complete prepared C++ owner; admission
comes from the per-prepared-item capability, not from lifecycle metrics. See
[KS final-state and force ownership](ks_diagnostics.md) for the current boundary.
Other qualified Python stationary-force compositions still execute outside the
native transaction and can invoke JIT consumers. Those paths need their own
complete prepared/AOT cutover: wrapping SCF alone does not establish an AOT-only
complete public force endpoint.

## Resource and diagnostic contract

Admission remains with `ResourceBudget`/`ResourcePlan` and native provider
preflight, not the lifecycle gate. `ExecutionResourceSnapshot` reports optional
host/device observations separately for numeric, scratch, pinned,
provider-retained, staging and capture-retained categories. Its category peaks
can overlap and **must not be summed**. The existing device ledger counts owned
allocation identities and simultaneously live intervals, not borrowed views or
the sum of mutually exclusive phase high waters. Unobservable vendor storage,
host allocator overhead and capture-internal allocations remain disclosed.
Precision controls and method-specific refinement stay with their existing
precision and scientific owners.

The native stationary CUDA controller additionally reserves its retained
grid-provider allowance in that ledger's admission capacity. Numeric v1/v2
receipts remain unchanged: these allowances have no numeric allocation identity
or requested-byte event. `generativeqc_resource_ledger_read_v3` reports six fields
in one locked snapshot: request limit, numeric live/peak, provider-reserved
live/peak, and combined simultaneously-live admission peak. Do not add this
reservation again to a planner-withheld allowance or call it measured provider
storage. Controller destruction releases it only after borrowed work drains;
closing an observation handle does not forgive a retained owner's charge.

For context-admitted native CUDA KS forces, descriptor-only capacity discovery
and authenticated prepared-item discovery share the native scientific rule and
the controller's capacity polynomial. Python reserves one complete consumer per
retained item, including grid-provider admission. Fleet alternatives select
cached or minimum storage for all items together, and native per-item limits
survive coordinate rebuilds. Native reservations stay inside the ledger limit;
they are not planner-withheld `runtime_allowance` estimates. The Python receipt
keeps numeric observations unchanged and adds the v3 `admission` record.
Unsupported or mixed fleets retain the incumbent force route and its staging
allowance, rather than inferring support from a raw profile or a method name.

Private observation ABIs `generativeqc_calculation_execution_lifecycle_v1` and
`generativeqc_batch_execution_lifecycle_v1` return nine host-control fields:
state, backend, device, bindings, invalidations, submissions, successful
transactions, failures and recoveries. They neither probe CUDA nor fabricate
allocation/transfer/fence/launch/work counts. Combine these with the actual
library source/artifact identity, native resource observations, provider
diagnostics and KS transport/work ledger. Unmeasured fields remain unavailable,
not zero. Native C++ owners also expose the gate's transition reason.

## Actual-path ownership inventory

This inventory covers the #926/#934 prepared endpoint boundary. The shared gate
replaces duplicated transaction/publication bookkeeping, **not** the following
legitimate scientific or backend-specific owners.

| Family | Endpoint adapter | Scientific/resource owners intentionally retained |
| --- | --- | --- |
| HF CPU/CUDA | `src/methods/hf_method.cpp` | `PreparedFockPlan`, CPU Fock/overlap owners, native CUDA RHF/DF buckets and original solve generations |
| XC/KS CPU/CUDA | `src/methods/dft_method.cpp` | Native semilocal/grid/XC, `CudaKsPlan`, source/grid leases, physical final-state validation and stationary force artifact lease |
| MP2/UMP2 | `src/methods/mp2_method.cpp`, `src/methods/ump2_method.cpp` | Reference/correlation source admission, phase-local providers, precision policy and last-good reference seed; unsupported targets still reject |
| CC | `src/methods/rccsd_method.cpp`, `src/methods/rccsdt_method.cpp`, `src/methods/df_rccsdt_method.cpp` | CC equations, physical residuals, shared response Krylov, source admission, provider streams and bounded fallbacks |
| GFN | `src/methods/xtb_method.cpp` | Context-owned `Gfn2RuntimeBridge`, native CPU/CUDA SCC owners and provider handle preparation/destruction |
| Tensor capture | `src/runtime/cuda_graph_region.cuh` | Original graph executable, stream/arena/library binding and capture eligibility; graph submission is not an endpoint completion fence |
| Explicit generated D3 | `python/generativeqc_compiler/geometry/d3_cuda.py` | Existing artifact lease, topology/switch-state guards and native tensor owner; rebuild after failure remains explicit |

Seeded post-HF batch calls and native ragged KS scheduling remain inside the
fleet transaction; they do not create fake per-item scalar lifecycle records.
The public default execution algorithms, mathematical identities and capability
domains are unchanged.

## Qualification

`tests/native/test_prepared_lifecycle.cpp` exercises legal/illegal transitions
and real CPU HF/KS owners. `tests/python/test_prepared_lifecycle_endpoints.py`
exercises real CPU and explicitly Slurm-qualified CUDA cold/replay, changed
geometry, reset, failed publication/recovery, stale original snapshot leases,
close, ragged isolation and two admitted budgets plus preflight refusal.
Existing independent scientific references, failure-injection, ledger cleanup
and toolchain-free tests remain required.

Run `benchmarks/prepared_lifecycle.py` in separate baseline/candidate processes
on the **same backend**, matching build options, request, geometry and output
selection. Cold includes preparation; replay includes the complete requested
energy or energy/force endpoint. Retain real transfer/fence/launch traces and
simultaneously-live allocation observations, and disclose uninstrumented scope.
CUDA tests, benchmarks, sanitizers and profilers require finite `srun`; successful
compilation or simulated backends do not establish device qualification.
