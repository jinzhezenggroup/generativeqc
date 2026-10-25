# CPU hot-loop allocation disposition

This audit is a **source-site inventory**, not a count of runtime allocations or
an instruction to eliminate every allocation. Baseline: `master` commit
`fb9586569769fccac67bc23411d902e08119024d`, 32
`python.loop-host-allocation` findings in the retained advisory work-audit
artifact. Method/identity changes require a new audit; these counts are not
portable benchmark measurements.

The table below records the disposition of those baseline sites only. Files and
loops have changed since, so re-run `python tools/audit_native_work.py
--summary-only` for the current inventory rather than reading these rows as a
live finding count.

## Owner and resolution

| Scope | Sites | Disposition |
| --- | ---: | --- |
| `response_operator.to_dense` | 1 | Hoist impulse vector, clear before each synchronous oracle action |
| `response_solver._block_solve` (projected panel) | 1 | Preallocate bounded projection scratch; reuse active views |
| `rks_hessian.rks_hessian` | 1 | Reuse bounded direction block, clear before each block solve |
| `rks_hessian_integrals._run_kernel_hvp` | 1 | Reuse component-weight scratch; shell-stream freezes values per chunk |
| `common.evidence.finite_difference` | 1 | Reuse gradient accumulator; detach every step through `.tolist()` |
| `dft.features` (spin density/orbital gradients) | 2 | Write directly to final spin-major gradient backing instead of restacking per spin |
| `dft.grid.partition_weights` | 1 | Share lazily created exact zero vector for coincident-center pairs |
| `dft.nonlocal_reference` | 1 | Clear/reuse point-coordinate accumulator |
| `dft.spatial.SpatialTasks.validate` | 1 | Clear/reuse omitted-AO bit mask for certified regions |
| `dft.spatial.build_spatial_tasks` | 4 | Reuse bounds, screening-off full mask, and zero discard diagnostics; every published task owns immutable copies |
| `tensor.autodiff._vjp_einsum` | 1 | One maximum-operand all-ones backing with shape views |
| `xc.contractions._geometry` | 2 | Reuse spin-contraction workspace and one directional AO panel |
| `xc.reference.exchange_reference` | 3 | Share LDA unit/zero coefficient buffers between spin components |
| **Consolidated source-level pass** | **20** | Source-level hoists and owned-buffer replacements; endpoint performance not yet claimed |
| `dft.nonlocal_integration.geometry` | 3 | Bounded nonlocal geometry AO-response scratch reused; endpoint performance not yet claimed |
| `rks_hessian_directional.native_rks_xc_hvp_components` | 1 | XC HVP directional impulse backing reused; endpoint performance not yet claimed |
| **Delivered separately** | **4** | Landed as independent changes; evaluate against one combined revision rather than adding the two halves as a performance estimate |

Eight original findings are **not fixable by simply hoisting scratch**:

| Scope | Sites | Why the buffer remains |
| --- | ---: | --- |
| `checkpoint.save_checkpoint` | 2 | Each serialized native warm state needs detached, independently owned density/coordinate payloads |
| `dispersion.D3CorrectionBatch.execute`, `D4CorrectionBatch.execute` | 3 | Each batch result supplies distinct writable native ABI output pointers; aliasing across items changes results |
| `progressive.initialize_from` | 2 | Source density and coordinates are independently captured native projection inputs; their lifetimes/owners cross source/target execution |
| `response_solver._block_solve` rank-zero branch | 1 | `np.empty((n, 0))` has **zero numeric elements** and publishes an independent empty-basis object for each RHS |

The eight retained entries are **known allocated-by-contract or zero-payload
representations**, not proof that all related data movement is optimal. They
stay visible in the advisory report. Do not add an allowlist that hides them.

## Validation and non-goals

The CPU regression selection must exercise source-count assertions for the
actual reusable buffers, scalar/blocked response, central finite differences,
coincident Becke centers, spatial full/screened masks, XC gradient ingredients,
TensorIR einsum VJP, VV10/rVV10 and Hessian direction/weight owners. Existing
independent endpoint tests remain mandatory. The static inventory is never
runtime peak memory, requested allocation bytes, or full CPU/CUDA owner
coverage. The runtime verification contract in
[replay_allocation_receipts.md](replay_allocation_receipts.md) remains necessary
to satisfy the full acceptance criteria. This source-only work does not
complete the runtime-verification contract, and it is not a wall-time speedup
claim without a matched experiment.
