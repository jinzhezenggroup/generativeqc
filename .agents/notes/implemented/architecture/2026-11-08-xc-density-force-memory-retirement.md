# Decision: retire optional XC density storage at the force-memory boundary

Status: implemented
Date: 2026-11-08
Agent: /root/audit_3041
Model: GPT-6 Astra Pro

## Problem

An admitted density panel provider retains a matrix cache and a conservative
provider allowance after SCF. The force-memory handoff previously reclaimed
anchors and a phase source but could leave this optional storage resident while
the force workspace lacked capacity. Releasing the owner directly would also
leave captured SCF graphs borrowing its resources, and destructor cleanup cannot
report a failed live CUDA release. Provider preparation exceptions need the same
public allocation/CUDA status mapping as other XC operations.

## Decision

Preserve the provider and captured regions when capacity is sufficient. Under
pressure, validate the final-state token, exclude capture, drain the producer
stream, and reclaim the existing optional anchors and phase source first. If
capacity is still insufficient, invalidate both captured SCF execution regions
before retiring the optional density owner. Keep generated strict/admitted
bindings, published XC values, and the final-state token available.

Live retirement uses a checked operation, separate from destructor cleanup.
Disable provider execution before teardown; clear each resource pointer only
after its release succeeds. Retain the owning pointers and original conservative
charges if any step fails, and propagate typed allocation/runtime failures at the
XC boundary. A later handoff completes pending retirement even if a partial
release already supplied enough capacity. Debit the owner's original reserved
charges once, after the entire checked release succeeds. A pending retirement
alone does not justify evicting unrelated anchors.

## Rejected alternatives

An unchecked `unique_ptr::reset()` could silently admit a failed live release.
Dropping the provider without invalidating captured execution regions could
replay freed resources. Unconditionally retiring it would change the established
no-pressure warm path.

## Evidence and limits

`test_xc_density_repreparation.py` executes the production XC admission,
retirement, dispatch, and force-memory bodies against resource/graph boundary
doubles. It covers preserved no-pressure behavior, pressure retirement, typed
preparation/release errors, partial teardown, exact charges, and retry.
`test_panel_product_cleanup.py` compiles the real panel owner, contraction
context, device scope, and allocation ledger with CUDA/cuBLAS boundary doubles
to exercise checked free/destroy failures and default-stream cleanup.

These host tests establish ownership and error behavior. They do not qualify GPU
arithmetic, performance, or a pressure-to-warm SCF endpoint. Existing GPU timings
remain evidence for their recorded source revisions. Convergence and numerical
acceptance gates are unchanged; future claims about the switched execution
schedule require corresponding real-device endpoint evidence.

## References

- PR #3041, force-memory and typed-error review threads.
- `CudaKsPlan::prepare_force_memory`, `CudaXcPlan::retire_density_provider`, and
  `CudaPanelProduct::release`.
