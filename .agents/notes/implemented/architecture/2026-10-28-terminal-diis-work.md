# Decision: owner-bound terminal DIIS work observations

Status: implemented
Date: 2026-10-28

## Problem

The ordered ring-Gram kernels already count logical reductions, but production
HF/KS launches did not supply device storage. A pointer alone would omit resource
admission, asynchronous destination lifetime and retry/Graph reset semantics.

## Decision

Use the existing optional progress journal to request a per-history device record
and owner-retained terminal readback. The arena layout queries and retained-host
inventory charge the storage. Freeze enablement during preparation. Count work
through ring retirement and internal resets; clear only at a new attempt so
discarded arithmetic remains visible. The live count/head describe the final ring,
not the number of reductions performed over its lifetime.

HF copies immediately before its existing result fence. KS copies from the
terminal result port, once per attempt, with one diagnostic-only terminal fence.
Partial copy failure drains the stream and publishes no successful receipt.
Serial/unordered fallbacks have no instrumented census and explicitly say so.

## Rejected alternatives

- Reusing a Gram buffer or a host pointer would violate device writes and aliasing.
- Iteration readbacks would add synchronization and interfere with Graph capture.
- Clearing on retirement would hide work already performed on discarded history.
- Inferring work from iteration counts would miscount inactive items and retries.

## Invariants and evidence

No change to the numerical algorithm, admission threshold or tolerances. Disabled
execution allocates no numeric counter storage and adds no transfers/fences.
`test_diis_work_journal.py` exercises observation validation and unavailable routes;
`test_hf_resource_layout.cpp` checks optional capacity; `test_cuda_diis_work.py`
qualifies complete RHF/RKS E+forces, cold/warm/moved/reset, failure and replay under
Slurm with independent PySCF energy/analytic-force references. The existing native
cached-Gram test owns wrap, retirement, invalid-history and staged-route coverage.

Revisit the transport if a versioned public work API replaces the diagnostic
journal. Do not reinterpret logical vector elements as DRAM traffic or timing.

References: #1874; `docs/developer/df_component_trace.md`.
