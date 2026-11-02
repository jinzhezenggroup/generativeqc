# Decision: exercise failure injection through the prepared resource owner

Status: implemented
Date: 2026-11-02

The public native CUDA force test had retained a private-ledger observation
wrapper after prepared batches moved to public device-resource handles. These
opaque handles have different native types. Passing the public handle to
`generativeqc_resource_ledger_bind_v1` caused an actual segmentation fault in
A100 jobs 224758/224759 before force execution; their 29 production tests had
already passed. This was a fixture ownership error, not a kernel failure.

The injected low-level batch call now relies on the same native resource binding
used by ordinary prepared execution. It still supplies sentinel caller output,
requires the precise allocation/runtime failure status, checks unchanged live
capacity and releases, and verifies warm recovery. It must not wrap that public
handle in `CpuResourceObservation`. A replacement A100 run on each unchanged
scientific source passes all 29 production cases and all three fault cases
(224765/224766). Original failure logs remain retained with their actual status.

Do not work around this mismatch by changing ctypes widths or by treating the
two opaque resource handles as interchangeable. Tests must follow the current
native ownership contract while retaining transactional failure coverage.
