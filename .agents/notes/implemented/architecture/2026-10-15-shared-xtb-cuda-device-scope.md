# Decision: share CUDA device-scope ownership with xTB

Status: implemented
Date: 2026-10-15

## Problem

Shared CUDA resources and the native xTB descriptor boundary independently
implemented temporary device selection and restoration. The implementations
serve different error contracts: resources throw with a caller-supplied checker,
whereas descriptor admission returns status codes and must report restoration
failures. Replacing the latter with an ordinary throwing guard would lose its
explicit cleanup diagnostics and retry semantics.

## Decision

`src/runtime/cuda_device_scope.hpp` owns the thread-local device-state lease.
`cuda_resources.cuh` continues to expose the same throwing resource API through
that header. The xTB `ScopedCudaDevice` becomes a status/diagnostic adapter to the
same owner; it no longer stores or restores an independent device-state lease.

The throwing constructor retains unconditional device selection, including the
same-device case, so resource preparation keeps its existing CUDA initialization
boundary and custom exception checker. The nonthrowing constructor preserves
xTB admission: reject negative ordinals without CUDA calls and avoid redundant
selection when the requested device is already current. Query failures and
selection failures remain distinguishable for status translation.

Explicit restoration is idempotent after success. Failure keeps restoration
pending so an explicit retry or destruction can restore the caller's device.
Destruction is best effort and consumes cleanup errors; status boundaries remain
responsible for consuming errors they handle explicitly.

## Rejected alternatives

- Reuse the old resource guard unchanged: it cannot report restoration failure
  or preserve xTB's nonthrowing boundary.
- Move the xTB guard wholesale into shared resources: its private status enum
  and diagnostics would reverse the shared-to-method dependency boundary.
- Normalize both constructors to the same-device fast path: that would alter
  the resource API's existing initialization behavior without qualification.

## Invariants

- Neither shared runtime nor tensor/provider infrastructure depends on xTB.
- A failed initial device query or selection never acquires a restoration lease.
- A successful explicit restore neither overwrites a caller diagnostic nor
  triggers a second restoration during destruction.
- The scope does not allocate device storage, initialize scientific data, alter
  Slurm visibility, or substitute CPU work for CUDA execution.

## Evidence

`tests/python/test_cuda_device_scope.py` compiles the actual shared header and
production xTB descriptor translation unit against a host-only CUDA control
double. It exercises nested leases, exception unwinding, same-device behavior,
negative and invalid ordinals, query/selection failures, custom exception
translation, allocation errors, explicit restoration failures, retry and
destructor cleanup. These checks qualify control/lifetime behavior, not real-GPU
execution or numerical parity.

`tests/python/test_grid_cuda_creation_device.py` includes the actual shared
device-scope header when exercising grid preparation and failure cleanup.
`tests/python/test_xtb_shared_ownership.py` guards the dependency boundary.

Qualification on the `890c425b0` master baseline passed 231 host tests across
those suites and the GFN2 bridge, coordinate-admission, native-runtime ownership,
CUDA provenance, electronic dependency boundaries, DF-source teardown,
ASan-instrumented DF-MP2 transfers, CPKS status/cleanup and KS point-batch resource
suites. Existing host probes now include the actual device-scope header instead
of extracting its spelling from `cuda_resources.cuh`.
CUDA 12.9 compilation through ccache passed for the production
descriptor translation unit and a resource/tensor-header consumer at `sm_120`.
Compile-only artifacts remain under the ignored
`build-cuda/qualify-xtb-infrastructure/` directory. No real-device validation was
performed: Slurm exposed only a drained GPU node during qualification.

The CUDA source provenance manifest retains the original upstream hashes and
records this shared-owner adaptation with refreshed hashes for the two adapted
descriptor files. Its integrity checks remain enabled.

## Installed and cached consumer closure

The shared device-scope header is a transitive dependency of the installed Tensor
CUDA runtime. It must remain in the wheel asset mapping, every existing resource
consumer's source/dependency inventory, and the stationary AOT contract inputs.
Otherwise a source-tree probe can pass while installed JIT compilation fails,
and later guard changes can reuse an artifact with stale device ownership.

The installed-header closure regression compiles only declared wheel assets.
The generated XC identity regression changes only the device-scope input and
requires both its retained source hash and complete artifact identity to change.
Host-only CUDA declarations include the destructor's last-error API; this is
header portability and identity coverage, not real-device execution evidence.

## Revisit when

A backend needs asynchronous device-state ownership or CUDA changes the
thread-local selection/initialization contract. Such a change must retain
explicit status-boundary cleanup reporting rather than hiding it in a destructor.
