# Decision: Python DFT batch forces select a native prepared force owner

Status: implemented (stacked on native batch #2230; numerical gates pending)
Date: 2026-10-11

## Problem

Even after the native batch reuses the single-system stationary-force sink,
`python/generativeqc/batch.py::execute` unconditionally sets
`native_compute_forces = False` for all DFT and dispatches the Python
stationary gradient source. That prevents Python clients from using the
same qualified C++ engine as the installed C++ SDK and native CLI.

A Python if-chain that mirrors PBE/PBE0, DF, spin, correction and CPU domain
would recreate the very multiple-source admission logic we want to remove.
Widening native support to match all Python force variants in one step would
silently omit unqualified scientific sources.

## Decision

Add a *private versioned* native C-linkage shape-free, computation-free
`generativeqc_ks_batch_supported_properties_v1` query. It obtains the actual
prepared item capability through a new optional `PreparedBatch` virtual
property method; only the native KS batch overrides it, delegating to the
existing `KsPreparedCalculation::supported_properties()`. No second
functional/backend predicate is introduced.

The Python batch checks all items after its immutable method/policy verification.
Only if every native prepared item advertises FORCES does it pass real force
buffers to `generativeqc_batch_execute`; in that admitted domain it does not
import, compile, launch or copy the separate Python stationary-gradient
consumer. If the symbol is absent (older installed native library), the
native method explicitly lacks the feature or one item is unqualified, the
existing Python force pipeline continues unchanged, preserving Python's
previously qualified broader CPU/CUDA force support.

Any malformed response or non-NOT_IMPLEMENTED C status raises immediately;
it is **never** treated as a scientific fallback. A native execution failure
is returned as a native per-item error, not retried with Python physics.

This changes only the dispatch/ownership for qualified native methods. It
does not claim performance gain, add a new method equation, change basis,
corrections, numerical tolerances, or drop the existing independent Python
force implementation until the full #2227 migration gates are complete.

## Required acceptance

1. Native C++ prepared-property query returns a stable, original qualified
   force bit only in the native CPU DF-PBE/PBE0 RKS supported domain, without
   SCF/allocations and with transactional invalid-index rejection.
2. Python tests prove the native query decides (no Python method-name whitelist)
   and reject fallback to the Python force callback in an admitted H2 case.
3. Numeric equality of Python API native-selected batch and existing Python
   full stationary-force result, plus independent reconverged finite
   differences, cold/warm/moved replay, bad neighbor, and buffer failures.
4. Existing Python CUDA/UKS/ECP/Direct and nonlocal/correction force tests
   continue using their original qualified route. Old native library remains
   supported as a compatibility path.
5. Source-level and runtime checks prove no silent CPU/PySCF or JIT force
   computation in the native-selected endpoint. Resource admission remains
   conservative and per-item error propagation remains intact.
6. A real GPU qualification is necessary before claiming native CUDA DFT
   force parity or switching those methods; this step claims neither.

## References

#926, #932, #933, #934, #1423, #2151, #2222, #2227 and stacked #2230.
