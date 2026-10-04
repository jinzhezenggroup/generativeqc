# Decision: bind optional cuTENSOR through the shared contraction table

Status: implemented
Date: 2026-10-05

## Problem

The affine provider introduced for #1887 prepared and replayed valid TensorIR
descriptors, but consumers would need another execution dispatch and lifetime
owner to use it. That would duplicate the shared matrix binding contract.

## Decision

Extend `PreparedContractions` with optional affine plans. All provider choice stays
in preparation; replay binds pointers and accounts original affine summands through
the existing API. Each shape publishes transactionally. An optional preparation
rejection has a distinct exception type, so consumers can re-admit a fallback
without catching scientific errors or failed execution. Live release is checked;
destructors remain best effort. Even the default CUDA stream must be drained.

Keep explicit per-plan ceilings for device workspace and provider storage and an
externally qualified host reservation. The owner must admit their sum for every
simultaneous plan, separately from descriptor and pointer tables. Zero host
reservation rejects; observed prepare-time device growth does not establish an
upper bound on lazy execution allocations.

CMake links an external cuTENSOR 2.8+ installation only on explicit opt-in. Ordinary
CUDA and CPU builds retain no dependency. Enabled native tests use identical
provider headers/macros as the library to avoid differing internal class layouts.
Wheel packaging is explicitly unsupported until its dependency contract exists.

## Rejected alternatives

- Method-local cuTENSOR dispatch would duplicate provider lifetime and counters.
- Automatically assigning the synthetic tests' resource limits to production would
  invent qualification for opaque host allocations and lazy provider growth.
- Silently rerunning on generated CUDA after an execution failure could duplicate
  partial work and hide invalid numerical output.

## Evidence and limits

The native affine test runs independent dyadic FP32/FP64 oracles through both
standalone and shared execution, including differing reduction-axis orders,
padding, transposed output and beta updates. It checks resource rejection,
three replays, shared semantic counters, stale context, stream/shape changes,
capture rejection and checked release. Provider-absent matrix tests exercise an
explicit cuTENSOR rejection before successfully binding the existing fallback.

Qualified on n1 through finite Slurm `main` / `gpu:5090:1` jobs with CUDA 12.9:
four host/device pytest tests passed, and the production CMake test target built
and passed with both compiler launchers set to verified ccache 4.5.1. Runtime
`cutensorGetVersion()` reported 20800. The local CPU-only configuration and three
host contract tests also passed. Ignored logs and before/after cache statistics
are retained in `.artifacts/1887-shared/`.

This slice enables shared preparation; production triples candidate admission,
opaque/lazy resource qualification and complete endpoint timing remain #1887 work.
No new precision or performance claim follows from build availability.

## References

- [Standalone provider rationale](2026-10-05-native-affine-cutensor.md)
- `src/tensor/cuda_contraction.cuh`
- `cmake/GenerativeQCCutensor.cmake`
