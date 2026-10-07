# Decision: retain source-major scalar reduction when CUB is unavailable

Status: implemented
Date: 2026-10-07

## Problem

PR #2067 centralized native RCCSD iteration block reduction in a strict-FP64
CUB provider. Its unconditional CUB include broke the supported CuMetal build:
job 112725571671 could not find `cub/block/block_reduce.cuh`. The independent
GCC job passed all 76 native CTests, then failed the ownership inventory because
the new provider header had no semantic shard.

## Decision and invariants

The shared tensor provider owns a compile-time capability selected by the
existing `GENERATIVEQC_CUDA_PROVIDER_CUMETAL` build definition. NVIDIA retains
the exact CUB type, algorithm, and strict rounded-add functor. CuMetal declines
block reduction without including CUB. The generator consumes the capability
and reuses its existing source-major serial body for all domain sizes, retaining
the same multiply/add order, coefficient placement, finite audit, and owner lane.

No new CC equations, host contraction, atomics, synchronization, workspace, or
parallel tree is introduced. The NVIDIA small-domain branch and independent
physical replay remain unchanged. Register the provider as shared runtime in
the CUDA ownership shards; do not suppress the inventory gate.

## Rejected alternatives

- Installing NVIDIA CUB into the CuMetal build would import an unsupported vendor
  dependency rather than preserve the backend boundary.
- A handwritten CuMetal block tree would introduce another reduction order and
  device qualification obligation into a portability-only repair.
- A dummy block sum returning one thread's partial would silently corrupt results
  if the caller forgot a capability check. The unavailable provider exposes no
  such implementation, so an unguarded use fails compilation.

## Validation and limits

Host tests compile the actual shared header with a poisoned CUB include for
CuMetal and execute every generated iteration scalar against legacy serial
lowering, including zero/warp/block boundary domains, non-owner lanes, NaN, and
infinity. Full-orbital reduce/einsum tests cover the runtime `o+v` extent. An API
double checks NVIDIA CUB type/algorithm selection and strict-add delegation; it
is not a CUB implementation or numerical qualification.

Real-device energy/residual/convergence and independent replay gates remain
required. The fallback restores compatibility and makes no CuMetal speedup,
CPU optimization, or complete-endpoint performance claim. Revisit only when a
supported CuMetal reduction provider has independent scientific qualification.

## References

- [PR #2067](https://github.com/jinzhezenggroup/generativeqc/pull/2067)
- [Observed blockers](https://github.com/jinzhezenggroup/generativeqc/pull/2067#pullrequestreview-5440596253)
- [Current implementation contract](../../../../docs/developer/rccsd_gpu.md#native-iteration-scalar-reductions)
