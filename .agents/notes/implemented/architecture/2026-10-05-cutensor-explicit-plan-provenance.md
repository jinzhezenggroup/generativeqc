# Decision: explicit cuTENSOR family and kernel rank

Status: implemented
Date: 2026-10-05

## Problem

cuTENSOR 2.8 exposes only required workspace through `cutensorPlanGetAttribute`.
The old `CUTENSOR_ALGO_DEFAULT` allowed the performance model to choose an
uninspectable algorithm/kernel. Reporting that default preference as an actual
algorithm would give misleading execution provenance.

## Decision

The optional native candidate fixes `CUTENSOR_ALGO_GETT` and kernel rank zero,
then reads both preference attributes back after plan creation. NVIDIA documents
an explicit algorithm as disabling algorithm selection and returning unsupported
when it cannot execute. Existing bounded preparation fallback handles that case.
JIT, plan caching and incremental autotuning stay disabled.

The prepared owner exposes resolved semantic/precision descriptors, actual
extents/strides and alpha/beta together with provider/runtime version, target
architecture, queried workspace, algorithm and kernel rank. The shared table
visits these records by compiler shape variant and slot. Inspection performs no
plan search; released/stale bindings refuse live provenance.

## Limits and rejected alternatives

The rank is relative to that algorithm, version, target and request; it is not
a global binary kernel identifier or sufficient cache key by itself. Scraping
debug logs or using opaque plan addresses would not provide a stable contract.
There is still no production cuTENSOR resource profile. Fixing this candidate
does not promote GETT over other providers or assert a speedup. More algorithms
may become separately qualified candidates with complete costs in the future.

## Evidence

`test_native_cutensor_binding.py` checks independent affine FP32/FP64 arithmetic,
transposed output, opposing reduction-axis order, nonfinite padding, repeated
replay, stable provenance, resource ceilings, stale context and release.
`test_df_cc_source_program.py` checks the streamed response through the fixed
candidate, including partial-plan rejection and same-precision fallback.
The existing triples qualification also passes: 34 tests with the fixed cuTENSOR
candidate, and 29 provider-absent tests with five optional skips. Strict/mixed
pinned H2O/NH3/CH4 energy gates and exact contraction work remain unchanged.

References: #1887; `src/tensor/cuda_cutensor.cuh`; NVIDIA cuTENSOR 2.8 headers
for `cutensorCreatePlanPreference` and `cutensorPlanPreferenceAttribute_t`.

This supersedes default algorithm selection in the
[original affine binding](2026-10-05-native-affine-cutensor.md) and addresses the
algorithm-provenance follow-up in the
[source-response decision](2026-10-05-source-response-prepared-execution.md).
