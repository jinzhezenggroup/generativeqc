# Decision: qualify BLAS products in host physical validation

Status: proposed (implemented as a default-off experiment)
Date: 2026-10-24

## Problem

Native sampling of the water96 r2SCAN-3c Direct cold endpoint attributed about
1.9 seconds to `scf::reference::multiply` under eigenframe and initial-density
admission. The endpoint finished, but py-spy exited with a child-reaping error;
its 6,664 samples are diagnostic evidence, not a clean timing population.
Removing validation would violate the scientific contract.

The development GPU builds had no build-linked OpenBLAS. Installing the existing
pinned build input `scipy-openblas32==0.3.34.0.0` exposed another issue: its header
declares a prefixed local-thread extension, but the ELF library exports
`openblas_set_num_threads_local`. The previous CMake probe rejected local thread
ownership and would leave automatic products scalar even with this provider.

## Decision

`GENERATIVEQC_SCF_VALIDATION_GEMM=1` captures an optional product schedule per
validation. Only SC, C^T SC, FC, the overlap square root, and its density
congruence products use the existing FP64 tensor service. No eigensolver,
acceptance gate, norm, rejection order, input matrix or reference oracle changes.
The default and unavailable-provider cases retain the original scalar products.
The existing task-parallel single-thread provider scope owns thread changes.

CMake first probes the usual provider spelling, then independently compiles and
links the unprefixed local-thread extension for a prefixed provider. It records
the selected spelling separately and revalidates it when configuring. The tensor
adapter uses exactly that admitted ABI; it never substitutes a global setter for
missing local control. Scalar builds remain linkable without BLAS dependencies.

## Rejected alternatives

Two simple cache-tiled scalar candidates, including a small register tile,
retained each element's reduction order but regressed the isolated 768/1024
matrix products on the local host. They are not retained in production.
An independently compiled BLAS probe reduced 768-order products from about
0.115 to 0.019 seconds, with maximum scalar/BLAS difference below 7e-15. This
is a microbenchmark observation, not an endpoint or GPU4PySCF parity claim.

## Invariants and qualification

Analytic dense nonidentity-metric eigenframes and ensemble densities exercise
physical acceptance, nonorthogonality, eigen residuals, occupation bounds,
trace, symmetry, singular metrics, malformed inputs and overflow. Provider
injection checks default/off/invalid selector fallback, missing local control,
restoration of caller thread state and rejection of corrupted products. A real
pinned BLAS build exercises the tensor and both validation owners.

Build-linked provider admission also affects other existing automatic CPU tensor
consumers. A GPU build with this dependency must therefore be independently
qualified as a whole. Compare selector 0/1 in the same binary, record build and
dependency identities, and keep previous-build timings separate. Promotion still
requires complete cold energy/analytic-force endpoints, work counts and repeated
statistics on all relevant methods and geometries.

## Independent stock-default qualification

Frozen source `f088d3fcd30435c125a5b92422e4d91088884977` isolates this change
on master `e85e5251e571f5b4a0008a19d83230d3ca8594e2`. It also fixes an existing
resident KS test to query its diagnostic capability; no production behavior is
added by that test repair. Main library SHA256 is
`639ec53fd7786b909b53be2539724b3c1dc9ae6a85eb94578e4a29ec39bb1288`.

All 328 calls pass independent energy and full-force gates: 240 paired calls,
56 seven-method Standard/Fast Direct/DF matrix calls, and 32 PBE/3c moved and
replay calls. Native ownership, resource, variational and graph checks pass;
prepared/unprepared independent force tests and all prescribed sanitizer runs
pass. Original build/harness failures remain distinct retained receipts.

Five alternating stock-default pairs give cold gains of 2.122%/2.142% for PBE
DF/Direct and 2.502%/2.777% for r2SCAN-3c DF/Direct. PBE0 improves
1.421%/1.958%, below the predeclared 2%/MAD gate. No warm comparison passes.
These are same-source/same-binary selector comparisons, independent of earlier
combined AO/spatial/DF-partition experiments. The schedule stays default-off,
and the wider Direct/DF cold goal against GPU4PySCF remains open.

Complete byte-preserved receipts, source identities, reproduction commands and
scope are in
[`scf-validation-gemm-20261024`](../../../benchmarks/results/scf-validation-gemm-20261024/README.md).
