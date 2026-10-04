# Decision: qualify CUTLASS through the actual streamed DF response region

Status: implemented
Date: 2026-10-05

## Problem

Standalone kernels and shared tables do not establish that a scientific consumer
can prepare all simultaneous plans, preserve its source traversal and publish
the complete response. Partial preparation also leaves CUDA-owned modules alive
when the region falls back to generated execution.

## Decision

Offer CUTLASS in the homogeneous region portfolio and use it in the existing
eight-contraction streamed DF source response. Scientific code still consumes
the shared region interface; provider admission remains in the tensor owner.
Require a matching compiled version, actual artifact digest and explicit
host/module bounds. The native region copies the digest and charges all plans.

The test adapter hashes its actual linked production source-response owner,
external headers, toolchain, commands and generated source. Qualification-only
resource/ranking controls exercise CUTLASS and partial-preparation fallback.
No production resource profile or default promotion is installed.

On fallback, subtract the table's retained module charge before re-admission.
Retain the original preparation ceiling in complete binding diagnostics whenever
partial loading occurred: final live storage alone is not an endpoint peak bound.
Provider-retention diagnostics include cache bytes even under generated fallback.

## Invariants

The original two source passes, mathematical derivatives and semantic summand
counts remain unchanged. Missing providers retain the incumbent; one-byte-below
optional admission selects the complete generated schedule before loading.
Hard loader or arithmetic failures must never become an apparently free fallback.

## Evidence

`test_df_cc_source_program.py` exercises the actual native consumer against the
independent complete response expression, derivative/energy gates, non-symmetric
inputs, all optional providers, exact budgets, partial rejection and sticky
failures. Complete response timings include preparation, uploads, callbacks,
publication and teardown; this boundary is not a molecular force timing claim.

### Retained endpoint result (n1, RTX 5090, CUDA 12.9)

Repeated complete probe calls recreate the region and its plans. Medians below
use five calls after each provider's first call in an ordered shared-process
sweep. They include upload, preparation, source callbacks, execution, publication
and teardown; they are neither isolated replay nor cold-process measurements.

| Provider | n=4, q=6 (ms) | n=16, q=24 (ms) |
| --- | ---: | ---: |
| cublas | 0.7934 | 1.4814 |
| generated.cuda | 0.4130 | 0.9314 |
| cutensor | 41.7902 | 42.7703 |
| cublaslt | 0.7904 | 1.4347 |
| cutlass-aot | 0.7051 | 1.7206 |

All providers performed 17 contractions / 3,456 summands and 53 contractions /
884,736 summands respectively, with identical source passes and independent
response gates. CUTLASS did not beat generated execution for these small complete
responses; cuTENSOR preparation also dominated. Preserve this negative result
and retain the incumbent until a larger workload and qualified resource profile
support a complete-endpoint choice. These tests use synthetic reservation bounds.

The first cuBLAS invocation also initialized the process's CUDA state, so first
provider-call times from this sweep must not be compared as equal cold starts.
Artifact: `1b03e5fbe14ffce1b6542998b5c1481170de9e5f1e43c8fd0a58eefc985195a5`.
Probe library SHA-256: `97b162f18b6ac4959c68cd40220c8d076e46b9b8dc5c330f98954cca4e069e8f`.
Raw samples, build identity and comparison script are retained locally under
`.artifacts/1888-cutlass-region/`; native qualification used finite Slurm jobs.

## References

- #1886, #1888, #1944
- [Shared table ownership](2026-10-05-cutlass-shared-table.md)
