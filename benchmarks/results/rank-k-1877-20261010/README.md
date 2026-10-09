# #1877 rank-k prepared CUDA capability qualification

This records a **standalone capability qualification** of the rank-k prepared
binding. It does not measure a complete SCF or SCC endpoint, does not promote
cuBLAS as a production default, and does not complete the two-production-
consumer adoption criterion in #1877.

## Final source and device

- Source commit: `41f920b512e25dc6da8c08130de9bbda15e71cbe` on
  `codex/issue-1877-rank-k-qualification` (base `880b46ef31d85541e87f99a53015d35b7de7cd98`).
- qz Job: `i1877-rankk-h100-1010h`, `SUCCEEDED` with process exit 0 on
  2026-10-10 03:37:14 Asia/Shanghai. Requested priority 4; the platform
  reported priority 20 / `NORMAL`. Quota: one H100, 20 CPU, 200 GiB, one
  point per hour. The Job finished and released its pod.
- Device: NVIDIA H100 80GB HBM3, compute capability 9.0, driver 570.124.06.
  Compiler: CUDA 12.9.86. Linked cuBLAS, cuBLASLt and shared cudart resolve
  from the task-owned CUDA 12.9 toolkit copy. The runtime reported 12090 and
  cuBLAS 120902. The source toolkit was copied read-only from the existing
  qz `issue-1882-stage-20261008/cuda-12.9` asset, then version and key bytes
  were checked in [provenance.txt](provenance.txt).
- qz source/results root:
  `/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1877-rankk-1010`.
  The clean CPU checkout had a 1,367-file SHA-256 manifest. Every file and
  the generated header passed the GPU-side integrity check. The manifest
  SHA-256 was `7ffba5b2314afe369ff071edfb039e8bcf1c2805d91c8a7d68dda83d6a88b4e7`.
  The [source check](source-check.txt) and [generated check](generated-check.txt)
  are retained verbatim; SHA-256 checks integrity, not source authentication.
- The generated header, object and binary hashes are retained in
  [source-artifact.sha256](source-artifact.sha256). The raw
  [qualification JSONL](qualification.jsonl) has SHA-256
  `50b2706e7f2bf6cc5a93bf929d19a9320d25f737d85aad04840a8651f9079c63`.
  The downloaded copies of those three artifacts matched the remote hashes.

## Acceptance and measured scope

The harness passed 16/16 distinct combinations: density and energy-weighted
density, row- and column-major coefficients/output, `3×5` and `17×9`
rectangular panels, two batches, and generated versus cuBLAS execution. Each
case checked a CPU `long double` reference, signed and negative-zero weights,
nontrivial `alpha=1.25` and `beta=-0.5`, asymmetric old lower-triangle data,
two captured graph replays, all-output preservation for nonfinite weights and
coefficients, and alias/order rejection. The shared harness also exercised a
dimension-overflow rejection, nonfinite host alpha rejection and execution of
the generated fallback after a zero-budget library admission failure.

The measured **device-resident prepared endpoint** includes output reset,
error reset, energy-weight materialization where applicable, signed scaling,
GEMM or both generated reduction passes, validation, mirroring and launches.
It excludes host/device input transfers and provider preparation; `prepare_us`
is reported separately. Each line times 20 repetitions after four warmups.

| Candidate | Endpoint across these 16 cases | Retained extra device resources |
| --- | --- | --- |
| generated CUDA | 12.79–21.50 µs | No provider handle or scratch |
| cuBLAS signed GEMM | 25.56–50.98 µs | 96 MiB allowance; 64–66 MiB observed retained; 384 or 7,072 B scratch |

cuBLAS was slower in every measured pair. These small shapes provide negative
promotion evidence, not a speedup claim for larger panels or an SCF endpoint.
For `17×9`, two batches require 2,754 logical upper-triangle products; the
generated validation/publication runs 5,508 products and scaling steps. The
library candidate performs a full 5,202-product GEMM plus 306 coefficient
scales and upper-triangle validation/mirroring. Exact per-case counts and
times are in [qualification.jsonl](qualification.jsonl).

The actual object compile invoked `sccache 0.16.0` around CUDA 12.9.86 `nvcc`.
[Before](sccache-before.txt) to [after](sccache-after.txt), it recorded one
compile request, one CUBIN hit, three CUDA/PTX-related misses, four actual
compilations, zero cache errors and zero unsupported calls. The object link
was separate; [linked library paths](linked-cuda-libraries.txt) record runtime
resolution. The short [link output](link-output.txt) is retained.

## Retained negative and preliminary runs

| Job | Result | Boundary reached |
| --- | --- | --- |
| `i1877-rankk-h100-1010a` | FAILED | Minimal CUDA image lacked `git`; no compile. |
| `i1877-rankk-h100-1010b` | FAILED | Minimal CUDA image lacked `python3`; no compile. |
| `i1877-rankk-h100-1010c` | FAILED | Compile found an undefined `CUDART_NAN`; [raw compile failure](failed-c-compile.txt). |
| `i1877-rankk-h100-1010d` | SUCCEEDED | [16-case CUDA 12.8 probe](probe-cuda12.8.jsonl); below the repository's CUDA 12.9 floor, so not the supported qualification. |
| `i1877-rankk-h100-1010e` | FAILED | CUDA 12.9 compiled and linked, but the newly added overflow test caught `std::overflow_error` while the native checked product correctly threw `std::length_error`; [partial JSONL](failed-e-qualification.jsonl). |
| `i1877-rankk-h100-1010f` | SUCCEEDED | CUDA 12.9 passed 16 cases before the final strict weight-stage precision and negative-zero test additions. |
| `i1877-rankk-h100-1010g` | SUCCEEDED | [16-case supported CUDA 12.9 probe](probe-cuda12.9-prelint.jsonl) before the final Python lint/format source change. |
| `i1877-rankk-h100-1010h` | SUCCEEDED | Final lint-clean source and supported CUDA 12.9 qualification above. |

Jobs and their full platform logs remain on qz. None of the intermediate
statuses is substituted for the final source-matched result. The GPU Jobs are
terminal; the task's CPU staging Notebook had a four-hour auto-stop timer from
2026-10-10 02:58:12 Asia/Shanghai, so its requested deadline was 06:58:12.
