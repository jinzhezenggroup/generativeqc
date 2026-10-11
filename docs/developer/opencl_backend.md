# Experimental OpenCL compiler/runtime contracts

The optional OpenCL adapter executes generated scalar expressions and FP64
reductions on a real GPU. It does not provide an OpenCL HF/DFT calculation.
CUDA remains the production accelerator backend and the default selection.
No OpenCL SDK or Python extension is required by ordinary builds or imports;
the adapter loads the installed OpenCL ICD with `ctypes` only when constructed.

## Target audit and evidence

The archived OpenCL proof used **NVIDIA OpenCL on an RTX 5090**: this is
another API on NVIDIA hardware, not evidence of cross-vendor portability.
Non-NVIDIA and domestic GPU validation is **not-run**. Device limits and
optional FP64, subgroup, compiler and library capabilities must be queried
per device; historical values are not universal defaults.

The source-matched [execution report](../../benchmarks/results/opencl-rtx5090/report.json)
preserves the exact driver/device/UUID, OpenCL C compiler, generated kernels,
compile/link logs, queried limits, profiling samples and cache roundtrips.
The [validation summary](../../benchmarks/results/opencl-rtx5090/validation-summary.json)
retains reference errors and test results. The
[vendor-source provenance](../../benchmarks/results/opencl-rtx5090/vendor-docs.json)
contains the audited documentation identities and dates. These archives are
historical qualification evidence, not current HF/DFT endpoint acceptance.

## Scientific and execution contracts

`ScalarCEmitter` holds the existing C-family scalar arithmetic emitter.
`CudaEmitter` preserves its CUDA import/API, while `opencl_lowering.py` emits
structured OpenCL address spaces, arguments, work-item indexing and FP64 policy.
There is no CUDA source-text replacement. `ScalarKernel` names an ordered
primitive input ABI and existing scientific graph roots; emission preserves
the scientific hash while source and schedule hashes vary with the target.

`backend.TargetInfo` represents unknown subgroup and resident-workgroup limits
with `None`. CUDA continues to supply its known limits. `RuntimeCapabilities`
and `ExecutionShape` validate workgroup size, local workspace and explicit
optional requirements before execution. Synthetic widths 8/16/32/64 and unknown
widths are tested without importing an SDK. A missing subgroup width permits
scalar workgroups and rejects subgroup-dependent schedules.

`OpenCLRuntime` implements separate `clCompileProgram`/`clLinkProgram` calls,
GPU queries, typed context-owned buffers/programs/queues/events, bounded blocking
transfers, padded launches, event profiling and multi-level FP64 reductions.
Reduction scratch and intermediate results stay on device without floating-point
atomics. An event retains submitted objects until completion; forged, stale and
cross-context handles are rejected. Failed compilation preserves diagnostics and
releases the program. Cleanup attempts every native release after vendor errors.
An indeterminate reduction wait failure drains and invalidates the runtime;
a terminal execution failure releases its intermediate resources and raises.

The library boundary uses `LibraryRequest` for explicit FP64 dimensions, row/
column layout, workspace limit and normalized residual tolerance. A future
provider must report workspace and native error status, and expose numerical
residuals for eigensolves/factorizations before its result can be accepted.
The current `UnsupportedLibraryProvider` implements no operation and raises for
GEMM, symmetric eigensolve and Cholesky. It never delegates to CPU arithmetic.

`LocalArtifactCache` is explicitly selected and local. Each bounded atomic
record includes backend, compiler, language/options, driver/runtime, device,
scientific/source/schedule identities, ABI and binary checksum. Loading checks
file ownership, shared-write permissions, regular-file type, size and identity;
symlink records are refused. The cache reuses the existing atomic publication
helper and does not change CUDA official/user profile precedence or automatically
download/activate executable artifacts from a remote profile.

## Reproduction and numerical boundary

CPU lowering and contract checks do not need OpenCL:

```bash
PYTHONPATH=python:. python -m pytest \
  tests/python/test_runtime_backend.py tests/python/test_opencl_lowering.py -q
```

With an explicitly allocated OpenCL GPU and verified device identity, run:

```bash
PYTHONPATH=python:. GENERATIVEQC_OPENCL_TEST=1 \
  python -m pytest tests/python/test_opencl_runtime_gpu.py -q
PYTHONPATH=python:. python tools/validate_opencl_backend.py \
  --directory /tmp/opencl-gate
```

An OpenCL ICD may not respect CUDA device visibility. Multi-GPU runs require
a scheduler-assigned UUID-to-OpenCL mapping, not a guessed ordinal. The
source-matched archive gives the exact earlier Slurm procedure, compiler
versions, cases, numerical tolerances, errors, resource measurements and
separately reported failures/skips. Its reference smoke uses host-generated
Boys inputs and host contraction of primitive results: the tested scalar
expression and reduction kernels are **not** a native integral provider or a
complete HF/DFT energy/force endpoint. Dated timings are not current default
performance claims.

## Next native HF integration

1. Select available target hardware and pin its ICD/toolchain/library versions.
   Repeat the capability and Slurm UUID audit, retaining missing hardware as
   not-run and absent provider operations as unsupported.
2. Supply a native linear-algebra adapter through the existing context/provider
   boundary, beginning with FP64 GEMM, then symmetric eigensolve and Cholesky.
   Validate layout, bounded workspaces, native failures and independent residuals.
3. Extend the shared integral graph consumer with on-device Boys evaluation,
   normalized contraction, spherical expansion and matrix write ownership.
   Pass independent raw-block gates before wiring an HF calculation.
4. Add explicit native prepared-plan selection with context-owned topology and
   geometry invalidation. Start with bounded one-system RHF and ordinary queue
   submission; keep graphs optional and expose unavailable methods clearly.
5. Run cold, unchanged and moved-geometry RHF/UHF/batch endpoints with density,
   energy and force gates, resource limits and full timing evidence. Promote
   only validated operator/method/target combinations; this issue promotes none.
