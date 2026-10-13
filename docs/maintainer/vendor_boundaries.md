# CUDA tensor and solver vendor boundaries

Run `python tools/check_vendor_boundaries.py` to check production native sources,
scientific compiler modules and `tools/generate*.py` entry points. The same check
runs in pre-commit. `--json` reports the current file/symbol reference inventory.

[`vendor_boundaries.json`](../../manifests/maintenance/vendor_boundaries.json)
classifies each file explicitly:

- `provider`: a retained primitive implementation with a semantic contract.
- `runtime_abi`: optional library ABI declarations and aliases.
- `infrastructure`: bounded non-scientific scheduling primitives.
- `diagnostic`: provenance or synthetic capability probes.
- `migration`: existing method-local coupling tracked by #1890.

The DF SCF adapter in `src/scf/cuda/df_scf_library.hpp` is retained as a reusable
matrix/eigensolver implementation. Its callers specify operands, strides,
alpha/beta and eigensystem work; its `DeviceSolver` owns numeric workspace and
retains the shared `solver::cuda::PreparedSymmetricEigenHandles` owner. That
move-only service uniquely owns solver/Params/Jacobi lifetime for GFN2, RHF/UHF,
ordinary KS and DF. Staged setup preserves the caller's stream order and Jacobi
settings. The enclosing device/stream owner settles work and explicitly resets
handles before releasing its stream or restoring the caller's device. The shared
service neither allocates numeric workspace nor synchronizes.

The streamed metric-panel transform (`df_generated_tiles.hpp`) and SCF
density-seed reconstruction (`df_scf_factor.cpp`) now submit their explicit
column-major FP64 views through `scf_matrix_product_view` in this existing
provider adapter. The scientific owners still specify transpose, M/N/K,
physical leading dimensions and alpha/beta, including the rank-truncated
Gram reconstruction and split-panel accumulation. The adapter borrows the
prepared BLAS context/stream, introduces no allocations or provider choice,
and maps library failure to the same `generativeqc_status` and context.
Both former direct-vendor source entries are retired from the manifest.
The same semantic adapter now owns the six ordinary GEMM submissions in
`df_plan_setup.cpp` (two full-rank rotations, packed and generated value
materialization, and the final truncated-rank tensor transform). Its batched inverse-root product now uses the same borrowed semantic provider
boundary; prepared-handle setup remains separately classified migration debt.
The existing DF SCF `scf_gemm_strided` path also delegates to the shared
batched-view implementation. The new contract accepts independent transpose
flags, bounded M/N/K/batch dimensions and physical strides in elements,
including zero input strides for broadcast, while preserving the original
cuBLAS submission and failure mapping. All eight direct GEMM submissions in
`df_exchange.cpp` now borrow that same adapter: seven strided-batched products
and one long-reduction product. The source-backed and streamed panel schedules,
FP64 precision, trace labels, alpha/beta and output reductions are unchanged.
The old `df_exchange.cpp` direct-vendor inventory entry is retired.
DF occupied-exchange follows the same path for streamed, resident and tiled
projections, two direct exchange matrix contractions and its lower-triangle
Gram. The shared provider retains the exact SYRK lower-triangle candidate
when supported, plus the pre-existing full GEMM fallback; the scientific
owner still performs its original mirror/reduction without changing visit order.
The prepared DF handle creation/stream/host-pointer setup and release now
reside in the same shared provider adapter. The plan continues to own stream
and resource lifetime, disposing dependent bindings before its handle and
stream; partial setup is drained by the same failure cleanup.
The DF response evaluator now borrows strict-FP64 column-major tensor views
from `tensor/cuda_runtime.cuh` for all ordinary/batched GEMM, GEMV and GEAM
submissions. These operations do not create another handle, select a provider,
change pointer modes, pack views or introduce stream synchronization. The
existing native-status exception boundary remains so the response bridge can
drain on error and preserve allocation versus execution failure. GEAM
availability is checked *inside* the shared provider layer using the existing
dependent capability rule; CuMetal retains the established generated gather
fallback. This is a mechanical retained-provider migration; canonical
precision/provider co-selection and method-level diagnostic algebra switches
are still separate #1889/#1890 work.

Retaining this adapter does not complete canonical binding integration
for its surrounding callers. Other SCF adapters with upstream implementation
selectors remain classified as migration work.

New files and symbols require an explicit classification and rationale. Existing
migration, diagnostic and infrastructure entries have exact reference counts;
adding another call or removing an old call requires a corresponding reviewed
inventory change. Remove retired entries when migrating a consumer. Do not
regenerate the manifest to accept a failing check without reviewing ownership.
Provider and ABI implementations may change the number of references to an
already classified symbol. No directory, including `src/tensor`, is exempt.

The scanner ignores native comments/string literals and Python docstrings. It
checks generator string fragments without executing Python or loading CUDA.
Function-address references, macro aliases and CUB/CUTLASS/CuTe namespace uses
are covered. Dynamically assembled API names, arbitrary external templates and
semantic changes at an existing callsite require review and runtime tests; this
lexical inventory does not prove the executed provider or replace scientific
acceptance. Implementation-selector API migration also remains separate work.
