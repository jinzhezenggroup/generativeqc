# Decision: bind reduction implementations below semantic schedule search

Status: implemented
Date: 2026-10-11
Base: master `637466c624c4c4c4dd4eb7c511a2c88675744325`

## Problem

The provider-selection inventory explicitly classified Tensor CUDA's
`reduction_provider` schedule/search axis as migration debt. Although generated
CUDA and CUB already advertised the same canonical request, code emission read
the implementation name from the scientific scheduling surface rather than
consuming the shared precision/layout/provider binding. Keeping this axis would
encourage equivalent selectors in other method owners.

## Decision

Remove the named axis from `TensorSchedule`, `TensorScheduleSpace` and schedule
search keys. The schedule still owns streaming, fusion and reduction unrolling.
Unbound plans retain the previously qualified generated reduction, with no
toolkit probing, device work or provider search in scientific planning.

`reduction_provider_candidates` projects one plan-bound canonical request into
both implementations. It attaches the actual cooperative execution topology so
cost/binding evidence cannot be reused after changing threads or unrolling.
The common `select_lowering_binding` owns precision, layout, capability,
version, complete-cost and strict-fallback admission. `bind_reduction_lowering`
installs this recipe in the immutable CUDA executable plan; it validates the
canonical offers and reconstructs the shared selection. Emission and static
shared-memory accounting read the selected recipe, not a method selector.
Diagnostics include the retained candidates, fallbacks and rejection evidence.
`tensor_compiler_identity` reuses the existing artifact owner's actual source,
NVCC/PTXAS, CUB header, host compiler, options and environment facts. A bound
recipe carries that hash, and `compile_cuda` rechecks it before allocating a
cache directory or calling NVCC. Supplying the binding's own compilation record
back to its validator would not establish current context; this extra check
prevents that tautological validation.

The typed recipe is not a device allocation or a new lifetime/cache owner.
Existing whole-plan compilation and prepared runtime execution still own those
resources. There is no production automatic CUB promotion in this change.

The source guard removes all four retired Tensor selector exceptions and rejects
provider-valued schedule fields even if their names change. The obsolete
default-promotion control is removed because that API no longer exists; the
historical negative CUB qualification remains linked from current-state docs.

## Identity and compatibility

Plan schema advances from 5 to 6. Prepared recipe identities are executable
inputs and omit costs; updating cost evidence without changing a selected
implementation or fallback does not invalidate identical code. Equations,
admitted precision and semantic scheduling remain independent of provider
identity. Old named-axis constructors fail explicitly rather than silently
ignoring the requested implementation. Repository consumers now use the
explicit qualification helper or shared binding.

The full emitted translation unit intentionally changes its embedded
`tensor_plan_identity` after the schema change. Six master-baseline fixtures
(FP32/FP64 elementwise, scalar reduction and cooperative reduction) retain
identical equation hashes, precision identities and source bytes after replacing
only that embedded plan hash. This is an executable rekey, not an arithmetic
rewrite or permission to reuse old compiled artifacts.

## Rejected alternatives

- Renaming the selector or moving it into another schedule dictionary preserves
  the ownership defect and allows schedule search to bypass shared admission.
- Accepting an arbitrary ready candidate by provider name allows forged
  algorithm/resource contracts that the emitter does not implement.
- Selecting from kernel time alone ignores cold preparation and complete
  replay work. The selector retains its full cost and strict-fallback rules.
- Promoting CUB on representability or lower register counts contradicts the
  retained complete-endpoint negative evidence from #971.

## Invariants and evidence

- Unbound generated lowering remains complete; missing CUB header evidence
  selects generated execution and retains the rejection.
- Alternative bindings require complete costs and provider identities. Only
  the existing generated implementation may be retained as an unmeasured
  qualified incumbent.
- Stale shapes/layouts, target, precision or cooperative topology fail before
  emission; canonical offers reject forged algorithms, determinism or storage.
- The cached binding index is read-only, and recipe lookup is not rebuilt for
  every emitted step. Validation reuses one semantic adapter for all bound sites.
- `tests/python/test_tensor_cuda_reduction_binding.py` independently checks
  FP32, FP64 and admitted FP32-compute/FP64-accumulation, cold/warm cost ranking,
  resource/legality evidence, stale recipes and executable identity. Its opt-in
  real-device tests compare generated/CUB ordinary execution and FP64 graph
  replay with NumPy. FP32/mixed graph requests retain the existing explicit
  precision-gate fallback; this change does not broaden graph admission.
- `tools/tensor_reduction_qualification.py` uses explicitly synthetic ranking
  costs only for fixtures. Device-free emission tests use synthetic versions;
  compiled fixtures obtain actual source/toolchain context identities. Ranking
  records are not timing/performance evidence or production admission.

Validation commands:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_tensor_cuda_reduction_binding.py \
  tests/python/test_joint_lowering.py tests/python/test_lowering_provider.py \
  tests/python/test_tensor_cuda_search.py tests/python/test_tensor_cuda_plan.py \
  tests/python/test_tensor_cuda_fp32.py \
  tests/python/test_provider_selection_boundaries.py \
  tests/python/test_default_promotion_inventory.py
PYTHONPATH=python:. python tools/check_compiler_structure.py
PYTHONPATH=python:. python tools/check_provider_selection_boundaries.py
PYTHONPATH=python:. python tools/check_vendor_boundaries.py
PYTHONPATH=python:. python tools/check_default_promotion_inventory.py
```

The broader source-matched local validation passed 411 tests and 54 subtests;
31 allocated-device cases were explicitly skipped locally. All compiler,
provider-selection, vendor-call and default-promotion guards passed, together
with Ruff lint/format and the public Python contract checker.

Real-device validation on `n1`/`node1` passed all 67 tests in the reduction-binding
and existing FP32 suites through Slurm job `7371`, `main`, `gpu:5090:1`, with a
finite ten-minute allocation and the assigned device visibility preserved.
The device-free ordinary/resident/reassociated/streamed/generated/CUB compilation
qualification also passed using CUDA 12.9 (`nvcc` 12.9.86). The initial remote
failure was a missing CUDA library search path, corrected by retaining the
toolkit's `lib64` in `LD_LIBRARY_PATH`, not by changing source or device visibility.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -c '
    export PYTHONPATH=python:. GENERATIVEQC_TENSOR_CUDA_TEST=1
    export GENERATIVEQC_TENSOR_ARCH=sm_120 CCACHE_BASEDIR="$PWD"
    python -m pytest -q --maxfail=1 --tb=short \
      --basetemp=.artifacts/provider-binding-20261011/pytest-final \
      tests/python/test_tensor_cuda_reduction_binding.py \
      tests/python/test_tensor_cuda_fp32.py
  '
```

Ignored receipts are retained under `.artifacts/provider-binding-20261011/`,
including changed-source provenance, the master fixtures, local/device logs,
compilation resources and ccache before/after snapshots. The verified ccache
4.5.1 launcher reused the existing cache. The shared NVCC compile/link calls
were reported as uncacheable, with no additional hits in this qualification;
launcher use is not presented as cache-hit evidence or a compilation speedup.

## Consequences and revisit conditions

Review found two executable consumers outside the initial focused suite still
constructing the removed schedule axis: the retained CUB endpoint qualification
runner and the ordinary real-device integration test. They now use the shared
binding fixture. The tuner applies a pure, bindings-only preparation hook after
semantic planning and before deduplication/static screening; concrete prepared
plan identities enter its evidence key. This preserves the complete existing
numerical/resource/ABBA qualification path without bringing provider names back
into schedules or duplicating a benchmark execution loop. CPU regressions cover
the runnable entry point, unchanged semantic plans, distinct recipe evidence,
resource pruning and endpoint gates. Real-device reruns remain separate evidence.
The executable plan also copies each incoming binding pair, not only the outer
sequence. A caller-owned mutable pair otherwise could alter the stored recipe
after validation while the cached emitter lookup still referenced the old one.

Review also reproduced a stale dependency identity: replacing the transitive
`cub/block/specializations/block_reduce_warp_reductions.cuh` left the former
single-header hash unchanged. The compiler context now uses NVCC's `-M` phase
with `-MF`/`-MT` to resolve runtime and runtime-plus-CUB include closures before
compilation. These are preprocessing-only probes, not object compilation or
device work. Header logical names and bytes are retained together; tool versions
are cached only under executable-byte and environment identities. Header
resolution refreshes on every context check so a newly shadowing include cannot
reuse an old manifest. Missing or unidentifiable runtime dependencies fail
closed. Optional CUB failures have a stable unavailable identity, allowing
generated execution while rejecting a selected CUB recipe. The probe has finite
process-tree timeouts and dependency count/byte bounds. Existing toolchain and
host-version evidence fields remain compatible. This inventories selected
compiler executables and preprocessing dependencies, not arbitrary wrapper
internals or a fully hermetic operating-system/toolkit image.

The 36 CPU dependency regressions cover transitive replacements, include
shadowing, filename/content swaps, same-banner executable replacement, portable
toolkit relocation, probe caching, missing CUB, malformed/missing/vanished
manifests, unsafe phase overrides and finite bounds. Independent regeneration of
six FP32/FP64 elementwise/scalar/cooperative fixtures also preserves equation and
precision hashes and emitted source after normalizing only the embedded plan
identity against the qualified base. No new real-NVCC or GPU result is claimed
for these review repairs. NVCC phase semantics follow the
[official compiler driver reference](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-compiler-driver-nvcc/index.html).

This retires the Tensor reduction-selector family under #1886/#1889/#1890.
It does not close those trackers: method-local CC/SCF selector debt and broader
provider/method endpoint qualification remain. No speedup or mixed-precision
scientific-domain expansion is claimed. Revisit default provider choice only
with source-matched numerical, resource and complete-endpoint evidence, while
preserving this provider-neutral boundary and explicit strict fallback.

## References

- #1886, #1889, #1890, #1991, #971.
- `docs/developer/tensor_cuda.md`.
- `docs/developer/provider_selection_boundaries.md`.
- `.agents/notes/implemented/performance/2026-09-22-cub-block-reduce-qualification.md`.
