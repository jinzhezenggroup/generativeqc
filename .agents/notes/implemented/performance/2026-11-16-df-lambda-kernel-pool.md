# Decision: intern identical local DF Lambda CUDA workers

Status: implemented; complete wheel and method endpoint gates remain separate
Date: 2026-11-16

## Problem and decision

The generated DF Lambda owner repeats identical device arithmetic across
retained responses, staged scalar/matrix actions, parameter VJPs, audits and
the optional FP32 auxiliary proposal. Reuse the existing exact emitted-kernel
pool with the translation-unit-local owner `df_lambda`. It abstracts only
the entry symbol and diagnostic node identifier; each original caller still
launches with its own operands, dimensions, configuration and node number.

Keep `staged_core_matrix` outside this pool. Its existing prepare/dynamic
reuse phases deliberately borrow its original kernel symbols with
`emit_kernels=False`. Neither phase may enter the local pool or lose its
original owner. Preserve all 286 core-owned definitions, retained frontier,
arena lifetimes, provider admission/refusal, precision boundaries and ordered
Q accumulation. This change does not remove actions, methods or architectures.

## Source and complete-object evidence

A frozen one-off migration proof compares every complete emitted signature
and body after abstracting only the symbol and diagnostic identifier. It also
compares all launches, their original diagnostic IDs and the entire remaining
host orchestration. These migration checks are not permanent source/body-hash
or private-call-count requirements.

| Complete generated owner | Before | After |
| --- | ---: | ---: |
| Device kernel definitions | 3,630 | 1,090 |
| CUDA source bytes | 3,728,519 | 2,660,506 |
| Original launch sites | 3,916 | 3,916 |
| Complete object bytes | 23,076,216 | 21,354,528 |
| Reaped compiler-worker maximum RSS, KiB | 1,791,132 | 1,283,252 |
| Embedded compute-80 PTX bytes | 14,773,862 | 6,326,175 |
| Extracted sm80 CUBIN bytes | 24,422,696 | 9,924,904 |
| Extracted sm90 CUBIN bytes | 25,075,608 | 10,294,000 |
| Extracted sm120 CUBIN bytes | 34,977,728 | 14,367,792 |

NVCC 12.9.86 and verified sccache 0.16.0 compile both complete owners using
the configured Release commands, retaining sm80, sm90, sm120 and compute-80
PTX. The single observed durations are 285.33/202.71 seconds, but baseline
has two stage hits/six misses and candidate zero hits/eight misses. Do not
call this an established cold-build speedup. The maximum RSS observation is
for the owned, reaped compiler process tree, not aggregate concurrent CI memory.
Do not confuse extracted raw image bytes with compressed object/wheel bytes.

Map all 10,890 original architecture/entry pairs to their actual candidate
entry before comparing `cuobjdump --dump-resource-usage` output. Stack, local
and shared memory are unchanged for every pair. Registers can change because
the node ID is now an argument: increases are at most one register on sm80,
four on sm90 and two on sm120. This is not a claim that register allocation,
occupancy or endpoint speed is universally identical.

## Executable numerical, diagnostic and ownership qualification

Existing scalar-kernel pooling, DF Lambda/core-reuse/frontier/matrix/mixed
precision and independent derivative suites pass **44 tests** with **31
unchanged explicit-GPU qualification skips**. No scientific gate is removed.

An additional one-off probe links each complete, three-architecture owner
against the same native test adapter and CUDA/cuBLAS dependencies. Before
and after run in separate processes, avoiding loader caches and weak-symbol
interposition. Finite Slurm job **17492** on authorized `node1`, preserving
its assigned GPU visibility, executes **572 cases per variant** across 45
actual generated action interfaces:

- Retained RHS/transposes and all parameter VJPs, including independent actions.
- Staged scalar and prepared-matrix actions, audits and FP32 auxiliary actions.
- Original retained-core prepare/dynamic phases borrowing the unpooled owner.
- Occupied/virtual shapes `(1,3)`, `(2,3)`, `(3,2)` and full/tail Q extents 3/1.
- Signed finite inputs, zero seeds, injected NaN and infinity, and arena guards.

Every finite result matches the separate NumPy TensorIR interpreter; maximum
absolute error is `5.56e-17` for FP64 and `9.32e-10` overall. **Every before/after
output is bitwise identical**, including nonfinite/error cases. Sticky original
node diagnostics and provider call/summand work counts also match exactly.
The generated source used for this proof equals the production emitter output.

These are complete generated actions, not a complete DF CCSD/Lambda solve,
Direct/HF endpoint or manylinux build/link/repair/installed-wheel qualification.
No GPU-method or whole-wheel speedup is claimed. Keep those endpoint gates.

Compiler structure and CI ownership suites pass **1,014 tests**, and the
604-module structure checker has zero dependency errors. The official Libxc
checker requires a regenerated retained snapshot because this newly modified
generator joins the historical changed-input lists in 28 revalidation rows.
The acceptance-view diff adds only `tools/generate_df_lambda.py` to those lists;
scientific summaries, receipts, endpoints, transition acknowledgements and
stale/current qualification claims are unchanged. Keep the required snapshot
update instead of ignoring or bypassing this integrity gate.
All **202** focused catalog/dependency/snapshot/resource-acceptance tests pass,
and the official checker passes against the regenerated snapshot.

Local frozen evidence is under `/tmp/gqc-wheel-df-lambda-pool/`, including
`equivalence.json`, `compiled-report.json`, `resource-comparison.json`,
`gpu-comparison.json`, scientific test logs, source/compiler/cache provenance
and separately executed GPU reports. Preserve its reproduction scripts while
the enclosing PR is under qualification.

## Separate wheel blocker

The older wheel run **43732294063**, head `a278661ba`, reaches object 708/1122
before `ptxas` compiling `src/scf/cuda/one_electron_derivatives.cu` is killed
with signal 9. Docker's retained state reports **OOMKilled=true** under the
12-GiB memory/swap-equal limit; the concurrent one-electron values object
finishes afterward. The build step lasts 14,642 seconds and fails, rather
than timing out or failing a numerical test. Lambda pooling does not resolve
this separate one-electron compiler-memory hotspot. The latest full-wheel
gate must not be called successful based on the action-level evidence here.

## Revisit when

If the core reuse phases gain a shared-symbol ownership contract, qualify
their complete retained-state behavior before pooling their current owner.
Any broader grouping must retain executable numerical/error coverage and
complete-object resources; source byte reduction alone is insufficient.
