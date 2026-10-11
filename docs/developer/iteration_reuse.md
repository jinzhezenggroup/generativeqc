# Shared iteration-invariant reuse

`tensor.iteration_reuse.analyze_iteration_reuse` proves which existing TensorIR
operations can be prepared once within an immutable-input execution epoch. The
analysis runs before CPU/CUDA emission and has no method, device, convergence,
DIIS, precision-selection or numerical-approximation policy.

## Contract

The consumer declares immutable input **names**. The analysis propagates complete
input dependencies and the existing `EffectKind` proof through immutable SSA.
Only proven-pure non-input operations whose ancestors are also reusable become
invariants. Everything depending on an iteration input remains dynamic. An
opaque or effectful ancestor prevents reuse even if all named inputs are fixed.
Optional effect declarations can downgrade purity, never authorize an unknown
operation. No algebra, operand order or output is changed.

The result contains ordered invariant/dynamic nodes, transitive dependencies,
logical retained bytes and a deterministic schedule identity. This identity
proves the program and selected dependency contract, **not the current numerical
values**. The analysis owns no cache and never authorizes reuse from equal shapes
or equal addresses.

An owner must establish immutable reference inputs for the entire epoch. Changing
geometry, basis, grid, method parameters, precision or executable/device binding
requires the relevant existing owner to refresh or replace that binding. Updating
density or CC amplitudes alone does not invalidate genuinely independent values,
but every operation depending on them still executes.

## Shared native storage lowering

`tensor.native_arena.plan_symbolic_arena` owns the runtime-shape storage schedule
used by the native RCCSD, triples/response, and DF Lambda generators. It consumes
ordinary TensorIR after the existing production preparation passes. Domain
adapters bind indices to runtime extent identifiers and supply the already
selected dependency order; they do not color slots or infer invariance.

The planner reserves exclusive slots for declared retained nodes before coloring
dynamic scratch by last reader, using `common.storage.analyze_lifetimes` for the
materialized ownership contract. The default plan shares identical symbolic
products. Equal representative sizes do not prove equal capacities at runtime.
An operation cannot overwrite its own inputs, and all borrowed outputs remain
live through return. Slot assignments are immutable and deterministic. The
storage identity describes capacities and assignments only: it does not establish
scientific identity, numeric validity, or an immutable-input epoch.

This lowering contract is explicitly materialized, dense FP64 storage. It does
not infer physical views, donate buffers, select providers, change execution
order, or choose fusion/rematerialization. Alias-aware lowering uses the existing
`common.storage` analysis; schedule selection and GPU profitability remain with
the shared ScheduleIR infrastructure. Unknown runtime dimension bindings,
non-FP64 intermediates, malformed orders, and invalid retained nodes fail closed.

The emitted checked element-capacity expressions feed the existing native byte
admission. They are not complete endpoint peaks or measured traffic. Native
owners must charge reference inputs, outputs, solver storage, descriptors,
preparation overlap, and dynamic workspaces, then allocate only the admitted
layout. Allocation/preparation failure must not make a retained value ready.

DF Lambda uses the same invariant proof and symbolic arena planner for its
FP64 staged matrix core. Primal inputs are immutable within one native action
owner; all adjoint seeds remain dynamic. The owner admits retained storage and
contraction descriptors against its complete budget. A resource miss retains
the full core evaluation; an optional allocation miss drops retention before
retrying the bounded matrix route. Numerical/launch failures propagate without
publishing partial output. This does not change batching, matrix-provider
defaults, stationarity gates, or the independently expanded audit.

## Native scratch planning

`plan_symbolic_arena(..., cross_size_reuse=True)` additionally coalesces whole
slots with disjoint owner lifetimes. Each resulting slot holds the runtime maximum
of its original checked extent products, so capacity cannot exceed their sum,
even when runtime dimensions reverse their representative ordering. This
conservative candidate does not search new schedules or infer physical aliases.
Borrowed outputs remain live through return; retained plans keep the default
coloring. DF Lambda's matrix-core capacity query also retains the frontier
scratch floor, so reducing the uncached layout cannot refuse the existing cache.

Native source qualification selects it with
`tools/generate_rccsd_native.py --cross-size-arena`. Generate matching CPU capacity
headers and CUDA sources in the same scope. Imported DF emitters can use
`tools.generate_rccsd_native.cross_size_arena_candidate()` around **all** associated
headers and sources. Never combine smaller capacity queries with baseline pointer
assignments. Storage identity uses schema v2 for maximum-product capacities;
scientific equation identities are unchanged. Standalone CLI generation without
the flag and direct planner calls retain the equal-product baseline.

CMake selects the paired native CC/DF/RHF emitters with the default-on
`GENERATIVEQC_NATIVE_CROSS_SIZE_ARENA` option. Set it to `OFF` to build the
equal-product fallback. The DF and RHF source generators
also accept `--cross-size-arena`; the selection covers all outputs from one
invocation. Use the CMake option rather than replacing individual generated
files in a production build.

`cc::LambdaOptions::serial_scratch_reuse` is a separate default-off owner candidate.
Conventional Lambda shares fresh-replay and response scratch. DF Lambda shares
expanded, staged, matrix, primal-replay and independent-audit scratch on its
existing single stream. Every borrowed result is consumed before another producer
reuses the region. Inputs, sums, seeds, immutable cuts and the optional invariant
frontier remain disjoint. The compiler supplies checked capacities; the native
owner supplies the serialized consumption contract. This is **not** a compiler
proof of arbitrary cross-stream or cross-method lifetimes.

DF admission charges one maximum-sized scratch region plus persistent storage,
host publications/descriptors and the existing provider allowance. A fixed aligned
scratch prefix after inputs shifts the persistent suffix without changing its
alignment. Allocation-refusal steps recompute admitted bytes and the shift before
binding pointers. Retention, audit, matrix and cut fallbacks retain their order;
no GPU synchronization or scientific work is added.

The internal `run_df_ccsdt_native` accepts the trailing
`lambda_serial_scratch_reuse` selector after its reference-experiment pointer.
`benchmarks/df_ccsdt_force_endpoint.cpp` accepts a trailing `0`/`1` selector and records
`lambda_serial_scratch_reuse_requested`. Existing Lambda numeric-capacity and
owned-device-byte fields describe the selected layout. `df_audit_arena_bytes`
remains a logical audit/replay capacity requirement, not an additional allocation
when scratch is shared.

The next benchmark argument selects a three-call lifecycle: cold, same-process
warm, and moved-warm after atom zero moves by `1e-4` Bohr in z. Both orbital and
auxiliary bases use the displaced geometry. These calls rebuild native scientific
owners; they are not prepared force replay. Recycle selection remains separate.

Cross-size arena reuse is qualified independently of serial scratch; serial
scratch remains experimental because it can change warm cache/provider admission.
Static capacity reductions, source equivalence, host tests or CUDA compilation
alone do not justify promotion. Promotion requires same-source FP64 numerical and
independent-audit gates, complete cold/warm/moved-warm E+F noninferiority, matching
iteration/action/task counts and the existing allocation journal. Observe numeric
allocation high-water separately from process, provider and CUDA Graph retention;
measure traffic, launches, registers and spills rather than inferring them from
arena sizes. Real-device runs require finite Slurm allocations on allowed nodes.

See the [memory-candidate decision](../../.agents/notes/implemented/performance/2026-10-11-memory-aware-native-arena-candidates.md)
for rejected approaches and qualification boundaries.

## Native RCCSD consumer

The conventional dense CPU RCCSD solver exposes an internal, default-off
`SolverOptions::iteration_invariant_reuse` candidate. When enabled, it declares
every input except `t1` and `t2` immutable for one synchronous solve. Its `Problem` owns the reference
vectors, and its reuse arena is local to that call. Every new solve, including a
changed reference, starts with a fresh preparation. DIIS/trial amplitudes always
run the dynamic graph. The separately expanded final physical-residual replay is
unchanged and uncached. Accepted-trial residual reuse is orthogonal: when DIIS
returns an unmodified trial, its already computed output remains reusable, and
no-DIIS iteration does not evaluate a redundant trial. Static preparation counts
only actual evaluator calls, never carried-output consumption.

The native generator lowers the same dependency schedule into CPU and CUDA
prepare/replay entry points. Only the explicitly enabled CPU candidate currently selects them. CUDA
owner integration, asynchronous failure/Graph qualification and real-device
acceptance remain separate work; generated CUDA parity is not a GPU execution or
speedup claim. Density-fitted CC retains its existing specialized schedule:
external corrections may depend on current amplitudes and cannot be declared
immutable just because they are provider inputs.

The symbolic arena reserves exclusive slots for retained nodes **before** coloring
dynamic scratch. Extending only their old last-use intervals would be unsafe:
an early operation in the next iteration could overwrite a later invariant's
slot. The complete CPU solver budget charges retained storage, dynamic scratch,
independent replay, reference inputs, outputs and existing solver/DIIS storage.
If the reuse layout does not fit, the original full evaluator is retained as the
bounded fallback. Only the selected arena is allocated. A failed preparation
never becomes ready.

Native diagnostics distinguish one-time preparations, dynamic evaluations,
executed invariant/dynamic operation counts and saved invariant operations. These
are conventional-dense semantic graph-work counts; they are not elapsed-time or hardware-instruction
measurements. On the representative conventional graph, 9 invariant operations
(8 einsum permutations/copies and 1 addition) are omitted from subsequent dynamic evaluations.
Actual memory scales with the runtime orbital dimensions.

## Portable work reduction

For the qualified conventional graph at `o=8,v=16`, the retained operations
transform 135,616 FP64 elements. Eight are one-input, unit-coefficient `einsum`
permutations/copies; the ninth is a one-input, coefficient-1 `add`, also an
identity transform. None performs a reduced-index contraction or nontrivial
mathematical floating-point addition/multiplication. Avoiding the operations
removes indexing, copying, checks and any remaining lowered identity arithmetic.

Each later evaluation avoids 1,084,928 logical read bytes plus 1,084,928 write
bytes for those transforms (2,169,856 bytes). Nine evaluations prepare once and
skip eight repetitions: 17,358,848 logical bytes. These are logical operator
traffic counts, not measured DRAM transactions; caches and compiler optimization
can change actual traffic. Dynamic consumers still read the retained values.

The existing unfused CUDA emission assigns one kernel to each of these nine
nodes. Its generated prepare/dynamic split can therefore omit nine launches per
later evaluation, or 72 across this nine-evaluation example, after one nine-kernel
preparation. This is a source-verified work reduction, **not executed GPU evidence**.
Future fusion/provider schedules must derive their own launch accounting. Device
residency, stream/capture ownership, full memory admission and physical validation
must be integrated before activating CUDA reuse.

## Scope and testing

HF/DFT already share the fixed-point controller in `src/solver/self_consistent.hpp`
and retain integral/Fock/grid resources through existing owners. This first implementation slice does
not replace those caches or integrate a new molecular AO cache into DFT. Tests
apply the same analysis to real restricted/unrestricted Fock composition graphs,
where Hcore can be invariant but density-dependent J/K/XC remains dynamic.

Run the focused shared proof tests with:

```sh
PYTHONPATH=python:. python -m pytest tests/python/test_tensor_iteration_reuse.py
```

The symbolic storage contract and actual CPU owner can be checked with:

```sh
PYTHONPATH=python:. python -m pytest tests/python/test_tensor_native_arena.py tests/python/test_rccsd_iteration_reuse.py
```

Native tests compare prepared and full evaluation over changing amplitudes and
fresh references, preserve independent physical acceptance, exercise byte-budget
fallback and inspect the common CPU/CUDA generated schedule. Native compilation
requires the repository's verified compiler-cache launcher.

## Promotion boundary

The internal option does not change public method defaults. The corrected
same-binary, same-budget, DIIS-6 probe has modest native-solve benefits, while an
initial noisier comparison had mixed signs. The earlier larger-shape slowdown
was not reproduced and is not evidence of cache misses or a pinned-layout
penalty. Saved graph work alone is not a general profitability rule.

The retained probe starts from an already-built synthetic `Problem`; it does not
measure the complete SCF → MO → CC molecular endpoint or force calculation.
Its separate fixed-work three-mode control compares the original traversal,
pinned layout with preparation repeated, and actual reuse. This control excludes
solver/DIIS/physical replay and cannot replace full-solve evidence. See
`tools/benchmark_rccsd_iteration_reuse.py` and the retained results for exact
raw vectors, provenance and scope. CUDA device execution remains unqualified.
