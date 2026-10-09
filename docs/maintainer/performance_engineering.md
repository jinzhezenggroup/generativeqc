# Performance engineering

GenerativeQC performance work must optimize scientific work and data movement, not
only kernel throughput or peak scratch size. This document records cross-cutting
rules for CUDA, generated integrals, response, DFT, and post-HF execution.

## Optimization method taxonomy

Treat performance as a stack of optimization opportunities rather than as a
kernel-only problem. The categories below are deliberately orthogonal: a complete
endpoint may need several of them, but each candidate should identify which
mechanism is expected to remove time and which measurements can falsify that
hypothesis.

| Symptom | First optimization questions | Evidence to collect |
| --- | --- | --- |
| Work grows faster than the scientific problem requires | Can the algebra be reassociated, factorized, symmetry-reduced, screened, sparsified, or solved with fewer expensive operator applications? | semantic work counts, symbolic degree, active-domain sizes, iteration/operator counts |
| The same expensive source is regenerated | Can one production owner generate it once and feed several consumers, or keep it resident across phases? | source evaluations/passes, cache/residency identity, bytes regenerated |
| Large intermediates are written and immediately reread | Can producer and consumer be fused, or can the producer contract directly into the final reduction? | intermediate bytes, traversals, launch count, endpoint time |
| GPU time is dominated by packing/copies or low useful bandwidth | Is the layout appropriate for the consumer, can indexed/packed/local layouts avoid dense work, and can staging be removed? | H2D/D2H bytes, gather/scatter bytes, achieved bandwidth, active occupancy of the logical domain |
| One generic kernel handles very different workloads poorly | Should execution plans specialize by operator, derivative order, shell/layout class, target, or size while sharing one scientific definition? | per-class work/time, register/local/shared-memory use, fallback fraction |
| Many short kernels or host waits dominate | Can a larger solver region, graph/device-tail execution, batching, or fewer host synchronizations preserve the same semantics? | launch count, synchronization count, CPU gaps, device-idle intervals |
| Transfer/preparation and device work alternate serially | Can bounded prefetch, asynchronous copies, or double buffering overlap independent work? | copy/compute overlap, idle gaps, buffer peak, stream/event trace |
| Compute is throughput-bound in numerically tolerant components | Can compute/storage/accumulation precision be selected independently while retaining strict publication checks? | precision provenance, FP32/FP64 throughput, numerical margins, complete endpoint time |
| Iterative response/CC work repeatedly applies an expensive operator | Can preconditioning, warm state, recycling, block/multi-RHS execution, or control-flow changes reduce the number of exact applications? | exact/provisional action counts, residual history, complete solve time |
| Runtime is good but specialization/compilation is excessive | Can equivalent generated variants be canonicalized, packaged AOT, cached, or left to bounded JIT fallback? | compile/cache identity, artifact count/size, cold compile time, runtime coverage |

The default order of attack is: reduce semantic work first; then remove
recomputation and materialization; then fix layout/data movement and choose the
right execution plan; only then spend effort on low-level kernel tuning. Launch
control, overlap, and mixed precision should be driven by a measured residual
bottleneck rather than enabled merely because the mechanism exists.

### Reduce mathematical and semantic work

The highest-value optimization is usually work that never executes.

- Reassociate or factor contractions when the compiler can prove a lower
  symbolic cost without changing the declared equation. TensorIR is the preferred
  owner for these rewrites.
- Exploit exact symmetry, packed domains, shell/pair topology, and repeated-index
  structure before expanding work into dense ordered domains.
- Use scientifically legal screening or active/local domains to remove work
  whose contribution is provably outside the requested contract. Screening
  policy must be part of the execution identity and must retain the required
  final physical validation.
- Distinguish logically different consumers. Coulomb and exchange, value and
  derivative, short- and long-range operators, or dense and occupied-factorized
  contractions may share one scientific source while needing different optimal
  execution plans.
- For iterative methods, count expensive operator applications as semantic work.
  A better preconditioner or reusable Krylov state can dominate a faster kernel
  if it materially reduces the number of exact actions.

Do not infer complexity from lexical loop depth alone, and do not trade exact
scientific semantics for a cheaper approximation without changing the method or
provider identity explicitly.

### Reuse source work and prepared state

Generate expensive source work at the widest lifetime for which its scientific
identity remains valid. Typical reusable state includes shell-pair topology,
geometry/Boys/Rys data, molecular grids, AO active maps, integral/source tiles,
metric/factorization results, transformed occupied blocks, and prepared solver
metadata.

Prefer one owner with explicit borrowing/lifetime rules over independent
method-specific caches. Reuse is valid only when geometry, operator, basis,
precision, screening, generation, and stream/event dependencies match. A stale
or ambiguously identified cache is a correctness bug, not a performance feature.

The detailed source-reuse and residency rules below remain normative.

### Fuse producers, consumers, and reductions

Materialization is justified only when its reuse value exceeds its storage and
traffic cost. Look for sequences such as

```text
produce full intermediate
write intermediate
read intermediate
permute / weight / reduce once
```

and prefer a generated fused region that applies the permutation, weight,
denominator, response cotangent, or final reduction while the source values are
live. Common targets include AO/grid consumers, derivative contractions,
triples/response tiles, normalized partition derivatives, and permutation-heavy
tensor equations.

Fusion must preserve a readable scientific owner. Do not replace a canonical
equation or AD graph with a second hand-maintained formula merely to obtain a
fused kernel. Keep an unfused or generic route as an oracle/fallback where the
specialized region has bounded admission conditions.

### Choose representation and layout for the consumer

Dense canonical tensors are not always the cheapest execution representation.
The planner may choose packed-symmetric, indexed, block-sparse/local, tiled, or
factorized layouts when the mapping is exact and explicit.

In particular:

- keep grid-local AO work in active/indexed AO dimensions instead of expanding
  inactive rows and columns only to skip them later;
- keep symmetric pair domains packed when downstream consumers can operate on
  the packed identity directly;
- gather or pack once for several contractions rather than once per consumer;
- prefer contiguous/coalesced device access and library-compatible layouts when
  they reduce total endpoint work, not merely one kernel's instruction count;
- avoid GPU -> host -> GPU round trips when ownership permits direct device
  consumption; and
- account for scatter/reduction contention introduced by a sparse or indexed
  representation rather than assuming fewer values are automatically faster.

A representation change must report both the reduction in logical work and any
new gather, scatter, metadata, or packing cost.

### Specialize execution plans without duplicating science

One scientific operator may have several execution plans. Specialization may be
by operator role, derivative order, shell/angular class, tensor/layout class,
precision, target architecture, or bounded size/resource regime.

Useful specialization includes generated shell-class kernels, separate J/K
schedules, value-versus-force schedules, target-specific tile/block geometry,
library-backed GEMM for regular contractions, and generated kernels for
irregular/fused domains. The selection boundary belongs in compiler/runtime
planning, not in molecule names or benchmark identities.

Specialization should shrink the expensive generic residual domain rather than
create a parallel scientific implementation. Retain a bounded generic fallback
for unsupported classes or resource regimes, and record how much production
work still uses that fallback.

### Tune GPU resource use after the work is right

Use low-level tuning after semantic work, representation, and dispatch are
credible. Inspect at least:

- registers per thread, local-memory spills, and stack usage;
- achieved versus theoretical occupancy and active warps;
- block/warp geometry and tail utilization;
- shared-memory capacity, bank behavior, and reuse;
- global-memory transaction efficiency and achieved bandwidth;
- instruction mix, FP64/FP32 throughput, and dependency/latency stalls; and
- atomic/reduction contention.

A high register count or low occupancy is a diagnostic lead, not an automatic
cause. A kernel can be latency-, bandwidth-, instruction-, launch-, or
work-amplification-bound with similar occupancy numbers. Validate resource
changes with profiler counters and the complete endpoint. Static compiler
resource reports help choose candidates but do not replace device measurement.

### Reduce launch and host-control overhead

When semantic work is already small, repeated launch and synchronization can
become the bottleneck. Candidates include:

- batching compatible work into fewer launches;
- compiler-generated fused solver regions;
- CUDA Graph replay for stable prepared regions;
- device-tail or persistent control when bounded failure semantics remain
  observable;
- eliminating redundant evaluations of the same accepted solver state; and
- replacing unconditional host fences with explicit event dependencies.

Do not hide a required convergence, error, publication, or lifetime check merely
to reduce synchronization. Profiling and clean timing should also remain
separate: instrumentation fences can change the schedule being diagnosed.

### Overlap independent preparation, transfer, and compute

Use asynchronous execution only when a timeline proves there is latency to
hide. Bounded prefetch and double buffering can overlap preparation or transfer
for tile N+1 with computation on tile N, but they also consume more retained
memory and make ownership more complex.

Charge every in-flight buffer to the resource plan, use explicit events for
ownership transfer, retain a single-buffer fallback for tight budgets, and
measure actual overlap rather than adding copy and kernel durations. If the
endpoint is compute-bound with no device idle gap, extra buffering is normally
the wrong optimization.

### Treat precision as an execution-schedule dimension

Precision is not a global on/off performance flag. When numerically legal, plan
compute precision, storage precision, and accumulation precision independently
per component. Throughput-heavy contractions may use lower precision while
sensitive denominators, reductions, residuals, stationarity checks, energies,
and forces remain in FP64.

Every mixed-precision path must expose its actual precision provenance, retain a
strict-FP64 route for the same scientific equation, and pass independent
numerical gates with margin on difficult cases. Tensor-core or SGEMM utilization
is implementation evidence; complete-endpoint speedup plus numerical acceptance
is the promotion criterion.

### Reduce expensive solver/operator applications

For response, CC, SCF, and other iterative endpoints, optimize the control
problem as well as the operator kernel. Investigate stronger legal
preconditioners, compatible warm starts, Krylov/subspace recycling, block or
multi-RHS solves, reuse of accepted residuals, and removal of duplicate
provisional/audit evaluations.

Always publish the exact final residual/stationarity check required by the
method. A provisional approximation can accelerate convergence only when the
final exact physical operator and acceptance gates remain unchanged. Report
both per-action cost and the number of actions; otherwise an apparent kernel
win can be erased by extra iterations.

### Optimize generated-code and compilation cost separately

Compiler/code-generation performance has its own objective. Canonicalize
equivalent specializations, package common qualified variants AOT, cache
immutable artifacts by complete compiler/source/target identity, and reserve
JIT for bounded uncovered domains.

Do not weaken runtime code quality solely to make CI compile faster, and do not
report a fast-build configuration as production performance. Conversely, a
runtime specialization whose compile/artifact explosion makes normal deployment
impractical is not a complete performance solution.

### Build a causal profile before choosing the next optimization

Use an attribution chain from the public endpoint down to the proposed
mechanism:

```text
complete wall time
  -> mutually exclusive phase wall time
  -> semantic work and data movement
  -> launches / synchronization / idle gaps
  -> kernel or library activity
  -> hardware counters and resource use
```

Parent wall intervals must be mutually exclusive. Nested profiler scopes,
CUDA-event intervals, API durations, and summed kernel times are explanatory
views and must not be added together as if they were independent endpoint
components.

Use source-matched clean timing for the endpoint and separate instrumented runs
for attribution. Nsight Systems establishes timeline and causality; Nsight
Compute or equivalent counters diagnose selected kernels. Static resource
reports and the [CUDA timing estimator](../developer/cuda_time_estimator.md)
can prioritize experiments but cannot establish a speedup.

A useful performance issue should therefore state the expected mechanism in
falsifiable terms, for example: fewer exact source evaluations, fewer bytes
materialized, a smaller active domain, fewer launches/fences, higher useful
throughput for unchanged work, or fewer expensive solver actions. If none of
those changes, a speedup hypothesis is incomplete.

## Work amplification is a first-class metric

A schedule can satisfy a memory budget while repeating expensive work many
more times than necessary. Planners and benchmarks should therefore report both
memory and work, including the relevant subset of:

- consumer tiles/panels and source tiles/passes;
- integral/source evaluations and generated values;
- GEMM/contraction counts and their dimensions;
- bytes generated, gathered, uploaded, downloaded, or regenerated;
- synchronization/wait counts; and
- complete cold, warm, changed-geometry, and batch wall time.

When system size or a smaller budget changes a tile shape, compare actual work
counts before interpreting a kernel slowdown. A sudden endpoint cliff often
means the executed algorithm changed its amount of work.

## Audit native high-order scalar work

Run `python tools/audit_native_complexity.py` to inventory leaf native loop nests
with depth four or greater. The default audit covers GenerativeQC production
sources under `src/` and excludes the vendored xTB native tree; use
`--include-vendored` for a broader diagnostic scan. `--format json` emits a
machine-readable inventory.

The audit separately classifies an avoidable rank-2 matrix-chain candidate when
a high-order scalar reduction writes a rank-2 target from at least three
pairwise-indexed inputs spanning four loop indices, without a genuine
three-/four-index source access. This distinguishes patterns such as a scalar
`C^T A C` implementation from an exact four-index ERI/J/K contraction. The
pre-commit `native-complexity-audit` hook runs
`--fail-on-matrix-chain`, so a reintroduced quartic rank-2 transform fails CI
and must move to TensorIR or the shared dense-linear-algebra owner.

The source audit is deliberately conservative: unclassified high-order loops are
reported, not automatically rewritten. Reports include both lexical loop depth and
an effective depth that removes simple fixed extents such as spin=2 or xyz=3. The
action classes are:

- `matrix-chain-candidate`: conservative structural match for a pure-algebra rank-2 scalar matrix chain; CI-blocking;
- `high-rank-output-materialization`: high-rank permutation/symmetrization pass;
  inspect producer/consumer fusion to remove a complete traversal/materialization;
- `high-rank-source-contraction`: genuine high-rank source access; inspect
  provider/factorization or fuse-consume ownership without assuming lower formal scaling;
- `fixed-extent-inner-loop`: lexical depth inflated by bounded constant dimensions;
  consider unrolling/fusion but do not label it O(N^depth);
- `high-order-loop`: report-only fallback requiring algebra/profile review.

TensorIR symbolic complexity remains the proof-carrying path for legal contraction
reassociation; native findings are review prompts for code that still sits outside
that IR.

### Review high-rank materialization candidates

The `Source work inventory (advisory)` job also enforces a bounded candidate-review
gate. Its name retains the wider advisory audit scope; the materialization review
step fails if a finding lacks a current source-bound disposition. Run:

```bash
python3 tools/ratchet_native_materialization.py --base-sha "$BASE_SHA" \
  --output .artifacts/native-materialization-review.json --fail-on-unreviewed
```

`BASE_SHA` must be the full immutable PR base commit, available in the local Git
object database. The tool extracts its `src/` and `include/` bytes without
executing baseline code, then scans both trees with the same current analyzers.
Missing source, baseline, malformed evidence or changed imported analyzer code
produces `INCOMPLETE`, which fails the enforcing command. It never silently
substitutes an empty baseline. The first adoption needs no older disposition
manifest; the actual baseline source census remains mandatory.

The receipt retains full baseline/candidate inventories, raw source and analyzer
hashes, counts, and separate added, removed, changed and unchanged identities.
Equal totals cannot hide replacements. A removed lexical finding is evidence of a
scanner delta, not proof of end-to-end retirement or numerical parity. CI uploads
the receipt with the existing `source-work-audit` artifact even on failure.

Review entries live in `manifests/native_materialization_dispositions.json`.
Each current site needs an owner, disposition, reason, producer, consumers,
residency, lifetime, resource owner, open evidence gaps and matching source
bindings. Produce the current anchors and source identities with:

```bash
python3 tools/ratchet_native_materialization.py --inventory-only \
  --output .artifacts/materialization-candidates.json
```

Inspect the changed producer/consumer code before updating a record. The site
identity uses path, function signature, target, loop variables and occurrence;
line numbers are display anchors. Context bindings cover entire declared source
files. For ordinary non-preprocessed source, comments and line shifts preserve
identity while code/literal changes require renewed bindings. Files containing
preprocessing spellings (including `#`, `%:` or `??=`), raw/line-spliced literals,
or line-sensitive builtins such as `__LINE__` conservatively bind exact source
bytes. Textual edits to those files, including comments and line shifts, require
renewed bindings. This avoids claiming preprocessing equivalence from a bounded
lexer. Remove records for removed sites: stale or unmatched
identities, duplicated records, and stale producer/consumer bindings fail review.
A scanner false positive must be fixed in the classifier, not exempted here.

Producer/consumer symbols are reviewer-declared links, not verified call edges or
ABI certificates. Source bindings do not claim transitive or whole-program
coverage. Whole-file changes can conservatively require review of unaffected
sites. All six current MP2 entries remain `retained-pending-evidence`: bounded
storage, a dense callback span, or a factorized alternative does not prove that
fusion is unsafe or slower. The dense first stage has a source-known production
RCCSD(T) caller in `src/cc/rccsdt_force.cpp`; its retention gap concerns safe and
profitable replacement, not whether that call exists. The gate enforces review bookkeeping only; issue
#1626 remains open for independent numerical, work/resource and complete-endpoint
performance evidence. Historical RCCSD(T) retirements are discussed in the
[decision note](../../.agents/notes/implemented/performance/2026-10-09-materialization-candidate-review.md).

## Prefer source-driven reuse

Expensive source work should normally be produced once and consumed by multiple
outputs before eviction. In particular, inspect loops of the form

```text
for consumer_tile:
    for every source_index:
        expensive_source_or_projection(source_index)
        consume(consumer_tile, source_index)
```

If the expensive operation does not depend on `consumer_tile`, either move it
outside that loop, retain/batch it under the resource policy, or demonstrate
with endpoint evidence that recomputation is faster.

The same rule applies to AO integral tiles feeding many MO blocks, auxiliary
projections feeding many response panels, grid/AO jets feeding several XC
consumers, and intermediates feeding CC/response equations.

## Reuse resident state before recomputation

Prepared plans often already own large, charged buffers and immutable metadata.
A downstream phase may borrow them when all of the following are explicit:

- scientific identity and generation match;
- the producer no longer needs the contents being overwritten;
- stream/event ordering protects the lifetime;
- resource accounting records borrowed capacity without double counting; and
- an explicit bounded fallback remains available when borrowing is impossible.

A component-local scratch allowance is not a reason to recompute a much more
expensive quantity if compatible resident storage is already owned elsewhere.
Memory safety, ownership, and work efficiency must be planned together.

## Keep oracle work out of production paths

Independent CPU/reference implementations are valuable correctness oracles,
but production CUDA setup must not construct transformed tensors, factorizations,
or derivative data solely because the oracle uses them. Record which consumer
owns each preparation result, and skip unused compatibility work on production
backends.

Likewise, avoid GPU -> host -> GPU staging when a CUDA consumer can use the
producer device data directly. Host-staged providers may remain explicit oracle
or compatibility paths, but performance claims must identify which route ran.

## Derivative and response consumers

Prefer consumer-driven derivatives:

```text
generated derivative primitive
    -> apply current response/adjoint weight
    -> bounded reduction
    -> physical gradient/HVP output
```

Avoid full coordinate-major derivative tensors when a generated kernel can
contract the final weight directly. Reuse common geometry, Boys, moment, or
shell work across requested derivative components when doing so improves the
complete endpoint rather than only an isolated kernel.

## Historical rationale

The DF-response work-amplification cliff fixed by PR #373 motivated several of
these rules. Its exact panel counts, GEMM/transfer amplification, benchmark
conditions, rejected alternatives, and revisit criteria are preserved in the
[DF-response work-amplification note](../../.agents/notes/implemented/performance/2026-09-15-df-response-work-amplification.md).
This document intentionally keeps the current policy rather than the migration
history.

## Performance qualification checklist

The [experimental CUDA kernel timing tool](../developer/cuda_time_estimator.md)
accepts explicit achieved-rate calibration for offline estimates. Its retained
RTX 5090 arithmetic/copy/streaming calibrations require explicit family selection
and qualify only the recorded warm FP64 probe domain. Refined-model acceptance
uses fresh held-out median/P95/maximum error gates of 5%/12%/20%, overall and per
family. Its engineering uncertainty band is not a statistical confidence interval.
Kernel estimates do not replace the complete endpoint and numerical acceptance
gates below.

The [RCCSD(T) CPU response bundle qualification](ccsdt_cpu_bundle_qualification.md)
records the complete force-endpoint comparator and its cold/warm artifact and
semantic-work gates.

The batch comparator records the loaded native binary hash and selected kernel
profile in `native_build`. Acceptance-matrix runs also write a per-point
`.progress.jsonl` journal before cold execution, so preparation/runtime failures
retain this identity. A source hash alone cannot distinguish builds with different
compiled kernel coverage; see the
[binary provenance decision](../../.agents/notes/implemented/compatibility/2026-09-17-benchmark-binary-provenance.md).

### SCF residual and stopping-rule interpretation

The batch comparator's schema v3 records `convergence_policy` explicitly.
Equal numeric tolerances or reported SCF iteration counts do not establish
equivalent stopping rules or equal Fock work. GPU4PySCF also evaluates an
initial potential before its counted cycles. Compare complete endpoints and
independently gated energies/forces; retain stock reference DIIS unchanged.
An explicitly requested `--reference-full-fock` suppresses both incremental
potential inputs, including on RKS backends that still reuse `vhf_last` when
`direct_scf=False`. Density fitting's own `direct_scf=False` policy alone does
not assert that an explicit full-Fock override was requested.

Convergence payloads with `residual_schema_version=2` distinguish GPU4PySCF's
`density_frobenius` from `density_rms`. RMS divides the backend norm by the
square root of all density-matrix entries, including spin blocks, using shape
metadata only. An unavailable shape yields null RMS, never an assumed size.
`orbital_gradient_norm` remains the backend's unnormalized global norm and is
not the native AO commutator RMS. Callback values describe the last reported
cycle, not an additional final physical audit. First-cycle energy change uses
the backend's preceding energy when available; a cold default guess is not
labeled as a warm density seed.

Historical payloads lacking the residual version marker retain their original
meaning: GPU4PySCF `density_rms` stored an unnormalized Frobenius norm. Do not
silently rewrite archived measurements. Summary readers accept v2 and v3
artifacts, preserve their residual fields and retain the new policy metadata.

Both cold and warm reference energy-plus-force timers include returning the
forces from device to host, matching native `execute()`'s public-output
boundary. JSON/list serialization remains outside timing on both engines.

### General acceptance gates

Numerical acceptance uses the maximum error across every measured repeat pair.
Matching iteration counts only classifies timing; it cannot exclude inaccurate
samples from the energy or force gate. Energy-only runs retain absent force
errors as `null`. Legacy integral/contraction timing field names contain complete
endpoints and explicitly report that no component split was measured. See the
[all-repeat acceptance decision](../../.agents/notes/implemented/compatibility/2026-09-17-all-repeat-accuracy-gates.md).

Before promoting a new default or auto-selection policy:

1. Record the baseline endpoint and exact scientific settings.
2. Record actual work counters for baseline and candidate.
3. Explain any change in algorithmic work caused by tiling or memory limits.
4. Validate numerical equivalence with an independent oracle/reference.
5. Test at least one larger size that can expose a planner/tile cliff.
6. Report cold, warm, and changed-geometry behavior when geometry-dependent
   preparation exists; report batch and constrained-memory behavior when those
   modes are supported.
7. Keep slower/correct fallbacks available where the promoted resource or
   identity preconditions do not hold.
8. Prefer complete endpoint evidence over isolated kernel speedups.
