# Decision: elide audited native copies returning to the original arena slot

Status: implemented
Date: 2026-10-10

The pair-enabled CUDA action and complete energy-only endpoints are numerically
qualified against a frozen baseline. Four matched timing observations are
retained; they are not formal/global performance or force promotion.

## Problem

A complete endpoint profile of the occupied-spectator-pair implementation
identified six singleton-add kernels costing 4.703272156 traced seconds in the
primary CCSD action. The copies occur in adjacent producer/copy/copy chains.
The original arena already returns each final value to its producer's slot:
the batched triples `(32,33,34)`, `(36,37,38)`, `(41,42,43)` have slots
`[12,13,12]`, `[13,15,13]`, `[15,16,15]`, respectively. Those indices describe
the measured graph, not an optimization rule or a stable node-number API.

Traced timing is diagnostic only. The separately measured complete endpoint is
the performance boundary, including native RHF, DF source, CCSD, independent
expanded replay, FP64 (T), and owner/caller teardown.

## Decision

`analyze_native_copy_roundtrips` derives immutable, reproducible execution
metadata from the existing mathematical graph and complete symbolic arena. A
triple must have all of these properties:

- The producer and two copies are adjacent in the actual native execution order.
- Both copies have one input and exact rational coefficient one, matching FP64
  value metadata except for the role that a materialized intermediate acquires.
- The producer's sole reader is the first copy; its sole reader is the final
  copy. Neither producer nor intermediate is an exposed output.
- The original producer and final copy have exactly the same arena slot.
- The producer is an audited scalar add, multiply, transpose, or two-input
  einsum without reductions. Inputs, matrix callbacks, and reductions are not
  accepted as earlier arithmetic auditors.

Triples are disjoint, including on longer copy chains. The immutable result
records graph, arena and copy-plan hashes. Unsupported future graphs simply
retain their copies. The proof introduces no alias, view, donation, extended
lifetime, arena recoloring, new allocation, or cross-iteration assumption.

The native CUDA emitter opts in by omitting only the two proven launches. It
keeps original kernels, pointer bindings, complete allocation, mathematical
serialization, and all other operations. Retained/reused/external kernel
schedules reject this opt-in rather than attempting a second lifetime proof.
Only the separately generated packed and Q-batched paired auxiliary actions
enable it; the raw action and every other existing consumer remain unchanged.

The production owner uses separately emitted CUDA operation counts for paired
work diagnostics. CPU graph counts remain unchanged. No new public diagnostic
fields or ABI changes are needed, and contractions/GEMMs retain their previous
counts. An empty proof naturally emits the original operation count.

## Numerical invariants

This is **not** generic singleton-add algebra folding. TensorIR's interpreter
starts addition from positive zero, whereas the native singleton expression is
`1.0 * source`. Folding at the mathematical IR level can change signed-zero
behavior. Leave the graph and interpreter alone.

Finite FP64 values, signed zero, and subnormals survive native unit copies
bit-for-bit. The scalar producer audits every output before either original
copy, so elision retains the first sticky arithmetic fault. Borrowed inputs and
matrix/reduction producers do not satisfy this earlier-auditor premise.
Original stream order and all ascending-Q additions are preserved. Existing
pair admission, projection bounds, explicit refusal/fallback behavior, full
DIIS storage, and independently expanded physical replay are unchanged.

## Evidence

`benchmarks/results/df-cc-native-copy-roundtrips-20261010/` retains identities,
all samples, reconstruction/build/qualification recipes, work counts, quantitative
gates, and a compact compliant numerical publication. The frozen candidate
parent is `d4b44b06489d44a517d3dfaecc6b7269f5b89b7a`; the baseline is the
already qualified pair-enabled implementation from #2178, not pre-folding code.

- 26 host proof/guard tests pass. Nine targeted CUDA solver checks and five
  complete native action checks pass. An additional generated original-versus-
  elided CUDA probe preserves exact output bits, canaries and sticky first
  errors for finite limits, signed zero, subnormals, nonfinite inputs, seeded
  faults, and longer copy chains. Memcheck reports zero errors.
- Eleven original generated artifacts and the pair CPU header remain
  byte-identical. The complete arena and production diagnostic ABI are intact.
- Finite Slurm job 2810 executes ABBA on the assigned node2 RTX PRO 6000 GPU.
  All four endpoints pass independent total-energy `1e-8`, (T)-energy `1e-10`,
  and expanded physical-replay `1e-10` gates. Total and (T) energy bits agree.
- Complete process wall medians are 165.499194 / 160.892676 seconds: an
  observed **2.78%** decrease. Complete CCSD medians are 69.408765 / 64.694207
  seconds: **6.79%**. There are only two fresh observations per selection.
- Every endpoint retains 20 iterations, 38 evaluations, 488 Q rows and 2,318
  primary Q8 tiles. Six omitted launches per tile remove exactly 13,908 launches;
  reported Q operations change from 186,538 to 172,630. GEMM/contraction work,
  Q scope, accumulation/projection observations, and numeric capacity are equal.
- Eliminating two copies in three `o^2 v^2` chains avoids
  `4 * 8 * 3 * 9^2 * 221^2 * 488 * 38 = 7,042,781,551,104` logical read+write
  bytes. This is not a DRAM-traffic measurement or a reduced-capacity claim.

Master advanced to `ee0ca19d5cade2eddcbd05585af2739ebe3f7619` during this
iteration. The new SCF-force/implicit-Hcore and optional cuTENSOR-discovery
changes do not modify these paired CC actions; cuTENSOR is OFF in the build.
The retained receipts remain frozen-parent qualification, not latest-master
recomposition. No unrelated suite is rerun merely because master advances.

After measurement, #2178 merged as
`10a86458edca7aab03de4742f56ca87d495a858e`. The follow-up branch is aligned
with that master. Comparing its tensor compiler, paired generators, CC native
sources and owner-test support against the frozen parent shows identical bytes,
so the bounded patch applies without a new scientific change. This source
comparison is not a relabeling or rerun of the retained endpoint campaign.

## Rejected alternatives and consequences

Generic TensorIR add folding would change interpreter semantics. General alias
or donation-based copy propagation would require a different allocation and
lifetime proof. Recoloring the arena could trade lower capacity for unrelated
traffic changes. None is needed when the final pointer already names the
producer's populated storage. Keep the allocated intermediate slot for now.

Speculative tiled transposes are a separate opportunity: the measured paired
transpose kernels total approximately 1.98 traced seconds, less than these
redundant copies. Do not mix a transpose/layout experiment into this proof or
reuse traced timing as untraced endpoint evidence.

Revisit if scalar finite-auditing semantics, singleton native expressions,
execution ordering, symbolic arena coloring, or retained/external kernel
schedules change. Do not silently broaden this proof to reductions, borrowed
storage, lower precision, generic views, or cross-iteration reuse.

## References

- `2026-10-10-df-cc-occupied-spectator-pairs.md`
- `python/generativeqc_compiler/tensor/native_arena.py`
- `tests/python/test_tensor_native_copy_roundtrips.py`
- `tests/native/native_copy_roundtrip_probe.cu`
- `docs/maintainer/performance_engineering.md`
