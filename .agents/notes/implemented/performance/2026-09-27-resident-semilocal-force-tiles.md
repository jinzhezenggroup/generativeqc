# Decision: resident same-stream semilocal force tiles use shared geometry lowering

Status: implemented
Date: 2026-09-27

## Problem

The CUDA stationary DFT force path already owns one generated scientific geometry
lowering for LDA, GGA, meta-GGA and composed semilocal functionals. The remaining
cost in the semilocal force slice was increasingly dominated by execution
orchestration around that shared lowering:

- a method-named feature-reuse seam was first exposed through the WB97M-V path;
- local grid tiles synchronized repeatedly for profiling sections even when no
  host feature/AO output was requested;
- the grid producer published its device error to the host before a same-stream
  stationary consumer could run;
- the stationary geometry consumer synchronized again after every tile;
- full-local stationary tiles repeatedly uploaded an explicit
  `[0, 1, ..., nao - 1]` AO map and gathered the complete resident density into
  a second local matrix even though the map was exactly identity.

These costs apply to the shared semilocal execution structure rather than to one
functional.

## Decision

Keep the existing generated stationary geometry pullback as the sole scientific
owner and optimize the runtime boundaries around it.

The resident execution contract is:

1. Feature reuse is selected by an ingredient contract, not by a method name.
2. Device-only feature leases may skip intermediate profiling fences.
3. A qualified same-stream consumer may explicitly own grid-error publication.
   In that mode the grid keeps its sticky device error resident and
   `GridTaskView.error` is propagated into the stationary error slot before
   AO/features are read.
4. Stationary semilocal geometry may enqueue multiple tiles on the borrowed grid
   stream and publish errors at one explicit drain instead of one host fence per
   tile.
5. A null `GridTaskView.ao_ids` denotes the complete identity AO map when
   `nactive == nao`. The grid then consumes the resident global density directly,
   skipping AO-map H2D and the full-density gather.
6. Explicit sparse/local AO maps retain their existing validation, gather and
   scatter behavior.

Detailed device profiling keeps the synchronous geometry path so phase attribution
does not silently change meaning. Host-output evaluation, ordinary XC consumers,
COSX and other callers that do not opt into consumer-owned error publication keep
their synchronous correctness boundary.

## Rejected alternatives

### Add a new native LDA/PBE/r2SCAN geometry implementation

Rejected. The generated stationary CUDA path already owns LDA/GGA/meta-GGA
geometry response, including tau. Reimplementing a second native scientific path
would split numerical ownership and make later functional additions require
parallel maintenance.

### Special-case WB97M-V

Rejected. WB97M-V exposed the redundant traversal/synchronization problem but is
only a stress composition. The reusable boundaries are ingredients, resident grid
state and geometry-response ownership.

### Remove every synchronization unconditionally

Rejected. Explicit host outputs require publication, detailed profiling relies on
synchronized event timing, and consumers that do not propagate the producer error
cannot safely borrow unvalidated resident buffers.

### Materialize a permanent identity AO map on device

Rejected. AO evaluation and factor gathering already accept a null map as identity.
Using a null sentinel removes both the upload and the duplicate map storage while
keeping explicit sparse maps unchanged.

## Invariants

- Scientific XC/geometry arithmetic remains compiler-owned and unchanged by these
  runtime optimizations.
- A deferred grid-error lease is admitted only for a device-only local feature
  execution with an explicit qualified same-stream consumer.
- The consumer must inspect or propagate `GridTaskView.error` before reading the
  borrowed AO/features.
- A null AO map is legal only as the complete identity map; partial maps require
  explicit sorted unique in-range IDs.
- The borrowed grid owner must outlive any stationary consumer using its stream.
  Exception unwinding therefore closes/drains the stationary consumer before the
  grid owner.
- Explicit sparse-map semantics, changed-geometry invalidation, failure isolation
  and numerical acceptance gates remain unchanged.
- Performance qualification must use complete endpoints and work/transfer/sync
  counters, not only microbenchmarks.

## Evidence

The stack carries structural and GPU tests for:

- ingredient-driven resident feature leases;
- deferred stationary geometry enqueue/drain ownership;
- removal of intermediate grid profiling fences on device-only leases;
- producer device-error propagation into stationary geometry;
- ordinary task/XC paths remaining synchronous;
- identity-map removal of AO-map upload and full-density gather;
- numerical equivalence of implicit identity and explicit full AO maps;
- close-order safety for the borrowed grid stream.

No endpoint speedup is claimed by this note before matched GPU qualification.
The expected work reduction is structural: fewer host synchronization points,
one fewer AO-map H2D for full-local tiles, and no O(2*nao^2) density gather for an
identity map.

## Consequences

The grid/runtime ABI gains two explicit reusable semantics: consumer-owned error
publication and null-as-full-identity AO maps. They make the semilocal stationary
path more device-resident without adding functional-specific science.

The deferred path has a stricter lifetime contract, so future consumers must not
opt in merely for speed. They must prove same-stream ordering and error propagation.

## Revisit when

- a consumer needs deferred publication across multiple streams;
- local AO screening makes the full identity map uncommon;
- a native whole-grid scheduler can eliminate host tile submission entirely;
- matched endpoint profiling shows remaining cost is scientific kernel work rather
  than orchestration/data movement.

## References

- #1423 shared CUDA DFT force productionization
- #1479 resident semilocal geometry response
- #1488 ingredient-driven resident feature lease
- #1494 deferred stationary geometry drain
- #1495 device feature lease profiling-fence removal
- #1498 grid device-error propagation
- #1506 deferred grid error publication adoption

Agent: ChatGPT
Model: GPT-5.6 Sol
