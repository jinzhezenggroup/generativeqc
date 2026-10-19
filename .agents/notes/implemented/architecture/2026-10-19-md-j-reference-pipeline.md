# Decision: integrate a reference-aligned MD J engine through exact-mask compatibility

Status: implemented (opt-in; real-device qualification remains required)
Date: 2026-10-19

## Problem

Porting only an upstream recurrence would leave primitive layout, density
contraction, hierarchy, source scheduling and projection mismatched. Conversely,
copying GPU4PySCF's float32-log screening changes GenerativeQC's exact public-AO
mask and directional density budget. The existing E/density/project pipeline and
reciprocal reuse already solve important parts of the dataflow and must not be
replaced merely to make the change look larger.

## Decision

Pin official GPU4PySCF at 9af748a71ca8f4881c604332c2f0cc1632e033ac and integrate
its architectural decomposition as a native J-value source engine. Keep original
public-shell/primitive identity and sparse weighted E transforms; sort a separate
primitive mapping, pack Hermite pair locations, refresh six hierarchy levels and
use generated compact radial math in one reciprocal tiled consumer for all 15
s/p/d pair-total-angular domains. The compiler owns math; native code owns all
storage, synchronization, task cursors and dispatch. Preserve Apache attribution,
license and pinned input hashes without adding a runtime upstream dependency.

The exact-mask compatibility adapter admits only the same uniform public region.
The retained residual source owns partial masks, and the unchanged global-pair
allowance governs each density direction. Linear FP64 hierarchy bounds preserve
leaf multiplication association. qD alone is insufficient at rounded thresholds,
so density maxima remain available for conservative coarse rejection. Sorted
active prefixes retain the incumbent geometry census without dropping full
packed storage needed by transforms and residual work.

The selector remains opt-in until real-device correctness, sanitizer and endpoint
performance gates pass. Optional hierarchy capacity cannot increase the existing
128-MiB allowance. A hierarchy-cap miss retains incumbent MD; an optional owner
allocation failure rolls back safely to the normal provider. Existing K,
precision and derivative dispatch remain separate consumers.

## Rejected alternatives

- Directly adopting upstream log q/qd admission: changes public-AO scientific
  semantics and is not a pure architecture port.
- Copying upstream same-shell half factors into original contracted-shell IDs:
  upstream shells are decontracted, so the symmetry factors are not interchangeable.
- A disconnected Python reference engine: gives no production lifetime or dispatch
  benefit and hides integration failures.
- Replacing cached weighted E with an explicit dense decontraction matrix: adds
  data movement and risks normalization/identity regressions without changing the
  mathematical factorization in the supported domain.
- Treating bounded storage as bounded useful work: full primitive tiles restore
  unnecessary quadratic search; retain geometry compaction and expose tile counts.
- Default promotion from host tests: does not establish native CUDA compilation,
  runtime behavior, graph replay, or full-size performance.

## Invariants

Public AO masks, global directional allowance, nonsymmetric/UKS density folding,
full sparse-transform identity, same-source multiplicity, owned-stream ordering,
optional rollback and explicit fallback diagnostics are contracts. A rejected tile
requires a block barrier before shared control is overwritten. Async staging must
outlive its stream copy even on exceptions. Windowed replay must not preload
uninitialized density outside the requested systems. Shared CUDA control types
must be trivial; host compilation alone does not enforce CUDA's initialization
restriction.

## Evidence and consequences

Permanent executable tests cover generated compact recurrence, independent
libcint full ERI/J, actual E/density/project composition, hierarchy boundaries,
64-bit task decoding, full native-body cooperative scheduling, active prefixes,
ragged tails and optional allocation/copy/preparation faults. Instrumented host
negative controls caught a missing rejected-tile barrier and out-of-window reads;
these complement, not replace, real device sanitizers. Tile shapes are deliberately
conservative and untuned. Full qualification procedures and the exact upstream
correspondence matrix live in `docs/developer/md_j_engine.md`.

## Revisit when

Real-device measurements support better tile/warp allocation or safe default
promotion; high-angular shells or multiple independent densities are added with
independent oracles; or the user explicitly approves a different screening
contract as a separate scientific behavior change.
