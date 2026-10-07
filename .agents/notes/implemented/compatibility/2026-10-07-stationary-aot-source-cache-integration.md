# Decision: compose exact AOT admission with lazy stationary source reuse

Status: implemented
Date: 2026-10-07

## Integration boundary

Integrate the reviewed AOT/provenance head
`5cb61b7972499e34d3961d97de99959b198062a1` with master
`d84e4006c611a0fdfbf6e12cb9c1ff891e715ba6`, including the landed #2080
semantic source cache and the complete native solver/weighted-Gram stack.
No native source or header is selected from the older PR base: every `src/`
and `include/` file remains identical to this master parent.

Exact catalog-covered plans retain fail-closed sealed AOT loading, source-only
compatibility checks, coverage validation without reduction IR, and replay of
build-time weight provenance. Unrepresented plans retain #2080's lazy integral
source provider and wrapper cache, including normal binary validation and the
explicit source-cache-disabled path. Neither cache runs in the AOT branch.

## Source contract reconciliation

The whole endpoint owner had independently changed on both parents. Re-audit
the combined function rather than accepting either frozen checksum:

- AOT parent: `a5759c161a2461143c871a632b7ea17513faa7eacc36fb63a621ec58e62f7dba`
- Master: `a43077fab9b73dbe52beacf12442fb2cd1e104e0eff296718b7251ecb4c1b22c`
- Integrated: `c4d9a037cee42a996325d5c8b0e02ef1fe93c3682a56a610f4444d5e6ba06b53`

Relative to the AOT parent, only the JIT-only lazy provider/cache options and
source-cache telemetry are added. Relative to master, only the coverage call
and AOT weight-provenance replay replace their IR-generating predecessors.
Native admission, complete-source requirements, work windows, allocation and
host reserves, and numerical tolerances are unchanged. The capacity qualifier
and its independent expected receipt are pinned to this combined source span;
their mutation tests remain enabled.

## Hardware-gate instrumentation

The AOT GPU gate previously patched eager emitters imported into the runtime.
#2080 removes those aliases. Guard the actual integral owner functions and the
lazy runtime provider instead, while retaining compiler, wrapper, weight-IR and
reduction-IR prohibitions. A host test invokes each installed guard so stale
instrumentation fails even when hardware qualification is intentionally skipped.

Scientific source/weight/plan comparisons cover all ten profiles against both
parents. Source equality and the host cache/admission tests are integration
evidence, not new GPU numerical results or an endpoint-performance claim. The
parent qualification receipts retain their original source/device scope.
