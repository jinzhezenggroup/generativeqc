# Decision: end bounded Direct HF page rows at the Schwarz tail and skip unused resident psss tasks

Status: implemented
Date: 2026-11-19

## Problem

The README Direct HF96 warm replay (water-96, def2-SVP, 768 AOs) was the last
H100 case in the three-process cohort where GPU4PySCF was faster: 1.380 s
against 1.353 s. The replay is one Fock build from the frozen cold density plus
forces. A node-traced nsys and a native host trace of it showed two removable
costs.

- **Row rescans.** Bounded exact-class compaction and the ssss/psss low-order
  force page kernel test each ket of a bra row against a row-fixed density tail
  (`bounded_page_density_tails`). A failing ket only `continue`d, so every
  thread scanned the rest of its stride.
- **Unused packing.** `pack_host_batch` rebuilt the resident psss task table on
  every execute: one record per psss bra and 128-ket ss chunk, 5.65M records
  (68 MB). That cost 42 ms of host time before the first CUDA call, and the
  arena reserved space for the table. Bounded streaming never reads it; its
  plan already sets `resident_psss_task_count` to zero.

## Decision

- **Tail break:** both bounded page kernels `break` at the first failing
  row-tail test (Fock: the |D| maximum tail; force: the density-product tail).
  Each class segment of `bounded_stream_shell_pair_order` is stable-sorted
  descending by `shell_pair_bounds` whenever the geometry changes, and the tail
  depends only on the bra pair. Every later ket of the stride therefore fails
  the same test. The exact quartet predicate still decides every ket that
  passes.
- **Resident skip:** `pack_host_batch` skips the resident task table, keeping
  the ket inventory, when `direct_topology_requires_bounded_streaming` holds for
  the total quartet count. Such a topology always executes through bounded
  streaming. Fixed-topology packing is unchanged.

## Rejected alternatives

- **Warp-level row scheduling** of the same two kernels, with one warp per bra
  row: HF96 warm 1.347 → 1.395 s and HF48 0.363 → 0.392 s.
- **Compile-time ssss/psss specialization** of the low-order page kernel, to
  avoid the runtime class switch: the ssss and psss source bodies themselves
  need 254 registers, so occupancy did not change. HF96 warm stayed at
  1.341-1.353 s against 1.341-1.351 s.
- **An sm_90 tuned AOT profile:** the README cohort builds
  `GENERATIVEQC_AOT_PROFILE=portable_cuda`, so a tuned profile would not reach
  it. The tuning driver's Fock harness also fails on master ("Fock benchmark
  force-section markers changed unexpectedly").

## Invariants

- The break is exact only while each bounded class segment is
  Schwarz-descending for the current geometry. A source-structure test checks
  the stable sort inside the geometry-changed block together with the `break`
  gates.
- A fixed-topology plan that can enable the resident-bra route must still
  receive the complete task table.

## Evidence

`benchmarks/results/h100-direct-hf-bounded-warm-20261119/`.

- **Combined:** HF96 warm 1.375-1.382 → 1.294-1.300 s, cold 6.43-6.49 →
  6.22-6.27 s. HF48 and PBE0-96 are unchanged.
- **Correctness:** Fock counts are identical, every arm qualifies, and
  same-phase forces agree within 6.3e-13 Eh/Bohr.
- **memcheck:** water-12 forced through bounded streaming reports 0 errors.
  water-96 reports 0 memory-access errors; its 5 API errors come from the
  existing xsyev capture probe.

## Revisit when

- Bounded streaming gains a resident psss consumer: build the table lazily for
  that plan instead.
- The bounded pair order stops being Schwarz-descending per class segment:
  return to `continue`.
