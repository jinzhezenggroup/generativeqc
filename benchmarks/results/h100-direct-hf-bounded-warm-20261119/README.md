# H100: bounded Direct HF force-page tail break and resident psss skip

The README Direct HF96 warm replay (water-96, def2-SVP, 768 AOs) performs one
Fock build from the frozen cold density, then forces. A node-traced nsys and a
native host trace of that replay found two removable costs.

- **Row rescans.** Bounded exact-class compaction and the ssss/psss low-order
  force page kernel test each ket of a bra row against a row-fixed density tail.
  A failure only `continue`d, so every thread scanned the rest of its stride.
  Each class segment of the bounded pair order is Schwarz-descending for the
  current geometry, so both kernels now `break` there. The exact predicate still
  decides every ket that passes.
- **Unused host packing.** `pack_host_batch` rebuilt the resident psss task
  table on every execute: one record per psss bra and 128-ket ss chunk, 5.65M
  records (68 MB), 42 ms of host time before the first CUDA call. Only the
  fixed-topology queue reads it. A topology whose quartet count exceeds the
  fixed tile grid always runs bounded streaming, so the table is no longer built
  there; the ket inventory is unchanged.

## Measurements

Fresh consecutive processes ran on one physical H100 80GB in qz Job
`gqc-h100-opt-1119-qc`, in the order base, candidate, base, candidate. Every
tree is master `cddde09a0` plus the named commits.

| Comparison | Base | Candidate |
| --- | --- | --- |
| tail break | master | + tail break |
| resident skip | + tail break | + both |
| combined | master | + both |

Combined, Direct HF96 r0/r1 arms, two rounds:

| Case (Direct) | master cold (s) | both cold | master warm | both warm |
| --- | --- | --- | --- | --- |
| HF 96 | 6.434-6.492 | 6.221-6.267 | 1.375-1.382 | 1.294-1.300 |
| HF 48 | 4.247, 4.279 | 4.251, 4.284 | 0.365, 0.365 | 0.364, 0.365 |
| PBE0 96 | 6.761, 6.847 | 6.830, 6.837 | 1.765, 1.771 | 1.771, 1.773 |

Warm values are medians of three calls. HF48 and PBE0-96 never select this
route and are controls.

- **Single-change A/Bs, HF96:**
  - Tail break: warm 1.376-1.384 → 1.345-1.349 s, cold 6.45-6.48 →
    6.31-6.34 s.
  - Resident skip: warm 1.346-1.353 → 1.294-1.295 s, cold 6.30-6.35 →
    6.23-6.29 s.
- **Counts and qualification:** iteration and Fock counts are identical
  (HF96 11 cold, 1 warm), and every arm of all three comparisons qualifies.
- **Forces:** same-phase forces of base and candidate agree within
  6.3e-13 Eh/Bohr in every comparison.
- **Files:** `endpoint.json` holds every arm, its samples and the comparison
  tables. `results-<comparison>.json.xz` adds all force vectors.

## Gates

- `memcheck.txt`: compute-sanitizer 12.9.79 memcheck of one cold and one warm
  README Direct RHF energy+force call on the measured candidate.
  - water-12 with `GENERATIVEQC_BOUNDED_DIRECT_STREAMING=force`: 0 errors.
  - water-96 on its default bounded route: 0 memory-access errors. The 5
    reported errors are CUDA API results inside the existing
    `probe_xsyev_batched_device_launch_graph` capture probe, which this change
    does not touch: `cusolverDnXsyevBatched` is not capturable (900), then the
    invalidated capture (901).
- `native-tests.txt`: `generativeqc_hf_resource_layout_tests` passes,
  including the new bounded-topology packing check, plus host source-structure
  tests.

These measurements do not qualify any other GPU, and neither change has an
architecture switch.
