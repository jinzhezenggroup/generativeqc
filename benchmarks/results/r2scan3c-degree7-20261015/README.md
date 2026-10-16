# Rejected degree-seven order-six experiment, 2026-10-15

The isolated cooperative six-only consumer is **removed**, not promoted.
Both README timing panels and production defaults remain unchanged. The
archived selector is not an available production option.

Unlike the earlier mixed-consumer experiment, this run builds a complete
source-matched Release/sm_120 image. Native identity
`3035ff1931cb74c8a39909d28d47ab6815d1f20f42f250a31963e072c3fa071d`
matches the frozen 1742-input inventory, every Python/native consumer hash
matches, and both arms use library SHA256
`018f81de976f28bd9147bdbcf5329cdb5adb524cf672688dadb74ca69da91ddc`.
Official matching stationary AOT manifests are rebuilt without editing them.

## Method and result

The candidate keeps the authoritative Coulomb IR but emits spatial degree seven
for angular order six: 1193 nodes / 19 levels / 120 roots, instead of degree
eight's 1991 / 24 / 165. A separately bounded 256-lane class-major consumer owns
only dd×pp, dd×ds and dp×dp. Orders zero through five retain their 128-thread
consumer; seven/eight and mixed-f ownership stay unchanged. Explicit refusal
guards retain the incumbent. This is reduced mathematical work, not a basis,
precision, threshold, grid or SCF change.

Both arms use canonical spherical def2-mTZVPP, D4 and gCP. Complete cold, five
warm, moved and five moved-warm calls include preparation and host force return.
Every returned E/F array passes the unchanged independent 1e-8 Eh / 1e-7 Eh/Bohr
gates. All 48 calls pass; maximum errors are 1.524e-11 Eh / 3.519e-11 Eh/Bohr.
SCF work is retained per sample and never normalized.

| Atoms | Arm | Cold (s) | Warm median [min, max] (s) | Moved (s) | Moved-warm median [min, max] (s) |
| ---: | --- | ---: | --- | ---: | --- |
| 12 | control | 9.178 | 1.608 [1.604, 1.640] | 6.714 | 1.618 [1.617, 1.629] |
| 12 | degree seven | 9.077 | 1.665 [1.657, 1.702] | 6.644 | 1.662 [1.660, 1.665] |
| 48 | control | 84.139 | 22.496 [22.398, 22.568] | 62.893 | 22.522 [22.434, 22.657] |
| 48 | degree seven | 94.738 | 32.635 [32.630, 32.641] | 73.033 | 32.635 [32.633, 32.639] |

The 48-atom warm regression is approximately 45%. No 96-atom candidate or DF
speedup is claimed; a failed small/medium gate does not justify promoting a
larger unmeasured case. Sharing a shorter ladder alone is insufficient.

## Validation and retention

- Slurm node5/1614: standalone raw-AD, independent symmetry-orbit force and
  exact primitive-work qualification, including both degree-seven/eight DAGs.
- Slurm node5/1615: six independent Libcint derivative cases and complete
  12-atom endpoints, including actual six-only recurrence work.
- Slurm node5/1616: both complete 48-atom populations.
- Host suite: 116 pass with `CMAKE_PREFIX_PATH=/home/jzzeng/miniconda3` for
  the existing incremental-CMake dependency fixtures. Earlier attempts failed
  OpenSSL/LibArchive discovery; the rerun log records the corrected environment.
- Slurm node5/1617: a separate intrusive DF/48 cold/first-warm capture passes
  numerical gates. cProfile and CUDA event timings are diagnostic, not clean
  complete timing populations or a measured DF speedup.

`observations.json.xz` retains all complete source-qualified samples, actual
work and independent cold/moved geometry references. `receipts.json.xz` retains
source/build/cache/test/scheduler records and the DF diagnostic. The exact
experiment is `experiment.patch.xz` against master snapshot `cb8781fb1`;
`manifest.json` binds archive hashes. Raw directories and the matching binary
remain under `.artifacts/r2scan3c-degree7/` locally and
`/data/jzzeng/gqc-r2scan3c-degree7-20261015` on n5. Do not overwrite this frozen
source/runtime with a later candidate.

The rationale and required boundaries are in
`.agents/notes/rejected/2026-10-15-r2scan3c-degree7-order-six.md`.
