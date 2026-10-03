# Decision: admit explicit active-AO maps on resident grid leases

Status: implemented (interface only; automatic screening is not enabled)
Date: 2026-10-03

## Problem

The resident-point Python entry point rejected every explicit AO map. Eight
stationary geometry entry points also required `nactive == nao`, although native
grid execution already gathers complete `D[I,I]` panels and emitted serial and
cooperative geometry kernels already map local columns through `view.ao_ids`.
Consequently callers could not use the established local-panel contract with
resident coordinates and analytic geometry consumers.

## Decision

Share the existing sorted/unique/in-range/capacity validator between host-point
and resident-point calls. Forward the selected host indices through the existing
native ABI; its charged device map buffer owns the uploaded copy. Preserve the
null full-identity route. Empty explicit maps publish zero grid features.

Use one native geometry-map shape predicate across synchronous, deferred,
explicit-owner, implicit-owner, and resident-weight entry points. A nonempty
partial panel requires a map; its global `nao` must still match the stationary
owner. The kernel retains global AO-to-atom routing and local panel strides.
No scalar mathematics, ABI layout, default mask, or tile policy changes.

## Invariants

- Gather the complete selected density submatrix, including cross-center terms.
- Preserve borrowed-stream ordering, density generations, producer errors, and
  lease lifetime. An index map does not authorize reuse across geometries.
- Screening policy is separate from explicit-map support. Bounding individual
  AO jets is not an energy/force error bound; default promotion needs independent
  complete endpoint, changed-geometry, resource, and numerical qualification.
- The full identity route remains available without a map upload or gather.

## Evidence

At `fb53bb548` composed with the frozen WB97M stack, n1 Slurm job 5538 ran nine
real RTX 5090 tests: resident full/subset/empty features and density-contracted
jets against retained independent CPU AO fixtures, and six restricted/spin
semilocal geometry comparisons against independent LibXC energy finite
differences. Noncontiguous selected indices span both LiH centers; CUDA receives
the original dense density while the oracle zeros omitted rows/columns.
All nine passed in 124.63 s. Three map tests under memcheck passed in 21.37 s,
with zero errors. Nineteen existing host protocol/emitted-kernel checks and the
compiler dependency checker also passed.

Those initial device tests used newly emitted current grid/geometry code and
the frozen `cac0727f3` native library for basis support. They are not a claim of
latest-master complete endpoint qualification. The exact source archive hash is
`2850c07c65c6f9adf490c1f159b2570f1e3da50fcaf64b327eacafd8a8c29647`;
the native library hash is
`08b276c3dbfd7b37ac5efa2f7e8a0c4518933673822b9428f872d48102a0d5d7`.
Fresh composed-native qualification is recorded separately when complete.

Slurm job 5540 measured every AO jet through order two on the full 48 x 16 x 32
grid, with def2-SVP spherical water clusters. At a *diagnostic* `1e-16` maximum
AO-jet threshold, the candidate work counts were:

| Atoms / full AOs | Tile | Mean retained AOs | Selected / dense sum(points x AOs squared) |
| --- | ---: | ---: | ---: |
| 24 / 192 | 1024 | 128.910 | 53.599% |
| 48 / 384 | 1024 | 152.845 | 18.740% |
| 96 / 768 | 1024 | 189.385 | 8.349% |
| 96 / 768 | 256 | 164.516 | 5.825% |

This scan evaluates all AOs first and measures possible column omission on the
actual grid. It does not qualify a screening policy or predict endpoint speedup.
Its separate 1 GiB diagnostic grid allowance does not change production budgets.
Fresh GPU4PySCF 1.8.1 initialization in this environment reports WB97M-V
functional 531 with `on_gpu=True`; historical endpoint records without that
marker are not retroactively relabeled.

The ignored local evidence is retained under
`.artifacts/resident-active-ao/`: source archives, source patch, binary hashes,
compiler commands, ccache before/after statistics, test/memcheck receipts,
`results/active-counts-v3.json`, and complete per-tile maxima in `maxima-*.npy`.
The two earlier count-script failures (JSON scalar conversion and insufficient
diagnostic budget) remain separate from successful interface-test evidence.

## Next qualification boundary

Use actual per-tile maps in complete WB97M calls, charge discovery and retained
maps, verify empty-panel geometry and independent total forces, then extend the
shared SCF grid path. Avoid adding another native AO or density-gather algebra.

## Fresh composition qualification

Slurm job 5541 completed successfully on the fresh `fb53bb548` + WB97M stack +
interface composition. Eleven interface/independent geometry tests passed in
158.36 s, including both spin modes with empty selected geometry panels. Seven
complete independent energy/force and geometry-rebuild/stale-state isolation
checks passed in 204.78 s, with no skips. The library SHA-256 is
`0c5b5f67d14c77352c903e8e0b037790ed2561165b96e10a94505c3d490d8f3b`.
Its source identity
`b359f13e392da960b1d456d6b7a269f5f4ee9fb256cf82638b7425d8737c6f2b`
was independently recomputed from all 1,349 manifest inputs at the committed
composition and matches the actually loaded library. All 442 compiler commands
were verified to invoke ccache; before/after statistics are retained.

This additional evidence qualifies the explicit-map interface and unchanged
complete endpoint composition. The separate fixed-geometry threshold experiment
is not part of this implementation and remains unpromoted.

## Capacity-report source binding review

Review identified that the legacy capacity report still fingerprinted the old
full-AO admission expressions in `stationary_geometry_external` and
`stationary_geometry_enqueue`. Re-auditing both complete bodies confirmed that
only AO-map admission changed: allocation, geometry lane count, reductions,
stream lifetime, complete point count and `G*A*(A-1)` pair counters are unchanged.
The public caller still plans `active_ao_capacity=n`; the census continues to
charge full global AO capacity and makes no screening or empty-tile work claim.

The report now binds `valid_geometry_ao_map` itself, in addition to the two
reviewed caller bodies. This prevents later weakening of the out-of-line
predicate from bypassing an unchanged caller fingerprint. New fail-closed
controls mutate global AO equality, active capacity and the required local map;
all are rejected before a report is issued. Existing allocation, execution,
Becke work, tile and counter negative controls remain intact. All 130 capacity
report tests pass locally; no numerical threshold or resource limit changed.
