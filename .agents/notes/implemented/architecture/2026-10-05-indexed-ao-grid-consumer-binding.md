# Decision: bind existing spatial and resident force maps to indexed grid domains

Status: implemented; correctness-qualified, P0-C performance acceptance pending
Date: 2026-10-05

## Scope

This extends the [indexed AO/grid foundation](2026-10-05-indexed-ao-grid-domains.md).
It is not completion of #1893 P0-C and is not a claim that ordinary native KS
CSR maps already execute through the generic TensorIR provider.

## Contract

The same data-only `AoGridBlockLayout` is admitted before CUDA enqueue for
host-point evaluation, borrowed feature tasks, detached feature publication,
spatial XC tasks, and resident nonlocal force composition. The native lease
exposes the producer's identical descriptor, including existing expiration.
Shape, indexed/dense routing, basis identity, evaluated derivative order, map
capability, and supplied epochs must agree with the execution owner.

`SpatialTask.block_layout` uses the existing immutable AO and point maps.
Its point offset addresses the task's ordered point permutation, not contiguous
rows of the original grid. A spatial map certifies only complete through-order
jet domains; one isolated Hessian entry cannot certify a deriv=2 force map.
The validated inventory remains the owner of the envelope certificate. The
descriptor neither reallocates coordinate arrays nor certifies a public forged
inventory without that inventory's existing preparation validation.

Keep `_tiles()`'s existing two-element inventory contract. Typed consumers opt
into `with_layout=True`; default consumers inspecting last-nonempty tiles do not
construct unnecessary descriptors. Slurm 6067 caught a missed XC consumer after
the first implementation changed tuple arity. Restore the old default and add
a host regression comparing both inventories. Do not patch every consumer into
a new tuple protocol merely to attach metadata.

## Executed duplicate removal

The owned restricted-spin witness already computes one GEMM and copies its
ordered output panels. Previously it still gathered both identical local density
matrices. Gather one matrix under that same witness; unrestricted, host-uploaded,
or otherwise unqualified inputs retain both matrices. No numerical-equality
heuristic, method-name dispatch, smaller arena, changed screening, new cache, or
projection-FLOP reduction is introduced. The second projected-panel copy remains.

`density_gather_elements` counts the actually executed single-spin gather.
`AoGridBlockLayout.logical_work` remains an explicitly two-spin logical graph,
not a claim about optimized native RKS execution. Qualification must distinguish
the two schedules instead of silently accepting either census. The endpoint
verifier's explicit `--density-gather-spins` selects the expected schedule.

## Evidence

- Frozen candidate source/native identity:
  `2f6d3972494674c9b043c18892594456a43fa619e14d674bfe623504f081f6c0`.
- Focused host gates: 172 pass; three explicit generated CUDA opt-ins skip.
- Compiler structure: 443 modules, zero dependency errors.
- ccache-launched native rebuild succeeds with independently matching source and
  embedded library identity; the transfer's binary hashes are independently checked.
- n1 Slurm 6070: 216 grid/spatial real-device gates pass. Restricted source
  ownership/replacement is checked bitwise across all feature masks and local
  map kinds. New actual map-producer tests qualify typed host/resident-point
  leases, stale-epoch rejection before enqueue, scatter, recovery, and expiration.
- The focused 19-case memcheck and racecheck repetitions both pass, with zero
  errors and zero race hazards/warnings.
- The native local-AO gate includes scaled-PBE PBE0, AO discovery, CPU E/V,
  spin features, nonlocal feature leases, capture, and bounded resources.

The full clean 48/96-atom paired endpoints in 6070 are still in progress when
this note is written. The distinct reference-domain observer in 6071 is
intrusive and must never supply clean endpoint speedup numbers. Its row counts
include GPU4PySCF's sorted/padded published local AO rows, not certified physical
AO support or measured DRAM traffic.

## Retained failures and remaining acceptance

6062 and 6066 passed their device gates and measured the old 48-atom control,
but failed because the portable snapshot omitted qualification helper scripts.
Both exact handles were terminal before proceeding. Their controls are checked
independently but are not pooled with 6070's paired GPU measurements. New
snapshots preflight all helper dependencies before chemistry execution.

Still required: ordinary native KS CSR producer integration; same-geometry
native/reference domain reconciliation; separately measured atom-gather and
owner-reduction time/traffic; complete force qualification; profitable dense,
tiny and high-occupancy routing; and the required complete-endpoint improvement
without disallowed cold/moved history regressions. This local duplicate removal
does not establish that the full reference gap is closed.

## Completed v4 receipts and separate v5 metadata replay

6070 completed on October 5, 2026 at 22:39:55 Asia/Shanghai, exit 0. All four
48/96 candidate/control campaigns pass 72 same-geometry reference pairings
each. Clean endpoint medians in seconds, with preparation included in cold:

| Atoms / variant | Warm | Moved-warm | Cold / Focks | Moved / Focks |
| --- | ---: | ---: | ---: | ---: |
| 48 control | 9.077594 | 9.060664 | 99.102496 / 25 | 58.610273 / 14 |
| 48 v4 | 9.222072 | 9.181609 | 121.243152 / 25 | 52.142950 / 12 |
| 96 control | 24.815707 | 24.830021 | 236.956292 / 29 | 123.380958 / 13 |
| 96 v4 | 24.793640 | 24.796159 | 243.808031 / 30 | 116.535938 / 12 |

48 warm regresses about 1.6%; 96 warm is effectively unchanged, not a material
reference-gap closure. The extra 96 cold Fock and slower cold endpoints prevent
calling this a qualified performance win. Preserve actual histories rather
than attributing all cold differences to the density gather. Candidate maximum
E/force errors are 8.185e-12 / 2.720e-11 at 48 and 1.055e-10 / 3.271e-11 at 96.
Raw records, explicit gather-spin gates and exact source/binary/device receipts
are retained in `.artifacts/n1-layout-native-6070/`.

The v5 replay change retains the immutable layout alongside an already retained
map. Every lookup still executes `select`'s binding, lifetime, tile and work-count
checks. Uncached dense budget/capability fallbacks retain no descriptor inventory;
there is no new scientific tensor or numerical map cache. A host-only mocked
9216-lookup replay reduces typed-minus-legacy median overhead from 32.209 ms
(v4) to 2.267 ms (v5). This is not an endpoint speedup and cannot explain the
entire v4 48-atom regression. The v5 independently matched source/native identity
is `65dec73d485ca144714d3b4b26b7b9f51b1b4efd586983ffc5b7b9ceb8329260`.
Its frozen n1 snapshot and Slurm 6082 are distinct from v4; do not relabel 6070's
GPU evidence as v5 qualification. Native CSR binding is a subsequent v6 change,
not part of either the v4 or v5 snapshots.

### Completed v5 endpoint qualification

6082 completed with exit 0 and 72 same-geometry numerical/work pairings for
each of its four candidate/control campaigns. All 216 grid/spatial tests and
both 19-case sanitizer runs pass. The descriptor replay cache is correctness-
qualified, but its complete endpoint result is not a material performance win:

| Atoms / variant | Warm | Moved-warm | Cold / Focks | Moved / Focks |
| --- | ---: | ---: | ---: | ---: |
| 48 control | 8.980650 | 8.952929 | 91.219142 / 23 | 51.298832 / 12 |
| 48 v5 | 8.947461 | 8.946205 | 107.480414 / 27 | 51.325698 / 12 |
| 96 control | 24.739989 | 24.601236 | 220.767273 / 27 | 135.505867 / 15 |
| 96 v5 | 24.617349 | 24.587611 | 227.810468 / 28 | 115.116059 / 12 |

Warm improves only 0.370% / 0.496%; moved-warm changes less than 0.1%.
The 48/96 cold extra Focks (23 to 27 and 27 to 28) and cold timing regressions
remain disqualifying for the intended complete-endpoint performance acceptance.
Do not infer that the metadata cache caused those numerical trajectories, nor
erase them because force sums pass. Maximum candidate E/force errors are
7.958e-12 / 2.744e-11 at 48 and 1.073e-10 / 3.320e-11 at 96.

The v5 candidate also passes 24 numerical/domain pairings against the separate
6071 census under the same strictly validated replay-count exception as v4.
Receipts live in `.artifacts/n1-layout-native-6082/` and
`.artifacts/n1-reference-census-6071-96/v5-candidate-96-same-geometry.json`.
`compare-p0c-clean.py` refuses unfinished jobs and intrusive profiler records;
it reports observations and actual Fock histories, not performance acceptance.
