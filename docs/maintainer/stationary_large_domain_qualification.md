# Large-domain stationary CUDA qualification

The generic public CUDA DFT force path admits a finite complete grid through the
shared stationary compiler/resource plan. Native resource ceilings remain 128
atoms, 1024 AOs, and 16384 primitive records. These ceilings describe allocation
and scheduling capacity, **not a numerical or performance qualification** for
every method, basis, or molecule inside them.

## Admission and ownership

- Public complete forces do not inherit the private diagnostic's 1,000,000-point
  and 100,000,000-total-pair-visit guards. Direct diagnostic calls retain these
  defaults and may supply explicit guards. `None` admits the finite complete grid.
- Every route retains 512 MiB additional-device and 256 MiB additional-host
  numeric budgets. These are consumer-local bounds, not process RSS or the total
  pre-existing SCF-owner budget. The concurrent Direct paired one-electron
  host-staging reserve is included and checked against its native report.
  Snapshot exports, driver/modules and allocation
  exclusions remain separately reported by the existing contracts.
- Above the old 32-atom / 128-AO / 4096-primitive diagnostic envelope, complete
  prepared native integral derivatives are mandatory. A missing provider,
  unavailable result, insufficient budget, or provider failure cannot select the
  AO-quartet fallback. The shared HF/DFT derivative owner remains authoritative.
- Deferred geometry submissions drain after at most 64 grid tiles or 100 million
  pair visits, whichever is tighter. The ordinary 256-point tiles, point order,
  final-state identity checks, equations and precision are unchanged. Tail tiles
  visit every point once. The native measured point and pair-visit totals must
  equal the admitted totals before publication.
- Cooperative Becke remains opt-in. The retained full-pair schedule covers 2–32
  atoms; a four-row tiled schedule covers 33–128 atoms without quadratic shared
  storage. The tiled schedule charges 23,952 bytes per block at 96 atoms and
  32,144 at 128, including control state. Actual-device resource rejection keeps
  the admitted generic path. The production default is unchanged, and the new
  tiled schedule remains pending real-GPU correctness and performance qualification.

For `P` points and `A` atoms the complete native pair-visit census is
`(1 + 2*P) * A*(A-1)/2`. Bounded submission does not reduce this total work.
`grid_work_plan` reports it beside point/tile/chunk counts and the maximum
per-chunk work. Complete-endpoint benchmarks retain the route, resource bounds,
and work record independently for cold, warm, moved and moved-warm endpoints.

## Source-only DFT-MP capacity audit

From a clean checkout, run the current-source audit in a fresh interpreter:

```bash
PYTHONPATH=python:. python tools/dft_mp_v1/qualify_capacity.py --output /tmp/dft-mp-capacity.json
```

The `stationary-capacity.v2` report audits the current compiler, native allocation,
endpoint and public forwarding contracts against the unchanged frozen DFT-MP-v1
inputs. It separately reports native owner capacity, the small AO-task fallback
envelope, private diagnostic guards, public complete-grid overrides, mandatory
submission windows and per-method host/device bounds. The Direct paired host
reserve is charged before host admission. The device bound includes the paired
reserve; the stationary/grid-only bound and remaining native-provider allowance
are also explicit. Logical AO-task reference counts are method-specific and are
not reported as executed native work. Required native-provider availability and
budget qualification remain `NOT_RUN`.

The frozen DFT-MP grid is distinct from the PBE0 ladder below: its 96-atom case
has **7,962,624 points and 72,619,135,440 pair visits**. With current defaults its
windows contain at most 10,752 points / 98,058,240 pair visits. All 35 frozen
FP64 force rows pass the current static capacity preflight; larger domains must
still return complete native integral derivatives at runtime. This result proves
neither GPU execution nor numerical or performance qualification. Private
`diagnostic_admission` retains its own whole-grid failures separately.

Package inventory is availability-only evidence: semilocal public rows select
packaged AOT, whereas global hybrids select the runtime-compiler path. Neither a
package declaration nor static admission qualifies its executable. Historical v1
reports, frozen inputs and receipts remain unchanged; do not reinterpret their
older admission fields as v2 results.

## Targeted large Becke schedule probes

The opt-in `test_cuda_large_tiled_becke_bounded_geometry_probe` test in
`tests/python/test_dft_complete_cuda.py` uses 67 fixed-density PBE0 geometry points
at 96 and 128 atoms. It compares generic and tiled cooperative execution with
cached/direct center geometry, tight/full point capacity, irregular tails,
repeated and changed geometry. The tests require the existing allocated-device
fixture, explicit CUDA test gate and matching compiler/library setup; they do
not submit a job or run the complete benchmark ladder.

Tiled execution retains two full pair-state evaluations per unordered pair per
point. Only the retained 2–32-atom schedule cuts these evaluations to one. Neither
schedule changes the logical pair-visit census, grid, precision or acceptance
tolerances. Run Compute Sanitizer memcheck/initcheck/synccheck on the targeted
probes before treating the new schedule as device-qualified. Host-thread checks,
resource admission and source work counts alone do not establish GPU speedup.

## Frozen PBE0 acceptance ladder

Use the existing `benchmarks.readme_pbe0` protocol and its checked-in offline
`benchmarks/results/pbe0-def2-svp-20261001/def2-svp-ho.json` basis:

| Atoms | Spherical AOs | Points, 48×16×32 per atom | Complete pair visits |
| --- | --- | --- | --- |
| 24 | 192 | 589,824 | 325,583,124 |
| 48 | 384 | 1,179,648 | 2,661,287,016 |
| 96 | 768 | 2,359,296 | 21,516,784,080 |

Host-only planner/allocation tests are prerequisites, not substitutes for the
following current-head real-device acceptance:

1. Build the exact candidate with the normal tuned CUDA build. Record its source
   identity, library SHA-256, compiler flags, architecture and loaded generated
   artifact identities. Do not compare timings against an unrelated binary.
2. Keep FP64, full Direct PBE0/def2-SVP, spherical representation, the full moving
   `48×16×32` grid, energy/density tolerances `1e-12`/`1e-10`, native/reference
   screening `1e-12`/`1e-14`, and 100 SCF iterations exactly as recorded by the
   existing protocol. No DF, basis truncation, grid coarsening or hidden CPU
   derivative fallback is admissible.
3. Run 24 atoms, then 48, then 96. At each size first require a complete,
   independently converged GPU4PySCF reference at both original and displaced
   geometries. Historical 24/48/96 references failed to converge in 100 steps;
   preserve those failures. If this recurs, stop that acceptance run and address
   reference convergence separately. A failed oracle is not a native timing or
   a numerical pass.
4. Require every native endpoint to converge and return finite energy and all
   host forces. Require maximum energy error ≤`1e-8` Eh and force error ≤`1e-7`
   Eh/bohr against that exact independent reference, across cold, five frozen
   density warm repeats, moved and five moved-warm repeats. Preserve failures,
   timeouts, actual iterations and branch-conditioned timings.
5. Inspect each endpoint's `native_force_components`: require
   `prepared-native-complete`, `native_integrals_required=true`, complete grid
   counts above, bounded chunk work, and no generated AO-quartet fallback work.
   Check actual peak/host/device budgets and cumulative-to-per-call work deltas;
   separately report pre-existing SCF state and snapshot storage.
6. On overlapping small cases, compare the candidate to the previous head with
   the same scientific inputs. Compare the private fixed/one-window and
   multi-window geometry schedules using the private diagnostic's explicit
   `max_pending_grid_tiles` / `max_pending_grid_pair_visits` controls; include a
   non-divisible final tile.
   Exercise host-grid and resident-grid branches, prepared reuse and changed
   geometry, constrained memory and an injected late error. Do not count
   instrumented/profiled runs as clean performance measurements.

Inside an already allocated finite GPU job, set `GENERATIVEQC_LIBRARY` to the
recorded candidate library and use the existing driver. This command does not
allocate a machine or submit a job:

```bash
export PBE0_BASIS_FILE="$PWD/benchmarks/results/pbe0-def2-svp-20261001/def2-svp-ho.json"
export PBE0_BENCHMARK_OUTPUT="$PWD/.artifacts/pbe0-large-domain-current-head"
export PBE0_POINT_TIMEOUT=1800
for atoms in 24 48 96; do
  PBE0_ATOMS="$atoms" bash benchmarks/run_pbe0_benchmarks.sh || break
done
```

The timeout is only an observation bound, not a scientific setting or permission
to omit slow phases. Increase it only within the actual allocated job's time and
resource limits. A 96-atom production/performance claim remains pending until
this ladder passes on a real GPU.
