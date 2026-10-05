# Weighted COSX ESP endpoint crossover

This receipt compares the complete weighted ESP region at 384/768 AOs, including
its extra publication pass when it uses the library recipe. Four qualification
masks isolate ESP and its interaction with projection/update:

| mask | projection/update | weighted ESP |
| --- | --- | --- |
| 0 | generated | fused generated |
| 4 | generated | batched matrix product + weight/finite publication |
| 3 | library | fused generated |
| 7 | library | batched matrix product + weight/finite publication |

Changing only ESP changes paired warm-median endpoint time by **−0.209% to
+0.0055%** across both projection/update choices. Changing all three operations
against generated changes it by **−0.242% to +0.034%**. These small diagnostic
endpoint differences do not qualify a production default promotion. The
production incumbent remains fused generated.

| AOs | tile | geometry | generated (s) | ESP library only (s) | ESP time change |
| --- | --- | --- | --- | --- | --- |
| 384 | 64 | original | 1.647615 | 1.646465 | -0.070% |
| 384 | 64 | moved | 1.645404 | 1.645494 | +0.005% |
| 384 | 256 | original | 1.654394 | 1.652013 | -0.144% |
| 384 | 256 | moved | 1.652661 | 1.650609 | -0.124% |
| 768 | 64 | original | 11.719156 | 11.706450 | -0.108% |
| 768 | 64 | moved | 11.719565 | 11.706863 | -0.108% |
| 768 | 256 | original | 12.346607 | 12.322207 | -0.198% |
| 768 | 256 | moved | 12.346461 | 12.320705 | -0.209% |

All **32 records / 192 samples** pass. Maximum paired raw/symmetric K error is
`5.55e-16`; energy error is `8.88e-16`. The independent CPU subset has maximum
K/energy errors `2.78e-17 / 8.67e-19`. The full receipt retains all four routes,
first-replay and five warm samples, preparation costs, every offer/rejection,
six scientific/semantic/precision identities, and semantic work counters.

Inputs are original/moved 48/96-atom water geometries with spherical def2-SVP,
recovered from retained Git blobs with exact hashes in `provenance.json`.
All routes use identical unscreened, unfitted COSX v1 semantics, symmetric
exchange, RHF spin-summed density and strict FP64. The positive diagnostic
density is `D[i,j] = 0.1*delta(i,j) + 0.001/(1+i+j)`.

The explicit **3×3×6 per-atom diagnostic quadrature** uses three partition
iterations and coincident tolerance `1e-12`, producing 2,592/5,184 points.
This is a coarse grid and a diagnostic density. Production quadrature,
converged SCF accuracy and complete SCF/force performance are outside this
receipt's qualification domain. Three spread-out points independently compare
the discrete CPU COSX oracle at each actual AO dimension and geometry; tile 2
exercises full and tail. Every timed full-grid sample compares raw/symmetric K
and exchange energy to generated under `1e-9 + 1e-11*abs(reference)` gates;
the verifier additionally requires each retained absolute error below `1e-9`.

Each timed build includes density upload, AO/ESP generation, all three algebra
operations, weighted/finite publication, symmetrization, matrix download,
synchronization and host energy contraction. Four routes rotate execution order.
The first sample is **first replay of a prepared plan**, following CPU-subset
qualification that initializes CUDA and the library. It is not process-cold
initialization. Warm medians use the other five samples. Basis/grid setup, plan
setup and provider preparation are separate costs. Changed geometry rebuilds
all owners and quadrature. CPU oracle evaluation occurs outside timed builds.
Component timing was not collected; derived work counts do not measure traffic.

Full/tail counters include three contractions per tile, one scaled output per
point/AO and one additional library ESP publication pass per tile. The 768-AO
tile-64 case has no partial tail and zero tail calls. Every route reserves the
same numeric buffers. Any library route shares a single 96 MiB provider
allowance across six sites; generated reserves zero. Host reservation is
160 KiB. `verify.py` checks these deltas, every identity, all calls/summands,
scaled elements and split publication passes, and reconstructs `summary.json`.
The integral/materialization counts in that summary are derived from the
unchanged scientific domain, rather than measured profiler counters.

All GPU work ran on n1/node1 through finite 20-minute Slurm
`main/gpu:5090:1` allocations, jobs 6000–6003, preserving assigned visibility.
Each AO/tile case used a separate GPU allocation; this receipt supports paired
route comparisons within each case, rather than a tile-size promotion. Successful
`srun` exits and assignment snapshots are retained; Slurm accounting is disabled.
Compiler commands all use verified ccache, with retained version and before/after
shared-cache snapshots. Source commit, source/binary/input/data hashes and exact
build/qualification scripts are retained. Full manifests and raw logs remain in
ignored `.artifacts/1884-cosx-esp-crossover/`.

The harness changes only diagnostic execution. Full value/derivative/failure/
lifetime regression, five shared GPU provider tests, and memcheck/initcheck with
zero errors qualify the weighted runtime in #1968, with that earlier source and
binary explicitly recorded in `provenance.json`. They are distinct from this
benchmark binary. See the retained
[decision note](../../../.agents/notes/implemented/performance/2026-10-05-cosx-weighted-esp-crossover.md).

```bash
python benchmarks/results/cosx-weighted-esp-1884/verify.py
python benchmarks/results/cosx-weighted-esp-1884/prepare-inputs.py .artifacts/1884-cosx-esp-crossover/inputs
```

To reproduce GPU measurement, use `provenance.json`'s source commit and input
hashes, recreate its tracked-source manifest in the ignored artifact directory,
and run the retained build script on n1/n2/n4/n5. Run the qualification script
with finite Slurm allocations as recorded; its arguments are atoms, tile and
three explicit quadrature dimensions. Preserve the scheduler's device visibility.
