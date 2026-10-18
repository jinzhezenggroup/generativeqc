# WB97M-V: explicit-grid complete E+F benchmarks

Measured on 2026-10-18 from master `2a62a17e8884cac9a0bfafa47caa5cd6a1ea4ad3`,
with benchmark-only grid controls and reference empty-AO compatibility changes.
The native library is a new source-matched Release/sm_120 build; no production
quadrature, precision, convergence or screening defaults change.

## Inputs and gates

- n1, NVIDIA GeForce RTX 5090, scheduler-assigned devices preserved. Every
  electronic calculation runs through `srun` with a finite allocation and point
  deadline. n4 independently runs the 12 CPU-oracle/GPU explicit-grid tests.
- Nested 3/12-atom water clusters, spherical def2-SVP, WB97M-V/RKS. Direct J/K
  and DF are separate populations. DF uses cc-pVDZ-JKFIT for both engines and
  the established matched full-range-DF / exact-long-range exchange contract.
- Each engine/grid/size/mode has one cold, five fixed-density warm, one moved
  and five moved-warm complete E+F calls. Cold includes construction, preparation
  and the first endpoint. Timers include synchronization, analytic grid response
  and host-returned forces. Compiler caches are retained; nothing is subtracted.
  Qualification/cache-priming calls are retained separately. Plotted populations
  start fresh processes after those calls, so initial compilation/setup effects
  do not masquerade as a quadrature-size speedup. This is cache-primed cold,
  not a clean-machine compilation benchmark.
- All plotted native calls pass independent same-grid `1e-8 Eh` energy and
  `1e-7 Eh/bohr` maximum-force gates. Full endpoint times and actual SCF
  iterations/builds appear in [grid-timings.md](grid-timings.md).
- Separate Direct-reference convergence sweeps use one warm replay and compare
  original/moved geometries against **64×16×32**, with explicitly chosen
  exploratory `1e-5 Eh` total-energy / `1e-4 Eh/bohr` maximum-force tolerances.
  These application tolerances do not relax same-grid engine parity.
  No denser DF reference is run: its same-grid parity is not a separate
  certification of fitted-model quadrature convergence.

## Grid choice

| Quadrature | Points/atom | Role |
| --- | ---: | --- |
| 48×12×24 | 13,824 | Angularly reduced timing population |
| 48×16×32 | 24,576 | Original-grid timing population, remeasured |
| 64×16×32 | 32,768 | Denser independent convergence baseline |

Both timing grids pass the stated original/moved 3/12-atom convergence gates.
The smaller rule has **43.75% fewer full-grid points**; the square of the
unscreened point count falls by **68.359375%**. These are sizing bounds, not
executed-pair counts or measured speedups. End-to-end timings also include SCF,
J/K, AO/XC, force response and host export.

Reducing radial resolution is not an equivalent shortcut: `(32,12,24)` and
`(32,16,32)` fail the exploratory force gate, and `(40,16,32)` fails on the
12-atom geometry. Retain their observations rather than selecting a grid from
energy alone. [grid-convergence.md](grid-convergence.md) lists all available
grid/geometry gates, including missing or partial evidence.

The native semilocal and VV10 owners currently share a quadrature. This study
does **not** provide independently coarse NLC and fine XC grids. A reference-only
NLC override would change the numerical model and is deliberately not used.
Nor does this small water study qualify a universal production default or
extrapolate speedups/accuracy to 48/96 atoms, other molecules or other bases.

## Reference compatibility and validation

GPU4PySCF 1.8.1 skips empty AO blocks by default. Dense radial tails exposed an
NLC density/coordinate shape assertion. Enabling strict ordering only on the
SCF instance was insufficient: analytic full-grid response constructs a fresh
NumInt, then encounters an empty-index `take_last2d` launch. The benchmark now
scopes the existing exact-zero-AO iterator adapter to those fresh integrators as
well, restoring the class on success or failure. No coordinates, weights,
cutoffs, XC/VV10 formulas or derivative mathematics are substituted.

The n4 explicit-grid regression covers spin-paired/open-shell, Direct/DF and
LDA/PBE/r²SCAN density, XC energy and potential against independent CPU PySCF.
An initial n4 attempt used the default toolkit and failed CUDA binary loading;
the recorded successful rerun explicitly selects CUDA 12.9.1. Initial dense
reference failures and the finite-deadline partial population remain in the
raw evidence; none becomes an accepted timing or convergence bound.

## Reproduction and retained evidence

See [grid controls and commands](../../readme_grids.md) and [reproduce.sh](reproduce.sh).
Use `tools.render_readme_grids --summary grid-timings.json.xz` to regenerate
the figure/table without a GPU. `manifest.json` binds `grid-timings.json.xz`
and the reduced reports. That single archive also contains `accuracy_observations`,
`grid_convergence`, `build_settings` and `raw_evidence`, including the hashes of
original journals and receipts. It retains every plotted E+F call, both engines'
force arrays, and semantic SCF/force work and traffic counters for executable
numerical replay. Optional iteration trajectories and subcomponent timing
profiles stay in the hash-bound journals. The figure is stored only once, as
the [homepage asset](../../../assets/readme-dft-performance.svg).
Full journals, logs, failed attempts, priming calls and measured consumer
snapshots remain under `/data/jzzeng/readme-vv10-grids-20261018` on n1 and in the
ignored local `.artifacts/vv10-grid` directory, not a new source-controlled run
bundle. Native source identity, binary hash, caller hashes, device/environment
metadata and original work/error data are retained rather than inferred from
the checkout name.

The README replaces the previous DFT chart with this grid-labelled study;
no previous timing chart is carried alongside it as a second comparison.
