# WB97M-V: FP64 anchors and anchor-relative delta-density J/K

This is an isolated experiment, not an enabled production path or a default
promotion. The frozen base is `5d1dbbcaf2f04fb83980c8f988795df28b41e392`.
Only the archived `src/dft/cuda_ks.cpp` changes; the dirty working checkout's
production sources and unrelated work are not modified.

## Protocol

- README water-96 geometry, spherical def2-SVP, 768 AOs, Direct WB97M-V/RKS,
  grid `48 x 16 x 32`, analytic forces, no density fitting.
- `PrecisionCalculator(policy="fp64"|"fast")`, native automatic Modified
  Huckel initialization, DIIS history eight, maximum 100 target iterations.
  Each arm uses a new process, calculator and prepared owner. Export support
  does not import a density or create a warm execution.
- Native policy defaults remain unchanged: FP64 energy/density/screening
  `1e-10 / 1e-8 / 1e-12`; Fast `1e-6 / 1e-4 / 1e-10`.
  Fast is the application-budget facade, not ordinary `precision="auto"`.
- Cold timing includes construction, preparation, the first synchronized
  energy/force endpoint, host/density export, teardown and cache release.
  Imports, basis JSON parsing, provenance collection, oracle comparison and
  result serialization are outside the timer. Existing compilation/artifact
  caches remain intact; no first-use time is subtracted.
- All four endpoints run in finite, exclusive Slurm job **1133** on node4's
  RTX 5090, preserving scheduler device visibility. Order: FP64 off/on,
  Fast on/off. This is one observation per arm, not a repeated statistical
  speed guarantee. The optional resident-source allowance is 2 GiB in all arms.

## Experiment

`GENERATIVEQC_EXPERIMENT_ANCHOR_DELTA=1` uses a full-density FP64 J/K/range-K
anchor. Provisional semilocal-seed and frozen-VV10 iterations evaluate the
**total current density minus the fixed anchor density**, in the policy's
ordinary J/K arithmetic, then add the anchor matrices in FP64. Nonlinear XC
always receives the total density. Full-target iterations build fresh
full-density FP64 J/K/range-K and refresh the anchor. Final physical checks,
force arithmetic, stopping gates and the ordinary nonlocal schedule remain.

This does not enable the existing generic incremental controller or remove
its mixed-precision guard. Separate state preserves the normal frozen-VV10
schedule and optional source residency. Delta updates do not advance the
anchor, avoiding accumulated screening/roundoff chains. The experimental
owner rejects unsupported spin/DF/nonlocal/provider/resource-ledger domains.
Positive scope/arithmetic events are checked; merely setting an environment
variable is not counted as executing the experiment.

## Water-96 results

| Policy | Delta | Complete cold E+F (s) | Target SCF Focks | Provisional / full target | SCF VV10 pair sweeps | FP64 / mixed J applications |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP64 | off | 996.437 | 28 | 23 / 5 | 4 | 28 / 0 |
| FP64 | on | 982.914 | 28 | 23 / 5 | 4 | 28 / 0 |
| Fast | off | 580.152 | 18 | 14 / 4 | 3 | 4 / 14 |
| Fast | on | 461.533 | 16 | 13 / 3 | 2 | 4 / 12 |

Every arm also performs seven native preliminary Huckel Fock builds; none
has a discarded target attempt, execution retry or native fallback. Preliminary
work is included in the timed endpoint, not in the target-SCF column.

Fast's observed elapsed reduction is **20.45%** (1.257x off/on), with two fewer
SCF Focks and one fewer VV10 pair sweep. The trace contains four full anchors
and twelve delta builds. Fourteen Focks are labeled mixed-stage, but two of
those use strict anchor J/K alongside other provisional arithmetic; the
operator inventory, not the stage label, gives four strict and twelve mixed
J applications. Full-range and range exchange together count twice that many.

FP64's observed reduction is only **1.36%**, with identical Fock and VV10-sweep counts.
This single ordered pair does not establish a meaningful FP64 speedup. Its
trace contains six full anchors and twenty-two delta builds. The experiment
does not demonstrate fewer FP64 SCF iterations.

## Independent accuracy and cost

Both sizes are checked against retained independently reconverged GPU4PySCF
references with identical geometry, basis identity and grid. The reference
is read after the target endpoint and is never used to seed it. The 96-atom
oracle is the geometry-zero accuracy receipt in frozen
`benchmarks/results/readme-wb97mv-20261014/summary.json.xz`, not that campaign's
timed-out population reference arm. It is historical accuracy evidence, not
a newly timed reference execution.

| Water-96 arm | Energy error (Eh) | Maximum force error (Eh/Bohr) | RMS force error (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| FP64 off | 9.55e-12 | 2.38e-11 | 8.79e-12 |
| FP64 on | 9.09e-12 | 2.31e-11 | 8.84e-12 |
| Fast off | 2.21e-9 | 6.46e-6 | 1.91e-6 |
| Fast on | 1.33e-9 | 3.19e-6 | 1.17e-6 |

All pass their unchanged independent policy gates: FP64 energy/max-force
`1e-8 / 1e-7`; Fast energy/max-force/RMS-force `1e-5 / 1e-4 / 3e-5`.
Fast+delta does **not** attain FP64-mode force accuracy: it still fails the
FP64 independent `1e-7` maximum-force gate. Its final physical residual RMS
is `1.77e-7`, versus `2.32e-13` for FP64+delta. The presence of strict final
arithmetic does not make the convergence contracts identical.

The five extra FP64 matrices and one scalar add **23,592,968 bytes**
(22.5 MiB plus eight bytes) to the 768-AO KS state arena. Matrix differences,
reconstruction and anchor copies add O(N^2) work. Strict anchors have a real
per-call cost, but are replacements for provisional J/K, not mandatory extra
SCF iterations. Here Fast's saved refinement work leaves the total strict
J count unchanged at four; do not claim two additional strict calls for this
96-atom endpoint. Peak whole-endpoint GPU memory and an isolated percentage
cost for the anchor operations were not measured.

Before water-96, four water-12 E+F arms pass independent gates in jobs 1130
and 1131. At that size Fast remains at seventeen SCF Focks and delta is
3.42% slower in the single pair. The first FP64-off qualification is much
slower than FP64-on despite identical reported Fock/sweep work; no water-12 FP64 speed claim or
first-use subtraction is made. Do not generalize the water-96 win to all sizes.

## Evidence and reproduction

`evidence.json` retains work, precision, numerical and paired-state reductions.
`experiment.patch` is the isolated source change. Raw results, forces,
densities, build/cache/authentication receipts and executable harnesses remain
in ignored `.artifacts/scf-delta96-20261018/` locally and
`/data/jzzeng/scf-delta96-20261018/` on n4. The authenticated build and full
source archive also remain at that path on n1. Native source identity is
`106787c62dc43ca7a1ec11d4b4491063f7e38c69a1e8b33a997bd052a590189f`;
library SHA256 is
`c0840cf5a2b7337302d2d0f5baf2ede0757df57422c34435eb8ca2858601e5b1`.
The overlay rebuild uses verified sccache 0.16.0 and recompiles every changed
header/source dependency plus the source-identity translation unit. Unchanged
objects are reused without changing the original build.

For a fresh output directory, run the retained `run-endpoint.sh` through
finite `srun` on the same allowed node; its arguments are policy, atom count,
delta flag and output directory. `suite96.sh` reproduces the four-arm order,
and `analyze.py` checks raw arithmetic/scope events and paired final densities.
See the proposed Agent Note for promotion boundaries and rejected shortcuts.
