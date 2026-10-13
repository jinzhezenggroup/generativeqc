# HF and DFT energy and analytic forces: benchmark issue follow-up

This publication builds on PR #2401 and adds source-matched measurements for
issues #2411–#2418, #2421 and #2426. Each point retains its own source revision,
native build identity, library hash, consumer hashes and Slurm receipt. The
current figures combine qualified new points with unchanged older observations;
they are not a complete remeasurement of one source revision.

![Direct HF/DFT cold and warm endpoints](dft-cold-warm.svg)

![Density-fitted HF/DFT cold and warm endpoints](df-cold-warm.svg)

The [direct](timings.md) and [DF](df-timings.md) tables retain cold, warm, moved
and moved-warm timings and SCF work. Warm means the median of all five frozen-seed
replays, with the entire minimum–maximum range. Unchecked phases use open markers;
failed or incomplete phases have no successful timing. No timeout was increased.

## Confirmed improvements

| Issue | Actual qualification and result |
| --- | --- |
| [#2411](https://github.com/jinzhezenggroup/generativeqc/issues/2411) | #2425's integrated pure-Coulomb DF repair passes all 12 calls at 48/96 atoms for PBE and r²SCAN. PBE/96 cold: 1225.048 → 53.891 s; r²SCAN/96: 1221.951 → 106.578 s. New warm medians: 9.884 / 14.213 s. |
| [#2412](https://github.com/jinzhezenggroup/generativeqc/issues/2412) | Integrated B3LYP DF force repair passes all 12 calls at 3, 12, 48 and 96 atoms; independent final-state/lifetime tests also pass. |
| [#2413](https://github.com/jinzhezenggroup/generativeqc/issues/2413) | WB97M-V DF passes all 12 calls at 3 and 12 atoms with the matching independent full-range-DF/exact-long-range-K oracle. The old default-reference mismatch remains in historical probes. Large WB endpoints remain unresolved. |
| [#2414](https://github.com/jinzhezenggroup/generativeqc/issues/2414) | [#2438](https://github.com/jinzhezenggroup/generativeqc/pull/2438) qualifies M06-2X FP64 local AO execution. Direct and DF pass all 12 independent calls at 3/12/48/96 atoms. DF/96 cold: 628.378 → 150.082 s; warm: 116.061 → 36.357 s. The older timeout remains retained. |
| [#2415](https://github.com/jinzhezenggroup/generativeqc/issues/2415) | Canonical r²SCAN-3c DF now passes complete independent populations at 3, 12 and 48 atoms, including D4/gCP and moved geometry. The 96-atom population is pending. |
| [#2418](https://github.com/jinzhezenggroup/generativeqc/issues/2418) | [#2436](https://github.com/jinzhezenggroup/generativeqc/pull/2436) packages exact M06-2X stationary AOT plans. Source `61b9387f3` gives 3-atom direct cold 4.378 s versus historical 229.691 s. A separate empty artifact-cache run passes all 12 calls with runtime compiler discovery and invocation forbidden. Later source-matched populations remain separately identified. |
| [#2421](https://github.com/jinzhezenggroup/generativeqc/issues/2421) | [#2437](https://github.com/jinzhezenggroup/generativeqc/pull/2437) adds automatic HF/DF value-response planning and exact-state final occupied projection reuse. All 12 calls pass at 96 atoms. Warm: 59.318 → 3.201 s; new cold 32.994 s, moved 28.670 s. |

These are before/after complete endpoint observations, not isolated attribution
of every source change. The newer integration uses the current native public
initialization policy, including Huckel; historical initialization and work
counts are preserved. M06-2X API-cold timing includes package loading. Its
separate compiler-forbidden control establishes AOT use, not a reconstruction
of the historical compiler-cache state.

The current reductions contain 29 fully qualified direct points (348 calls)
and 29 fully qualified DF points (348 calls). Maximum accepted errors are
2.274e-11 / 9.159e-11 for direct and 2.261e-10 / 1.573e-10 for DF, in Eh and
Eh/Bohr respectively. All pass the unchanged 1e-8 Eh / 1e-7 Eh/Bohr gates.

## Remaining qualification

- #2416: [#2439](https://github.com/jinzhezenggroup/generativeqc/pull/2439)'s
  bounded SCF AO-map staging and read-only force-grid leases remove the
  diagnosed host capacity cliffs. Direct/48 passes all 12 calls (86.302-s cold,
  26.070-s warm median). Direct/96 returns cold in 500.751 s and passes all
  11 returned E/F calls, but the 1800-s point limit interrupts the final moved
  replay. Its complete warm series has a 105.539-s median; integral derivatives
  contribute about 77.8 s per call. Initial capacity failures and the final
  incomplete population remain retained. The incomplete moved-warm series has
  no successful median.
- #2415: DF/96 gets past the repaired force-grid admission, but its latest
  source `227c55c06` run times out during cold SCF with no completed record.
  The packed-versus-streamed value planning cliff remains under investigation;
  the data contains no claimed successful 96-atom DF r²SCAN-3c endpoint.
- #2417 and #2426: WB97M-V large reference/native populations remain incomplete.
  Diagnostic reference/48 has approximately 820 billion active ordered pairs
  and about 23 s per VV10 feature call. Native/96's first measured SCF stage
  takes 116.0 s in nonlocal pairs versus 4.24 s in semilocal density/features
  and 2.45 s in nonlocal AO potential assembly. Diagnostic times are excluded
  from the figures. They establish quadratic pair work, not a qualified fix.
- The superseded M06-2X/DF 96-atom timeout and an earlier slower 48-atom
  population remain in the summaries. A same-source 48-atom replay already
  reduced cold/warm to 102.832/12.012 s before the local-AO change; this variation
  is not attributed to that change. Source `9e3c5e6e6` subsequently gives
  45.783/8.920 s at 48 atoms and a complete passing 96-atom population.
- A private fully FP64 reciprocal refinement saves only 8.3% in synthetic VV10
  features and 5.8% in geometry. This is insufficient endpoint evidence; the
  original production arithmetic is retained.

## Scientific and timing boundaries


- Nested water clusters: 3, 12, 48 and 96 atoms, batch size one, restricted FP64
  calculations. All methods use full def2-SVP spherical shells (24 AOs per water)
  except r2SCAN-3c, whose defining def2-mTZVPP basis has 34 AOs per water.
- Both engines use the same explicit 48×16×32 per-atom moving quadrature, three
  original-Becke partition iterations, no radius adjustment and analytic grid
  response. This is a matched-grid comparison, not a claim about each package's
  default grid or chemical grid-convergence accuracy.
- The public native endpoint and GPU4PySCF both include SCF energy, analytic
  forces, synchronization and force return to the host. No native calculation
  uses a reference density or reference energy/force implementation.
- Cold includes calculator/engine construction, preparation and the first
  endpoint. Process startup, provenance collection and input snapshot loading
  are excluded. Existing compiler/artifact caches are retained; these are API
  cold starts, not empty-cache compilation or process-launch timings.
- Each warm series reuses one frozen engine-local post-cold density. The moved
  geometry displaces the first hydrogen by 0.001 Bohr along z; its own post-move
  density is frozen for the five moved replays. Densities are not progressively
  improved across measured repeats.
- Native tolerances are 1e-12 Eh for energy, 1e-10 for density and 1e-12 for
  screening, with 100 SCF iterations maximum. GPU4PySCF uses 1e-12 Eh energy,
  1e-10 orbital gradient and 1e-14 direct-integral screening tolerances. Its stock
  DIIS is unchanged. Full-density potential rebuilds suppress both incremental
  inputs, as in the existing PBE0 protocol, and their work stays timed.
- r2SCAN-3c includes D4 (`s6=1`, `s8=0`, `s9=2`, `a1=0.42`, `a2=5.65`,
  `ga=2`, `gc=1`) and gCP. Its reference combines GPU4PySCF electronic calculations
  with CPU dftd4 4.3.0 and the independent CPU gCP qualification provider. Both
  corrections and their geometry-dependent setup are evaluated inside **every**
  reference endpoint. The native endpoint uses CUDA D4 and native CPU gCP; its
  corrections also remain inside every timed endpoint. The figure labels the
  reference CPU corrections.
- WB97M-V includes self-consistent VV10 and the shared molecular density-screening
  policy. Direct and DF use separate, matching independent oracles. No mixed
  precision, experimental AO selection or preliminary-density override is enabled.


## Provenance, validation and retention

All GPU work uses finite Slurm allocations on n1/node1, partition main,
RTX 5090, eight CPU threads, with assigned device visibility preserved.
The reference stack is GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0 and
CUDA 12.9. Core/AOT builds are Release sm_120 and use verified ccache;
runtime JIT retains the verified sccache cache where needed.

The integrated repair campaign is bound to `b27ee33a2`; HF/map/AOT qualification
is bound to `61b9387f3`; the first dense-grid controls use `877775209`;
final feature-grid populations use `227c55c06`; M06-2X local-AO populations use
`9e3c5e6e6`. The frozen sources are available on the
[qualification source branch](https://github.com/jinzhezenggroup/generativeqc/tree/codex/benchmark-qualified-source-20261013).
Full revisions, source inventories, library hashes and
individual raw-file identities are in [manifest.json](manifest.json) and both
compressed summaries. Later source changes do not relabel these measurements.
The source audit reconstructs the native source inventory and verifies all
43 recorded consumer files from those Git trees. Host integration tests and
independent CUDA tests qualify the changes; intrusive
profilers, setup failures and compiler-forbidden controls stay outside clean
benchmark populations.

[followup-prior.json.xz](followup-prior.json.xz) losslessly retains #2401's prior
summaries, manifest and report, including failed probes, timeouts and all scalar
observations. The old unqualified DF probes also remain in the current DF summary
as historical observations. [source-campaigns.json.xz](source-campaigns.json.xz),
[force-memory.json.xz](force-memory.json.xz) and the
[reference replay audit](reference-replay-audit.json) retain earlier evidence.
[followup-receipts.json.xz](followup-receipts.json.xz) retains authentic test
receipts, the compiler-forbidden control, source audits and the rejected
reciprocal probe. Failed fixtures and missing-library setup attempts remain
labelled as failures, rather than being rewritten as final acceptance. Native
arrays, build products and full trial logs stay in ignored local storage.

## Reproduction

Use a source-matched Release native library and all stationary AOT libraries
**and their JSON manifests**. Run each campaign in a fresh directory. Preserve
independent reference JSON before launching native measurements. Set
`README_BENCHMARK_METHODS`, `README_BENCHMARK_SIZES` and
`README_BENCHMARK_DENSITY_FITTING` (`none` or `cuda`) as needed; the default point
limit is 1800 seconds and five replays per warm phase.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:35:00 \
  bash benchmarks/run_readme_dft_benchmarks.sh both
PYTHONPATH=.:python python -m tools.render_readme_dft \
  --summary benchmarks/results/readme-dft-20261013/summary.json.xz \
  --output .artifacts/readme-figure
PYTHONPATH=.:python python -m tools.render_readme_dft \
  --summary benchmarks/results/readme-dft-20261013/df-summary.json.xz \
  --output .artifacts/readme-figure
```

Both engines can run independently (`reference` or `native`). Missing reference
geometries remain unchecked. A later oracle never rewrites original null gates.
Complete independently checked 12-call populations replace historical points.
A newer incomplete population can add complete checked phases where no complete
population exists, with its timeout and partial rows retained. All old outcomes
remain recoverable; no incomplete warm series is reduced to a successful median.
