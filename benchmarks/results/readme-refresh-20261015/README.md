# HF/DFT README benchmark refresh (2026-10-15)

This campaign replaces the native HF and seven DFT series in both README
Direct/DF panels. **GPU4PySCF was not rerun**: its historical timings, SCF
work, failures and independent reference arrays remain unchanged.
DF-CCSD(T)/CBS is unaffected.

## Results and limits

- **63 complete native populations plus one partial DF/96 population, 761
  successful energy + analytic-force calls:** 744 pass independent `1e-8 Eh`
  energy and `1e-7 Eh/bohr` force gates. The 12 ωB97M-V DF/48 and five DF/96
  calls are explicitly **unchecked** (no matching oracle). DF/96 has a complete
  cold endpoint, but not a complete five-repeat warm population.
- Maximum checked differences: `2.2555e-10 Eh` in energy and
  `1.6854e-10 Eh/bohr` in force. Another 384 small-system cache-priming
  calls are recorded separately, not included in timing populations.
- These are absolute endpoint timings, **not** a balanced before/after
  performance experiment. Differing SCF work and cache readiness are not
  normalized away; neither an individual kernel gain nor a universal speedup
  is implied.

![Direct cold and warm E+F](https://raw.githubusercontent.com/jinzhezenggroup/generativeqc/36a80785990124d03e9b005b12951957b584167e/benchmarks/results/readme-refresh-20261015/dft-cold-warm.svg)

The superseded Direct figure is retained losslessly as `dft-cold-warm.svg.xz`;
the preview uses its existing Git snapshot. The verifier checks both stored
and expanded identities. This storage-only compaction makes room for the
native r²SCAN-3c update without removing observations or raising the 64-MiB cap.

![DF cold and warm E+F](https://raw.githubusercontent.com/jinzhezenggroup/generativeqc/4e79f783640f71d41cd5dc6d1fe798e05faf55a1/benchmarks/results/readme-refresh-20261015/df-cold-warm.svg)

The superseded DF figure is also retained losslessly as `df-cold-warm.svg.xz`,
with stored and original expanded hashes checked by the existing verifier.
This 2026-10-19 storage-only compaction makes room for the explicit-grid README
study without changing historical observations or the aggregate evidence cap.

Full [Direct timing/work table](timings.md) and
[DF timing/work table](df-timings.md).

## Frozen source

| Input | Revision |
| --- | --- |
| #2531 (Direct-J / strict-FP64 XC correction) | `ba0dadbdcdef7af18715b72a2366595719df4bde` |
| #2532 (resident final validation for pure-J KS) | `27d3b8558b7db2ae4b7d4051ae9122522379a809` |
| Combined measurement commit | `1593d3b2fea9ed3e9232efc49eedede1beb1787b` |
| Combined source tree | `f158f1c1b69700df6d74b0ebe42c68d370be09f3` |

The combined source includes master `890c425b0` and can be reconstructed
using `git merge-tree --write-tree 27d3b8558 ba0dadbdc` after fetching
those commits. The PRs later advanced to `01ed327f4` and `1be47f590`;
[source-input-audit.json](source-input-audit.json) verifies that these
test/publication-only changes alter no measured native input or benchmark
consumer. Measurements retain the original frozen identity, not the later
PR heads or current master. The historical #2532 `none/96/moved`
nonregression failure remains an acceptance hold; this refresh does not
turn that gate into a pass.

The 2026-10-17 DF/96 supplement uses the same production source and binary.
Only its benchmark consumer adds an explicit DF workspace option; the exact
executed consumer is retained as `df96-benchmark-consumer.py.xz`, bound to the
observation's source-file inventory. Later formatting/docstring edits to the
checkout are not relabeled as the executed consumer.

## Measurement protocol

- **Hardware/build:** Slurm-assigned RTX 5090 GPUs (n1/n4/n5), finite
  allocations; CUDA 12.9.1, Release/sm_120, source-matched native core and
  all 24 stationary-force AOT plugins. Compiler-cache and exact binary
  identities are retained in [build-settings.json](build-settings.json);
  the original 16 allocations plus two DF/96 allocations, assigned devices
  and log hashes are retained in `job-receipts.json.xz`.
- **Cases:** 3/12/48/96-atom water clusters, spherical def2-SVP except
  r²SCAN-3c (def2-mTZVPP); both DF engines use cc-pVDZ-JKFIT.
  The moving, unpruned `(48, 16, 32)` Becke grid, FP64,
  public initialization and convergence thresholds are unchanged.
- **Requested per population:** one cold, five fixed-density warm, one moved-geometry
  and five moved-warm complete E+F calls. Cold includes Calculator creation,
  preparation and first execution. Warm results report five-sample medians
  and ranges; recorded SCF iterations/Fock builds are not normalized away.
- **Cache and accuracy:** 3/12-atom priming runs in separate processes;
  measurement Calculators, densities and checkpoints are fresh. Priming is
  never subtracted from cold. Native accuracy uses retained independent
  energy/force arrays, never GPU4PySCF or a reference density in production.

## DF/96 supplement (2026-10-17)

Native-only Slurm job **9108**, on n1 / scheduler node1, ran from
2026-10-16 23:30 to 2026-10-17 03:30 (+08:00). Its **14400-second** point deadline
expired during the fifth warm call (exit **124**), inside a 04:10:00 allocation.
The raw journal still says `running` at `warm/4`; the external outcome receipt,
not an invented successful final status, establishes the timeout.

| Completed phase | Complete E+F (s) | SCF iterations / Fock builds |
| --- | ---: | ---: |
| Cold | 4717.471 | 37 / 37 |
| Warm repeat 1 | 2145.750 | 1 / 1 |
| Warm repeat 2 | 2148.055 | 1 / 1 |
| Warm repeat 3 | 2146.223 | 1 / 1 |
| Warm repeat 4 | 2146.284 | 1 / 1 |

All five records have successful status, converged SCF and finite exported
96-atom host force arrays. They have **no independent accuracy qualification**:
accuracy gates and energy/force errors remain null. The cold phase is plotted
with an open marker; four warm samples are retained individually, never
promoted to a five-sample median. Moved/moved-warm were not reached.

This is a **non-default resource population**: a 12 GiB DF workspace hint,
4 GiB additional device and 4 GiB additional host force allowances, and
`GENERATIVEQC_KS_RESIDENT_MAX_BYTES=2147483648` (2 GiB). These are not total-process
memory caps. The value path reports `streamed: true`; cold force work includes
about 1967.84 seconds of integral derivatives. Bounded memory does not imply
bounded work. No production defaults, FP64 policy, grid, initialization or
numerical thresholds changed, and no reference engine was run.

The preceding default-policy job **9073** failed force admission after
1027.926 seconds despite SCF convergence:
`stationary CUDA byte budget is not representable`.
Its raw negative evidence is retained in `df96-default-force-failure.json.xz`
and excluded from endpoint timings. The original 63 populations and sixteen
job receipts remain unchanged, rather than being replaced by this supplement.

## Evidence and reproduction

`manifest.json` records checksums for publication files and four unchanged
historical reductions. Superseded SVGs are preserved losslessly as
`.svg.xz` with verified original expanded hashes, under the 64 MiB cap.
`native-summary.json.xz` contains native timings, work counts, acceptance
residuals and provenance; `job-receipts.json.xz` contains GPU/Slurm receipts.
Large raw arrays, SCF histories and build logs remain in ignored campaign
storage; offline replay validates retained scalars and identities, **not**
omitted raw vectors.

From the repository root, replay the figures and tables without a GPU or a
reference-engine calculation:

```bash
PYTHONPATH=.:python python benchmarks/results/readme-refresh-20261015/refresh.py
PYTHONPATH=.:python python benchmarks/results/readme-refresh-20261015/refresh.py \
  --output .artifacts/readme-refresh-replay
```

Replay requires Matplotlib, Scour and defusedxml. For raw-data reduction,
use `reduce.py --raw <campaign-directory> --df96-raw <df96-directory>`;
`restore_references.py`
reconstructs accuracy receipts from retained independent arrays.
`reproduce.sh <method> <none|cuda>` runs native-only measurements in an
existing Slurm allocation with `README_REFRESH_ROOT` set to the frozen
source/runtime package.

For the explicit-budget DF/96 point only, run on n1 with the same frozen
source/runtime package present:

```bash
README_DF96_OUTPUT=/data/jzzeng/readme-df96-fresh \
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=96G --time=04:10:00 \
  bash benchmarks/results/readme-refresh-20261015/reproduce-df96.sh
```

The script restores the hash-bound executed consumer and never runs GPU4PySCF.
It verifies both archived and expanded consumer checksums, reserves a new output
directory exclusively, and refuses to reuse prior receipts, even after interruption.
For a longer attempt, increase both `README_BENCHMARK_POINT_TIMEOUT` and the
finite Slurm wall limit; this publication does not claim the missing phases
completed. Offline replay still requires no GPU or new reference calculation.
