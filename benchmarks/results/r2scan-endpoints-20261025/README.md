# Large pure-Coulomb endpoints and refreshed HF/DFT figures (2026-10-25)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **128 populations / 512 complete energy-and-force calls** were remeasured
under the **#2786 default-only framework** at frozen implementation
`068d24de3956f02626bd0c46b595a16d3f6f23ce`. The baseline includes #2785 and
#2786, as well as the earlier HF/DF48 repairs. Eight methods, Direct/DF and
3/12/48/96 atoms each have a fresh process for both engines, with one cold and
three frozen-density warm calls. All HF points were remeasured too.
The original 2×4 layout is preserved. CCSD(T)/CBS is unchanged.

## Observed target endpoints

Seconds per complete energy-and-force call; warm is the median of all three
calls. Min/max ranges and every sample remain in the tables and raw receipts.

| Case | GenerativeQC cold | GPU4PySCF cold | GenerativeQC warm | GPU4PySCF warm |
| --- | ---: | ---: | ---: | ---: |
| direct r2scan-3c, 96 | 33.168 | 31.916 | 13.804 | 14.728 |
| df r2scan-3c, 96 | 26.108 | 26.353 | 8.269 | 12.756 |
| df r2scan, 96 | 14.925 | 16.690 | 5.002 | 6.737 |

Both DF targets are faster in this cohort. Direct r2SCAN-3c96 warm is faster,
while cold remains slower. The small DF r2SCAN-3c96 cold advantage is an
observation from one fresh process, not a statistical significance claim.
`comparisons.json.xz` also retains the three targets' previous README cohort,
without relabeling its revision or mixing its times into the new figures.

| Mode | Phase | GenerativeQC wins / measured pairs |
| --- | --- | ---: |
| direct | cold | 29 / 32 |
| direct | warm | 32 / 32 |
| df | cold | 29 / 32 |
| df | warm | 32 / 32 |

GenerativeQC is faster in **64/64 measured warm cases**. The 96-atom Direct
ωB97M-V warm ratio is **4.18×**. All completed points are shown, including
accuracy failures and slower results; this is not universal leadership.

## Retained implementation

Large pure-J DF endpoints use a resource-aware streamed work policy and
metric-amplified cumulative primal omission bounds. Forces retain their complete
derivative source domain. Direct cold work benefits from bounded pure-J
incremental proposals, ordered public-AO Schwarz preparation, larger complete
force capacity and wider large meta-GGA batches. Final full-density physical
and canonical closure, explicit budgets and bounded fallbacks remain intact.

The [DF decision note](../../../.agents/notes/implemented/performance/2026-10-25-large-pure-j-df-work.md)
and [Direct decision note](../../../.agents/notes/implemented/performance/2026-10-25-large-direct-cold-endpoints.md)
record the mechanisms and rejected alternatives. Further Schwarz-width,
provisional-cutoff and force-layout experiments were not promoted. Their raw
records, including cancelled partial trials, remain in the exploration bundle;
none contributes a timing point to these figures.

## Accuracy and validation

Independent status: **standard: 58 qualified / 6 failed / 0 unverified; gpu4pyscf: 58 qualified / 6 failed / 0 unverified**. The same default-grid/fitting failures
as the previous cohort remain visible. Every complete sample is checked against
energy ≤1e-5 Eh/atom, maximum force ≤1e-4 Eh/Bohr and RMS force ≤3e-5 Eh/Bohr,
with quarter-gate reference-grid stability and CPU/GPU spots. Original oracle
producer identities and audited input-reuse contracts are unchanged. Numerical
reference reuse does not reuse any old benchmark timing.

The final matching library passes **83 Python resource/default-policy checks**,
public Schwarz Cartesian/spherical s/p/d/f, batch/capacity/nonfinite checks,
**48 weighted DF finite-difference gates**, and native density-fitting/RI-JK
planning checks. The final host suite had 672 passes and 25 native-loading
failures in the library-free local environment; all 25 are covered by the
successful final 83-test run. Earlier metric-aware DF/force/NumPy/libcint and
resource reruns also pass. Initial setup failures and their matching reruns
remain in `regressions.json.xz`, rather than being discarded.

All **24 original/warm/displaced/fresh-displaced E/F checks** at 96/120 atoms
pass independent GPU4PySCF gates. The 120-atom geometry extends water96 with
24 atoms translated by 35 Bohr. These checks use the validated implementation
snapshot preceding the final formatting/fixture/import cleanup; its exact base
revision and reconstruction patch are retained. They are separate from figure
timings. CUDA memcheck reports zero errors for public Schwarz and the 18-case
DF metric/replay/gradient selection. No new full-device-graph synccheck or
racecheck pass is claimed. Trace/clean replay agrees in all nine exploratory
calls (maximum energy difference below 1e-12 Eh and force difference below
8e-12 Eh/Bohr); instrumented timings are excluded.

## Provenance and reproduction

The frozen archive's **12,950 files** match the measurement source and actual
incremental-build inputs. `provenance.json.xz` retains archive/audit identities,
core/AOT hashes, build configuration, actual compiler commands and verified
sccache 0.16.0 before/after statistics. The statistics belong to a shared cache,
not an isolated per-build hit-rate experiment. Raw archives, binaries and the
full source audit remain in ignored local/n1 artifacts.

Seven finite Slurm workers pair the two engines consecutively on the same
allocated card and alternate first-arm order across pairs. The native execution
schedules explicitly retain `GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` and
`GENERATIVEQC_SCF_VALIDATION_GEMM=1`; these are not unset-source defaults.
`jobs.json.xz` retains commands, scheduler visibility, allocations, outcomes and
runner scripts. [Conditions](conditions.md) describe solver policies, full grid
response, grids, resource policy and the limits of this one-process cohort.

```bash
PYTHONPATH=python:. python benchmarks/results/r2scan-endpoints-20261025/reproduce.py \
  --output .artifacts/r2scan-endpoints-replay --verify
```

This recomputes qualification, wins, target comparisons and work tables, verifies
all publication hashes, and redraws both original figures byte for byte using
their recorded hashes. The [final refresh](../readme-refresh-20261025/README.md)
supersedes the homepage figures and verifies that CBS is unchanged. If Direct cold work is revisited later,
preparation and SCF operator work remain the useful focus; this campaign stops
at the validated improvements above.
