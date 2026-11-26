# H100 WB97M-V DF geometry replay (#3185)

The original native release at `53442660ae08d965890632b6bb538c2f45aa253d`
failed at water96 step 8 and n-C32H66 step 3 on both repeats. A fresh owner
completes either geometry. Rebinding from the preceding geometry triggers
`DF metric rank crossing: retained/discarded subspaces are unresolved` with
or without the preceding density. The extra SCF retry cannot change the
prepared LR provider. The independent spectrum keeps the full-range metric
well separated. The failing LR modes are 0.426 and 0.758 resolution bands
from the existing cutoff.

Branch base `a7d4d0e5d886fe89398dfaa12cdfdc48cbbc245c`.
Measured fix source `e8049fa996846c9a64d9b02416fa96527a874dbc`.
The fix rechecks automatic LR DF admission before solving each moved geometry
while the item is fitted. An unresolved fit selects exact LR before the SCF.
The item retains exact LR across subsequent replays. The auxiliary basis,
full-range fit, thresholds and quadratures are unchanged. Explicit fitted LR
keeps its unresolved analytic-response error.

## Acquisition

Use Release/sm90 AOT builds with the same C++ and CUDA options for both arms,
one H100, one numerical thread, and the exact frozen coordinate sequences.
The [regression fixtures](../../../tests/reference_data/wb97mv-df-3185/README.md)
retain the two minimal pairs and the dense Direct references at the failing
steps. The replay driver authenticates the original frozen sequence and loaded
build before using each sample. It times preparation, geometry rebind, SCF,
analytic force and return to host. After a failed endpoint, record the failure
and use only the last successful density if attempting subsequent steps.
A failed endpoint cannot contribute a qualified time.

Run `acquire_qz.sh TASK_ROOT HISTORICAL_ROOT COLD3_INPUTS ACTION` in a finite
qz H100 batch Job, with the two release arms at `TASK_ROOT/master` and
`TASK_ROOT/candidate`. Each arm contains a source archive expanded into
`source` and a verified production `build` plus release `build-receipts`.
Set `GQC_QZ_JOB_NAME` and `GQC_QZ_PROJECT` from the actual allocation.
`prove` checks the failing geometry in fresh, moved and moved-cold owners and
runs the regression tests staged under `TASK_ROOT/regression/tests`.
`measure` alternates the baseline/candidate processes on one card with two
full trajectory repeats and three cold3 water96 processes per arm, each with
one cold and three warm calls. Prewarm processes are kept separate.
`oracle` independently reconverges the failing geometries at levels 4/2 and
5/3 (XC/VV10). The common absolute gates are 1e-5 Eh per atom, 1e-4 Eh/Bohr
maximum Cartesian force error and 3e-5 Eh/Bohr force RMS. Grid stability uses
one quarter of each gate.

## Complete trajectory results

Both candidate repeats per trajectory completed all eleven geometries (cold
step 0 plus moves 1–10), for 44/44 successful energy and analytic-force
endpoints. The baseline completed 36/44 attempts. It was allowed to continue
from its last good density after a failed endpoint. The moved sums exclude
step 0. Only the candidate has complete moved sums.

| System | Baseline failed moves in each repeat | Baseline time at first failure, s | Candidate time and iterations at first failure geometry | Candidate moved sums, s |
| --- | --- | ---: | --- | ---: |
| paper-water96 | 8 | 33.447 / 33.269 | 31.481 / 31.394 s, 10 / 10 iterations | 224.092 / 224.010 |
| n-C32H66 | 3, 5, 10 | 37.618 / 37.554 | 56.681 / 57.001 s, 10 / 10 iterations | 459.023 / 464.171 |

The failed public baseline endpoints do not publish a converged iteration
count. The terminal cold attempts at the first failure retain 23 history
rows for water96 and 24 for n-C32H66 in both repeats. Later alkane failures at
steps 5 and 10 also report the same native rank guard. Their independent LR
spectral distances are 0.330 and 0.743 resolution bands.

After the one-way retreat, successive exact LR steps can take longer than a
successful fitted step. In repeat 1, water96 step 10 is 18.317 s on the baseline
and 30.410 s on the candidate. The corresponding successful alkane step 8 is
16.084 s and 48.974 s. The two routes retain their own density histories.

Both minimal tests fail on the baseline and pass on the candidate. The
candidate completes the isolated, moved-cold and preceding-density inputs.
Native snapshots at the two moved target geometries authenticate fitted
full-range J/K, a 1e-10 relative cutoff, and exact LR. The preceding geometries
use fitted LR. Energy-only and force requests make the same admission decision
before SCF. All sixteen candidate comparisons at moves 1, 5, 10 and the
original failure move pass the independent grid and observable gates.

| System | Worst energy error, Eh/atom | Worst max component force error, Eh/Bohr | Worst force RMS, Eh/Bohr |
| --- | ---: | ---: | ---: |
| paper-water96 | 3.3481e-6 | 2.6477e-5 | 1.0946e-5 |
| n-C32H66 | 8.1072e-6 | 3.1587e-5 | 9.5321e-6 |

The historical dense Direct oracles for moves 1, 5 and 10 have the same
frozen coordinate hashes and independent scientific controls. The target
moves 8 and 3 were solved fresh in a separate GPU4PySCF 1.9.0 allocation. Native
timings always use the pinned release images and their own densities.

## Cold3 water96 timing

All three processes per arm report 15 SCF iterations/Focks cold and one
iteration/Fock in each of the three unchanged-geometry warm calls. The two
profiles used the same stock water96 fixture, auxiliary input and H100 UUID
as each other, following the [cold3 protocol](../paper-hfdft-cold3-20261120/README.md).

| Timing population | Baseline median, s | Candidate median, s | Relative change |
| --- | ---: | ---: | ---: |
| Cold | 20.3656 | 20.3045 | −0.30% |
| Warm process medians | 4.4049 | 4.4136 | +0.20% |

The 0.009 s warm-median change is within the observed same-card repeat spread.
The timing population is descriptive and has three fresh processes per arm.

## Retained evidence

[Per-step times and iterations](steps.csv), [gate and cold3 results](summary.json),
and the hash-bound [publication manifest](publication.json) capture both arms.
Run `python report.py` from this directory to recompute the summary from the
compressed samples. The provenance includes source and library digests,
compiler inputs, AOT audits, driver hashes, point-rate estimates and receipt
hashes. The complete raw acquisition and build logs remain in task-local ignored
storage and the task qz directory. Only the two regression geometry pairs enter
the checkout. The branch incorporates later upstream CPU/resource-planning
changes, and the H100 numbers remain pinned to the two measured source commits.

The first build attempt failed to find the configured compiler-cache executable
because the staged path was one level shallow. It exited before compiling. The
first oracle attempt resolved GPU4PySCF 1.8.1 and was stopped. The two verified
1.9.0 oracles are from the re-run. The scheduler receipts preserve transient
queue warnings and the actual effective priorities. All eight task jobs were
deleted after terminal receipts were collected. Their total container runtime
was approximately 3.269 H100-hours, or 3.269 points at the listed one-point
hourly rate. Container runtime estimates are not billed transaction records.

This publication requests an aggregate storage increase equal to its indexed
result files. The per-file limit and incoming review budget still apply.
