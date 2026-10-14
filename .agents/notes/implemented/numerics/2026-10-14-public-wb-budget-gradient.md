# Decision: public dense-grid WB97M-V budget-selected geometry

Status: implemented within the qualified dense-grid small-domain boundary
Date: 2026-10-14

## Why the earlier candidate was not enough

The coarse-grid private candidate in `../../proposed/2026-10-14-composite-fp32-geometry.md`
was numerically sound but near-neutral at the complete endpoint. Slurm 8385
now profiles two frozen E/F calls per mode, retaining native strict SCF and
the ordinary optimized FP64 comparator. At 96 AOs on the coarse grid,
integral/SCF kernels dominate; VV10 takes only 38.6 ms across both calls.

With the actual README GridSpec(48,16,32) at 24 AOs, VV10 pair kernels take
710 ms across two strict calls: approximately 48% of recorded GPU kernel time.
The shared strict geometry kernel totals 228 ms. With only the semilocal
consumer narrowed, float geometry totals 42.8 ms and the remaining strict
geometry totals 40.6 ms; instrumented endpoints are 0.776/0.703 s. These are
**profiler diagnostics**, not clean endpoint qualification or a new VV10 FP32
implementation. Pair evaluation and SCF remain the next major arithmetic costs.

## Admission and provenance

The public consumer checks the complete canonical MethodIR (including range
omega, both exchange coefficients and VV10 parameters), native exact J/K proof,
molecular nonlocal density policy, strict stopping criteria, budget and bounded
domain. It admits only 24–96 AOs, 2–12 atoms, through-d shells, and the explicit
dense README grid. Coarse/modified/DF/UKS/corrected models are not promoted.

The existing compiler-owned semilocal geometry kernel executes FP32; there is
no copied WB97M-V equation or relabeling of strict nonlocal work. Schema 4 uses
the distinct `python-composite-stationary-cuda` owner. Only a completed pair of
whole-grid geometry consumers may publish evidence. Returned and rejected
semilocal work remain separate; no SCF completeness or FP32 AO storage is
invented. Existing typed numerical failures permit one complete strict retry.
Missing optional transport, including before source construction, has a separate
drain/rebuild fallback and does not become a numerical failure or mixed-work win.

## Integration failure and correction

Slurm 8394 failed the first public water fixture: the facade reported
`observed_met` after two strict solves, not real selected FP32. The run was
interrupted during the next costly oracle to diagnose that failure; its partial
JUnit/log are retained. Slurm 8397 exposed the exact reason: the facade selected
AUTO, but the native application scheduler executed **zero** mixed Fock builds
for a nonlocal model. The composite admission correctly required strict SCF.

Do not weaken admission to pretend AUTO qualifies this model. The facade now
selects FP64 for resolved nonlocal models, matching the arithmetic it already
executes, while independently passing the application derivative budget.
Ordinary Calculator AUTO is unchanged. Slurm 8398 confirms one candidate solve,
73,728 actual FP32 geometry points, and strict SCF/nonlocal provenance. This
diagnostic alone is not independent observable qualification.

The initially queued README timing job 8395 was cancelled around allocation
while repairing selection. Retain any partial `readme-prepared-8395` files; do
not combine them with the corrected population.

## Independent complete README evidence

Slurm 8386 completed the 24-AO private dense population, but hit its 35-minute
limit during the 96-AO CPU oracle. Its retained `running` JSON is partial, not a
passing full campaign. Slurm 8399 passed the complete public water test, then
was interrupted during the expensive 96-AO CPU oracle. Its one-test success
does not qualify the other three representations or the fault-injection suite.

Slurm 8400 completed **all 48 native endpoints**, ordinary FP64 and public Fast
at 3 and 12 atoms, before interruption during its final CPU-oracle verification.
The raw CLI rows remain `completed_unchecked`; they are not overwritten.
An existing GPU4PySCF environment provides a much cheaper independent oracle.
Slurm 2960 cross-checked it against the two completed CPU water geometries:
maximum differences were 9.095e-13 Eh and 9.367e-10 Eh/bohr. Slurm 2961 then
completed both actual README independent reference populations (12 calls each).

`verify-wb-readme.py` requires exact scientific protocol equality, including
geometries, basis identity, both grids, nonlocal density policy and tolerances.
It applies the existing strict or application-budget gate to **every** native
row and writes the separate input-hashed `wb-readme-8400-verified-2961.json`.
All 48 pass. FP64 uses unchanged 1e-8 Eh / 1e-7 Eh/bohr gates. Fast uses its
explicit 1e-5 Eh / 1e-4 maximum / 3e-5 RMS Eh/bohr budget. The maximum Fast
errors across these populations are 3.127e-12 Eh, 7.574e-8 maximum and 2.915e-8
RMS Eh/bohr; strict force errors are below 9.160e-11 Eh/bohr.

| AOs | Phase | FP64 seconds | Fast seconds | FP64/Fast |
| ---: | --- | ---: | ---: | ---: |
| 24 | Cold | 7.423815 | 7.396425 | 1.0037 |
| 24 | Frozen warm | 0.763167 | 0.696263 | 1.0961 |
| 24 | Moved | 5.523844 | 5.499335 | 1.0045 |
| 24 | Moved frozen | 0.762803 | 0.695940 | 1.0961 |
| 96 | Cold | 54.235546 | 53.787949 | 1.0083 |
| 96 | Frozen warm | 6.058855 | 5.787511 | 1.0469 |
| 96 | Moved | 40.338849 | 40.096403 | 1.0060 |
| 96 | Moved frozen | 6.053770 | 5.788187 | 1.0459 |

Every warm phase includes all five frozen-seed repeats. Both native modes ran
on the same n1 RTX 5090. The reference ran on n2 PRO6000: **no cross-hardware
engine speed ratio is claimed**. Cold/moved Fock counts match (13/11 at 24 AOs,
19/14 at 96); all warm calls use one Fock. Every Fast row has one solve and
73,728/294,912 actual FP32 semilocal geometry points, respectively. Cold/moved
savings are essentially neutral; do not present the warm gain as SCF speedup.

## Source-matched native qualification

Slurm 2960's subsequent native test did not run: the n1 binary requires
GLIBC_2.38 and n2 has 2.35. This is a loader prerequisite failure, not failed
numerics. Slurm 2962 completed a source-matched full Release/sm120 rebuild with
verified ccache, GCC 11.4 and CUDA 12.9, followed by **all 14 direct public/private
tests passing** in 965.91 s. This includes four dense public models/representations,
both tight-budget/explicit audits, missing optional capability before and after
construction, real pending-work recovery, failed-peer/energy evidence lifetime,
and the five prior private coarse-grid regressions. The new n2 library is
`3d854b3b05ceb17d2c330a4b231d752d65e18eec01a121c35be893fb7e0fe792`;
it must not be substituted for the n1 timing library's identity.
The now-unneeded n1 queued duplicates 8426/8427 were cancelled while pending.

The unchanged native library is
`f993f2ba55c378b4c8102a682226ad8ed7193903d1dc674632445b3fec886bf6`.
The timing's Python hashes are retained with each raw receipt; later fitted
integration does not retroactively change that source identity. The direct
integration host suite passed 353 tests and all-followup pre-commit passed.
Raw evidence lives in ignored `.artifacts/readme-budget-fast-20261013/` and its
n1/n2 counterparts. Fitted composite integration has a separate source snapshot
and qualification; it must not be inferred from these direct results.

## Remaining scope

Do not call this all-FP32, all-size README coverage or a 2–7x endpoint gain.
HF DF and larger domains remain open. Canonical fitted WB97M-V and r2SCAN-3c
are separately qualified in `2026-10-14-fitted-composite-budget.md`. Future
VV10 narrowing should type the existing compiler-owned TensorIR pair closure,
retain bounded tails/fallback and strict reductions, and independently recheck
the complete E/F endpoint before admission. The dense profile, not the old
coarse-grid ratio, identifies that next bottleneck.
