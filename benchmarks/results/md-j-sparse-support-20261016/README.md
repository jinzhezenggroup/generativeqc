# Structural-support MD-J transforms: complete Direct endpoints

Date: 2026-10-16. Qualification: **complete for the standalone sparse-layout path**.

## Scope and source identity

This is a sparse **exact coefficient layout**, not DF, RI-K, COSX, a changed
screening tolerance, a larger optional cap, or another scientific CUDA kernel.
Only structurally impossible Cartesian-component/Hermite entries are removed.
Actual public AO expansions determine support, including unions of spherical
terms. Surviving coefficients retain their original FP64 summation order.
Each primitive retains an explicitly weighted zero for signed/nonfinite behavior.
The existing generated reciprocal Coulomb recurrence and normal-K source remain.

Frozen baseline: `b84ea05eb230addcc8599aab913a5c13f997dbe7`, including #2534 and
#2553. It is **not** a claim to be the latest master. At qualification,
`origin/master` also contains #2559 (array `.T`) and #2544 (CPU response recycle
vectors); neither changes these SCF/force paths. This branch does not contain or
claim combined qualification with the separate #2541 → #2549 → #2556 → #2558
force stack, or #2520/#2561 XC work.

Baseline library SHA256:
`eb3570c74d80f78c211f7844ecfc18ed2463b3d0a049f38617f6a71e598d50dc`.
Candidate library SHA256:
`4949afdcaa8b41893db1032f7d7d689777e9031c10b150b1ab7c9ee6e642414d`.
Scientific and test inputs match the frozen `candidate-current/source` snapshot.
Full binary/configuration/scheduler receipts are retained with this evidence.

## Why the default 96-atom endpoint changes

The README `Calculator` does not supply a `resource_budget` or `resource_plan`.
Its existing private MD-J source already fits at 48 atoms. At 96 atoms, dense
transforms exceed the original 128 MiB optional cap and use normal J instead.
Structural support moves the same existing MD source inside that unchanged cap.
Explicit public-ledger owners still keep their existing admission guard and
bounded normal fallback; spare live bytes do not authorize extra retained data.
No functional, molecule, AO-count or GPU-model selector is introduced.

| Layout | Original bytes | Candidate bytes |
| --- | ---: | ---: |
| 48 spherical / 544 AO | 45,878,756 | 31,303,512 |
| 96 spherical / 1088 AO | 182,845,316 | 124,792,120 |
| 48 Cartesian / 576 AO | 50,249,316 | 30,380,018 |
| 96 Cartesian / 1152 AO | 200,228,996 | 121,109,074 |

The 96-spherical inventory changes 13,712,336 dense coefficient bodies to
5,957,648 supported coefficient bodies plus 431,952 explicit weighted-zero
writes (6,389,600 stored slots). Its 1,690 shared uint16 map entries cover
131,328 shell pairs. These are exact preparation/storage work counts, **not
hardware FLOPs or quartet-enumeration speedup claims**. Indexed reads remain a
cost; the existing contraction/recurrence sources are not rewritten here.
The frozen default 96-atom baseline does not execute the dense MD preparation:
it falls back because of the cap. Thus this layout inventory is not falsely
presented as measured baseline hardware instruction savings.

## Complete endpoint population

All times are seconds including synchronized E+analytic forces, host publication,
D4/gCP where applicable, preparation for cold/moved and the original convergence
trajectory. Warm columns are medians of **five** calls, not isolated kernels.
Each arm retains cold + five warm + moved + five moved-warm calls (12 total).
The population bound remains 1800 seconds. No timing is divided by iterations.

| Case | Arm | Cold | Warm median | Moved | Moved-warm median |
| --- | --- | ---: | ---: | ---: | ---: |
| r²SCAN-3c 96 | baseline | 409.462 | 87.519 | 257.727 | 87.531 |
| r²SCAN-3c 96 | candidate | 311.849 | 83.562 | 205.257 | 83.534 |
| r²SCAN-3c 48 | baseline | 76.529 | 21.447 | 57.052 | 21.425 |
| r²SCAN-3c 48 | candidate | 76.739 | 21.452 | 57.185 | 21.441 |
| PBE0 48 | baseline | 23.831 | 4.608 | 22.607 | 4.598 |
| PBE0 48 | candidate | 23.889 | 4.614 | 22.642 | 4.589 |
| HF 48 | baseline | 11.171 | 0.745 | 7.224 | 0.745 |
| HF 48 | candidate | 11.192 | 0.746 | 7.218 | 0.745 |

Primary changes are **−23.84% cold, −4.52% warm, −20.36% moved and −4.57%
moved-warm**. Five warm observations span 83.478–83.699 seconds; five moved-warm
observations span 83.489–83.657 seconds. This is a qualified cold improvement,
**not achievement of the 10% warm aspiration**. All 48-atom changes are within
0.28%; PBE0 cold changes +0.244%, rather than a claimed reproduction of the
historical 3.06% regression on a different source pair.

All 96 completed source-matched calls (four cases × two arms × 12 calls) pass
independent E ≤ 1e-8 Eh and max-force ≤ 1e-7 Eh/Bohr gates. Per-case candidate
maxima and actual SCF/Fock-build counts, identical between arms:

| Case | Max E error (Eh) | Max force error (Eh/Bohr) | Cold iterations/builds | Moved iterations/builds |
| --- | ---: | ---: | --- | --- |
| r²SCAN-3c 96 | 2.456e-11 | 6.339e-11 | 24/24 | 13/13 |
| r²SCAN-3c 48 | 1.592e-11 | 3.693e-11 | 19/19 | 13/13 |
| PBE0 48 | 6.821e-12 | 2.462e-11 | 17/18 | 12/12 |
| HF 48 | 1.819e-11 | 2.184e-11 | 21/22 | 16/17 |

Every warm/moved-warm call uses 1 iteration/1 physical Fock build. The maximum
baseline energy/force errors over all calls are 3.866e-11/6.015e-11. These are
absolute independent errors, not merely agreement between native sources.

The 48-atom r²SCAN, PBE0 and HF controls use the same n4 GPU for both arms.
Primary 96-atom clean populations use the same n5 GPU. All GPU work uses finite
Slurm `srun` allocations with scheduler-assigned visibility; no n3 work occurs.
r²SCAN uses canonical def2-mTZVPP spherical AOs and the unchanged moving Becke
48 × 16 × 32 per-atom grid, strict FP64, screening 1e-12 and original SCF gates.
The PBE0/HF representative controls use the same def2-SVP inputs in both arms.

`raw/` retains every endpoint and its independent two-geometry E/F reference.
HF reference status is explicitly `accuracy_only_historical_reconstruction`:
both independent geometry oracles exist, but no fresh reference HF timing
population is claimed. Every HF call still passes the ordinary E/F gates.
The initial strict-loader HF setup rejection and a missing-launcher 48-atom
setup failure occurred before calculation; their logs remain in `trials/` and
the validation archive. Neither is silently counted as a completed population.
`python benchmarks/results/md-j-sparse-support-20261016/summarize.py` recomputes
all geometry/protocol/convergence/accuracy gates and medians into `summary.json`.

## Source-matched profile and remaining work

The intrusive cold/first-warm profile is **not** the clean timing population.
`profile/` retains the original records, trace events, force census and per-call
summaries. The summary uses mutually exclusive host wall-time intervals and
separate device activity windows. Device seconds must never be added to host
wall time. Unmeasured counters remain null, never zero.

| Device window | Baseline cold | Candidate cold | Baseline warm | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| SCF J (24/1 actual builds) | 198.520 | 99.807 | 8.231 | 4.158 |
| XC/AO/grid | 108.234 | 108.232 | 4.509 | 4.509 |
| Generated low force | 4.630 | 4.633 | 4.631 | 4.631 |
| Order-6 spatial J′ | 21.648 | 21.726 | 21.781 | 21.773 |
| Order-7 cooperative J′ | 28.034 | 28.034 | 28.035 | 28.034 |
| Order-8 materialized J′ | 5.772 | 5.772 | 5.772 | 5.773 |
| Other stationary force scope, including XC/Becke | 11.513 | 11.514 | 9.318 | 9.320 |
| One-electron pair metadata | 2.269 | 2.270 | 2.269 | 2.267 |
| Resident final-state validation | 0.018 | 0.018 | 0.018 | 0.018 |

The device J row combines equivalent exclusive nested J scopes: normal J is
tagged `direct_j`, whereas MD remains under `cuda_ks_exact_j`. K is not evaluated
by the pure-J r²SCAN endpoint; representative combined J/K controls are separate.

Complete exclusive **host wall** ledger (not device sums):

| Wall scope | Baseline cold | Candidate cold | Baseline warm | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| Preparation | 8.842743 | 8.920885 | 0 | 0 |
| Native SCF/publication | 318.490444 | 219.651607 | 12.837843 | 8.764471 |
| Stationary force/publication, excluding nested final-state stages | 78.062657 | 78.116434 | 72.308808 | 72.297560 |
| Final-state decode + snapshot | 1.594917 | 1.572083 | 0.964337 | 0.931048 |
| Final-state validation | 0.622042 | 0.606234 | 0.607266 | 0.609377 |
| D4 + gCP | 0.449001 | 0.451102 | 0.447899 | 0.447852 |
| Construction, remaining public driver and final return/sync | 0.006946 | 0.007530 | 0.006028 | 0.005780 |
| **Profiled total** | **408.068750** | **309.325875** | **87.172182** | **83.056088** |

All original per-call intervals and reconciliation errors are in the retained
summaries; rounding the table is not a replacement for those ledgers.
Synchronization API observations are separately recorded in
`profile/synchronization-and-resources.json`. Both arms observe 13,780 cold and
4,808 warm synchronize calls. The opt-in MD work census performs a pageable
copy/fence, moving blocked time between API names. Its lower event-wait sum is
**not** a separate production synchronization speedup and must not be added to
the wall/device totals. Preparation and all validation/force obligations remain.

Order-6/7/8 scientific kernels and all their measured executed work are unchanged
in both cold and warm profiles:

| Order | Executed primitive-pair products | Executed components | Coulomb preparations | Component contractions |
| --- | ---: | ---: | ---: | ---: |
| 6 | 2,400,504,912 | 1,121,296,224 | unmeasured | unmeasured |
| 7 | 1,250,544 | 490,435,456 | 1,250,544 | 1,937,135,584 |
| 8 | 95,976 | 107,642,016 | 95,976 | 270,222,624 |

These are actual counter definitions from the existing force census, not
hardware FP64 FLOPs. Some order-6 shell/preparation counters are unavailable.
The complete per-class counters retain nulls. There is no claim that this patch
reduces force arithmetic; the remaining warm bottleneck is still J′.

No zero-local-memory claim is made from relocatable object reports or Nsight's
launch-metadata zero: emitted PTX has explicit 5,760-byte preparation arrays in
both sources. Static preparation registers remain 64; density 54 → 48,
projection 42 → 40 and residual maximum 102 → 110. These are resource bounds,
not achieved occupancy or executed traffic measurements.

## Correctness, resource and negative evidence

- Focused host/API/resource suite: **343 passed, 9 opt-in GPU skips**.
- Resident CUDA/independent screened Libcint suite: **16 passed**, covering
  restricted/unrestricted, spherical/Cartesian, J/K, fallback and screening.
- Same focused binary under memcheck/synccheck/racecheck: **16 passed each**,
  zero errors/hazards/warnings. Host translated preparation passes ASan/UBSan.
- Independent long-double polynomial oracle checks 36 angular/layout/geometry
  combinations, structural signed zeros and nonfinite weighted-zero propagation.
- SCF/compiler structure checks: 250/541 modules, zero errors. Default promotion
  inventory: 67 entries, 117 audited controls; Ruff and clang-format pass.

Raw J diagnostics use an explicit matched 1 GiB test budget, not the README
default. Their MD-versus-normal 3e-11 comparator failed on atomic densities:
6.90e-11 at 48 and 9.46e-11 at 96. These failures are retained, not relabeled as
passing equivalence. At the three largest-disagreement AO pairs per size, an
independent screened Libcint oracle agrees with MD to about 1e-20 or better;
normal J supplies the discrepancy. Existing normal density-contribution
screening differs from MD's AO-only oracle. Dense-vs-sparse MD at 48 differs by
at most 1.60e-13. No production tolerance changes; complete E/F gates decide
endpoint qualification. Selected-pair checks do not prove whole-matrix parity.
`validation/resident-md-work.json` retains actual angular counter rows. For the
96-atom dense diagnostic, one MD J call executes 2,225,962,462 uniform radial
roots, 4,013,275,668 density-consumer directions, 95,324,371,844 contraction
summands and 242,399,046 residual roots (43,920,982 admitted residual tasks).
Its 526,842,030 residual candidates are enumeration, not executed recurrence
roots. Normal J has no equivalent census here; no invented before/after FLOP
ratio is asserted. These diagnostics are not full SCF endpoint populations.

An unnecessary optional-public-ledger prototype was removed after verifying the
README call graph. Its 31 GPU tests and sanitizer passes do not replace the
focused source's 16-test qualification. Its partial 96 population is incomplete.
The first focused attempt is also excluded: `scancel` left predecessor timeout
children alive, including a verified n5 GPU process. A canceled n4 predecessor
profile completed at 15:51:47, after the focused job began at 15:44:47. That
overlap invalidates its 542.922-second cold control, which is retained in
`trials/`, not hidden. Guarded replacement allocations first verify no existing
compute contexts; source revisions and numerical conditions are unchanged.

An intermediate build verification omitted checkout-root `CCACHE_BASEDIR`,
causing avoidable cache misses. Only the owned Ninja was interrupted; the shared
cache was not cleared/stopped. Both final source-matched builds use verified
ccache 4.5.1 and explicit CXX/CUDA launchers; logs/stats distinguish reuse from
compilation. Runtime caches are retained, not reset to manufacture cold wins.

## Reproduction

The retained `run-endpoint.sh`/`run-probe.sh` scripts document exact environments,
native-library selection, canonical inputs, reference handling, 1800-second
population bound and Slurm receipts. Paths point to the retained frozen source
exports under `/data/jzzeng/gqc-direct-md-20261016/`; adapt filesystem roots,
not scientific parameters. CUDA 12.9.1, GNU 11.4, Release sm120, shell/native and
stationary CUDA AOT enabled (`r2scan_rks;pbe0_rks`), CPU stationary AOT disabled.
Existing compiler caches are reused, with `CCACHE_BASEDIR` set to each checkout
root. No release, tag or merge is performed.

The archived guard records are from successful uninterrupted replacement jobs.
If canceling a run on these Slurm controllers, verify owned timeout/profiler
descendants actually exit before launching another allocation: `scancel` alone
was demonstrably insufficient. Do not override device visibility or let an
orphan continue outside its finite allocation.

## Default scope and next bottleneck

The optimized layout is enabled by default wherever the existing optional MD-J
source passes its original method/angular/resource guards, including the
unbudgeted README 96-atom endpoint. Existing explicit disable controls and
bounded fallbacks remain. Explicit public-ledger admission is **not** enlarged.

The historical GPU4PySCF warm number 26.59 seconds is not a newly rerun matched
reference: 83.562 seconds is still about **3.14×** that historical result. The
standalone profile leaves 21.773 + 28.034 + 5.773 seconds in order-6/7/8 J′,
and about 72.298 seconds in exclusive warm stationary-force/publication wall.
The smallest next step is a source-matched qualification stacking this layout
with the independently delivered #2558 force chain, then the remaining
XC/AO/grid (~108.232 cold / 4.509 warm device seconds). Do not subtract separate
PR campaign timings to invent a combined endpoint measurement.
