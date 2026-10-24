# Decision: further reduce Fast provisional integral-source work

Status: implemented
Date: 2026-10-24

## Baseline and question

Start from `cc9b0c041`, including PR #2712's independently qualified `1e-8`
proposal-source policy and its lifecycle/replay test fixes. The pilot started while that PR was open; this is a follow-up to its policy.
The integration baseline used for final qualification is recorded below.
The prior 96-atom profile contains 1,035,084,412 FP32 source values, about
3.5 seconds of source preparation and seven mixed Focks before two strict
closing Focks. The source and repeated contractions remain material costs.

Test `1e-7`, `1e-6` and `1e-5` source predicates against `1e-8`. All are only
proposal heuristics: no accumulated-error guarantee is inferred from a Schwarz
cutoff. Fewer source rows are useful only if the complete endpoint improves;
extra mixed or strict iterations can erase the gain.

## Invariants and protocol

Keep the existing model, size, resource and lease-lifetime admission. Never
reuse approximate values for strict closure or analytic derivatives. The
primary screening, two fresh physical closing Focks, final validation and
independent energy/force gates remain unchanged. No reference density enters
native execution. Retain failed candidates and all measured samples.

Build each exploratory predicate separately with a verified compiler cache;
record its changed input and binary identity. Compare fresh processes in
alternating order under finite Slurm allocations with their assigned devices.
Include complete preparation/cold energy/forces and two frozen warm calls.
Pilots select a candidate but do not constitute its final qualification.

For a promising candidate, freeze the final source and run five alternating
baseline/candidate pairs at 12/48/96 atoms. Reduce warm repeats per process,
then compare process medians and MADs. Require the 96-atom cold gain to exceed
max(2%, twice the sum of relative MADs), and no sampled size/phase regression
beyond that floor. Add original/moved/frozen-moved/reset PBE0/B3LYP/M06-2X,
Standard/DF exclusions and optional-source refusal controls.

Independent full-density GPU4PySCF gates remain `2.5e-6 Eh/atom`,
`2.5e-5 Eh/Bohr` maximum force and `7.5e-6 Eh/Bohr` RMS force. This is
stock-grid solver qualification on sampled geometries. It establishes neither
grid convergence nor a universal method ranking. Promote only after these
gates pass; otherwise retain the prior policy and the rejected evidence.

## Pilot decision and integration baseline

The first pilot used eight fresh processes (forward/reverse cutoff order) on
n1, and the cross-size/model pilot used fifteen on n4. All 69 complete E/F calls
passed the predeclared independent gates. The 96-atom n1 cold medians were
25.064915, 23.793457, 24.066762 and 23.142286 seconds for `1e-8`, `1e-7`,
`1e-6` and `1e-5`. The latter two needed a third strict Fock; `1e-7` retained
seven mixed and two strict Focks. On n4, the 48-atom single-process cold values
were 9.6802, 9.1583 and 9.3546 seconds for `1e-8`, `1e-7` and `1e-5`.

Select `1e-7` for final qualification. It preserves the sampled strict work,
beats `1e-5` at 48 atoms and retains larger force-error margins. Do not introduce
a size-specific cutoff merely to select the best pilot sample per size. These
exploratory timings are not the final promotion population.

PR #2712 then integrated master as `46f0501a92c289675647d594fec14d04ca365639`,
including the public Schwarz and DF raw-J work. Rebuild both policies from that
same base before final qualification. No old/new baseline timings are pooled.
The final source manifest expands driver coverage to all tracked benchmark
Python inputs (2,797 total production/build/driver inputs).

The first n4 cross pilot could not load the library: its older Python NVIDIA
wheel preloaded a cuSOLVER without `cusolverDnXsyevBatched_bufferSize`. All
fifteen processes failed before calculation; none is numerical/timing evidence.
The corrected launcher pins the coherent CUDA 12.9 userspace libraries before
wheel discovery and retains the node's driver and scheduler-assigned visibility.
A private GLIBC compatibility loader is required for the n1-built core on n4.
Retain the failed attempts and exact corrected launcher with the receipt.

## Final decision

Use `1e-7` for the admitted proposal source. The implementation changes only
`mixed_source_proposal_screening`; all admission, lease retirement, strict
closure and derivative ownership remain with the existing owners. It supersedes
the numeric predicate in the [prior decision](2026-10-23-fast-provisional-source-screening.md),
without broadening its domain or weakening its acceptance gates.

| Atoms | Phase | 1e-8 median (s) | 1e-7 median (s) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 12 | cold | 2.095577 | 2.087023 | 0.41% |
| 12 | warm | 0.373354 | 0.374638 | -0.34% |
| 48 | cold | 9.028520 | 8.685979 | 3.79% |
| 48 | warm | 2.306222 | 2.311384 | -0.22% |
| 96 | cold | 23.520148 | 22.281159 | 5.27% |
| 96 | warm | 6.530802 | 6.539532 | -0.13% |

Five alternating pairs at each size pass the predeclared gain/nonregression
criteria. The final receipt contains 51 processes and 183
complete E/F calls, including lifecycle, excluded-owner, capacity and rollback
controls. Maximum independent errors are 1.84173e-10 Eh/atom,
1.13915e-05 Eh/Bohr maximum force and 3.38419e-06
Eh/Bohr RMS force. Small and warm differences below the noise floor are not
speedup claims. Native incremental/DF cold-factor/Fast anchor regressions and
eight public GPU cases also pass.

The separate 96-atom source diagnostic records 1,035,084,412 versus
813,497,268 values (21.41% fewer),
saving 886,348,576 FP32 storage bytes. Cold work
remains seven mixed and two strict Focks. The receipt keeps every process's
iteration, precision and source counts; diagnostic intervals are not pooled with
clean endpoint timing.

The CUDA 12.9 / GNU 13.3 / sm_120 Release build uses verified sccache 0.16.0.
Both libraries and all three force AOT manifests have recorded binary/source
identities. The final 2,797-input map differs only in the proposal policy header.
Finite Slurm allocations retain their assigned device visibility. Clean timing
runs on n4; controls and native/public GPU regressions run on n1.

[Retained evidence and executable replay](../../../../benchmarks/results/fast-source-floor-20261024/README.md)
include the raw observable arrays, references, all samples, failed setup attempts,
source/binary/AOT identities, cache snapshots and exact runners. The replay uses
explicit exceptions, so numerical and population gates remain active under `-O`.

## Revisit when

Broader geometries, shell classes or models justify changing the floor or domain,
or an endpoint profile shows source work is no longer material. More aggressive
floors require new complete-endpoint and independent observable qualification;
the lower source count of `1e-5` alone does not justify its extra strict work.

## Evidence storage prerequisite

The receipt initially exceeded the existing aggregate limit by 296,064 bytes.
Reuse PR #2711's exact storage-only patch at `26c8522cb8cd552243266418ee11316a0e7b229f`
to reclaim 1,152,291 bytes. A one-off comparison confirms all 85 original XC #161
files remain byte-exact against the measured parent; all 2,797 qualified inputs
remain unchanged. The compaction's existing corruption/restoration tests and
original manifest gates remain active. No sample, scientific gate or budget is
removed or relaxed.

## Integrated master requalification

After #2712 merged and #2700 added DF charge partitioning, rebuild both policies
at master `937102f71218a7f50e19fd8bb4b2697d0df3dd08`. The only baseline production-input changes from the first
cohort are `src/scf/cuda/df_md_j.cuh` and `src/scf/cuda/df_md_owner.cu`; the three
force AOT binaries remain byte-identical. Repeat the entire protocol on the
integrated source: five alternating pairs per size, all lifecycle/exclusion/
capacity/rollback/diagnostic controls, three native suites and eight public GPU
tests. This adds a separate 51-process / 183-endpoint cohort; the two cohorts
are never pooled for timing. All 366 qualified endpoints pass their independent
observable gates, in addition to the 69 separate pilot endpoints.

| Atoms | Phase | 1e-8 median (s) | 1e-7 median (s) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 12 | cold | 2.232567 | 2.223318 | 0.41% |
| 12 | warm | 0.377756 | 0.378566 | -0.21% |
| 48 | cold | 9.128436 | 8.764538 | 3.99% |
| 48 | warm | 2.315570 | 2.311254 | 0.19% |
| 96 | cold | 23.599389 | 22.378540 | 5.17% |
| 96 | warm | 6.525197 | 6.526264 | -0.02% |

The integrated cohort passes the same gain/nonregression floors. Its maximum
errors are `1.84177e-10 Eh/atom`,
`1.13915e-05 Eh/Bohr` maximum force and
`3.38419e-06 Eh/Bohr` RMS force. The 96-atom diagnostic still
records 1,035,084,412 versus 813,497,268 provisional values and seven mixed/two
strict Focks. Both builds' compiler-cache snapshots and source verification
receipts are retained.

The linked receipt retains both complete cohorts and their separate binary/source identities.
