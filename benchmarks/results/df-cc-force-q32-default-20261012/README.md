# CUDA DF complete-force automatic primal Q32 cap

Force extension of #2197, isolated from #2298/#2299. Frozen measured parent:
`34dd51ed3ebc5b03b17f095de2cd7b39903923c8`, plus the retained one-file production
selector overlay. Both timed endpoint binaries and generated scientific kernels
are identical. Candidate requests **real automatic zero**, not explicit Q32;
baseline requests Q8. Lambda cap Q32, ordinary DIIS8, FP64, 64 GiB, native
ethane230/o9/v221/Q488. Four cold complete force processes, Slurm 2902/node2:

| Median seconds | Explicit Q8 | Automatic Q32 |
| --- | ---: | ---: |
| Complete process wall | 275.668013 | 257.361953 |

Complete observed saving **6.6406%**; both candidate samples beat both controls.
This is descriptive scoped evidence, not statistical/global performance or a
latest-master endpoint claim. Actual primal tile is 32. CC GEMMs
54,862 -> 25,792, Q tiles 2,806 -> 1,096; physical GEMM and semantic contraction
summands are unchanged. Evaluations 38, Lambda actions 22, orbital actions 13.

CCSD logical numeric capacity rises **4,392,162,937 -> 8,231,194,433 bytes**
(+3,839,031,496, ~3.58 GiB). Its device arena rises
3,981,577,984 -> 7,820,543,744 bytes. Complete logical capacity remains
19,509,839,897 because a later force phase determines its maximum; this must
not hide the CC-stage increase. Physical peak VRAM is not measured. Explicit
positive caps, dimension/budget admission and bounded allocation fallbacks stay.

All 38 independent force/public cases and four large same-Hamiltonian energy,
fresh residual, Lambda/Z and independent force-direction gates pass. The initial
12 private small-force cases retained their explicit Q8 adapter; all 12 are
additionally qualified through the rebuilt adapter with
`GENERATIVEQC_TEST_CCSD_Q_CAP=0`. Public cases already use the production default.
Both sets of receipts are retained. Water memcheck: zero errors. Independent
compiled work queries recompute every CC aggregate. Baseline independent suite
is reused because its binaries are unchanged.

Raw receipts include exact lengths/SHA-256, runner, finite Slurm recipe, device
and binary identities, test XML, all endpoint JSON/resources, work queries,
baseline source and separately rebuilt test-adapter source. Reconstruct the
production owner with `production-source.patch.gz`. Reproduce with the retained
updated qualifier and matching staged binaries:
`srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1
--cpus-per-task=2 --time=00:30:00 bash qualify-feature.sh q32-default
candidate-q32-default 0`. Preserve Slurm's assigned device visibility. No Release,
release asset/tag, publishing workflow or external archive is created.
