# Decision: preserve bounded point batching as density-product jets grow

Status: implemented; qualified complete endpoints, bounded incumbent retry retained
Date: 2026-10-15

The [live-panel capacity extension](2026-10-24-xc-capacity-by-live-panels.md)
subsequently includes AO jets in the automatic allowance. The historical
measurements and density-jet-only decision below remain unchanged.

## Problem

The rejected force-point launch experiment in
`../../rejected/2026-10-15-stationary-point-launch-occupancy.md` improves its point
probe but regresses complete M06-2X endpoints. That change is reverted. A real
README96 DF warm profile instead attributes 3.701023 device seconds and 13,824
launches to the SCF equal-spin point evaluator across three physical XC/Fock
evaluations. Its original AO domain has 9,216 tiles. The existing fixed 32-MiB
optional batch allowance is shared by AO, density-product, feature and local
potential panels. Four density-product jets consume the allowance much earlier
than the ordinary one-jet case, limiting independent point supply even though
the requested batch count remains 32. This is distinct from #2486's rejected AO
compaction, matrix-tile, dddd and order-seven-page changes.

## Decision

Scale only the ordinary unbudgeted automatic optional allowance by the validated
density-product jet count: 32 MiB for one jet, 128 MiB for four. Keep the existing
compiler planner, original AO maps, compact/ragged admission, stream/scatter
order, FP64 point algebra and requested maximum of 32 original tiles. This is a
resource-shape rule, not a method, spin, molecule or device whitelist.

Explicit `GENERATIVEQC_CUDA_XC_BATCH_BYTES` values remain exact byte caps and
are not multiplied. Tile-count and zero-byte opt-outs retain their semantics.
Public device ledgers still admit zero optional bytes; unused fleet/force
capacity is not lent to this path. The larger default does not enlarge a public
resource estimate or skip accounting of the allocated optional panel.

If the automatic larger allowance cannot retain a batch, retry the previous
32-MiB allowance before falling back to one tile. Without this retry an OOM
could replace a formerly fitting small batch with a slower one-tile executor.
Runtime/driver failures and host allocation failures still propagate; they are
not converted into arithmetic retries. Do not reinterpret explicit byte caps.

## Source identity and endpoint protocol

Control source is master `5f80ba629b5f20412215cdc690f8fa458ac3fdf8`.
The capacity candidate has a separate archived checkout/fresh cached Release
build, not the rejected launch experiment's image. All real GPU work uses finite
Slurm allocations and preserves assigned visibility. No independent density
seeds production, and no convergence/accuracy tolerance is relaxed.

The initial same-binary capacity-only pilot uses the existing explicit byte
override, with 128 MiB before 32 MiB. The 128-MiB DF96 population passes all
twelve independent full E/F gates with 18 cold / 12 moved / three warm Focks;
warm medians are approximately 24.52 s instead of the baseline's approximately
29 s. This motivates qualification, not an assertion about the new default's
source-matched endpoint population.

Both images use GCC 13.3, CUDA 12.9.1, sm_120, Release and the same five
stationary AOT profiles in both s/p and s/p/d domains. The fresh capacity build
uses verified ccache 4.5.1 with explicit CXX/CUDA launchers and checkout-root
`CCACHE_BASEDIR`; its cumulative cache counters increase by 536 hits and two
misses. The measured production file SHA-256 is
`5db62d088389c9fa6d8b240de58c1fa20fc8e5300c5756207d5f39fd53613918`.
The control/capacity native library SHA-256 values are respectively
`dc20c3978813a3a998a01b1f9984123662cb2d24e72bf69775b8816e53e05d4e`
and `6d2e59986eda54ccfb36ce03f4661495b7adce07a066055233c1ef5413d69ed0`.
Installed-package audit authenticates all ten artifacts, with native cubins and
no required driver PTX JIT. Audit/build logs remain diagnostic artifacts.

The complete README96 protocol uses spherical def2-SVP, 768 AOs, the unpruned
48x16x32 moving grid, strict FP64, energy/density/screening tolerances
1e-12/1e-10/1e-12, and cold + five warm + moved + five moved-warm calls.
DF uses cc-pVDZ-JKFIT with the original auxiliary-response contract. Each
engine freezes its own post-cold/post-move density for warm replays; independent
reference densities are never imported. The original independent gates remain
1e-8 Eh and 1e-7 Eh/Bohr. Complete cold time includes construction/preparation;
process startup and provenance collection are excluded. Compilation caches are
retained. Eight OMP/OpenBLAS/MKL threads are used, and only one owner/worker is
resident at a time.

## Completed default populations

All rows below are full E+F endpoint times in seconds, not kernel estimates.
Warm columns are medians of all five calls. Both DF arm orders pass every gate
and retain 18 cold / 12 moved / three warm Focks.

| DF96 population | Arm | Complete cold | Warm | Moved | Moved-warm |
| --- | --- | ---: | ---: | ---: | ---: |
| n5/1596, control first | control | 123.211800 | 29.017255 | 91.641966 | 29.007583 |
| n5/1596, control first | capacity | 96.374500 | 24.528651 | 73.742932 | 24.518545 |
| n1/8561, capacity first | control | 123.724984 | 28.736473 | 91.619474 | 28.747289 |
| n1/8561, capacity first | capacity | 102.746640 | 24.214325 | 73.649822 | 24.208127 |

This is a 15.47--15.74% warm and 15.48--15.79% moved-warm time reduction.
Complete cold improves 16.96--21.78%, and moved improves 19.53--19.61%.
Maximum DF errors across both images/orders are 2.046364e-11 Eh and
1.457888e-10 Eh/Bohr. These gains are relative to the already merged integer
power/CSE baseline, not a re-credit of #2505/#2511/#2464.

The first direct96 order (n1/8561, capacity first) also passes all 24 gates:
control/capacity warm medians are 16.932912/16.208507 s, moved-warm
17.201116/15.429428 s and moved 94.930269/76.886866 s. Complete cold is
125.034848/111.038131 s, **but uses 19/18 Focks**; that comparison is not
equal-work proof. Warm Focks remain one and moved Focks twelve.

The opposite direct96 order (n1/8588, control first) passes all 24 gates:
control/capacity complete cold is 124.897988/110.488939 s (again 19/18 Focks),
warm 16.943778/15.582227 s, moved 94.695283/76.670775 s and moved-warm
16.950283/15.673172 s. Both orders reduce warm time by 4.28--8.04% and
moved-warm time by 7.53--10.30%. Do not replace this range with one favorable
sample or claim matched direct cold trajectories.

## Production profile, distinct from clean timings

The source-matched capacity profile (n1/8577) captures the same three-Fock warm
consumer. Point submissions fall from 13,824 to 864 (two original tiles per
submission to 32), and point-device time falls from 3.701023 to 0.264643 s.
The original grid still contains 9,216 tiles per XC evaluation; the scientific
point domain is not pruned. Original active maps, AO visits, density-product jet
work and actual XC evaluation counts are retained. Compact-contraction versus
tiled-fallback admission is still decided by the existing compiler planner.
No whole-endpoint bitwise-identity claim follows from unchanged scalar point
algebra; admitted contraction schedules can round differently.

The separate profile changes CUDA event-trace settings relative to the original
diagnostic capture; neither profile's instrumented endpoint time establishes
the clean timing claim. Unavailable integral execution counts remain unknown.

## Fallback, numerical and cross-shape qualification

Executable host tests compile the production KS preparation policy and actual
planner/ledger with allocation doubles. Both spins verify four-jet capacity,
unchanged one-jet selection, exact explicit opt-outs, refusal of the larger
allocation with a fitting 32-MiB retry, and final one-tile fallback. The focused
resource/planner/compact suite passes 60 tests; the broader source-matched native
suite passes 416 tests with eight explicit skips. The promotion inventory audits
rollback-control ownership rather than pinning the C++ default-expression
spelling; executable policy tests protect actual capacities and fallback behavior.
The UKS fixture may use the same
point count while gaining compact-contraction admission: do not assume a larger
allowance must strictly increase every selected group size.

The 48-atom M06 RKS/closed-shell UKS, PBE/PBE0/r2SCAN forward/reverse cohorts
complete all 240 gates (n1/8563 and n1/8578). They use 16 perturbed waters,
sto-3g, a 24x8x16 grid and unchanged 2e-9 Eh / 2e-7 Eh/Bohr gates. M06 UKS
warm reductions are 2.11--2.55%, moved-warm 2.59--4.00%; M06 RKS is essentially
flat. The one-jet PBE/PBE0 controls show roughly 0.2--2.6% warm variation,
including small slowdowns despite unchanged one-jet resource selection. Do not
declare universal non-regression from those populations. Four-jet r2SCAN warm
variation is +0.2--0.6%, and moved-warm is -0.1--+0.2%, not a demonstrated gain.
An additional opposite-order cohort (n1/8589) uses eight explicitly allocated
CPUs and core binding, with all 240 independent gates passing. PBE warm
differences shrink to +0.64/+0.67%, moved-warm +0.16/+0.27%; PBE0 warm
+0.44/+0.66%, moved-warm +0.21/+0.40%. r2SCAN spans -0.73--+0.74% across
the two warm phases; M06 RKS spans -0.45--+1.84%. These observations are
retained, not relabeled as gains.

Four further fresh workers all use the **identical control library** under the
same eight-CPU/core-binding policy (n1/8591). All 96 calls pass the unchanged
gates. Their per-worker PBE warm/moved-warm medians span
0.536269903--0.548018809 s and PBE0 spans 0.621749036--0.628920861 s,
larger than the corresponding small paired differences. A/A changes include
both signs. This does not prove a zero-percent regression or supply a
statistical confidence interval; it prevents diagnosing sub-percent sequential
differences as a demonstrated policy loss. The single-jet allocation/planner
selection is also unchanged by executable contract. Revisit if a sufficiently
controlled experiment resolves a repeatable loss outside this run variability.
The queued n5 repeat (1600) is canceled before allocation: it executes no GPU
work and is neither a failing nor passing numerical population.

Retain complete times, Fock counts, independent gates, resource admission and
unavailable work as unknown. Existing
48-atom own-SCF analytic references may be replayed only as E/F oracles; their
densities are never imported. Missing force telemetry for a native integrated
route is unknown, not an execution failure or fabricated zero work. A first
small harness attempt stops after 24 successful M06 calls because it assumes
that telemetry exists; the corrected complete cohorts do not fill it with zeros.

Compute Sanitizer n1/8570 passes eight complete cold/warm calls for M06 RKS48,
closed-shell M06 UKS48, r2SCAN RKS48 and genuinely polarized H3 M06 UKS
(multiplicity two), with zero memory errors. The H3 oracle reconverges its own
PySCF SCF and moving-grid analytic gradient; sharing the quadrature is not
sharing a density. Instrumented times are not performance samples.

## Retention and reproduction

`benchmarks/results/xc-point-capacity-20261015.json.xz` retains all 576 clean
default-policy calls, including complete cold preparation, both orders, every
small control and every negative timing. It is 4,972 bytes (SHA-256
`ba5214d437be8cb80709a3b1fbc7daf37ecdf81e35463766568fbbe2e0dcc5e3`),
within the existing aggregate evidence budget without deleting older negatives
or raising that budget. The same-image noise comparison retains all sixteen
per-worker phase medians as its four independent worker observations, rather
than treating their nested replays as independent workers. Raw full E/F arrays,
all A/A calls and the motivating same-image byte-cap pilot remain diagnostic
artifacts, not additional default-speedup claims.

The compact JSON uses the source/build identities above. Its reference table is:
DF96 `49d8850c933b6f6e508264ccc405e81eae1c421a76c0d3eae1e4e2af0733bf6c`,
direct96 `4f31261489c0c3d0a9d7af5ba17979110c5953edc787dab365a0702b0ecc43be`,
and the independent 48-atom fixture
`ae426e4b951539f1c0a6f59a0950940cee6316a880b3c9777793e4b8cb194508`.
Each cohort also binds its original JSON hash and Slurm job. Scalar columns
`[scale, integers]` decode as `math.ldexp(integer, -scale)` without rounding.
Per-method records are cold, five warm, moved, five moved-warm. `prepare`
contains the cold preparation only; other preparation values are zero and
complete time is `seconds + prepare`. Iterations equal Fock counts unless an
explicit column overrides them. Status zero, convergence true and gate true
apply to every row unless explicitly overridden. Warm-state flags are bitmasks
by row index. `maximum_errors` contains exact cohort maximum energy/force errors,
not per-row error arrays. Work rows contain only the named observed fields;
`null` is unknown, especially absent integrated-force telemetry and integral
execution counts. A/A median order is worker 1--4, then PBE/PBE0, then
warm/moved-warm; all 24 calls per A/A worker pass the same scientific gates.

Reproduce README populations with the source-matched Release native library and
packaged AOT inventory using `benchmarks/run_readme_dft_benchmarks.sh reference`
followed by `benchmarks/run_readme_dft_benchmarks.sh native`, each in a finite
Slurm GPU allocation. Set `README_BENCHMARK_METHODS=m06-2x`,
`README_BENCHMARK_SIZES=96`, `README_BENCHMARK_REPEATS=5`, and separately
`README_BENCHMARK_DENSITY_FITTING=cuda` and `none`. Use distinct output roots
and reverse image order for the second pair. Do not set any XC batching
override when qualifying the default. To reconstruct the measured candidate,
apply only this change's `src/dft/cuda_ks.cpp` diff to the base revision and
verify the production-file identity above; test/documentation changes are not
part of the measured native image.

The small fixture is deterministic: start with O `(0.1,-0.1,0)`, H
`(0.1,0.2,1.7)`, H `(1.6,-0.2,-0.5)` in Bohr. For water index 0--15, scale
these positions by `1 + 0.002*index` and translate by
`(8*(index % 4), 8*(index // 4), 0)`. Move atom index one by
`(0.015,-0.01,0.02)` for the second geometry. Use sto-3g, the explicit
24x8x16 grid, multiplicity one, native direct FP64 and
`Calculator.prepare_batch(..., warm_start=True)`. Execute energy and forces,
synchronize before/after timing and freeze warm updates after each first
geometry call. Run M06 RKS/UKS, PBE, PBE0 and r2SCAN. Independent reference
energies/analytic moving-grid gradients come from
`tests/python/test_dft_complete_cpu.py::independent_global_hybrid_gradient`,
which runs its own PySCF SCF; M06 uses Libxc `450,236`. Reusing that reference's
E/F values is permitted, never its density. The genuinely polarized H3 check
uses H `(0,0,-1.4)`, `(0.15,0.1,0)`, `(0,0,1.4)` and multiplicity two.

Raw scripts, source patches/build/cache receipts and successful/failed diagnostics
remain in `.artifacts/xc-point-warp-20261015/` and
`/data/jzzeng/qc-xc-point-warp-20261015/`. The rejected launch experiment and
setup/time-limit failures remain separately identified. The initially missing
profile basis path and an invalid package-audit profile spelling are setup
failures, not failed physics or successful qualification. No release, tag or
external backup is created.

## Consequences and revisit

Retain the bounded automatic 32-MiB retry and explicit overrides. This buys
independent point supply with up to 96 MiB more optional storage per ordinary
four-jet owner; it does not lend unused public/fleet/force reservations to XC.
No universal speedup, literal zero-regression guarantee or bitwise complete-SCF
identity is established. Revisit with source-matched, own-seed, full E/F
populations on additional shapes/devices, especially a resolved regression,
memory-pressure failure or evidence that equal jet work should use another
resource policy. Do not substitute a functional whitelist or rerun #2486's
negative candidates as though they were new work.
