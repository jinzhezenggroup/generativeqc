# Shared stationary XC/VV10 force grid qualification

This source-matched campaign compares the guarded default with
`GENERATIVEQC_STATIONARY_SHARED_GRID=0` in the same native image on actual A100
and RTX5090. It is independent of unmerged PR #3020. The cohort is 96-atom water,
WB97M-V, spherical def2-SVP, full Direct, Standard strict FP64, public default
quadrature and convergence. Cold includes constructor + prepare + SCF + analytic
forces + host export; warm replays a frozen independently produced native seed.
No production PySCF/GPU4PySCF imports are permitted.

## Complete endpoints

| GPU / allocation | Arm | Cold / s | Warm samples / s | Warm median / s |
|---|---|---:|---|---:|
| A100 / LSF225900 | off | 111.904928 | 15.116480, 14.991422, 15.012093 | 15.012093 |
| A100 / LSF225900 | default | 112.897774 | 14.967766, 14.804610, 14.831869 | 14.831869 |
| RTX5090 / Slurm16589 | off | 271.771667 | 14.915725, 14.829264, 15.367384 | 14.915725 |
| RTX5090 / Slurm16589 | default | 193.335207 | 13.614976, 13.685334, 13.649965 | 13.649965 |

A100 warm median improves **1.20%** (0.180224 s), RTX5090 **8.49%** (1.265759 s).
Every warm candidate/control ratio is below the retained 1.05 guard. These are
three frozen-warm samples per arm in one sequential paired cohort, not a universal
zero-regression proof. A100 cold is **0.89% slower**, and cold plus three warm
calls is **0.30% slower**: this change does not establish an A100 cold or aggregate
win. The very large RTX5090 cold difference exceeds what this removed panel work
can explain; retain it as an observation, **not a causal 28.9% cold speedup**.
No parameter sweep or repeated cohort was used to select a favorable cold result.

Both arms retain 18 cold Focks (13 provisional / 5 target), four SCF VV10 pair
sweeps and one Fock per warm call. These pair counts are SCF-only, not an
endpoint-wide force census. The separate warm force diagnostics record one exact
pair-cache hit and zero force pair sweeps in both arms.

## Executed work

Both GPUs' separate same-owner warm diagnostics show:

| Semantic work | Off | Default |
|---|---:|---:|
| Evaluated grid points | 2,156,672 | 1,078,336 |
| Evaluated tiles | 2,108 | 1,054 |
| Active point-AO visits | 497,699,712 | 248,849,856 |
| AO jet values | 4,976,997,120 | 2,488,498,560 |
| Projection FMA pairs | 539,021,746,688 | 269,510,873,344 |

The 1,078,336-point physical grid and its partition/weight response are unchanged.
Both models still execute their own geometry contractions. There is no extra
full-grid resident panel, device allocation or resource tier. Different point or
topology owners and mixed grid/geometry precision retain the two-pass fallback;
there is no GPU-model or molecular-size whitelist.

Complete endpoint timing runs with DF/progress tracing disabled. After all timed
samples, one additional frozen-warm replay of the same owner records intrusive
force work. That diagnostic is not a clean timing sample: DF traces can disable
optional MD-J. Projection FMA pairs are semantic counts, not hardware instructions.
Both geometry consumers still run; reduced panel work is not a proportional
endpoint claim. The automatic-artifact work ABI does not admit this curated
WB97M-V recipe; the initial collector failure is retained, not promoted as data.

The frozen independent GPU4PySCF observables in
`../direct-range-warm-anchor-20261105/publication.json.xz` supply energy and
analytic-force gates only, not a fresh competing latency ranking. Acceptance:
energy/atom <= 1e-5 Eh, maximum force error <= 1e-4 Eh/bohr, force RMS <= 3e-5.
This is same-prescription default-grid implementation parity, not independently
grid-converged accuracy or universal device performance.

The offline replay checks **40** independent E/F repeat-pair comparisons, including
the separate diagnostic observables. Worst errors (energy/atom, force max, force
RMS) are **7.130e-12 Eh, 3.553e-7 Eh/bohr, 1.558e-7 Eh/bohr**. A historic
reference's latency is never relabeled as a current GPU4PySCF ranking.

`test_stationary_shared_grid_cuda.py` additionally reconverges an independent
same-moving-grid oracle for original/displaced structures, disable,
minimum-capacity and distinct-grid fallback. The existing native nonlocal-reuse
suite covers fitted J/K plus Direct LR, exact-key replay, pair-schedule changes,
failed solves and geometry-owner replacement. These intrusive tests are not used
as complete-endpoint timing evidence.

Eleven production-image numerical/work cases pass. The first run's failure case
cannot find a test-only runtime injection symbol in the production build; this
infrastructure failure remains in the bundle. A separately identified image,
reusing production objects except `cuda_ks.cpp` compiled with
`GENERATIVEQC_TEST_HOOKS=1`, passes the failure/recovery case in Slurm16588. It is
never used for endpoint rankings. Host policy/binding/options tests: **271 passed**.
Slurm16587's earlier automatic-only ABI collector failure and cancelled follow-on
are retained as incomplete, unqualified endpoints rather than timing samples.

## Identity and reproduction

- Production source: `391b1896a8cc8606cc22f0a1f7747c8bdd5480fd1242a677c03aefd2d73f84c8`.
- Native ELF: `9d89c53fbf5e98090c5738c1627610b174a0bd0b039b70d4843cf9a756fd913f`.
- Measured base: `32b4942de70ff4ac6ec4701063f4b537daf096a9`.
- PR synchronizes documentation/evidence-only master `b681a1fc34279aa2b77de44b9aa5f42df6b31093`;
  the complete production source identity remains exactly unchanged.
- Build: Release, CUDA 12.9.1, GCC12, `80-real;120-real`, verified sccache0.16.0;
  cache statistics, actual scheduler/device receipts, runners, raw work traces,
  test failures and independent-reference identity are retained in `evidence.json.xz`.

The initial finite full-image SCP transfers were truncated. Independent remote
hash checks blocked GPU submission. The final A100 payload reconstructs the
**exact same ELF bytes**, reusing byte-identical sections of the previous frozen
image only after checking its full SHA256, then checking the full reconstructed
target SHA256. This is a transport optimization, not a new build, precision change
or source relabeling. The source archive omits historical benchmark results only;
the complete production closure is independently authenticated before execution.
The borrowed build owner is restored and the owned transfer server is stopped.

For scheduled numerical checks, set `GENERATIVEQC_TEST_WB97MV_CUDA=1` and run
`tests/python/test_stationary_shared_grid_cuda.py` and
`tests/python/test_native_stationary_nonlocal_reuse_cuda.py`; failure injection
requires the separately identified test-hook build. All real GPU execution must
remain inside a finite `srun` or the ECNU guide's finite LSF allocation, preserving
scheduler device visibility. The exact off/default complete endpoint wrappers
are in the evidence bundle; they use the public defaults benchmark and forbid
production reference imports. Tracking memory: #2976.

Run `python benchmarks/results/shared-force-grid-20261106/replay.py` to replay
the retained scientific, work, capacity and endpoint gates without a GPU.

The promotion is the guarded strict-FP64 default, with
`GENERATIVEQC_STATIONARY_SHARED_GRID=0` retaining the causal control/rollback.
