# Original-input KS final-state controls (2026-11-09)

**Overall decision: inconclusive; no issue auto-closure or production promotion.**
This is a current-source numerical diagnostic for #2840 and a bounded,
incomplete attempt at #2915, not a reconstruction of the historical #2910 or
#2287 binaries. Original historical failures retain their original identities.

## Measured source and device

- Base revision: `24a22da0e0684ad8d195e0d401312e5a278da2d0`, dirty native
  candidate reconstructed by `native-source.patch`.
- Baseline source identity:
  `a305bde4a845e554e5f04ca4ab593b26166b7e31902635dcd5a900194532aea3`.
  Loaded library SHA-256:
  `d6383c769ed44e089a61b731f2e3614032ee7e2cbc0b9ac321955fa1a58d5604`.
- Candidate source identity:
  `254e1db0bcf1401bca2e350a56132334cdd9d58f2a909f309aa9902cf32c0d75`.
  Loaded library SHA-256:
  `7bd4802cf734ffe38de4da24f0f8231642814fc5decc2f6510000392148681ef`.
- RTX 5090, UUID `GPU-d0b9e1b7-2083-6198-da50-c3e55693691c`, driver
  `580.95.05`, 32607 MiB; Slurm node1, job 16985, assigned visibility `1`.
  All GPU commands ran through finite `srun` allocations without modifying
  scheduler-assigned visibility. No A100 or PRO6000 qualification is implied.
- Matching Release/sm120 builds, GNU C++ 11.4.0, CUDA 12.9.86, verified
  sccache 0.16.0 launchers, native test hooks and native stationary-force
  profiles `pbe_rks;pbe_uks`. `GENERATIVEQC_CUDA_FAST_COMPILE=ON` makes this
  **diagnostic numerical evidence only**, not production/performance promotion.
  Integral-shell and automatic-XC AOT and Python stationary compatibility AOT
  were disabled. Python 3.13.9, NumPy 2.3.5, PySCF 2.14.0.

## #2840: UKS controls

The retained water-dimer cation uses Cartesian/spherical def2-SVP, PBE-UKS,
Q=1/M=2, grid 32/8/16, no DF, screening `1e-14`, energy/density `1e-11`/`1e-9`,
maximum 200 iterations. Receipts retain its geometry. Candidate controls cover
masks 0/127, E/E+F, and cold/warm/repeated-warm/moved/warm-moved phases, without
changing scientific iteration budgets or gates.

The same-device baseline probes are four cold consumers with the default
mask setting, **not all eight candidate controls**:

| Representation | Consumer | Baseline disposition |
| --- | --- | --- |
| Cartesian | E | nonconvergence, iteration 35 |
| Cartesian | E+F | nonconvergence, iteration 35 |
| Spherical | E | success, iteration 107 |
| Spherical | E+F | nonconvergence, iteration 85 |

The candidate passed **8 tests in 102.04 seconds: 40 final states and 20
analytic-force states**. Each phase authenticates its own native final state,
retains the independent unshifted canonical-density gate `1e-8`, and checks
independent PySCF energy/density/Fock with tolerances `1e-8`/`1e-6`/`1e-6`.
For forces, an independent PySCF PBE stationary solve (gradient tolerance
`1e-9`) and full moving-grid analytic derivative check use `3e-5 Eh/Bohr`.
The native density seeds only the same stationary basin; native Fock,
derivatives and acceptance flags do not replace independent equations. Python
scientific force fallbacks are forbidden in the native endpoint.

- Maximum canonical-density reconstruction error:
  `9.26770282738687e-9`, below `1e-8` without a waiver.
- Maximum independent force error: `1.2819691391774413e-8 Eh/Bohr`.
- Cartesian cold/moved convergence: iterations 31/23. Spherical cold: 89--107;
  warm replay: one iteration. These are not performance claims.

The exact native `generativeqc_ks_cuda_tests --df-range-exchange` fixture also
passed for RKS/UKS DF-primary J/K plus Exact Direct LR-K, including independent
CPU energy/exchange/state checks. This is **not** a complete independent RSH
force receipt and does not finish #2840's separate original RSH force gate.
Actual CUDA Graph execution is not asserted by a requested mask; those
execution/fallback dispositions were not independently captured in this run.

## #2915: original-grid water96 remains unqualified

The added opt-in test pins the original spherical def2-SVP orbital/auxiliary
bases, 512 MiB DF budget, default Treutler level-3 grid, FP64, force tile 1024,
energy/density `1e-9`/`1e-5`, two-item cold batch, invalid-import atomic
rollback, frozen replay, atom-1 z displacement `0.025 Bohr`, and moved/fresh
comparison. A distinct arm changes only SCF controls to `1e-12`/`1e-10`.
The existing small-grid ownership fixture remains a separate experiment.
The energy/force gates stay `1e-8 Eh` / `1e-7 Eh/Bohr`.

Corrected instrumentation ran under job 16986 on the same device/visibility.
Its original-default **baseline** arm timed out after 300 seconds during cold
SCF (approximately six completed Fock iterations). RI-J took about 40 seconds
and complete Fock iterations about 43 seconds because three-center values
were repeatedly streamed under the original cap. The tighter baseline arm
started but was cancelled at `2026-11-09T23:38:33+08:00`. No corrected
candidate water96 arm reached an endpoint. These are **unavailable numerical
results**, not failures or passes at `1e-7` and not an endpoint timing claim.

Earlier instrumentation failures and timeouts are not scientific outcomes.
The writer now uses the calculator's resolved grid. No original-grid default,
tighter-SCF-only, historical-causality or cross-device gate is complete.

## Retention and reproduction

`samples.json.gz` retains all 40 final diagnostics, full force/reference vectors,
four baseline observations and compact water96 provenance. Original iteration
streams remain under `.artifacts/ks-original-state-gates/final-paired/candidate-uks/`,
with byte identities/history hashes in this publication. Logs/binaries remain
local, not in external releases. `evidence.json.gz` retains the scoped envelope
and shared cache counters/stream identities (not isolated speedups); full
`--show-stats` outputs remain in artifacts. `publication.json` is inconclusive.

Build with the recorded configuration and verified sccache launchers. With the
PR's tests, PySCF and shared library available, run on the measured Slurm host:

```bash
export PYTHONPATH=python:.:tests/python
export GENERATIVEQC_LIBRARY=$PWD/build/ks-state/libgenerativeqc.so
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --nodelist=node1 --time=00:10:00 \
  env GENERATIVEQC_KS_STATE_CUDA_TEST=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest -xq tests/python/test_ks_scientific_gates_cuda.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --nodelist=node1 --time=00:05:00 \
  build/ks-state/generativeqc_ks_cuda_tests --df-range-exchange
```

For water96, use `GENERATIVEQC_GRID_CUDA_TEST=1` as well and select
`tests/python/test_ks_seed_admission_cuda.py -k original_grid`; set
`GENERATIVEQC_KS_STATE_EVIDENCE_DIR` for receipts. The failed 300-second
command limit is not an adequate qualification allocation for that endpoint.
Do not change the original scientific task to fit the allocation. Any future
qualification needs a finite, sufficiently provisioned allocation, matched
libraries and both original-grid arms.

The selected host contract/controller tests passed (190 tests in 10.68s).
The CUDA ownership audit checked 431 files: no scientific CUDA kernel LOC or
ownership-shard changes were added by this repair.
