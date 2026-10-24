# Guarded cold default and Fast/DF qualification, 2026-10-23

PR #2696 now admits the public native Hückel cold density by its provenance.
Automatic incremental J/K remains limited to the documented Standard/Direct
domain. Fast Direct, Standard DF and Fast DF retain their existing controllers;
their coverage below establishes compatibility, not incremental acceleration.

Three paired, alternating-order rounds on n1 RTX 5090, PBE0/def2-SVP spherical
water clusters, give these complete endpoint medians in seconds:

| Atoms | Explicit off cold | Unset cold | Cold change | Warm unset/off |
| --- | ---: | ---: | ---: | ---: |
| 12 | 3.17989 | 3.09057 | -2.81% | 0.99524 |
| 48 | 12.87392 | 11.23742 | -12.71% | 0.99845 |
| 96 | 29.68486 | 26.22357 | -11.66% | 1.00713 |

Each cold sample is a fresh process and includes calculator construction,
preparation, energy and analytic forces. Python/CuPy startup and compilation
are excluded. Two subsequent warm calls reuse the frozen native cold density.
All three cold and six warm samples per size/selector are retained in
[qualification.json.gz](qualification.json.gz), including variability and work.
The JSON is losslessly gzip-compressed (`gzip -dc qualification.json.gz`).
Predeclared gates were cold median unset/off <=1.02, warm <=1.05, and an
improvement for the larger cold cases. All pass. Cold Fock counts remain 10/9/9
with six deltas when admitted; warm remains one full Fock and zero deltas.
Quartet counters were unavailable, so no quartet-count reduction is claimed.

All **210 complete E/F endpoints in 44 fresh native processes** pass independent
tighter full-density GPU4PySCF references. The campaign includes unset/off,
cold/frozen warm/moved/moved-warm/reset, all three Fast/DF combinations at
12/48/96 atoms, and B3LYP/M06-2X at 12 atoms. DF uses identical frozen auxiliary
bases on both engines. References never supply production initial densities.
Maximum errors are 1.3793e-10 Eh/atom, 1.1372e-5 maximum force and 3.0045e-6 RMS
force (Eh/Bohr), below 2.5e-6, 2.5e-5 and 7.5e-6 respectively. These are sampled
stock-grid solver gates, not quadrature-convergence or universal speed claims.
Six earlier pilot endpoints are excluded from this population.

## Source, build and retained evidence

Measured base: `b97a176db5840a45247c828001b38d49c1a64a83`, plus the native
cold-provenance fix. All 2424 compared production/build inputs match the
qualification checkout. The later integration base
`b964aee3ef80f3f0b71f8cdf06ddf97811c02468` changes only the host controller test.
The compact record binds the production file hashes, input-manifest and patch
hashes, core library, force-AOT manifests/libraries, raw samples and references.

Core library SHA256:
`117feee7922ac187fb781ece52be59e345be04aceb98d6bf0fb2d8ff93d14e9b`.
Build: GCC 13.3.0, CUDA 12.9.1, Release `120-real`, PCH off, sccache 0.16.0.
Generated compiler commands used the verified cache launcher. Source-matched
PBE0, B3LYP and M06-2X stationary force AOT artifacts were built; an incompatible
older AOT candidate was rejected without overriding its contract identity.
Before/after cache snapshots are retained for the fix and AOT builds. Shared
stats include concurrent activity; the original full-build before snapshot was
overwritten by the relink retry, so no whole-build cache-hit rate is claimed.

Raw forces, resolved settings, failures, logs, source snapshot and runner scripts
remain at `n1:/data/jzzeng/pr2696-fast-df-20261023/` and in the qualification
worktree's ignored `.artifacts/remote/`. Timing job 13152 preserved Slurm's
`CUDA_VISIBLE_DEVICES=2`. Native receipts are also retained at
`n4:/data/jzzeng/pr2696-native-20261023/` and `.artifacts/n4/`. n4 ran the same
core with a private copy of the build host's userspace loader/libraries for its
newer GLIBC ABI, using n4's GPU driver. Every GPU run used finite Slurm `srun`.
Slurm accounting is disabled; the raw endpoint records retain job identity.

Validation: 339 host tests passed, 24 skipped before integration; 141 final
controller/policy/inventory tests passed after integrating the remote fixture
fix. The independent CUDA Python suites passed 21 tests. Native incremental
physical-state/restart, DF warm-factor/fallback, Fast anchor/oracle/lifetime,
Fock composition, SPD values and mixed-census/strict-exchange selections pass.
The broad provider diagnostic timed out at five minutes after globally forcing
generated J; it is not counted as a full provider-suite pass. The final fixture
scopes that selection to its generated-channel census, whose focused run passes.
No production numerical or final-state gate was relaxed.

## Reproduction

Use a source-matched CUDA build with the three stationary force AOT manifests,
the Python package, GPU4PySCF 1.8.1 and PySCF 2.14.0. Verify sccache (preferred)
or ccache before compiling, and export `GENERATIVEQC_LIBRARY` to that build.
From the checkout root, an example complete pair is:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 bash -lc '
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PYTHONPATH="$PWD/python:$PWD"
    python -c "from pathlib import Path; from benchmarks.readme_defaults import prepare_inputs; prepare_inputs(Path(\".artifacts/ks-default/inputs\"))"
    revision=$(git rev-parse HEAD)
    python -m benchmarks.ks_incremental_default --arm reference --atoms 12 \
      --inputs .artifacts/ks-default/inputs --source-revision "$revision" \
      --output .artifacts/ks-default/reference.json
    for selector in off unset; do
      python -m benchmarks.ks_incremental_default --arm standard --atoms 12 \
        --selector "$selector" --inputs .artifacts/ks-default/inputs \
        --source-revision "$revision" --reference .artifacts/ks-default/reference.json \
        --output ".artifacts/ks-default/$selector.json" --lifecycle
    done
  '
```

For timing, omit `--lifecycle` and repeat three alternating-order rounds at
12/48/96 atoms; never pool extra lifecycle samples into those populations.
Use `--arm fast` and/or `--mode df` with a matching independent reference for
compatibility. `--method b3lyp` and `--method m06-2x` select other hybrids.
Run the permanent GPU tests in a finite allocation with
`GENERATIVEQC_DFT_CUDA_TEST=1 python -m pytest -q tests/python/test_ks_incremental_default_cuda.py tests/python/test_ks_incremental_convergence_cuda.py`.

See the [decision note](../../../.agents/notes/implemented/performance/2026-10-23-cold-default-seed-provenance.md)
and [current user contract](../../../docs/user/ks_options.md#experimental-incremental-direct-jk-and-resource-budgets).
