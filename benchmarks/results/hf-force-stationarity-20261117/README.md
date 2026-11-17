# Default HF force-stationarity qualification, 2026-11-17

This is independent scientific qualification for issue #3092, not a performance
ranking or a claim that every system satisfies a universal force-error bound.
The repair separates force admission from energy-only orbital-gradient stopping;
it does not change public defaults or add production reference-oracle work.

## Source and acquisition

- Native code commit: `6b52f067c581de21a6c06d11db08d3c7a6873ea4`.
- Base: `50482bc351756772660c45b3e8430fb388e9cf99`.
- Native source identity:
  `15f2cf4ebb59e16c3c3d31a50aaa3dcc508b4e7c07c539e63a82910d73e745c6`.
- Final device-test recipe:
  `tests/python/test_hf_default_force_stationarity_cuda.py`, SHA256
  `644f782b50ab124dc9d70564a24a06d5723b3f7386332607517a9ca1ad4ca370`.
  This is retained acquisition provenance, not a fingerprint requirement on
  future production refactors.
- CUDA 12.9.86, GCC 12, Release, no fast compilation, verified sccache 0.16.0.
  Build job 17644 and finalization job 17662 used n1 Slurm allocations.
  Both AOT-profile tests and all three compiled final-state/force-convergence
  tests passed. Complete CMake settings, cache statistics and actual verbose
  launcher command samples are retained. Hessian AOT leaves and stationary
  CPU-force AOT are disabled in these bounded qualification builds.
- Final n1 job 17670: RTX 5090, driver 580.95.05, SM120, `main/gpu:5090:1`.
- Final n2 job 5324: RTX PRO 6000 Blackwell, driver 595.99.02, SM120,
  `main/gpu:pro6000:1`. Both jobs have finite 25-minute limits and genuinely
  successful driver exits. Slurm visibility is preserved. Actual CUDA-driver
  UUIDs, rather than an assumed GPU index, bind observations to the inventory.
- For each profile, both cards execute the **same ELF bytes**. Auto SHA256:
  `3a474081c78a59b3c9c3cbe0f90d93b5412a575f15f0503348bdc2443e37af2e`;
  portable SHA256:
  `32828213de7d0122de423dbc913bba85ff4c1cdbe53ace21c663b79672d667a4`.
  n2 uses the recorded isolated host runtime, not a source rebuild. No A100,
  H100, different-SM, release-distribution startup or speedup claim is made.

`qualification.json.xz` retains raw references and observations, acquisition
scripts, build/source receipts, numerical work counts, device metadata and
failed-setup observations. Large ELF files and complete compile logs remain in
ignored local acquisition storage; no Release or external backup was published.
The aggregate evidence cap is unchanged. `storage-compaction.json` records
94,140 bytes reclaimed solely by removing insignificant JSON whitespace from
five historical records, preserving every numeric/string token and parsed value.
Original byte identities and their existing-Git recovery revision are retained;
no scientific observation, failed case or gate was deleted or changed.

## Coverage and unchanged gates

For each card/profile, four fresh pytest processes execute six cases:
Cartesian Direct RHF water, Direct UHF OH and explicitly matched DF RHF water,
each with def2-SVP and def2-TZVP. Each case covers cold, frozen warm, moved,
moved warm, energy-only and energy-to-forces calls. Warm reuse is asserted.
Production imports of PySCF are forbidden; independent references run in
separate processes. Stock production controls are omitted and then checked as
energy `1e-9`, density `1e-5` and screening `1e-12`.

| Quantity | Unchanged gate | Maximum observed error |
| --- | --- | --- |
| Absolute energy, Eh | `1e-9` | `1.1220891e-10` |
| Maximum force component, Eh/Bohr | `1e-7` | `4.8487086e-9` |
| Maximum translation residual, Eh/Bohr | `1e-7` | `2.2470914e-13` |

All **96 cases / 576 endpoints**, including **480 E+force endpoints**, pass.
The original TZVP initial cold/warm case passes in all 32 observations, maximum
force error `1.3859373e-9`; initial cold SCF now takes 12 iterations rather than
9. The earlier-source negative regression remains a failure at `6.7580089e-7`.
Its receipt predates the final combined DF recipe and is labeled accordingly.

The DF oracle matches the explicit Cartesian auxiliary model and native default
relative metric cutoff `1e-10` in the unit-self-overlap auxiliary gauge. PySCF/CINT
independently constructs the integrals and spectral factors. Complete reference
forces use reconverged displaced energies at five-point steps 0.01/0.02 Bohr,
requiring unchanged retained rank and step agreement within `5e-8`, followed by
Richardson extrapolation. Raw displaced energies are retained and replayed.
This is not qualification of arbitrary/default auxiliary choices or fitting
error relative to direct HF.

Initially spherical/Cartesian-mismatched and full-rank DF oracle comparisons
remain separate failed setup observations, not relabeled production passes.
Likewise, the earlier host-cache-prerequisite failure retains its nonzero exit;
the full scientific job was rerun with verified sccache and exited zero.
The expanded host convergence/error/retry suite also passed all 246 tests in
finite n1 job 17671; force owners retain the same typed-error and arena-cleanup
guarantees without re-enabling relaxed convergence.

## Reproduction

Replay the raw scientific gates without loading a native library or GPU:

```bash
python benchmarks/results/hf-force-stationarity-20261117/replay.py
```

For a new source-matched Release library, run the device test through Slurm,
with matching Python/host-runtime dependencies available. Repeat for each ELF
and supported card; do not override scheduler-assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:10:00 \
  env -u GENERATIVEQC_CUDA_RHF_CONVERGENCE \
  GENERATIVEQC_RESOURCE_CUDA_TEST=1 GENERATIVEQC_PROFILE=off \
  GENERATIVEQC_LIBRARY=/absolute/path/to/libgenerativeqc.so \
  GENERATIVEQC_SCF_ORACLE_RECEIPTS=/absolute/path/to/new-receipts \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/python/test_hf_default_force_stationarity_cuda.py
```

Future source changes require new acquisition, not reassignment of these
historical source-bound observations. Historical failed profile-qualification
data elsewhere in the repository remains failed and unchanged.
