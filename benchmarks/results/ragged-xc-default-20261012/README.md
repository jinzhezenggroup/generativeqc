# Bounded Ragged XC default: current-master complete PBE0 E+F

Measured source `ce01eaf5a` is based on master
`a50eb54f362a1b27edcb0f06353e7a04aa9979e2`. Same-binary A/B runs compare
`GENERATIVEQC_CUDA_XC_RAGGED_BATCH=0` against the **unset/default** control.
No ACE implementation is included. RTX PRO 6000 Blackwell on n2, CUDA 12.9,
strict FP64, def2-SVP/spherical, original grid/screening/SCF gates and full
analytic forces remain unchanged. Slurm jobs 2931/2932 have finite limits;
their visibility is preserved. Compiler caches are reused, not cleared.

## Five interleaved clean trajectories per arm

Cold includes preparation; every phase includes the full energy-plus-force
endpoint. Diagnostic trajectories are separate and excluded from medians.
All samples and actual iteration histories are retained, including outliers.

| Atoms | Phase | Incumbent median (s) | Default median (s) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 96 | cold | 64.624547 | 61.236247 | 5.243% |
| 96 | warm | 12.805590 | 12.615486 | 1.485% |
| 96 | moved | 57.768114 | 55.378792 | 4.136% |
| 96 | moved-warm | 12.827722 | 12.607120 | 1.720% |
| 48 | cold | 24.091125 | 21.109162 | 12.378% |
| 48 | warm | 4.498975 | 4.345828 | 3.404% |
| 48 | moved | 23.277822 | 21.304932 | 8.475% |
| 48 | moved-warm | 4.488108 | 4.338941 | 3.324% |

The descriptive floor is `max(2%, 2*(relative_MAD_A+relative_MAD_B))`,
2% for these populations. Both 96-atom warm populations remain below it:
this is **not** an all-workload significant-speedup claim. The user explicitly
authorizes the bounded scheduling default based on the cold/moved gains.
Opt-out and resource/provider fallbacks remain available.

All 96 scientific records (including diagnostics) pass unchanged independent
1e-8-Eh energy and 1e-7-Eh/bohr **all-force-component** gates. Maximum errors
are 1.319e-11 Eh and 3.824e-11 Eh/bohr. The existing independent GPU4PySCF
references are checksum-bound; no oracle enters production computation.

## Executed work and limits

Cold iterations/Fock builds are 17/18 (96 atoms) and 19/20 (48 atoms).
Warm/moved-warm execute 1/1. Moved usually executes 12/12: one incumbent
96-atom sample executes 15/15 and one candidate 48-atom sample 14/14.
These are neither excluded nor rescaled. Separate diagnostic moved samples
execute 12 exact-J/K invocations and 12 XC invocations in both arms.
The compact endpoint records retain per-phase operator times/counts and receipts;
overlapping GPU/host intervals must not be added into endpoint times.

At 96 atoms the candidate admits 7,248 large compact tiles at the original
geometry (maximum 536 AOs). Reserved XC bytes are 24,989,968 versus
25,311,448. At 48 atoms they are 26,975,200 versus 21,131,264: both obey
the unchanged 32-MiB cap. Reservations are not complete endpoint peak memory.
Other devices, UKS/response, a complete peak-memory envelope and uncached
compilation cost are not qualified by this campaign.

Native independent E/V, ragged cross-block/nonidentity-map/tail tests and
capture/replay pass. Memcheck reports zero errors; racecheck reports zero
hazards/errors/warnings. The preserved hashes and archived exact logs identify
the tested binaries. CI and host tests are reported separately in the PR.

## Storage and exact raw recovery

The repository's unchanged 64-MiB aggregate evidence cap leaves insufficient
checkout space for another complete raw campaign. Current compact XZ records
retain every sample's timing, status, final residuals, iterations, Fock builds,
operator receipts and source/binary identity. Full force vectors, per-iteration
histories, diagnostic traces and exact sanitizer/cache/Slurm logs remain in
the existing ancestral evidence commit
`47c7da365ff8b8b26579ddd18ca8240fb19348c1`; they are **not discarded**, rounded
or moved to an unauthorized external archive. This follows the repository's
Git-history recovery policy rather than raising its budget or deleting another
campaign. `raw_recovery` binds the exact uncompressed payload SHA per case.

## Recheck and reproduce

From this PR's checkout with its Git history available, run:

```bash
cd benchmarks/results/ragged-xc-default-20261012
sha256sum -c files.sha256
cd ../../..
PYTHONPATH=python:. python benchmarks/results/ragged-xc-default-20261012/summarize.py
```

The summarizer reads the checksum-bound complete raw records with `git show`,
compares every compact field against them, checks measured source hashes at
`ce01eaf5a`, reference identity, every complete force vector and original
convergence status, then recomputes medians/noise and work. It needs Python
dependencies and these local Git objects, not a GPU or network. A shallow clone
must fetch the two named commits first. Release CUDA builds use
the existing CMake `generativeqc`, `generativeqc_dft_cuda_tests`,
`generativeqc_stationary_pbe0_rks_manifest` and
`generativeqc_stationary_pbe0_rks_spd_manifest` targets with verified ccache
launchers. If generated weighted headers are missing, use the repository's
`tools/generate_weighted_eri_kernels.py` before building; do not patch physics.

`validate-n2.sh` and `qualify-n2.sh` retain the exact executed commands and
environment paths. Stage a fresh source/binary tree and use finite `srun`
allocations on node2, `main`, `--gres=gpu:pro6000:1`, 8 CPUs, with respective
20/55-minute limits. Never override scheduler device visibility. For fresh
reproduction use source commit `ce01eaf5a`, retaining these instructions outside
the checkout. The ancestral raw JSON/traces use unchanged gzip payloads.
