# H100: WB97M-V cold endpoint, resident VV10 grids, Direct nonlocal tiles and cold VV10 anchors

Diagnostic cohorts on one H100 80GB HBM3 (SM90, driver 595.58.03, 700 W) in one qz
Job, `gqc-h100-wb97mv-1117-v1`. Every process is fresh and runs consecutively on the
same physical GPU (`CUDA_VISIBLE_DEVICES=0`). Native arms use the public Standard
defaults; the reference is stock GPU4PySCF 1.9.0 (CuPy 13.4.1) with its defaults.
WB97M-V/def2-SVP, spherical, water clusters from `benchmarks.readme_hf_scaling`.
Cold is constructor + preparation + SCF + analytic forces (full grid response) +
host export; warm is the median of three same-owner calls from the engine's own
frozen cold density. These are single-GPU point observations, not statistical
rankings.

`results.json.xz` retains every process: per-call seconds, energies, SCF iterations,
Fock/Veff builds, VV10 work counters, resolved SCF tiles, raw-record hashes and the
independent qualification status with maximum errors. `acquisition.tar.xz` retains
the build, runner, endpoint, profiling and collection scripts and arm lists.
Independent qualification uses the tracked
[readme-refresh-20261026](../readme-refresh-20261026/README.md) oracles with the
reviewed reference-reuse contract and the original gates (energy 1e-5 Eh/atom,
maximum force 1e-4, RMS force 3e-5 Eh/Bohr). Every completed native and reference
process qualifies.

## Changes and their measured effect (water-96 Direct)

| Source | Change | Cold (s) | Warm median (s) | Focks / VV10 SCF sweeps |
| --- | --- | ---: | ---: | --- |
| `a7241b4fd` (#3073 cohort) | — | 48.306 | 9.383 | 16 / 4 |
| `6810f3c5a`, `GENERATIVEQC_VV10_PAIR_GRID=fixed` | master `2791504cc` | 45.804, 45.683 | 7.642, 7.634 | 16 / 4 |
| `6810f3c5a` | resident VV10 pair grid | 43.941, 43.799 | 7.640, 7.628 | 16 / 4 |
| `6810f3c5a`, explicit 1024-point SCF tile | | 41.276 | 7.436 | 16 / 4 |
| `126b902d0` | automatic Direct nonlocal 1024-point tile | 41.459, 41.293 | 7.474, 7.439 | 16 / 4 |
| `126b902d0`, cold seed stage skipped | | 39.466, 39.267 | 7.482, 7.438 | 13 / 4 |
| `8fcfd5825` | cold VV10 anchor default | 39.508, 39.499, 39.356 | 7.492, 7.477, 7.437 | 13 / 4 |
| `8fcfd5825`, `GENERATIVEQC_VV10_COLD_SEED_STAGE=1` | rollback control | 41.598 | 7.429 | 16 / 4 |

Same-image ABBA pairs isolate each change: the resident grid saves 1.86/1.88 s cold
(4.1%) with unchanged warm calls; the 1024-point tile 2.6 s cold and 0.2 s warm; the
cold anchor 2.0 s cold (three fewer provisional Focks, same four pair sweeps).
The warm improvement relative to `a7241b4fd` predates these changes.

## Paired comparison with GPU4PySCF on the final source `8fcfd5825`

| Pair (order) | GenerativeQC cold | GPU4PySCF cold | GPU4PySCF/native | Warm (native / reference) |
| --- | ---: | ---: | ---: | --- |
| r0 (native first) | 39.508 | 38.066 | 0.963 | 7.492 / 11.609 |
| r1 (reference first) | 39.499 | 39.911 | 1.010 | 7.477 / 12.001 |
| r2 (native first) | 39.356 | 40.029 | 1.017 | 7.437 / 12.230 |

The median cold ratio is 1.010: water-96 Direct cold is at parity, not a stable win.
Warm is 1.55–1.64× faster. Other sizes on the final source:

| Case | GenerativeQC cold / warm | GPU4PySCF cold / warm (cohort) |
| --- | --- | --- |
| Direct water-48 | 14.749 / 3.416 | 16.487 / 4.591 (same cohort) |
| Direct water-12 | 2.197 / 0.361 | 11.808 / 2.350 (`126b902d0` cohort) |
| Direct water-3 | 1.429 / 0.110 | 7.857 / 1.573 (`126b902d0` cohort) |
| DF water-96 | 23.900 / 5.926 | 35.459 / 11.472 (`126b902d0` cohort) |
| DF water-48 | 9.398 / 1.961 | — |

The cold anchor is also measured before promotion on every size. It is faster for
Direct water-96/48/3 and DF water-96/48 (DF water-96 27.35 → 23.98 s, three instead
of four pair sweeps), and within noise for Direct water-12 (2.156 → 2.199 s, one
fewer Fock but one more pair sweep).

## Rejected and diagnostic arms

- `GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=0` (no resident range source):
  156.887 s cold, 15.332 s warm. The resident composite source is essential.
- A per-component GPU4PySCF profile (device-synchronized wrappers, cold 37.83 s)
  attributes 20.69 s to 11 VV10 potential calls (1.88 s each), 5.11 s to XC, 3.76 s
  to K, 1.04 s to J and 4.88 s to the gradient's VV10 term. Its VV10 domain has
  1,048,696 points with rho >= 1e-10 and 955,804 after its additional |w| > 1e-14
  screen. These synchronized timings are attribution only, not a timing vote.
- The first cohort's final Nsight Systems arm failed because the image has no
  `nsys`; the second cohort's capture is retained remotely as an unconverted
  `.qdstrm` and is not used here.

## Remote retention

`/inspire/hdd/global_user/czxs25220150/projects/GenerativeQC/h100-cold/20261117-wb97mv/`
retains every raw record, plan, stdout/stderr, qualification, build receipt and
log (`wb-v1`, `wb-v2`, `wb-v3`). The first build attempt, rejected because macOS
AppleDouble `._*` files had entered the staged source, is retained under
`attempt0-appledouble/`; the accepted build's identity matches the Git tree.
