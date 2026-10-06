# Becke physical endpoint comparison — 2026-10-06

Decision: **inconclusive for performance; no promotion**. Primitive remains default off, PR remains draft, #1894 remains open for the full acceptance audit.

Finite n1 main/gpu:5090:1 jobs 6173, 6184 and 6193 all exit zero. Eight native protocols retain 96 full force vectors and all actual solver histories; two fresh independent GPU4PySCF reference protocols retain 24 observations. These are repeated observations, not 96 independent molecular cases.

All vector gates pass unchanged: energy 1e-8 hartree; force 1e-7 hartree/bohr. Maximum energy/force errors across clean and intrusive observations: 1.08684616862e-10 / 3.21890986088e-11.

## Clean complete E+F

Cold includes outer preparation. Warm/moved-warm retain all five raw samples; table uses medians, never best repeats. Both arms use the current retained #1830 phased implementation including later log scheduling, not a relabeled historical binary.

| Atoms | Phase | Phased seconds | Primitive seconds | Actual iteration/Fock builds (phased / primitive) |
| --- | --- | ---: | ---: | --- |
| 48 | cold | 120.084466 | 113.640823 | 27/27 / 25/25 |
| 48 | warm | 11.665443 | 11.202456 | 1/1 / 1/1 |
| 48 | moved | 53.655117 | 53.369826 | 12/12 / 12/12 |
| 48 | moved-warm | 11.236248 | 11.177308 | 1/1 / 1/1 |
| 96 | cold | 280.447748 | 301.997070 | 30/30 / 33/33 |
| 96 | warm | 39.788979 | 39.998613 | 1/1 / 1/1 |
| 96 | moved | 146.283094 | 161.277940 | 14/14 / 16/16 |
| 96 | moved-warm | 39.894366 | 40.084957 | 1/1 / 1/1 |

Grouped arms reverse order at 96 atoms; these are **not interleaved paired measurements**. The 48 warm median difference is affected by retained baseline spread; 96 warm and moved-warm medians are slightly worse. Cold solver work differs at both sizes and moved work differs at 96, so do not attribute their wall differences to the source mechanism or normalize their histories.

The moved API reports zero outer preparation, not zero geometry setup: internal owner rebinding/construction and source preparation remain inside the full endpoint and are retained in the summaries. Initial source-owner lookup/construction is about 14 seconds in each clean cold call; no fresh compiler-only total is claimed.

## Separate intrusive phases

Per-tile events/fences perturb execution. Times below are medians over the five original-warm repeats, in milliseconds; totals are medians of per-call sums, not sums of component medians. Never add these observations to clean endpoint wall time. Moved-warm and all raw intervals are in the compressed observations.

| Phase | 48 phased ms | 48 primitive ms | 96 phased ms | 96 primitive ms |
| --- | ---: | ---: | ---: | ---: |
| point_center_distance | 37.598400 | 37.641472 | 94.374718 | 94.893278 |
| pair_primal_switch_log | 197.386110 | 189.179899 | 1340.632145 | 1359.694181 |
| atom_log_reduction | 56.620930 | 56.618946 | 207.697282 | 210.976131 |
| normalization | 192.815456 | 190.588735 | 751.497183 | 762.568162 |
| reverse_derivative | 68.065729 | 64.541121 | 388.357374 | 356.356984 |
| atom_gather | 103.682689 | 115.946786 | 415.302404 | 547.373646 |
| point_motion_publication | 105.986913 | 106.904706 | 395.945629 | 402.929375 |
| Total per call | 762.287715 | 760.886594 | 3593.864880 | 3734.906351 |

Reverse time improves in this intrusive scope but atom gather becomes slower. This does not establish a reduced complete geometry-response bottleneck; no unchanged losing fusion/log/cache experiment is promoted.

## Work and provenance

| Atoms | Points per force | Dense primal/reverse visits per phase | Batches | Becke phase launches |
| --- | ---: | ---: | ---: | ---: |
| 48 | 1179648 | 1330642944 | 4608 | 32256 |
| 96 | 2359296 | 10758389760 | 9216 | 64512 |

Both arms have identical dense pair domains and launch counts. Extra gather direction reads offset the smaller logical reverse panel. These logical bytes are **not hardware traffic**; actual branch-dependent log/sqrt execution counts are unavailable, not inferred from visits. No privileged profiler workaround. AO/XC (#1893) and Direct primitive-pair work (#1892) remain separate owners/experiments.

Original core source: a65b075ca0b38f61508a5e596ba408d4f8214807 plus source.patch. Core library SHA256: 806f28a2191ecbdaea5964e4ab8e3b53629ca1ea44b66aee1cda069df8db3f1d. All 1400 scientific/build input hashes match the frozen measurement and later 5828f274 source. Original drivers, 382 ccache compiler-launch commands, actual toolchains, separate assigned device visibility, terminal accounting and aggregate before/after cache stats are retained. No per-call generated artifact binding was newly instrumented; use source and core identities at their stated scope.

The published revision/dirty patch was reconstructed separately and all scientific/build hashes pass. Reproduction shell passes bash -n; the entire recipe was not rerun as an additional campaign. Keep the bundle outside the reconstructed checkout, apply its source patch, then use reproduce.sh with actual CUDA_ROOT/PYTHON installations. No releases or external archive assets are created.
