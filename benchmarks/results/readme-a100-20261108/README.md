# All non-CBS README endpoints on A100

Measured on 2026-11-08 (Asia/Shanghai), not extrapolated from the historical
RTX 5090 README figures. **The A100 is not ahead in every case.**

## Scope and results

- Eight methods × Direct/DF × 3/12/48/96 atoms: 64 paired cases, 128 fresh
  processes, 512 complete energy + analytic-force + host-export endpoints.
- Cold: **56/64 timing point wins**; warm: **59/64 timing point wins**.
  Warm is the median of three calls replaying each engine's own frozen cold
  density. A single cold is a point observation, not a statistical ranking.
- All 64 pairs pass all 16 cold/warm cross-comparisons at the unchanged common
  E/F gates. Independent dense/CPU qualification admits 116/128 populations
  (58/64 pairs); 12 failed and none were silently dropped or called unverified.
  Qualified timing wins are therefore 51 cold and 54 warm, not 64.
- The six independently failing pairs are HF DF at all four sizes and M06-2X
  Direct/DF at three atoms. The historical oracle qualification also retains
  these failures; matching the other engine is not independent dense accuracy.
- CBS is excluded. Defaults, grids, fitting inputs and observable gates are not
  weakened; native processes reject PySCF/GPU4PySCF imports. There is no CPU
  reference calculation inside the measured native production endpoint.

## Provenance and controls

Measured revision `bcb43acffbb913aadf35836c0bbddecb9be40853`; source identity
`89ad140b13db66d929acfa3e99eb797aff5dc91b4a6071b1a528dac8d1e22751`;
normal Release SM80-real + SM120-real ELF SHA256
`0e49d1f3397e20ccfd7eb89ed38feb1f3c4a31ac004dad47e40782cc0f55db88`.
The fresh combined image includes HF #3038 and r2SCAN #3041; an older image
was not relabelled. Publication parent `61b3a62fbc9b0620d51753f3f14d1074e1ecc301`
adds master `97c34eeb0` and host CI repairs, with no production-source delta;
its native source identity was independently checked unchanged.

CPU-only Slurm build16729 succeeds with 95.22% sccache hits; the borrowed
source/runtime is restored. CPU-only LSF setup228199 and GPU jobs228200–228203
all finish successfully on gpu22. Every pair shares its actual LSF allocation,
visibility and physical GPU UUID; first-arm order is balanced32/32. Math library
threads remain one. `GENERATIVEQC_PROFILE=off` excludes user-local tuned-image
replacement; opt-in/diagnostic paths are not substituted for the linked default.
All first cold populations, warm samples, SCF/operator counts, settings, failures,
original logs, submission/transfer receipts and exact byte checksums are retained.

Reference packages are PySCF2.14.0 / GPU4PySCF1.9.0 / CuPy13.6.0 / NumPy2.5.4.
The historical independent oracles used GPU4PySCF1.8.1 / NumPy2.4.6. Reuse is
explicitly reviewed against the exact old producer/scientific contract and
unchanged geometry, charge, spin, orbital/auxiliary inputs and representation.
Original grid-stability and CPU/GPU spot gates remain active, including failures.
Old oracle bytes and timings are not rewritten; no old GPU timing enters this
new table. See the capsule's version2 `reference-reuse.json` and source-bound
oracle hashes. Frozen auxiliary bytes match all32 historical DF cases exactly.

## Reproduction

The existing exact-FP64 codec groups predictors by scientific case and engine.
The storage-only repack preserves all388 original text receipts, every FP64 word
and the entire experiment metadata. `storage-receipt.json` retains the previous
compressed identities; `sha256.txt` authenticates the current containers. No
failed population, independent reference or timing is removed or regenerated.

```sh
python benchmarks/results/readme-a100-20261108/replay.py
PYTHONPATH=python:. python -m pytest -q tests/python/test_readme_a100_evidence.py
```

The CPU-only replay authenticates all original text/FP64 bytes, execution order,
image and scheduler identities, reruns all64 timing reductions and1024 paired
E/F checks, and recomputes independent qualifications from the tracked historical
oracles. It does not run SCF or require a GPU. Capsule compression is lossless;
no printed rounding substitutes for original force/energy evidence. The baseline
capsule has388 exact files and125172 FP64 words, stored in471016 compressed bytes.

## Remaining targets

Eight cold losses: HF Direct48/96, HF DF96, PBE0 Direct/DF96, B3LYP Direct/DF96,
and M06-2X Direct96. Five warm losses: HF Direct/DF96, PBE0/B3LYP Direct96,
and r2SCAN-3c Direct96. These form nine distinct cases. HF Direct96 is the
largest cold gap:16.277175s versus6.018824s (2.704×); its native work includes
seven preliminary and fifteen target Fock builds, versus ten reference SCF
iterations/eleven effective-potential builds. This diagnoses work, not yet a
causal attribution. Prior blind incremental-HF toggles were rejected and are
not repeated as an assumed fix. The user requests attacking these residuals,
while protecting5090 and retaining every baseline observation in this capsule.

Within3% cold points (three cases) and warm points (one case) are labelled in
the machine-readable summary. In particular, r2SCAN-3c Direct96 cold is only
14.596999s versus14.716811s; do not advertise that as a robust statistical win.

## All64 pairs

Each timing cell is native/reference seconds. Independent status is native/ref.

| Method | Mode | Atoms | Cold native/ref (s) | Warm native/ref (s) | Paired E/F | Independent native/ref |
|---|---|---:|---:|---:|---|---|
| hf | direct | 3 | 0.910549 / 7.992519 | 0.035774 / 1.383125 | True | qualified/qualified |
| hf | direct | 12 | 1.252794 / 4.798864 | 0.072733 / 1.167333 | True | qualified/qualified |
| hf | direct | 48 | 6.188330 / 5.463523 | 0.591522 / 1.220441 | True | qualified/qualified |
| hf | direct | 96 | 16.277175 / 6.018824 | 2.282014 / 1.651905 | True | qualified/qualified |
| hf | df | 3 | 0.286086 / 14.204613 | 0.022713 / 0.286810 | True | failed/failed |
| hf | df | 12 | 0.549116 / 3.313295 | 0.091165 / 0.313946 | True | failed/failed |
| hf | df | 48 | 1.481357 / 6.949344 | 0.313397 / 0.493819 | True | failed/failed |
| hf | df | 96 | 7.422615 / 5.533299 | 1.602722 / 1.385077 | True | failed/failed |
| pbe | direct | 3 | 1.158210 / 13.561128 | 0.045692 / 0.503968 | True | qualified/qualified |
| pbe | direct | 12 | 1.423552 / 4.313834 | 0.161937 / 0.681845 | True | qualified/qualified |
| pbe | direct | 48 | 3.899869 / 6.268958 | 0.748825 / 1.310749 | True | qualified/qualified |
| pbe | direct | 96 | 7.359367 / 8.831047 | 2.384254 / 2.692558 | True | qualified/qualified |
| pbe | df | 3 | 0.632995 / 8.022483 | 0.052802 / 0.301089 | True | qualified/qualified |
| pbe | df | 12 | 0.936401 / 3.759590 | 0.125230 / 0.413116 | True | qualified/qualified |
| pbe | df | 48 | 3.265978 / 5.335969 | 0.686688 / 1.066410 | True | qualified/qualified |
| pbe | df | 96 | 6.673871 / 7.896043 | 2.112246 / 2.196636 | True | qualified/qualified |
| pbe0 | direct | 3 | 1.282808 / 6.272188 | 0.059573 / 1.098866 | True | qualified/qualified |
| pbe0 | direct | 12 | 1.897612 / 7.677611 | 0.219424 / 1.225758 | True | qualified/qualified |
| pbe0 | direct | 48 | 5.823122 / 8.771791 | 1.283120 / 1.900231 | True | qualified/qualified |
| pbe0 | direct | 96 | 13.345868 / 11.167846 | 3.851119 / 3.191341 | True | qualified/qualified |
| pbe0 | df | 3 | 0.604766 / 3.200909 | 0.056651 / 0.409401 | True | qualified/qualified |
| pbe0 | df | 12 | 0.958641 / 3.717173 | 0.151780 / 0.485977 | True | qualified/qualified |
| pbe0 | df | 48 | 3.092321 / 5.278081 | 0.763507 / 1.193330 | True | qualified/qualified |
| pbe0 | df | 96 | 9.179725 / 8.447658 | 2.967126 / 3.144313 | True | qualified/qualified |
| b3lyp | direct | 3 | 1.347897 / 6.793063 | 0.064605 / 1.108772 | True | qualified/qualified |
| b3lyp | direct | 12 | 1.916262 / 7.329178 | 0.227178 / 1.260980 | True | qualified/qualified |
| b3lyp | direct | 48 | 5.940226 / 9.571355 | 1.307048 / 1.902378 | True | qualified/qualified |
| b3lyp | direct | 96 | 13.508506 / 11.848855 | 3.913587 / 3.243415 | True | qualified/qualified |
| b3lyp | df | 3 | 0.585671 / 3.488379 | 0.060458 / 0.372887 | True | qualified/qualified |
| b3lyp | df | 12 | 1.005850 / 3.729351 | 0.116764 / 0.498150 | True | qualified/qualified |
| b3lyp | df | 48 | 3.216075 / 5.107177 | 0.745380 / 1.217107 | True | qualified/qualified |
| b3lyp | df | 96 | 9.175714 / 8.736263 | 2.981426 / 3.098597 | True | qualified/qualified |
| r2scan | direct | 3 | 1.303828 / 3.367861 | 0.078629 / 0.544001 | True | qualified/qualified |
| r2scan | direct | 12 | 1.698068 / 4.282926 | 0.204576 / 0.722541 | True | qualified/qualified |
| r2scan | direct | 48 | 4.333289 / 6.824417 | 0.929337 / 1.686446 | True | qualified/qualified |
| r2scan | direct | 96 | 7.152479 / 10.962509 | 2.478609 / 3.408996 | True | qualified/qualified |
| r2scan | df | 3 | 0.774965 / 3.335643 | 0.090517 / 0.320192 | True | qualified/qualified |
| r2scan | df | 12 | 1.175720 / 3.984225 | 0.166415 / 0.546478 | True | qualified/qualified |
| r2scan | df | 48 | 3.670338 / 6.095205 | 0.866951 / 1.382035 | True | qualified/qualified |
| r2scan | df | 96 | 6.488417 / 9.621844 | 2.213066 / 2.995980 | True | qualified/qualified |
| r2scan-3c | direct | 3 | 1.311855 / 3.802893 | 0.125473 / 0.767082 | True | qualified/qualified |
| r2scan-3c | direct | 12 | 2.484574 / 5.036407 | 0.830253 / 1.050127 | True | qualified/qualified |
| r2scan-3c | direct | 48 | 5.487605 / 9.431656 | 2.018898 / 2.529744 | True | qualified/qualified |
| r2scan-3c | direct | 96 | 14.596999 / 14.716811 | 6.555448 / 5.567463 | True | qualified/qualified |
| r2scan-3c | df | 3 | 0.694461 / 3.498883 | 0.098474 / 0.392546 | True | qualified/qualified |
| r2scan-3c | df | 12 | 1.483564 / 4.113366 | 0.194131 / 0.545739 | True | qualified/qualified |
| r2scan-3c | df | 48 | 4.515073 / 6.690341 | 1.378358 / 1.619451 | True | qualified/qualified |
| r2scan-3c | df | 96 | 11.295484 / 11.617113 | 3.410684 / 3.918484 | True | qualified/qualified |
| m06-2x | direct | 3 | 1.368814 / 6.262649 | 0.082043 / 1.252922 | True | failed/failed |
| m06-2x | direct | 12 | 2.024303 / 7.557809 | 0.243894 / 1.300432 | True | qualified/qualified |
| m06-2x | direct | 48 | 7.336714 / 9.836685 | 1.456834 / 2.282322 | True | qualified/qualified |
| m06-2x | direct | 96 | 14.736431 / 13.855340 | 3.940225 / 4.011983 | True | qualified/qualified |
| m06-2x | df | 3 | 0.769907 / 3.434765 | 0.081283 / 0.377814 | True | failed/failed |
| m06-2x | df | 12 | 1.012837 / 3.930966 | 0.134124 / 0.606495 | True | qualified/qualified |
| m06-2x | df | 48 | 4.087665 / 6.195426 | 0.890624 / 1.603121 | True | qualified/qualified |
| m06-2x | df | 96 | 9.575339 / 11.323277 | 3.018577 / 3.899790 | True | qualified/qualified |
| wb97m-v | direct | 3 | 1.998032 / 6.740792 | 0.174661 / 1.478835 | True | qualified/qualified |
| wb97m-v | direct | 12 | 3.254910 / 10.282327 | 0.617772 / 2.370806 | True | qualified/qualified |
| wb97m-v | direct | 48 | 27.438414 / 28.061296 | 5.779490 / 8.857563 | True | qualified/qualified |
| wb97m-v | direct | 96 | 79.716762 / 87.999737 | 13.561213 / 28.266242 | True | qualified/qualified |
| wb97m-v | df | 3 | 0.922144 / 3.788863 | 0.171395 / 0.643209 | True | qualified/qualified |
| wb97m-v | df | 12 | 3.548475 / 12.933268 | 0.592731 / 1.361898 | True | qualified/qualified |
| wb97m-v | df | 48 | 18.237878 / 24.984558 | 2.945807 / 8.424334 | True | qualified/qualified |
| wb97m-v | df | 96 | 61.576440 / 86.366824 | 10.761456 / 29.346198 | True | qualified/qualified |
