# H100 three-process HF/DFT figure on master `a6d5a3c00`

A fresh H100 acquisition with the same protocol, driver, frozen inputs and stock
GPU4PySCF 1.9.0 / CuPy 13.4.1 reference as the
[2026-11-15 cohort](../paper-hfdft-cold3-20261115/README.md), on unmodified public
master `a6d5a3c006a6f95e066ab00c729dd4a6be86c6e6` (after #3111/#3112/#3114).
The earlier H100 cohort measured `efeaa3c71` plus a recorder overlay, before the
ωB97M-V changes. Its figure (`assets/paper-hfdft-cold3-h100.svg`) and data are
unchanged; this cohort adds `assets/paper-hfdft-cold3-20261118-h100.svg`.

| GPU | Original timing table | Plot summary | Formal processes | E+F calls | Cold / warm median wins |
|---|---|---|---:|---:|---|
| H100 | [timings.csv](h100/timings.csv) | [summary.csv](h100/summary.csv) | 384 | 1536 | 64/64, 62/64 |

There are 128 separate prewarm processes; their times do not enter these tables.
Each formal process makes one cold and three warm complete energy+force calls;
three fresh processes per engine and pair, balanced engine order with the middle
round reversed, one physical H100 80GB in qz Job `gqc-paper-cold3-a6d-1118-a`.

## Changes against the 2026-11-15 H100 cohort

| Case | Phase | GenerativeQC | GPU4PySCF | 2026-11-15 (native / reference) |
| --- | --- | ---: | ---: | --- |
| Direct ωB97M-V96 | cold | 34.194 s | 40.492 s | 47.896 / 42.343 s |
| Direct ωB97M-V96 | warm | 4.964 s | 12.327 s | 9.446 / 12.507 s |
| DF ωB97M-V96 | cold | 21.739 s | 36.392 s | 29.344 / 37.088 s |
| DF HF96 | warm | 0.976 s | 0.766 s | 0.971 / 0.774 s |
| Direct r2SCAN-3c96 | warm | 4.021 s | 3.676 s | 4.028 / 3.782 s |

Direct ωB97M-V96 cold, the only cold loss in the earlier H100 figure, is now a
win. DF HF96 and Direct r2SCAN-3c96 warm remain the two losses; these are medians
of three processes, not universal rankings.

## Data and reproduction

The files, schemas and checks are those of the 2026-11-15 cohort: `samples.json.xz`
holds all fields of all 384 formal records (full energies and force vectors) and
the successful-command census; `inputs.json.xz` is byte-identical to the earlier
cohort's frozen inputs; `provenance.json.xz` binds the build, hardware, reference
reuse contract and figure; `publication.json` / `evidence.json` bind storage.
Removing only the collector's `process_repeat`, `raw_file` and `raw_sha256`
additions reproduces every original formal JSON file byte for byte.

```bash
python -m benchmarks.paper_hfdft_replay --verify-only \
  --directory benchmarks/results/paper-hfdft-cold3-20261118
```

This checks all 384 processes and 1536 E+F calls, recomputes every statistic and
energy/full-force error, and reads back all 256 medians and 512 range endpoints of
the SVG. As before, 348 process qualifications pass and 36 fail: both engines'
DF-HF energy approximation and three-atom M06-2X grid/force failures are retained.

The measured source is the public commit itself (`native_export: git-archive`, no
patch). `--reconstruct-source <dir> --directory benchmarks/results/paper-hfdft-cold3-20261118`
extracts `a6d5a3c00` with `git archive`, then verifies native identity
`c5f05cc9e6971b4fc3319057d54f35b908710fb04ae7ffd3402afe4319406ad6` and both
recorder hashes; a git-archive record that also carries a patch or export rule is
rejected.

## Storage

All original scientific values are retained in bounded XZ JSON; every file is
below the 1 MiB hard limit. The aggregate addition and the large-record review
(`benchmarks/evidence-reviews/paper-hfdft-cold3-20261118.json`) are explicit.
