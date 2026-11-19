# H100 three-process HF/DFT figure on master `cddde09a0`

A fresh H100 acquisition with the same protocol, driver, frozen inputs and stock
GPU4PySCF 1.9.0 / CuPy 13.4.1 reference as the
[2026-11-18 cohort](../paper-hfdft-cold3-20261118/README.md), on unmodified public
master `cddde09a0f9a77f65d0f6776e5cc883ee67c28dd`, after #3115, #3116 and
#3118–#3123. The earlier cohorts, figures and data are unchanged; this cohort
adds `assets/paper-hfdft-cold3-20261119-h100.svg`.

| GPU | Original timing table | Plot summary | Formal processes | E+F calls | Cold / warm median wins |
|---|---|---|---:|---:|---|
| H100 | [timings.csv](h100/timings.csv) | [summary.csv](h100/summary.csv) | 384 | 1536 | 64/64, 63/64 |

There are 128 separate prewarm processes; their times do not enter these tables.
Each formal process makes one cold and three warm complete energy+force calls.
There are three fresh processes per engine and pair, in balanced engine order
with the middle round reversed. All ran on one physical H100 80GB in qz Job
`gqc-paper-cold3-cdd-1119-b`.

## Changes against the 2026-11-18 H100 cohort

| Case | Phase | GenerativeQC | GPU4PySCF | 2026-11-18 (native / reference) |
| --- | --- | ---: | ---: | --- |
| DF HF96 | warm | 0.644 s | 0.768 s | 0.976 / 0.766 s |
| Direct r2SCAN-3c96 | warm | 3.494 s | 3.770 s | 4.021 / 3.676 s |
| Direct HF96 | cold | 6.429 s | 6.535 s | 6.610 / 6.661 s |
| Direct HF96 | warm | 1.380 s | 1.353 s | 1.396 / 1.496 s |
| DF B3LYP96 | warm | 1.129 s | 1.955 s | 1.692 / 1.909 s |
| Direct ωB97M-V96 | cold | 33.526 s | 40.268 s | 34.194 / 40.492 s |

The two earlier warm losses, DF HF96 and Direct r2SCAN-3c96, are now wins.
Direct HF96 warm is the only loss, by 2%. Its three GenerativeQC processes are
tight at 1.372–1.428 s. The GPU4PySCF processes vary from 1.312 to 1.840 s, and
their median is lower than in either earlier H100 cohort. Direct HF96 cold is
the narrowest cold win. These are medians of three processes, not universal
rankings.

## Data and reproduction

The files, schemas and checks are those of the earlier cohorts:

- `samples.json.xz` holds all fields of all 384 formal records (full energies
  and force vectors) and the successful-command census.
- `inputs.json.xz` is byte-identical to the earlier cohorts' frozen inputs.
- `provenance.json.xz` binds the build, hardware, reference reuse contract and
  figure.
- `publication.json` and `evidence.json` bind storage.

Removing only the collector's `process_repeat`, `raw_file` and `raw_sha256`
additions reproduces every original formal JSON file byte for byte.

```bash
python -m benchmarks.paper_hfdft_replay --verify-only \
  --directory benchmarks/results/paper-hfdft-cold3-20261119
```

This checks all 384 processes and 1536 E+F calls, recomputes every statistic and
energy/full-force error, and reads back all 256 medians and 512 range endpoints of
the SVG. As before, 348 process qualifications pass and 36 fail: both engines'
DF-HF energy approximation and three-atom M06-2X grid/force failures are retained.

The measured source is the public commit itself (`native_export: git-archive`, no
patch). `--reconstruct-source <dir> --directory benchmarks/results/paper-hfdft-cold3-20261119`
extracts `cddde09a0` with `git archive`. It then verifies the 2256-input native
identity `3d8267815e3bf83dd7b089d6f1c2e380aab652ce8754737e2e92617d72d4db24`
and both recorder hashes.

## Storage

All original scientific values are retained in bounded XZ JSON; every file is
below the 1 MiB hard limit. The aggregate addition and the large-record review
(`benchmarks/evidence-reviews/paper-hfdft-cold3-20261119.json`) are explicit.
