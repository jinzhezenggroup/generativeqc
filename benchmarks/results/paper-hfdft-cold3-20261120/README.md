# H100 three-process HF/DFT figure on master `471ea1a0a`

This is a fresh H100 acquisition on unmodified public master
`471ea1a0aa9b872998be27abc990dcc837182a4c`, after #3126. Protocol, driver,
frozen inputs and the stock GPU4PySCF 1.9.0 / CuPy 13.4.1 reference are the same
as in the [2026-11-19 cohort](../paper-hfdft-cold3-20261119/README.md). Earlier
cohorts, figures and data are unchanged. This cohort adds
`assets/paper-hfdft-cold3-20261120-h100.svg`.

| GPU | Original timing table | Plot summary | Formal processes | E+F calls | Cold / warm median wins |
|---|---|---|---:|---:|---|
| H100 | [timings.csv](h100/timings.csv) | [summary.csv](h100/summary.csv) | 384 | 1536 | 64/64, 64/64 |

The 128 separate prewarm processes do not enter these tables.

- Each formal process makes one cold and three warm complete energy+force calls.
- There are three fresh processes per engine and pair.
- Engine order is balanced, with the middle round reversed.
- The run used one physical H100 80GB in qz Job `gqc-paper-cold3-471-1120-a`.

## Changes against the 2026-11-19 H100 cohort

| Case | Phase | GenerativeQC | GPU4PySCF | 2026-11-19 (native / reference) |
| --- | --- | ---: | ---: | --- |
| Direct HF96 | warm | 1.291 s | 1.665 s | 1.380 / 1.353 s |
| Direct HF96 | cold | 6.190 s | 7.234 s | 6.429 / 6.535 s |
| DF HF96 | warm | 0.642 s | 0.765 s | 0.644 / 0.768 s |
| Direct r2SCAN-3c96 | warm | 3.485 s | 3.788 s | 3.494 / 3.770 s |
| Direct ωB97M-V96 | cold | 33.471 s | 41.563 s | 33.526 / 40.268 s |

Direct HF96 warm, the only loss in the 2026-11-19 cohort, is now a win.

- **Direct HF96 warm.** The native processes measure 1.288–1.298 s. The
  GPU4PySCF processes measure 1.622–1.769 s. In the previous cohort they ranged
  from 1.312 to 1.840 s. Even the slowest native call here is faster than the
  fastest previous reference call.
- **Direct HF96 cold.** The native processes measure 6.188–6.199 s, against
  7.121–7.723 s for GPU4PySCF. A previous reference process reached 5.831 s, so
  this cold ranking depends on reference variability across cohorts.
- **Narrowest margins.** The narrowest warm margin is Direct r2SCAN-3c96, where
  GPU4PySCF is 1.087× slower. The narrowest cold margin is Direct HF96, at 1.169×.

These are medians of three processes, not universal rankings.

## Data and reproduction

The files, schemas and checks are those of the earlier cohorts:

- `samples.json.xz` holds all fields of all 384 formal records, including full
  energies and force vectors, and the successful-command census.
- `inputs.json.xz` is byte-identical to the earlier cohorts' frozen inputs.
- `provenance.json.xz` binds the build, hardware, reference reuse contract and
  figure.
- `publication.json` and `evidence.json` bind storage.

Removing only the collector's `process_repeat`, `raw_file` and `raw_sha256`
additions reproduces every original formal JSON file byte for byte.

```bash
python -m benchmarks.paper_hfdft_replay --verify-only \
  --directory benchmarks/results/paper-hfdft-cold3-20261120
```

The verifier does the following:

- checks all 384 processes and 1536 E+F calls;
- recomputes every statistic and every energy and full-force error;
- reads back all 256 medians and 512 range endpoints of the SVG.

As before, 348 process qualifications pass and 36 fail. The failures are the
DF-HF energy approximation in both engines and the three-atom M06-2X grid/force
cases.

The measured source is the public commit itself (`native_export: git-archive`,
no patch). Run
`--reconstruct-source <dir> --directory benchmarks/results/paper-hfdft-cold3-20261120`
to check it. This extracts `471ea1a0a` with `git archive` and verifies the
2256-input native identity
`9bc35927feb6c549f54a031dfa2f9c66b4d3b1a6fe6bb326b3471c0cc80eb4c2` and both
recorder hashes.

## Storage

All original scientific values are kept in bounded XZ JSON, and every file is
below the 1 MiB hard limit. The aggregate addition and the large-record review
(`benchmarks/evidence-reviews/paper-hfdft-cold3-20261120.json`) are explicit.
