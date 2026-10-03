# Ordinary resident-grid tile admission

Matched complete PBE0/def2-SVP energy-plus-analytic-force measurements on n1
RTX 5090, finite Slurm jobs 5542/5543, 2026-10-03. Base `fb53bb548`; source
`4019520ef710bdddedd1e9d1fde2d71769d01ed2c7014fe2a408d6b45c8eb191`;
library `bda4ae74fd384116ca5758b327cd721d1ec15f961e2805276fffbc4d58452ba4`.
Both modes use that same source/binary and GPU allocation per size.

Each mode has cold, five warm, moved, and five moved-warm calls. All 144 native
and 72 fresh GPU4PySCF reference calls pass every same-geometry repeat-pair gate
(`1e-8 Eh`, `1e-7 Eh/Bohr`). Maximum errors are `1.051e-10 Eh` and
`3.037e-11 Eh/Bohr`. Native warm calls all take one SCF iteration; reference
iterations vary and are retained, not normalized. Observed PBE0 XC cache ID 406
has `on_gpu=True`; this records the actual backend, not an inference.

## Warm complete endpoints

| Atoms | 256 points (s) | Candidate (s) | Reduction | Reference (s) |
|---:|---:|---:|---:|---:|
| 3 | 0.347256 | 0.277766 | 20.01% | 1.142227 |
| 6 | 0.749709 | 0.559970 | 25.31% | 1.913221 |
| 12 | 1.670831 | 1.462801 | 12.45% | 1.412798 |
| 24 | 5.036699 | 4.585956 | 8.95% | 2.201190 |
| 48 | 17.961313 | 15.177445 | 15.50% | 5.961485 |
| 96* | 77.349014 | 66.358783 | 14.21% | 13.607823 |

Sizes 3–48 compare explicit 256 against automatic 1024 under unchanged
512 MiB additional-device / 256 MiB additional-host caps. **96* uses both caps
explicitly raised to 1 GiB for both variants**, comparing explicit 256 and 1024.
Default automatic 96 remains 256: full-AO 1024 needs 603,451,392 numeric grid
bytes alone, and also exceeds the default host cap. It is not a default-budget
performance result. Ordinary default selection remains explicitly 256.

## Limits and negative results

- Dense `G*M^2` AO work and `G*A^2` partition visits are unchanged. This is
  scheduling work, not active-AO screening or lower asymptotic complexity.
- At 96, moved time worsens from 289.241844 to 341.733907 seconds, with SCF
  iterations increasing from 14 to 18. Moved-warm changes from 77.501910 to
  66.664322 seconds. These samples are not discarded.
- Ordered processes share on-disk compilation/artifact caches. Retained cold
  times do not measure an isolated cold-compilation speedup.
- This fb53-based campaign precedes the separate #1767 claim-barrier repair;
  no timing is relabeled as belonging to its corrected source/binary.
- The complete endpoint remains slower than the reference at larger sizes.
  Reference warm iteration counts differ; see every retained row.

## Recheck and reproduce

Run `python benchmarks/results/pbe0-resident-tiles-20261003/verify.py`.
The verifier checks lossless member hashes, scientific/binary/allocation
identity, every numerical repeat pair, work counts, medians and iterations.

`campaign.json.xz` contains one JSON mapping paths to the **exact original
UTF-8 text** of records and receipts; `storage.json` binds every member hash.
Decode with Python `lzma.decompress` and `json.loads`, then select a named
member without extracting or executing it. Reproduction harness, finite Slurm
job script, source patch, build/cache receipts and full summary are included.
No binary, profiler output, external archive, or Release is required.
