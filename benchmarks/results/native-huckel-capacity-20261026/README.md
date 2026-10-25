# Native Hückel capacity: CPU endpoint qualification

Measured candidate: `7e555f3841e3109dea82fa94d12de8ad78ff0353` (clean).
Baseline: `40c31f7bd6130903ade0cd87ef60fb06c56d9d8b` (clean).

Both ABBA and BAAB pass the 1.05 non-regression limit for RHF H2 E/F,
PBE H2 energy, PBE H2 E/F and ragged PBE H2/water energy. Each block has
21 measurements after 3 warmups. The maximum pooled candidate/baseline median
ratio is 1.0299035232137126. All 672 complete endpoint samples retain their
2,520 item publications, planned peaks, final diagnostics and semantic work
fields in `samples.json.gz`; publications and planned peaks agree exactly.
No unknown Fock count is inferred. This is scoped non-regression evidence,
not complete performance promotion or CUDA qualification.

Timing includes calculator construction, resource planning/preparation,
cold/warm/genuinely moved execution, and prepared-owner closure. Source-matched
CPU Release libraries use the scalar backend and one OpenMP/OpenBLAS thread.
Library hashes and source paths are retained per measured block. Full build/test
logs remain in ignored `.artifacts/huckel-capacity/`; 119 CTests and 299 targeted
Python tests passed, with 12 explicit GPU skips. The independent atomic-oracle
and live-preparation-allocation tests remain in the source tree.

## Reproduction

Build both revisions with CUDA disabled, Release mode and a verified compiler
cache. For each arm in ABBA followed by BAAB, run the following from that arm's
checkout, using a fresh output file and the matching source SHA:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  GENERATIVEQC_LIBRARY="$PWD/build/native-ks-catalog/libgenerativeqc.so" \
  python -m benchmarks.native_resource_composition \
  --samples 21 --warmup 3 --source-sha FULL_SHA \
  --output .artifacts/benchmarks/ARM-ORDER-BLOCK.json
```

For each case/order, pool the two blocks per arm and compare median seconds.
Require candidate/baseline <=1.05 and equality of every sample's `phases` and
`peak_bytes` across arms; source-bound plan identities are deliberately separate.
The full raw signatures allow those checks to be repeated independently.
Whole-process allocation/provider memory, cold compilation cost and independent
per-iteration residual traces were not measured; the publication envelope keeps
full performance promotion unqualified. The stored decision does not change a
production selector or claim completion of #2227 R2/R6.
