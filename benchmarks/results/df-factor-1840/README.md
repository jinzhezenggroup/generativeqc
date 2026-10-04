# Ethane230 DF factor precision repair

This records a source-only numerical qualification for
[#1840](https://github.com/jinzhezenggroup/generativeqc/issues/1840), with no
RHF/CC solve, endpoint timing claim or complete-force qualification.
Independent numerical review is pending. The original failing reference is
preserved; the passing result uses an independently justified corrected
reference, not relaxed tolerances.

## Final actual GPU result

n2 Slurm job **2268**, one RTX PRO 6000 Blackwell, CUDA 12.9.1, driver 595.91.07.
The actual rebuilt library SHA-256 is
`368f16fd2410b3c9a241e310918346a1bc1565309f5fc278928eaa1b35766216`.
The base is `ffe7ed9c981619d715c5a43fa6bcf39da29190d6` plus this PR's exact
code. All 3198 final source/config/test inputs were hash-compared with the
archive used on n2. The input tree hash and modified-file hashes are retained
in [provenance.json](provenance.json); the run is not mislabeled as a clean
committed HEAD. No node3 compilation or job was used.

Every element is checked with the original allowance
`3e-10 + 3e-10 * abs(reference)`:

| Quantity | Maximum absolute error | Failures |
| --- | ---: | ---: |
| Bov | 2.6626704313637006e-11 | 0 |
| Bvv | 8.557535192618704e-11 | 0 |
| ovov | 1.2029982045247412e-13 | 0 |
| ovvo | 1.2029982045247412e-13 | 0 |
| oovv | 1.3680723220943491e-12 | 0 |
| ovoo | 2.350867648326192e-13 | 0 |
| oooo | 1.149302875091962e-12 | 0 |
| Boo (additional) | 1.7520512818336442e-11 | 0 |
| Actual unprojected Bov | 2.6824410748194083e-11 | 0 |
| Actual unprojected Bvv | 8.831619506891042e-11 | 0 |

[native-replay.json](native-replay.json) retains worst indices, paired values
and error/allowance maxima. The existing physical pair projection is applied
to both published-factor paths. The last two checks independently capture the
actual GPU values before that projection, so acceptance does not rely on it.

C remains byte-identical, rank is 488 and the actual native cutoff is
`1.087578826627117e-7`. The original budget is 68,719,476,736 bytes; admitted
numeric capacity is 1,194,388,199 bytes. Exactly 230 raw rows and one whitening
call were captured. Both raw components are charged: 413,043,200 output bytes.
This adds 897,920 bytes of row storage. Semantic counts are 24,472,809,600
transform summands and 5,873,584,104 block summands, not hardware FLOPs.
Instrumented timing includes interposer copies/synchronizations and is not a
benchmark.

## Before-fix evidence and reference correction

The original Rys-repaired arrays reproduced Bov 44 failures, Bvv 988025 and
oovv 5915. Their maximum errors were respectively `8.102559946772392e-10`,
`5.8931808133303246e-8` and `2.9842685544956282e-9`.
Even the newly repaired native factors still fail against the known-inaccurate
old reference: Bov 44 failures (`7.991624907405931e-10`) and Bvv 985960
(`5.8848537890388714e-8`). Those results are explicit in the final report.

The [decision note](../../../.agents/notes/implemented/numerics/2026-10-04-df-frozen-factor-precision.md)
separates independently verified original-reference integral error, original
reference transformation roundoff, native MO dot cancellation and raw rounding
amplification. It includes the 90-digit dominant primitive and rejected routes.
The [corrected-reference record](reference.json), generated in n2 CPU job2266,
pins unchanged C, basis metadata and recipe/helper hashes. Its root moments
have maximum relative error `5.225705865606745e-15` against a 90-digit oracle.
The reference is independent of the production compiler and has no pair
projection. Its NPZ SHA-256 is
`cb66df98e25968f03a92e46742d596c35475d376a8e9fe3e60b343a16cfeea62`.

## Validation and reproduction

- Job2267: final CUDA build with ccache, Release, sm120,
  AOT=ON / `portable_cuda` empty registry. The prior AOT=OFF duplicate-stub
  problem is not silently reclassified as a passing build.
- Job2270: 124 host tests passed, including independent 90-digit primitive and
  reference-oracle tests, source IR/layout and capacity boundaries. Its 14 GPU
  skips subsequently passed in job2271 against the same final library.
- Job2268: 12 molecular factor/block cases and 22 physical source/response
  cases passed, then the complete frozen source replay above passed.
- Job2271: 14 streamed MO response, ownership/failure and nonfinite tests passed.
- Compiler dependency check: 427 modules, zero errors. CUDA ownership inventory:
  313 files. [Ownership delta](ownership-delta.json) retains unchanged semantic
  classifications: +8 scientific CUDA code lines and +27 runtime lines. The
  scientific growth is bounded public-layout high/low writes and accumulation;
  integral and compensated-dot formulas remain compiler-owned.

Use the [stable reproduction tools](../../../docs/maintainer/df_frozen_precision.md).
The original 1,353,823,563-byte frozen archive is hash-verified; large arrays and
transient logs remain outside Git under
`n2:/data/jzzeng/df-factor-1840-20261004/`. No public archive URL is claimed.
The exact build/replay shell scripts are retained there. Earlier adapter-link
and interpreter setup failures are recorded in provenance and were resolved
before scientific qualification.

**Benzene264 and complete large-system forces remain unqualified.** No inference
from these source or small-response passes closes either gate. Independent
review of the numerical repair/reference correction is required before closing
#1840.
