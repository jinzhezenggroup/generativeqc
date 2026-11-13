# Shared FP64 anchor default: complete trial, no promotion

2026-11-13. **Reject Standard default promotion on performance grounds.**
Scientific/lifetime gates pass, but Standard has no meaningful endpoint gain.
Keep this candidate a Draft; preserve Fast's incumbent optimization.

## Source, protocol and measurements

Base `e41c546c40c1fa04db5e319506aac1f9ed330d50` plus decompressed
`measured-source.patch.xz` reconstructs the measured dirty source (including
runner and test changes). Native identity, library/fixture hashes and compiler
cache version are pinned in `source-provenance.json`. Normal SM120 Release,
RTX 5090, fast-compile and test hooks off for timings; a separate hook image
passed native lifecycle tests and restoring normal produced its original hash.

WB97M-V RKS exact Direct, complete spherical def2-SVP, moving grid `[48,16,32]`,
24/48-atom water clusters (192/384 AOs). Twelve predetermined alternating
OFF/auto pairs, each with a cold owner and three same-owner warm executions:
24 cold + 72 warm endpoints. Cold time includes construction/preparation,
synchronized E+F, host/density export and teardown. No oracle or diagnostic
timing is included. Both arms have a 2-GiB optional resident cap.

| Profile | AOs | OFF/auto cold medians (s) | Median paired reduction | Warm auto/OFF |
| --- | ---: | ---: | ---: | ---: |
| Standard | 192 | 39.550315 / 39.625693 | +0.0028% | 1.008360 |
| Standard | 384 | 206.254153 / 206.443312 | -0.0185% | 1.000990 |
| Fast | 192 | 38.513397 / 34.919245 | +9.4126% | 0.999809 |
| Fast | 384 | 137.145760 / 119.334644 | +13.0275% | 1.000398 |

Reduction is median per-pair `1-auto/off`, not the ratio of marginal medians.
Predeclared acceptance requires at least one Standard gain > max(2%, twice
the sum of relative MADs), no significant other cold regression and warm
auto/OFF <=1.05. Standard fails the profit gate. Fast reproduces incumbent
benefits, not new speedups. Do not reuse the historical Standard gain claim.

All 96 endpoints pass independent and paired energy/max-force/RMS-force gates.
Independent Standard gates: `1e-8/1e-7/1e-7`; paired: `1e-9/1e-7/3e-8`.
Fast independent/paired: `1e-5/1e-4/3e-5`. Worst independent errors (Eh and
Eh/bohr) are Standard `1.2574e-10/1.4857e-11/5.9135e-12` and Fast
`2.2865e-9/5.8983e-6/1.6841e-6`. Full gates remain in `acceptance.json.xz`.

The 192-AO GPU4PySCF reference was fresh (SHA-256
`d8d92d538a1acf3324ae0112effe9e7bed6f2061a25d540650a1ce1034b51ff0`).
Fresh384 hit its finite 1,000-second timeout; reuse of the independent retained
`../shared-fp64-anchor-trial-20261105/reference48.json.xz` followed exact
scientific protocol matching (expanded SHA-256
`15871994df192d5684f7e163658949c80d9e406357d666683033c08602859237`).
That bundle also retains the complete basis and original protocol/backend
identities. Fresh192's protocol exactly matches its retained192 counterpart
(energy differs by 4.55e-13 Eh). These are validation-only, not production
dependencies. The original fresh192 reference remains ignored locally.

## Semantic work, resources and test scope

Standard iterations/Focks stay 27/27 and 28/28 at 192/384 AOs; Fast changes
18/19 to 16/17. Separate non-timed traces show Standard auto has 6 anchor +
21/22 delta builds, Fast 4 + 12; OFF and warm have none. Deltas still apply
J/K/range-K, not omitted applications. Strict stays FP64 with physical closure.
Complete precision inventories/Fock timelines and semantic work are retained
per endpoint. Mixed-owner narrow J/K counts remain null/unavailable, not zero.

Sampled NVML process peaks (MiB) OFF/auto: Standard192 14940/14940, Fast192
14842/14842, Standard384 16938/16944, Fast384 17216/17222. Context is included;
instantaneous peaks may be missed. **No exact endpoint allocation peak or clean
full compilation-cost claim.** Separate native fixture 17331 passes exact
optional capacity, allocation/failure invalidation, bounded fallback, force
retirement and warm/recovery. Shared verified ccache stats are not exclusive
per-job savings. The Libxc identity refresh does not requalify stale evidence.

Host selection: 809 passed, six GPU skips, not the entire repository. Four
public selector cases passed in stopped job 17335, and corrected moved/warm/
fresh cases passed separately in 17336/17337: six cases, not an all-green run
of 17335. `qualification.json.xz` retains available stdout and every failed
attempt. Initial missing-input/binary deployment prechecks failed closed.
The first moved fixture omitted coordinates on replay (requesting the original
geometry); correction passes moved coordinates again and asserts warm start,
without weakening the oracle. Timeouts/interrupted attempts are not successful
samples. Final Standard384 pair 17338 was assigned before seeing its results;
the incomplete old attempt is excluded. Slurm accounting is disabled.

## Retention and reproduction

`samples.json.xz` selects all 96 endpoint timings, energies, independent force
error metrics, convergence, identities and full precision/work inventories;
it pins each original raw receipt by SHA-256. `comparison-summary.json.xz`
retains every paired numerical error and decision. To fit the existing storage
budget, raw force/density arrays and duplicate diagnostics/JUnit/build logs
stay ignored, not part of this selected bundle. This is a compact negative-result
receipt, not a replayable dump of all raw outputs. `validation.json.xz` has
numerical pass, performance fail and production not-run. Diagnostic phase
counts, trace hashes/byte counts and memory caveats are retained separately.

Reconstruct exact source, set `PYTHONPATH`, `GENERATIVEQC_LIBRARY`, library paths
and `EXPERIMENT_ROOT` (the directory containing `def2-svp.json` and independent
`reference24.json`/`reference48.json`), deploy exact derivative artifacts and pinned upstream inputs, and use a
verified compiler cache for any build/JIT miss. Copy/run `reproduce.sh` from
the reconstructed checkout with a new absolute output directory. On allowed
Slurm n1/n2/n4/n5 only (never maintenance n3), preserve assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:45:00 \
  bash /path/to/reproduce.sh 24 /absolute/fresh/output/directory
```

Repeat with 48 atoms and an appropriate finite limit. Regenerate references
with `python -m benchmarks.fast_anchor_default.reference` through finite `srun`
using the retained basis/protocol; a timeout is not permission to fabricate one.
Original selected paths: `retry-24/*`, `retry-48/*`, except Standard384 round2
uses `retry-48-round2/*`. Neither scientific nor profitability gates are lowered.
