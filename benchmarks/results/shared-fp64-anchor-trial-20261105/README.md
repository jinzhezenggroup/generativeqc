# Shared strict-FP64 anchor trial, 2026-11-05

**Decision: inconclusive complete trial; strict default unchanged.** The shared
FP64 J/K/range-K implementation is explicitly selectable for eligible Standard
(`fp64`) owners using `GENERATIVEQC_KS_FIXED_ANCHOR_DELTA=1`. Zero disables it;
unset retains the existing qualified Fast default and leaves strict reuse off.
Arithmetic, SCF tolerances, total-density nonlinear XC/VV10 and final physical
closure do not change. This is not a claim that only precision differences remain.

## Observations and limits

| Standard workload | OFF cold median (s) | ON cold median (s) | Median paired reduction | Warm ON/OFF |
| --- | ---: | ---: | ---: | ---: |
| Water-24, 192 AOs | 50.320683 | 48.756945 | 3.0507% | 0.999808 |
| Water-48, 384 AOs | 161.042278 | 156.908628 | 2.6165% | 0.998864 |

These are three alternating-order paired rounds, each with three same-owner
warm calls. Reduction is the median of paired ratios, not the quotient of the
displayed marginal medians. Cold timing includes construction, preparation,
synchronized complete energy/forces, host force/density export and teardown;
intervening warm calls and independent oracle work are excluded. Warm timing
includes synchronized complete energy/forces and host force/work export.

All 12 Standard cold and 36 warm calls pass independent energy/max-force/RMS-force
gates `1e-8/1e-7/1e-7`; paired gates are `1e-9/1e-7/3e-8`. Independent stock
GPU4PySCF uses the identical complete spherical def2-SVP snapshot and moving
grid `[48,16,32]`, never native seeds. Standard J/K work is unchanged by ON/OFF:
water-24 uses 27 iterations, 28 Focks, 28 J and 56 K applications; water-48 uses
28 iterations, 29 Focks, 29 J and 58 K applications. Warm calls use one iteration,
two Focks, two J and four K applications. Full/provisional/precision work is
retained in the lossless raw receipts; this is not omitted-operator work.

All Fast OFF/ON endpoints in this native-library-only fixture lack the packaged
`wb97mv_rks` stationary CUDA force artifact and fail closed. They supply **no**
performance comparison; SCF convergence is not complete E/F qualification.
This does not establish an anchor-induced regression. Peak endpoint memory
and a single fresh total compile cost are also unavailable. The small,
shape-specific Standard observations do not promote a general speed/default.

## Evidence and provenance

`samples.json.xz` retains full successful and failed cold/warm scientific,
work and gate receipts. `summary.json.xz` retains statuses, comparisons and the
interrupted/retried history. `profile-work.json.xz` retains frozen-image CUDA/NVTX
counts and full CSV/trace/SQLite checksums: Standard ON has 21/22 delta-density
preparations, OFF has zero, and all warm ranges have zero. These prove selection,
not a public native anchor-counter census. Full reports/databases stay ignored;
profiler timings never enter performance samples. Native fixture counters and
capacity/lifetime gates are separate evidence, not a measured memory peak.

`qualification-receipts.json.xz` retains small native/build/oracle/cache receipts
and selected actual launcher commands. Large logs have explicit byte counts,
checksums and tails, not full log archives. It preserves the initial 1800-second
build and 390-second water-48 oracle timeouts and their successful continuations.
The 40-minute campaign expires during the final Fast owner; a separate retry
preserves that interrupted original rather than overwriting it.

Base: `c511f9a330ef4e6c7d48e6ed4a5f778bff260001`, dirty measured source. Exact
native/driver identities are in `source-provenance.json`; exact library/device
identity is in the full samples. Normal SM120 Release, fast-compile/test hooks
off for timing; CUDA 12.9.86, GCC 13.3.0, Python 3.12.12, sccache 0.16.0.
n1 Slurm jobs: native fixture 15934; timing 15935/retry 15937; diagnostic traces
15936; independent water-24 oracle 15931 and water-48 retry 15932. All runs have
finite `srun` limits and preserve assigned visibility; maintenance n3 is excluded.

## Reconstruction

Copy the bundle into an ignored reconstruction directory and expand its
lossless XZ members there (`xz -dk -- *.xz`), leaving the publication unchanged.
Apply `measured-source.patch` to a clean base checkout. It reconstructs changed
native/Python inputs, the exact driver and native fixture. Its three native
inventory file blocks reproduce the inventory-patch hash in provenance.
Verify the reconstructed native/driver identities; do not apply the patch on
top of an already updated checkout.

The original scripts use task root
`/home/jzzeng/generativeqc-qualification/standard-fast-unified-anchor-20261105-c511f9a33`,
with checkout `source`, Python environment `.venv`, basis `def2-svp.json` and
oracle files `reference24.json`/`reference48.json`. Restore that layout or adapt
all script roots consistently. Required numerical packages are CuPy 13.6.0,
PySCF 2.14.0 and GPU4PySCF 1.9.0 plus repository development dependencies.

1. Verify/reuse sccache; `build-remote.sh` records configuration and
   `finish-remote.sh` records the successful source-authenticated native test
   and hook-free production rebuild. A fresh build need not repeat the timeout.
2. Use retained basis/oracles for exact replay, or regenerate independent
   references with `reference-remote.sh` and `oracle48-retry.sh` in fresh paths.
   Schedule the latter with 22 minutes; its coarse seed is GPU4PySCF-only.
3. Run `benchmark-remote.sh` in a fresh root through `srun --partition=main
   --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:55:00`.
   Keep every failure. `continue-remote.sh` shows non-overwriting continuation.
4. Run `profile-remote.sh` separately through finite `srun` (40 minutes), never
   concurrently with host timing. Reduce with `python trace-work.py
   "$ROOT/evidence/profiles"` and `python analyze-trial.py "$ROOT"`.

The official publication manifest binds every retained byte to an explicit
inconclusive performance decision, not a production promotion.
