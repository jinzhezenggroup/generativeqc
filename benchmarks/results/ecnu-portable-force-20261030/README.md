# Portable force seed and small-work routing

`report.py` prints complete default-grid energy/force warm medians. Each seed
cohort contains all 64 README combinations (eight methods, Direct/DF, 3/12/48/96
atoms), cold plus three warm calls from the engine's frozen cold density, with
one math thread. A100, RTX 5090 and PRO 6000 use their recorded scheduler jobs.
GPU4PySCF references are the preceding A100/5090 observations; PRO 6000 has no
reference measurement. These are static default-grid observations, not a
universal accuracy-qualified ranking or unmeasured-GPU performance guarantee.

The 21-class seed passes the same 58/64 independent gates per device. HF DF
fails the dense-direct fitting-energy gate at all sizes; M06-2X three-atom
Direct/DF fails the default-grid RMS-force gate. No gate changed. WB97M-V DF
retains different long-range fitting contracts between engines.

The final small-work routing change has a **separate ten-case cohort** per
GPU: HF Direct/DF at all four sizes, plus PBE0 Direct at 3/12 atoms. It is not a
new 64-case matrix. The earlier seed's small-HF regression and interleaved
old/new pairs remain visible. Both original noisy A100 PBE0 DF3 observations
and all three paired repeats are retained; no outlier is discarded.

The XZ JSON retains every endpoint time, energy, iteration/Fock count and
available grid/nonlocal work, paired error envelopes, final seed and routing
per-repeat oracle errors, original source/binary/scheduler identities, and
one-off emission comparisons. Superseded 20-class prototypes retain all
populations, work, gate outcomes and paired errors; their receipt-set hashes
bind the original case-to-receipt maps. Full raw receipts, the producer, failed
instrumented experiments and compiler logs remain in ignored
`.artifacts/ecnu-stationary-parity-20261030/`. This compact summary does not
reconstruct force vectors. The source revision alone does not identify the
uncommitted measured binary; native build identities remain authoritative.

The corrected original portable baseline and references are read from the
verified Git snapshot in [the preceding bundle](../ecnu-df-projection-20261030/).
Do not use that bundle's older invalid-force HF48/96 baseline.
See [the decision note](../../../.agents/notes/implemented/performance/2026-10-30-portable-force-seed-and-work-crossover.md).
