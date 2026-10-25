# Rejected native admission: XC schedule diagnosis

Historical baseline: `a97112c4e703f90d21ba48f032d49d2d2e9c19fe`.
Historical candidate: `d474d0b265e59e0ed36c819839f9ad2a5293b75f` (#2586, closed/unmerged).
Diagnostic Slurm job 1858 ran on n5/main/gpu:5090:1 with a 15-minute limit.
Neither historical source nor its installed products was modified.

Five configurations of two complete public SDK endpoints ran in forward and
reverse order, each with 21 samples after three warmups: default baseline,
baseline without point batching, baseline without AO maps, baseline without
both, and the original candidate with native admission. All 420 samples retain
cold/warm/moved preparation/execution/closure, identical iteration counts and
E/F parity within 1e-12. These are diagnosed PBE RKS H2/STO-3G Direct E/F and
DF energy cases; they do not replace the original full qualification matrix.

The candidate is about 26% slower for Direct and 42% for DF, exceeding the
unchanged 1.05 gate. Disabling both optional schedules in the original baseline
reproduces candidate timing within 0.2%. Point batching alone explains most
of the effect. Separate Nsight traces show 25 versus 720 XC point evaluations
and 6149 versus 7293 total kernel launches; the candidate and disabled-control
counts agree. Profiling overhead is excluded from clean timing samples.

See the [Agent Note](../../../.agents/notes/rejected/2026-10-26-prepared-admission-xc-reservations.md)
for the future-owner reservation constraint. Removing a ledger guard or spending
currently unused capacity does not supply a safe optional-panel allowance.
No replacement implementation or resource/performance promotion is included.

## Reproduction and retained data

Decode `samples.json.gz` with Python gzip/json. It contains all observations,
exact diagnostic runner and original installed SDK probe source, executable and
library identities, original compile receipts, Slurm visibility, and the three
separate Nsight summary tables. `summary.json` retains each exact command,
configuration, clean median and profiler counts. The runner verifies the
preserved products before and after execution. Rebuild the two source revisions
with the original recorded toolchain or use authenticated historical products;
compile through a verified cache and run all GPU execution/profiling through
finite Slurm allocations preserving assigned device visibility.

The original products and full traces remain in ignored n5 qualification
storage named by the exact runner. The original failed campaign remains intact.
The compact bundle is sufficient to recompute the retained diagnostic comparisons;
reproducing runtime measurements naturally requires the scientific build/GPU.
