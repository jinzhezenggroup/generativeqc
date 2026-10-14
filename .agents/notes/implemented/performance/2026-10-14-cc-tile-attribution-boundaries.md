# Decision: separate CC timeline observations from a pipeline proposal

Status: implemented
Date: 2026-10-14

## Problem

Issue #1765 requires evidence of producer/transfer/wait latency on the larger
energy-only DF-CCSD(T) endpoint before spending memory on double buffering.
Inclusive host scopes, CUDA API calls and concurrent kernels are different
views of the same wall interval. Their sums cannot form an endpoint phase bill.

## Decision

`benchmarks/cc_tile_pipeline_attribution.py` partitions an explicitly bounded
integer-nanosecond clock by sets of observed interval kinds. Nested and concurrent
ranges retain their set membership until every range of that kind ends. Its
exclusive ledger sums exactly to the recorded complete window; the empty set
is unattributed wall. Inclusive sums, unions and intersections remain separate.

The versioned input contract requires a separate source/library/executable/input/
environment/device/run receipt, exact endpoint boundary, explicit coverage and
capture flags. Nsight extraction also binds SQLite SHA256/size and selects an
explicit process, device, submission threads and complete-process NVTX range.
Unrecognized columns/copy kinds and duplicate records fail closed. A missing
table does not prove zero transfers. Supported DMA directions retain actual
byte counts independently of logical traffic estimates and DRAM counters.

Producer scopes require exact labels from an independently audited capture
recipe. CUDA synchronization API durations are observed host intervals;
`cudaStreamWaitEvent` enqueue duration is not a device event wait. The SQLite
adapter separately parses CUPTI synchronization intervals and types when that
table exists; absence makes wait coverage incomplete rather than zero. The
production CC owner currently provides work/phase diagnostics without precise
producer NVTX ranges, so those producer ranges cannot be reconstructed from
idle spaces.

`benchmarks/cc_tile_pipeline_campaign.py` invokes the existing native molecular
benchmark rather than building another scientific implementation. It binds
input/oracle/binaries and a whitelisted environment; checks the loader choice;
explicitly selects energy-only strict FP64/Q8/DIIS8; retains each run's native
output, failure and independent energy/(T)/physical replay gates. Clean runs
and the separate parent-NVTX diagnostic are distinct experiments.

## Rejected alternatives

- Summed kernel or API durations as additive wall phases double-count overlap.
- No-recorded-GPU-work time as a producer stall invents causality and CPU work.
- Runtime stream-wait API duration as GPU wait time confuses enqueue with wait.
- Automatically recommending double buffering from overlap/gaps ignores data
  readiness, ownership, resource reservation and actual endpoint benefit.
- Reusing the old copy-elision receipt as current-source qualification would
  erase source/library identities and claim the same optimization twice.

## Invariants and evidence

The executable tests use a hand clock and an independent discrete cell oracle,
plus SQLite/identity/tail/loss/failure/opaque-phase controls. Scientific tests
retain the existing absolute gates: total energy 1e-8 Eh, (T) 1e-10 Eh and
expanded physical replay 1e-10. These host checks validate the reporter and
gate handling, not execution of a large CC endpoint on a GPU.

The historical `benchmarks/results/df-cc-native-copy-roundtrips-20261010/`
contains a pair-enabled-parent ethane230 observation: 165.499 to 160.893 s,
logical avoided traffic 7.042e12 bytes and a separate six-copy diagnostic
4.703272156 s. Neither the traffic nor the diagnostic is a new DRAM/critical-
path/prefetch measurement. That frozen parent/library is not current master.

## Consequences and revisit conditions

Critical-path hideable time, physical DRAM traffic and a two-buffer reservation
remain null until independently measured. Missing phases yield INCOMPLETE
rather than manufactured zero. No asynchronous schedule, changed mathematics,
precision/default promotion, force/Lambda/response qualification or parent
issue closure follows from this reporter. A later separately owned slice may
add causally mapped producer/device-event instrumentation and then determine
whether a bounded second buffer is worthwhile.

## References

- Issue #1765 and parent #1401.
- `docs/maintainer/performance_engineering.md` causal profiling contract.
- [NVIDIA Nsight Systems SQLite schema and copy kinds](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).
