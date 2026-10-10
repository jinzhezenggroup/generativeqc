# Decision: Add requested-byte observation without widening allocation claims

Status: implemented
Date: 2026-10-10

## Problem

Issue #1630 requires distinct allocation count, cumulative requested bytes and
observed peak ownership. The existing native device ledger exported only count,
live, peak and rejections. Peak cannot recover requests that were already freed,
nor can it distinguish retained storage from allocations in warm replay.

## Decision

Add a successful-request byte counter to the existing numeric-buffer owner and
an additive, length-checked five-counter `generativeqc_resource_ledger_read_v2`.
The four-counter v1 ABI stays unchanged. Python uses v2 when present and keeps
the exact legacy export when it is absent. Each observation binding resets
request bytes and counts but retains live charges and their initial peak.

Only allocation requests that successfully enter the ownership registry count.
Failed CUDA allocations, budget rejection and host registry failure do not.
Frees do not reduce request bytes. Counter overflow invalidates the v2 read
rather than changing resource selection, numerical execution or producing an
apparently zero-allocation replay. The ordinary allocation path adds no dynamic
observation storage and performs no additional CUDA calls.

## Rejected alternatives

- Extending the existing four-counter read in place would overrun v1 callers.
- Inferring requested bytes from peak/live would fabricate missing evidence.
- Rejecting an otherwise legal allocation on observation overflow would make
  instrumentation change the production execution contract.
- Treating v2 as a complete replay receipt would hide its absent event journal,
  phase/source/build identity and host coverage. Both snapshot modes remain
  `INCOMPLETE` in the strict receipt consumer.

## Evidence and remaining gates

`tests/python/test_device_ledger_requested_bytes.py` compiles the unchanged
production C API translation unit and CUDA allocation wrapper with explicit
host allocation doubles. It covers byte/count/peak separation, asynchronous
allocation, retained-owner reset, rejection/registry cleanup, overflow, v1 ABI
and Python capability fallback. `test_replay_allocation_audit.py` protects
strict parsing and incomplete-evidence behavior.

Local regression selections passed: 89 tests, a further overlapping 198-test
resource/receipt selection, and 8 allocation-reuse tests. Compiler structure
validation checked 504 modules with zero dependency errors. Ruff lint/format
and `git diff --check` passed.

A real CUDA allocation probe passed on node1 through Slurm job 6859, main
partition, one `gpu:5090:1`, time limit 3 minutes, preserving Slurm's assigned
`CUDA_VISIBLE_DEVICES=5`. The actual C API translation unit and allocation
wrapper reported two freed requests totaling 64 bytes with peak ownership 40;
a third retained 64-byte request brought the cumulative bytes to 128. Rebinding
reported retained live/peak 64 with zero allocations and requested bytes; release
reduced live to zero without altering peak or requests. Source, probe and logs
are retained in ignored `.artifacts/issue1630/` in the isolated checkout and
its node1 qualification copy. This is real allocator integration, not an HF
endpoint or performance measurement.

The complete native CUDA library then built successfully from baseline
`f87d51ab616cc9aa774bd344a15e268cef7eaf05` plus this isolated patch, using
verified ccache 4.5.1 and the release sm_120 preset. Shell and stationary
CUDA/CPU force AOT were disabled for this qualification; no performance/default
promotion is claimed. Library SHA-256:
`ad3c4ec8160cae4725a77ab280228b3e6bffeb6c1cfa8bdc7887bc40616704dd`.

Slurm job 6864 (node1/main, one 5090, 15-minute limit, assigned visibility 5)
passed 22 resource/prepared tests, including RHF/UHF cold/warm energy, first/warm
force routes, changed geometry and native metadata/concurrent-binding checks.
The retained `prepared-capture.json` independently records eight route windows
for the H2/water/H2 batch. Both methods have zero new owned-device allocations
and zero requested bytes in warm energy and force replay; retained capacities
remain nonzero. First-route requested bytes/counts are 37,208/2 and 33,360/4
for RHF, and 49,632/2 and 45,784/4 for UHF. Force-route peak ownership is
41,144 and 53,568 respectively, distinct from requested and final live bytes.
Energies agree exactly with matched ordinary prepared execution and the maximum
force difference is below `2.2e-15` Eh/bohr. These are observation-on/off
non-regression checks, not a new independent scientific-method qualification.
Raw snapshot assessments correctly remain `INCOMPLETE` for the strict receipt
contract despite these successful scoped endpoint gates.

`test_hf_resources_cuda.py` requires zero count and requested bytes on warm
prepared energy and force replay, with existing numerical comparisons. Real
device execution must run through Slurm; host doubles do not satisfy that gate.
Closing #1630 still requires the declared runtime capture/identity/host evidence
and the real prepared-path qualification, not this counter alone.

## Revisit when

The existing measurement owner gains bounded event capture and independently
pinned endpoint/phase identity, or complete host allocation observation becomes
available. Do not replace it with a second scientific allocator.
