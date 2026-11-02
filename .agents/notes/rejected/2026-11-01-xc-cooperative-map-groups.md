# Decision: retain serial discovery of equal XC AO maps

Status: rejected
Date: 2026-11-01

## Problem

An RTX 5090 r2SCAN-3c/96 profile attributed 0.908 s to ordered matrix scatter
over one cold and one warm endpoint. The scatter kernel repeatedly compares
consecutive AO maps from one lane in each row-owner CTA. Cooperatively testing
columns could shorten those scans without changing floating-point work.

## Experiment and disposition

The candidate assigned columns to block lanes and used `__syncthreads_or` for
a uniform group boundary. It preserved map grouping, output ownership, tile
addition order, kernel counts and allocated scratch. It introduced no GPU-name
condition or scientific approximation. The source patch and full receipts are
retained in `benchmarks/results/xc-map-groups-rejected-20261101/`.

Do not promote this candidate: its complete endpoint differences are too small
to establish a useful gain. It adds synchronization complexity without resolving
the measured GPU4PySCF gaps. The production emitter retains its prior serial
group discovery. These single pilot populations do not establish statistical
equivalence, a repeatable speedup, or universal nonregression.

## Evidence

All 24 parent/candidate populations (48 complete energy-and-force returns) pass
the unchanged independent oracle gates. The task is water96, DF Standard,
SCF tile256, force tile1024, eight allocated CPU slots, single-threaded math
libraries, one cold call and one frozen warm repeat per population.

Candidate versus parent endpoint change (negative is faster):

| Device | Method | Cold | Warm |
| --- | --- | ---: | ---: |
| A100 | PBE | -0.62% | -0.51% |
| A100 | B3LYP | -0.31% | -0.03% |
| A100 | r2SCAN-3c | -0.16% | -0.02% |
| A100 | M06-2X | -0.37% | -0.36% |
| RTX 5090 | PBE | -0.29% | -0.07% |
| RTX 5090 | B3LYP | +0.08% | +0.34% |
| RTX 5090 | r2SCAN-3c | -0.21% | -0.10% |
| RTX 5090 | M06-2X | -0.17% | -0.10% |
| PRO6000 | PBE | +0.05% | +0.40% |
| PRO6000 | B3LYP | -0.00% | -0.10% |
| PRO6000 | r2SCAN-3c | -0.12% | -0.00% |
| PRO6000 | M06-2X | -0.08% | +0.22% |

Blackwell controls share each Slurm allocation (5090 job1395, PRO6000 job3823).
A100 parent job224130 and candidate job224131 are separate allocations: the
first candidate native executable failed before device execution because its
loader did not match the supplied libc. The corrected loader invocation passed;
both original failure and successful follow-up are retained. Do not label that
A100 comparison same-allocation or discard the failed attempt.

Point-batch and ragged native suites pass on all three devices, including
independent E/V oracles, exact serial parity, RKS/UKS, tails, large AO boundaries,
capture/replay and bounded allocation fallback. Blackwell synccheck and memcheck
each report zero errors. Host simulator tests also covered 257-entry maps
differing only after warp/block boundaries; that candidate test patch is retained.

## Revisit when

A new workload demonstrates substantial map-comparison cost within scatter, or
a design discovers boundaries once per batch without extra per-owner votes.
The current profile instead prioritizes matrix contractions: local diagonal
potentials total 4.387 s and density products 2.494 s. These intrusive kernel
spans identify work to investigate; they are not clean endpoint timings.
