# Decision: batch the loads of the stationary geometry reduction

Status: implemented
Date: 2026-11-18

## Problem

`geometry_reduce` sums the per-lane geometry partials of one KS force tile into
`9*na` gradient columns, one thread per column, in lane order. Only 14 blocks
exist for 96 atoms, so each thread's chain of dependent loads is latency-bound.
On H100 it took 128 ms of each r2SCAN-3c96 warm call.

## Decision

- **Loads:** each thread loads 16 lanes into registers, then adds them in the
  original order. The FP64 summation order and its rounding are unchanged.
- **Tail:** lanes left over after the last full batch keep the scalar loop.
- **Scope:** no launch geometry, scratch, synchronization or reduction tree
  changes. The host-emulated probes therefore keep calling the kernel one thread
  at a time.

A cooperative multi-block split would remove the 14-block limit. It would
change the summation order, and the host tests call the kernel per thread, so
it is not done here.

## Evidence

`benchmarks/results/h100-stationary-geometry-reduce-20261118/`.
- **Warm timings:** r2SCAN-3c96 3.84-3.85 → 3.77-3.78 s, Direct PBE0-96
  2.11-2.12 → 2.06 s and DF B3LYP-96 1.70 → 1.64 s.
- **Kernel time:** `geometry_reduce` falls to 71 ms per r2SCAN-3c96 warm call.
- **Forces:** cross-build differences stay within run-to-run spread.
- **Host tests:** the five stationary host-probe suites pass unchanged, 95 tests.

## Revisit when

A per-block reduction is wanted. It needs a new ordered tree and host probes
that model blocks.
