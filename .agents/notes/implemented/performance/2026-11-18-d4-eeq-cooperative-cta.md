# Decision: evaluate CUDA D4 EEQ charges with one cooperative CTA per system

Status: implemented
Date: 2026-11-18

## Problem

An H100 nsys capture of the README water-96 r2SCAN-3c Direct endpoint showed
`eeq_prepare_kernel` and `eeq_compose_kernel` at 113 ms each per energy+force
call, about 0.23 s of a 4.04 s warm endpoint (GPU4PySCF: 3.90 s). Both kernels
ran the whole EEQ provider in one GPU thread per system: an (n+1)^2 partial-pivot
LU and 3n response back-substitutions through global memory. The prepare pass
also formed dq/dR, although fixed-charge D4 consumes only q and the compose pass
recomputes the complete provider before its vector-Jacobian product.

## Decision

The serial provider is split into shared `eeq_stage` functions (validation,
CN/matrix assembly, response right-hand sides, one response solve). The host
provider calls them in the original order. The CUDA provider runs one 128-thread
CTA per system and calls the same functions on lane zero for every
order-dependent stage; it distributes only work whose stored values do not depend
on traversal order: plain stores, the row updates of each LU elimination step
(same pivot choice and update expression) and the independent response-column
solves. The prepare pass no longer forms dq/dR. Every provider output is
therefore the same IEEE value as before. Worker slots, workspace bytes, launch
counts, failure isolation and the persistent queue are unchanged.

## Evidence

`benchmarks/results/h100-d4-eeq-cta-20261118/`. H100, fresh processes on one
physical GPU, master `a6d5a3c00` against the candidate (`a43d2736…`, which differs
from the committed tree only by clang-format whitespace):

- D4 batch execute, water-96 and a ragged 3/96/12/48-atom batch: 230.6 → 26.0 ms.
  EEQ charges are bitwise identical and replay-deterministic. Energies and
  gradients already vary between replays of the unchanged build (atomic
  accumulation in fixed-charge D4; spread up to 1e-17 Eh and 2.4e-19 Eh/Bohr);
  the new build stays within that spread.
- r2SCAN-3c water-96 Direct ABBA: warm 4.036/4.028 → 3.836/3.826 s, cold
  9.93/9.91 → 9.68/9.74 s. `correction_energy` is bitwise identical; all arms
  pass the existing independent qualification.

## Rejected alternatives

- Overlapping the unchanged kernels with SCF on the D4 stream: hides latency only
  when another stream has work, keeps one SM busy for 0.23 s and still serializes
  the final apply.
- An adjoint (A^T lambda = dE/dq) gradient: removes the 3n solves but changes the
  floating-point result and needs its own oracle qualification.
- Host EEQ: adds a host dependency to the CUDA production path.

## Revisit when

Systems approach `kEEQMaximumAtoms` with many concurrent items; a blocked LU or a
batched cuSOLVER owner may then beat one CTA per system.
