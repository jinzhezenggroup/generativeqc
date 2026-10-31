# Decision: retain the H100 DF gap after final-projection reuse

Status: proposed
Date: 2026-10-30

## Problem

Issue #2892 needs an isolated remaining optimization after #2888. A successful
projection handoff or reduced nominal GEMM work does not establish that the
complete endpoint beats GPU4PySCF.

## Evidence and decision

The source pair `15ab66a6`/`2241fa6a` passes all 240 full E+F calls in the
three-process PBE0 matrix. Candidate DF 96 Standard has baseline/candidate
median speedup 1.198610, but GPU4/candidate is only 0.448375. Direct 48 wins;
the other three physical cases do not. Keep these unfavorable observations.

Separate intrusive 32-call diagnostics establish actual completed final-U
arithmetic reuse on every observed candidate default DF cold/warm call.
Admission alone is insufficient: use the same valid stream response row's
completed arithmetic counter and atom-coordinate completion. The off control
and baseline report zero. Native prepared force capability is independently
observed true.

The current native provider still submits separate K and J derivatives. Its
occupied response can form Coulomb and exchange weights, but native stationary
transport retains distinct semantic source slots and range correction depends
on the separate exchange source. Do not relabel a combined gradient as a pure
Coulomb channel or relax the one-shot final-state lease to force fusion.
Joint contraction remains a hypothesis requiring interface, numerical,
lifetime/resource and complete-endpoint qualification.

## Invariants and revisit conditions

- Preserve independent full vectors, exact geometry/fit/operator/precision,
  immutable source identities and explicit bounded fallbacks.
- Existing normalized Becke-adjoint experiments already exist; inspect their
  qualification boundary before proposing a duplicate primitive.
- CUDA-event component intervals include submission idle time. Host/GPU views
  overlap; do not assign unaccounted time to CPU packing or claim occupancy.
- Panel counts are not proof of repeated raw forward integral generation.
- Do not generalize this PBE0 slice into the fixed 30-case majority result or
  qualify a later/integrated source without its own gates.

Full vectors, clean paired samples, exact Python producers, CUDA/host streams
and offline replay: `benchmarks/results/h100-source-pair-20261030/`.
