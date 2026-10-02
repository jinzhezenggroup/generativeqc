# Decision: derive RHF Coulomb response from the rooted final projection

Status: implemented
Date: 2026-10-02

## Problem

The qualified single-B final-K response already borrows the exact B*C factor.
It formerly unpacked the whole retained fitted tensor again to compute the
Coulomb charge, then applied the metric root independently of exchange.

## Decision

For the validated determinant D = w C C^T, finish S_Q = C^T B_Q C and compute
U_P = sum_Q (M^-1/2)_PQ S_Q once. The Coulomb potential is w tr(U_P), and the
same U supplies the exchange metric and pseudo-density contractions. Generated
CUDA owns the scalar diagonal reduction; native code orchestrates existing
storage and BLAS operations.

Keep the Q-fast pair-major gather: its source index is (i*r+j)*naux+Q, whereas
the downstream occupied matrix is column-major i+j*r within Q. Removing it by
assuming exact AO symmetry breaks deliberately nonsymmetric diagnostics.

## Invariants and fallback

- Reuse requires the existing exact density, final-state token, generation,
  occupation and one-shot scratch lease checks; matching dimensions alone do
  not authorize it
- Reuse remains one-term restricted, full-rank, retained-root and single-B;
  UHF, batch, truncated rank, stale leases, spectral-root and final-projection
  off controls retain their exact existing routes
- The finish output and rooted result share the existing staging interval at
  distinct times; the gather input/output and metric-root input/output remain
  disjoint. No new persistent allocation or widened response budget is added
- Rank-squared staging stays live through Coulomb and exchange and is only
  consumed as derivative panels after the exchange metric contraction

## Evidence

Host tests execute emitted BLAS calls with an independent scalar BLAS stand-in.
The composed finish/root path was compared with a raw-integral, direct-linear-
solve energy finite difference, including an occupied-space gauge rotation.
Separate tests cover nonsymmetric projection orientation, metric conditioning,
rank/shape boundaries and panel tails. The real-device molecular oracle test
now asserts the removed charge/projection passes on the admitted route and
retains explicit off/spectral/dense fallbacks.

At 768 AO and 3,712 auxiliary functions, the semantic work counters expose the
avoided 1,096,138,752 packed fitted-source elements and 2,189,426,688 unpack
outputs, plus the charge-only pair of auxiliary matrix-vector GEMMs. These are
logical work counts, not measured DRAM traffic. NVIDIA endpoint force accuracy,
memory and alternating warm/changed-geometry timing remain unmeasured for this
change; host tests and hosted compilation are not device receipts.

## Revisit when

The projection owner supports multiple terms/spins or a persistent immutable
projection. Any broader admission needs its own density/lifetime and resource
proof.

## References

PR #1694; issue #1690; prior final-projection reuse in PR #1433.

Agent: dot
