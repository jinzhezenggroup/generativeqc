# Decision: bounded packed occupied projection selected by FP64 capability

Status: implemented
Date: 2026-11-01

## Problem

The generated packed-B occupied projection is useful on devices with limited
FP64 throughput, but leaves substantial contraction time on full-throughput
FP64 devices. A universal unpack-and-BLAS replacement is not portable: on the
5090, even 128-row blocks regress the measured projection by about 20–50%.
A GPU-name allowlist would also miss the underlying resource tradeoff.

## Decision

Borrow the already charged `exchange_intermediate` scratch while it is dead
before the K Gram. Unpack at most 128 AO rows and contract each panel through
the existing batched matrix-product interface. Keep U disjoint and retain its
final-force consumer lease. No new tensor allocation or whole-B expansion is
introduced. Explicit BLAS with less than one full-auxiliary row of scratch
retains the generated kernel.

Automatic admission uses CUDA's reported FP32/FP64 throughput ratio in (0, 2],
nbf >= 256, naux >= 1024, occupied rank >= 64 and <= nbf, and charged scratch
for at least eight full-auxiliary AO rows. These are conservative work/resource
bounds, not a universal latency model. Unknown/unsupported capabilities and
smaller domains retain generated execution. Other CUDA query errors propagate.
Capability and requested policy are frozen in the owner; changing the requested
policy invalidates captured native and Python plans. Explicit `generated` and
`blas` controls remain available on all devices.

## Rejected alternatives

- Unconditional BLAS: rejected by the 5090 microbenchmark, including tails and
  both coefficient layouts. FP64 arithmetic capability alone does not remove
  the extra unpack traffic and library-launch overhead.
- Whole dense expansion: unnecessary allocation and data movement; it could
  displace resident factors or the final-response representation.
- GPU-model selection: the relevant distinction is a physical capability and
  contraction size, with bounded fallback, rather than a product name.
- Treating isolated kernel gains as endpoint gains: retained comparisons include
  constructor, preparation, SCF, complete forces and host export, plus warm calls.

## Invariants and evidence

All arithmetic remains FP64. Precision, grids, screening, initial guess and
physical convergence are unchanged. Projection FLOPs and K work keep their
original census; added unpack panels, values, write bytes and scratch are counted.
Host controls exercise tails, row/column coefficients, insufficient scratch,
BLAS failures, scratch/output canaries and final-U lifetime. Nine independently
reconverged energy/force cases per device exercise cold/warm/moved states,
response spaces, packed replacement and policy changes. Blackwell memcheck
passes the same nine cases with zero errors.

A100 LSF224180 automatic versus generated complete endpoint medians improve
cold PBE0/B3LYP/M06-2X by 11.95%/11.80%/11.11%; warm by 5.32%/5.27%/4.67%.
A100 WB97M-V (LSF224181) improves cold by 3.35% and warm by 0.67%.
Physical iterations and Fock work are identical. 5090 Slurm15042 and PRO6000
Slurm3841 retain the generated path: B3LYP cold changes +0.154%/+0.077%, warm
-0.071%/+0.036%. These small variations do not establish speedup or degradation.
The final report retains conditioning, full ABBA populations, original identities,
complete force vectors, work counts, and earlier explicit-only/loader controls.

## Revisit when

Extend the automatic work domain only with source-matched independent-reference
complete endpoints and bounded resource/lifetime controls. Additional devices
may justify a bandwidth/launch-aware model; current measurements do not prove
an optimum on every FP64-capable GPU. GPU4PySCF parity and integration with
OpenBLAS or opt-in lazy range anchors require separate measurements.

## References

- `benchmarks/results/bounded-packed-projection-20261101/`
- `docs/developer/df_tuning.md`
- Issue #2426
