# Decision: fuse meta-GGA density products across AO jets

Status: experimental
Date: 2026-10-01

## Problem

The native CUDA semilocal path represents a meta-GGA density contraction with four work jets:
`D*phi` plus `D*dphi/dx`, `D*dphi/dy`, and `D*dphi/dz`.  The existing scalar and tiled
kernels assign the jet dimension to independent threads/blocks.  Consequently each meta-GGA
jet repeats the same symmetric density-row/tile loads even though only the AO panel changes.
r2SCAN and other rho/sigma/tau functionals pay this cost while LDA/GGA use one work jet.

## Decision

When the immutable native layout requests exactly four work jets, the scalar fallback can
compute all four dot products under one density-row owner. The attempted tiled fusion remains
in source for measured crossover work, but production retains the prior tiled schedule after
RTX 5090 qualification found a 2.92x kernel regression in the 24-AO regime.

The one-work-jet LDA/GGA path and the generic non-four-jet fallback remain unchanged.  No new
workspace, host transfer, functional-name dispatch, or scientific expression is introduced.

## Rejected alternatives

- Do not duplicate an r2SCAN-specific kernel.  The optimization belongs to the shared
  `work_jets == 4` meta-GGA execution contract and therefore also applies to other qualified
  rho/sigma/tau consumers.
- Do not fuse density features, XC point evaluation, or potential assembly into the same kernel
  in this slice.  Those consumers have different register/occupancy pressure and already have
  independent schedule ownership.
- Do not infer an endpoint speedup from reduced density loads alone.  The tiled fused launch has
  one z-plane per spin instead of one per spin/jet, reducing block-level parallelism by four and
  therefore requires matched GPU qualification before this Draft candidate is promoted.

## Invariants

- Each jet retains the same ascending `nu` / tile-`k` reduction order as the prior kernel.
- Strict FP64 keeps the same symmetric-density expression and FP64 accumulation.
- Mixed density compute retains FP32 density/AO products with FP64 accumulation.
- Work-panel layout, feature semantics, tau's one-half convention, and downstream potential
  algebra are unchanged.
- LDA/GGA one-jet execution remains on the prior scalar/tiled kernels.
- Failure publication continues through the existing device error slot.

## Evidence

Structural regression coverage retains both fused and generic kernels while requiring the
production tiled launch to keep the qualified spin-times-jet z-domain. The scalar fused
implementation remains a bounded candidate outside the tiled schedule.

For one meta-GGA density-product output tuple, the conceptual symmetric density load is reused
across four AO jets instead of repeated four times.  AO loads, four dot products, work writes,
and downstream feature work are unchanged.  This is a data-reuse/work-scheduling claim, not a
DRAM-bandwidth or complete-endpoint speedup claim.

Promotion requires matched RTX 5090 measurements covering at least r2SCAN RKS/UKS and a
representative second meta-GGA, with small and medium AO regimes.  Report density-product device
time and complete warm endpoint time; reject or shape-gate the fusion if the fourfold reduction
in block count causes material regression.

## Consequences

The meta-GGA density-product kernel uses four accumulators and fewer blocks.  It should reduce
repeated density traffic, but may trade that benefit for register pressure or reduced occupancy
on small shapes.  Keeping the old generic path makes rollback and comparison explicit.

## Revisit when

Revisit the scheduling choice when matched sm_120 evidence identifies a crossover by AO count,
point-tile count, spin count, or compiled occupancy.  At that point #168 may select a measured
shape-specific schedule rather than relying on an unmeasured static heuristic.

## References

- #1423
- #1479
- #1480
- #168
- `python/generativeqc_compiler/dft/ao_cuda.py`
- `python/generativeqc_compiler/dft/xc_contraction_cuda.py`

Agent: ChatGPT
Model: GPT-5.6 Sol


## 2026-10-02 qualification follow-up

Matched RTX 5090 evidence for PR head `26fd7f3` measured the fused tiled density
kernel at about 2.92x the baseline in the 24-AO r2SCAN RKS case, with the launch
domain shrinking from 128 to 32 blocks. Complete warm endpoints showed no
meaningful speedup. Production therefore retains the previous tiled schedule;
no unmeasured AO-count threshold is introduced. The fused tiled implementation
is retained only as an experimental candidate for a future measured crossover.

Agent: ChatGPT
Model: GPT-5.6 Sol
