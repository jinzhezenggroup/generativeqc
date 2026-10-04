# Proposal: contract order-four force sources through shared weighted roots

Status: proposed; local experiment, not qualified for shipping
Date: 2026-10-05

## Problem

The bounded full-range J/K derivative owner evaluates higher-order primitives
per AO quartet. The diagnostic order-four replay is expensive, while the
shared low-order external-weight consumer already reuses shell-pair geometry
and radial moments across AO components and the two source channels.

## Experiment

Generate force-only weighted roots for pppp, dspp, dsds, dpps and ddss using
the existing Weighted IntegralIR, algebra and pressure-aware ordering. Bind
them to the existing canonical task/weight/primitive/scatter adapter. Drain
these whole shell tasks once per scalar lane before the generic warp fallback;
the fallback skips exactly those classes for full-range separate J/K sources.

The scratch bound is two fixed component-weight arrays (at most 81 doubles per
source), scalar geometry and two nine-component gradients per task. No new
resident or quartet-capacity allocation is introduced. Compiler register/local
memory cost may still overwhelm the reuse benefit and must be measured.

## Invariants and scope

- Same shell and AO screening, orbit weights, source coefficients and FP64.
- Independent J/K outputs, including disabled source guards.
- Canonical shell orientation and repeated-atom translation recovery unchanged.
- HF combined force, range sources, f-containing order-four and higher classes
  retain their current consumers.
- No new recurrence implementation or functional-specific dispatch.

Independent scientific tests and full cold/warm/moved E+F comparisons are
required before a shipping PR. The experiment deliberately remains separate
from the root-reuse and Wick-pruning candidates and from #1833 promotion.
