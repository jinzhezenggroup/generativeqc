# Decision: CUDA planners share the native compact basis counter

Status: implemented
Date: 2026-10-29

## Problem

The existing allocation-free native shape query shares AO-count arithmetic with
normalized systems, but the Python resource adapter deliberately supplied it
only for CPU requests. CUDA planners with an already-selected library still
ran the portable formula for the same expanded contractions.

## Decision

Pass the selected library for both execution backends. Representation and
expanded shell/primitive rows determine these counts; the query has no device
selector and performs no CUDA operation. Retain native-free/older-library
compatibility without loading another library. A selected query's rejection
remains final and cannot fall through to portable counts or provider discovery.

## Boundaries

No C++/CUDA equations, native query, ABI, basis expansion, mathematical identity,
provider admission or scientific tolerances change. This transfers only compact
shape arithmetic. It does not establish complete resource-plan authority or
admit a new method/basis/force context. R3 is excluded.

## Evidence

Independent PySCF dimensions cover Cartesian/spherical STO-3G, def2-SVP and
def2-TZVP under both planner backends. Actual HF/KS planning refuses selected
native failures before device inventory; portable arithmetic and library
discovery are poisoned. The existing signed-limit, transactional rejection,
old-library and prepared-state recovery gates remain. Real CUDA warm-state and
complete endpoint qualification must retain their own source/binary identities.

## References

This extends [the original CPU cutover](2026-10-16-native-cpu-basis-shape.md)
under #2227 R2/R6; it does not replace that historical receipt.
