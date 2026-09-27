# Decision: generate Direct primitive-pair cache geometry

Status: implemented
Date: 2026-09-27

## Problem
After recurrence/contraction retirement, the shared Direct scientific-support family
contained only the primitive-pair cache kernel and two force-result structs. The
cache still computed exponent/product-center/decay geometry in maintained CUDA,
while the structs were data layout rather than scientific equations.

## Decision
Emit the primitive-pair cache kernel from the scientific compiler and keep
`direct_pair_cache.cu` as a launch-only runtime boundary. Rename the gradient
layout header to `direct_gradient_types.cuh` and classify it as runtime data
layout without claiming formula retirement for those structs.

## Invariants
- Pair-cache exponent, product-center, coefficient and decay arithmetic is unchanged.
- Launch geometry, stream and allocation ownership are unchanged.
- Gradient-result memory layout is byte-for-byte unchanged.
- Screening, selectors, queue policy and endpoint semantics are unchanged.

## Evidence
Structural tests require pair-cache formulas to appear only in the generated header,
the native translation unit to be launch-only, and gradient types to carry runtime
ownership. The #356 shared-native-recurrence family becomes empty and is removed.

## References
- #356
- #1469
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol
