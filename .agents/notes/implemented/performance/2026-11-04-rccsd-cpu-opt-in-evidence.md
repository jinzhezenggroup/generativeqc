# Decision: retain the measured CPU RCCSD opt-in boundary

Status: implemented
Date: 2026-11-04

## Problem

#762 already has a real compiled TensorIR solver behind `backend="cpu-native"`,
but no endpoint evidence justifies replacing the default NumPy iteration. The
old campaign harness also removed its temporary JIT cache without retaining
binary identities and described solver-process RSS without excluding compiler
children explicitly.

## Decision

Retain the existing default and explicit native selection. Enrich the benchmark
receipt with independent input identities, actual loaded BLAS thread pools when
observable, shared-library hashes before cleanup, source/harness identity and a
finite worker timeout. Keep the independent expanded equation as the final
convergence oracle. No production scientific/default behavior changes.

## Rejected alternatives

Do not promote a slower complete endpoint for compiler-source uniformity, infer
physical process-tree memory from `RUSAGE_SELF`, or describe an artifact miss as
a flushed compiler cache. Do not infer that supplied-MO RCCSD measurements are
an exhaustive CPU scientific-ownership audit or a complete HF+CCSD pipeline.

## Evidence

[Retained evidence and offline replay](../../../../benchmarks/results/cpu-rccsd-native-closeout-20261104/README.md)
cover two pinned fixtures, 1/4 BLAS threads, complete reference/cold/cached
solves and independent residual gates at scientific source
`9cfb77727661cc64e47938d9d0ba65a157427439`. The measured cached native route
is slower in all four enriched fixture/thread observations. These individual
samples support not promoting the default; they are not universal performance
ratios. Scientific acceptance and original tolerances are unchanged.

## Revisit when

A changed compiled iteration/lowering shows reproducible complete cold/warm
endpoint improvements with the same independent numerical, work and resource
controls. The separately scoped current-source ownership inventory remains
outstanding under #934. #762 was closed on 2026-11-04 as superseded tracking,
not as acceptance of that inventory or as a native-default promotion.
