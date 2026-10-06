# Tensor CUDA reduction provider moved below TensorSchedule

Date: 2026-10-06

Agent: ChatGPT
Model: GPT-5.6 Sol

## Problem

Tensor CUDA previously carried `reduction_provider` in `TensorSchedule`. That
made a vendor/generated implementation identity part of generic schedule search,
plan identity, and method-visible tuning even though the shared lowering layer
already represents generated CUDA and CUB as competing providers for one semantic
reduction request.

This inverted the intended ownership:

```text
TensorSchedule -> provider already chosen -> LoweringCandidate diagnostics
```

instead of:

```text
TensorSchedule / semantic plan -> lowering candidates -> prepared implementation
```

## Change

The schedule now carries only cooperative reduction shape/policy
(`stream_reductions`, threads, unroll and related topology). A typed
`ReductionLoweringBinding` owns the generated/CUB implementation choice below
the schedule boundary.

- production defaults to `ReductionLoweringBinding("generated")`;
- CUB remains qualification-only and is pinned explicitly at emission/compilation
  or tuning;
- `TensorPlan.identity` and `TensorSchedule` no longer encode provider name;
- compiled artifact identity includes the lowering binding, so generated and CUB
  binaries cannot alias in cache;
- static resource/source estimates and lowering diagnostics consume the same
  explicit binding;
- tuning takes a bounded product of schedule × precision × reduction-lowering
  candidates without reintroducing provider identity into the schedule;
- the default-promotion inventory now audits the CUB choice as
  `tensor-lowering:reduction-provider-cub`, not a schedule control;
- the provider-selection debt manifest removes the retired Tensor CUDA entries.

## Numerical/runtime behavior

The default remains the existing generated cooperative reduction. No scientific
equation, precision contract, tolerance, default schedule, or promotion decision
changes. Existing CUB qualification remains available through the lowerer binding
and retains its previous negative endpoint evidence.

## Follow-up

This slice removes the provider name from TensorSchedule but does not claim that
all Tensor CUDA implementation choice is already produced by the generic
`LoweringBinding` selector. The next #1889 step can replace the specialized
`ReductionLoweringBinding` qualification adapter with a prepared binding derived
from the shared `LoweringCandidate` registry once runtime preparation owns that
binding end-to-end.
