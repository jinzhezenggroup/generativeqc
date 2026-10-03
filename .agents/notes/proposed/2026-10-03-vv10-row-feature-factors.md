# Proposal: contract VV10 feature factors once per row

Status: experimental; complete endpoint comparison pending
Date: 2026-10-03

## Problem and decision

Every molecular VV10 pair previously multiplied its parameter partials by the
same row-local omega/kappa derivatives. The candidate instead sums weighted
partial derivatives with respect to omega and kappa in ascending partner order,
then applies the local-scale chain once. Compiler-owned TensorIR emits the
final rho/sigma contraction through `build_nonlocal_row_feature_program`;
the native owner retains admission, traversal, reduction and publication.

This changes feature summation rounding and has its own lowering identity,
`vv10-bounded-row-chain-fp64-v1`. Pair energy summation and geometric accumulation
remain ordered. The experimental bundle also includes the separately qualified
unit-reciprocal energy policy and #1736 full/LR force proposal. All use FP64.
Partner staging is absent; its complete-endpoint rejection remains applicable.

## Bounded fallback

Factoring is restricted to molecular VV10 feature consumers, at most 2^32
active partners, coefficient magnitude at most 2^32, row density and local-scale
derivative magnitudes at most 2^128, and partner-weight magnitude at most 2^64.
Every visited pair must satisfy the existing rational domain: r2 in [0, 2^32],
positive omega/kappa in [2^-32, 2^32]. The resulting denominator is bounded
away from zero; conservative bounds keep the new contractions below 2^740.

An unsupported pair rejects the entire factored row. All accumulators reset and
the original chain is evaluated from partner zero. No partial factored output
is published, and each row makes at most two traversals. Screening, negative
weights and active positive-zero rows retain their existing meanings. CPU,
rVV10 and unmasked primitive consumers use their existing chain.

## Evidence and limits

112 host tests pass, including independent 90-digit energy/partial checks,
signed-zero screening, density and gradient energy finite differences, and
exact ordered fallback for oversized partner weights/out-of-domain pairs.
Observed fallback pair visits exceed the ordinary pass and never exceed twice
its work. No molecular executed-pair count is inferred from grid capacity.

The candidate library hash is
`33b3963fd3cafa337280b837b409ea5729b8e5baade9c5b5476e987b39384a81`,
parent `a4c705fd5` plus archived source patch. Node1 job 5455 passes six independent
complete molecular tests plus rebuild/stale-state isolation (7 total, 202.22 s)
before a full24 comparison with `vvunit`. This comparison removes staging as
well as adding row factoring; its result must not be described as a one-change
experiment. The earlier controlled staging comparison showed no useful gain.
Default48 is separately scheduled on node5. Every complete sample requires
1e-8 Eh / 1e-7 Eh/Bohr accuracy before promotion; performance is unproven.

Source archive, patch, compiler commands, ccache statistics and results live in
ignored `.artifacts/wb97m-row-factors/`. There is no released or merged row-factor
policy. Promote only after clean complete endpoints, changed geometry and a
larger case qualify the actual staging-free source.

## Same-allocation full24 result

Node1 job 5455 completed both variants. `vvunit` warm samples were 46.058390,
45.933724 and 45.941576 s (median 45.941576 s); `vvrow` samples were
46.628314, 46.989511 and 46.579320 s (median 46.628314 s). Cold was
317.135352 -> 323.439316 s. This is a roughly 1.5% warm regression for the
staging-free factored bundle, so row factoring alone is not promoted.
All five candidate/reference pairs pass, maximum energy error 2.683010e-11 Eh
and force error 4.160841e-10 Eh/Bohr. Larger qualification remains in flight.

A subsequent separate experiment performs O(N) admission and compiles away the
pairwise bounds/retry branches on uniformly admitted molecular grids. Any gain
there belongs to that combined policy, not to this unsuccessful standalone
measurement. Retaining factoring for that experiment is not acceptance of a
slower production default.
