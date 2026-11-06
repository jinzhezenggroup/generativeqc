# Proposal: explicitly fused FP64 VV10 packet accumulation

Status: proposed
Date: 2026-11-06

## Problem and diagnostic evidence

WB97M-V96 full-Direct still has a large cold cost after the LR force
improvement in #3017. One finite RTX5090 diagnostic, Slurm16019, used the
already-qualified source `ff6117e3ba5b481e6124f3bf80f4613fa27586d77bff88f5187f9290abbf5afa`
and ELF `4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a`.
It completed cold plus one frozen-seed warm call, with 18/1 Focks and
four/zero VV10 pair sweeps. The four feature sweeps took 84.536 seconds
inclusively, approximately 21.13 seconds each. The first stationary-force
call also contains an uncached geometry pair sweep.

This is intrusive diagnostic evidence, not clean endpoint timing. In
particular, DF_TRACE requests a quartet census and thereby bypasses optional
MD-J. Its generated-J time must not be presented as the default J bottleneck.
Nested/inclusive regions cannot be added. Clean endpoint qualification must
keep both tracing controls unset and retain the actual default MD-J route.

## Candidate

Reuse the existing canonical bilateral pair DAG and its bounded admission.
Explicit FP64 FMA contracts distance-square accumulation and the two legs'
weighted feature/geometry sums. The scalar emitter's existing
`fused_accumulation` lowering also contracts eligible single-use products.
No FP32, fast-math, new cutoff, pair omission, grid change or solver change
is introduced. The runtime's global implicit FMA setting remains disabled.

Keep the incumbent `parallel-packet` template and selector as an exact
same-image scheduling control. `parallel-packet-fma` selects the candidate;
unset/empty selection is a candidate default pending qualification. Existing
mixed arithmetic retains its previous packet. General/ordered extreme-input
fallback, signed/positive-zero rows, output disjointness, finite publication,
O(N) storage and all cache invalidation rules remain unchanged. Changing the
selector invalidates density-dependent pair reuse as before.

## Acceptance and fallback

Require independent Decimal scalar derivatives, actual-device independent
discrete-functional arrays and finite differences, signed/zero/boundary rows,
partial warps, output canaries, cache invalidation and production automatic
admission. Then one bounded same-image incumbent/candidate complete cold plus
three-warm E+analytic-force cohort on actual A100 and RTX5090. Preserve
observable gates, SCF/VV10/J/K work and scientific settings; require no5090
endpoint regression. Include a semilocal control, which has no VV10 consumer.

If complete endpoints do not benefit, reject the candidate rather than
shipping a permanent opt-in or exporting microbenchmark claims. No fresh
GPU4PySCF ranking or original1.576-second-gap closure is inferred.

## References

Issue #2976; parent #3017; new branch `codex/direct-cold-next-20261106`.
Ignored full diagnostic and subsequent qualification artifacts are under
`.artifacts/direct-followup-20261106/`.
