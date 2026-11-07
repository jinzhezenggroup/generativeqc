# Rejected: explicitly fused FP64 VV10 packet accumulation

Status: rejected
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
is introduced. The candidate does not modify compiler FMA settings. The initial
assumption that the VV10 translation unit disabled implicit FMA was incorrect;
the actual compiler command below is decisive.

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

## Rejection and why not to repeat it

The actual Release compile command for `vv10_runtime_cuda.cu` has no
`--fmad=false`. That option applies to other scientific translation units, not
this one. In the measured binary, original and explicitly fused feature kernels
both use 144 registers on SM80 and 136 on SM120, with zero stack/local storage.
Geometry uses 170/170 on SM80 and 172/176 on SM120. Source-level FMA spelling
alone therefore did not remove the hypothesized work.

Complete same-image RTX5090 Slurm16020 confirms no benefit: WB97M-V cold
193.947278 -> 194.417461 seconds and three-warm median
14.350758 -> 14.376013 seconds. The semilocal control is unchanged. Numerical
gates pass, but correctness and unchanged throughput do not justify a new
default or permanent opt-in. Revert this production candidate; preserve the
measured source/ELF and raw A100/5090 endpoints as rejected evidence.

Actual A100 LSF225807 likewise does not qualify it: cold111.798143 ->
113.152722s and warm median15.038729 ->15.015639s. No cold benefit exists on
either measured device. Both complete jobs exit0; the original native paired
gates and32 independent static-reference comparisons pass. These are correct
but non-beneficial results, not a performance promotion.

This failed hypothesis is not a parameter-tuning invitation. The retained
136--176-register packet frames instead motivate a separately bounded
two-row versus four-row scheduling diagnostic. It changes live storage and
occupancy rather than spelling already-contracted arithmetic differently.
It requires its own source, independent gates and complete endpoint evidence
before any default claim.

That bounded scheduling diagnostic subsequently completed and was also rejected:
the roughly 0.11% component differences do not justify another full dual-card
campaign or a production default change. See the
[compact-packet rejection](2026-11-06-vv10-compact-packet.md) before repeating it.
