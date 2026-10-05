# Decision: evaluate phased Becke logarithms in the pair domain

Status: proposed; endpoint qualification pending
Date: 2026-10-05

## Problem

The 96-atom public-force profile attributes about 1.52 s of summed GPU time to
the phased atom log-product traversal. Each point/atom thread serializes the
logarithms of all its neighboring factors. Fewer exact-zero operations alone
did not improve this schedule; see the separate rejected-probes note.

## Decision and invariants

Use the canonical `point_pair` with retained logarithms in the existing parallel
pair-primal phase. Store the two log values in the two words previously occupied
by ratio partials. The atom phase sums these values in the same triangular order
and preserves every zero count. The pair reverse phase rematerializes ratio
partials through `geometry.coordinate`, using the existing emitted ratio graph,
only for a nonsaturated pair. No new scientific formula or approximate cutoff is
introduced. `pair_adjoint<false>` retains the canonical derivative and its
single-zero/rounded-zero behavior.

The same four pair words are overwritten with four pullbacks after normalization.
Same-stream ordering protects that lifetime. Native allocation, admission and
bounded fallback are unchanged. Pair production remains once per point/pair;
atom log and gather traversals still visit each incident pair in order. The atom
pass reads an extra log word; reverse rematerializes the cheap ratio from the two
retained distances and geometry metadata. This trades traffic and simple work
for parallel transcendental evaluation, not lower asymptotic work.

## Initial evidence

The prototype passed 96 host tests (including the independent Decimal gate and
admission boundaries) and 21 CUDA tests, with 28 skipped iteration specializations.
Four processes ran in baseline/candidate/candidate/baseline order in one finite
n1 RTX 5090 allocation. Each internally compared cooperative/phased routes
using ABBAABBA, over 64 sampled 256-point tiles and synthetic signed seeds.

| Atoms | Baseline phased medians, ms | Candidate phased medians, ms |
| --- | ---: | ---: |
| 48 | 11.043664, 11.043936 | 10.084928, 10.098704 |
| 96 | 28.178048, 28.235072 | 24.708257, 24.730736 |

Maximum absolute gradient differences were zero. Scratch stays 10,422,272 /
39,716,864 bytes for these isolated tiles. Work remains 18,481,152 / 74,711,040
pair productions for the sampled grids. These are isolated results, not endpoint
speedups. The final source additionally removes the unused atom-phase log
callback; it must be qualified separately before promotion.

Emitted prototype source, binaries, JSON samples, compiler/cache receipts and
commands are retained under
`/data/jzzeng/qc-becke-pair-logs-1894/.artifacts/1894-pair-logs/` on local and n1.

## Remaining gates

Complete paired 48/96-atom E+F cold/warm/moved/moved-warm runs, independent
reference gates, final-source CUDA/owner and sanitizer checks, and resource/work
verification are still pending. A full tuned current-source native build is in
progress; the generic-library rejection in the earlier square-product experiment
does not count as an endpoint measurement.

## Final-source composed qualification (Slurm5928)

The final phase sources in PR #1950 (`ef91a6617`) are byte-identical to the
composed candidate `e92cf75b0`, measured against order-five control `8f90f0ac3`.
The full stack retains #1830/#1833/#1847/indexed policy. All 288 independent
E/F pairings pass (max E 1.06411e-10, max F 2.72972e-11). Final-source device
coverage comprises 21 emitted CUDA tests, 24 shared-owner cases and all four
sanitisers with zero errors/hazards; 28 iteration-specialization skips remain
explicit. Host Becke/resource gates pass 79 + 17 cases.

96-atom complete warm improves 25.672614→25.182758 s (1.91%); moved-warm
25.750174→25.199025 s (2.14%). Grid response improves about 0.49 s.
48-atom grid response improves only 0.067–0.068 s; the substantially larger
48-atom endpoint median gaps include noise outside this treatment. Do not claim
those full gaps or the isolated 12.3% as its stable endpoint benefit.
Cold/moved trajectories and slight 48-atom regressions remain in the evidence.
No iteration normalization, standalone-master timing or automatic profile
promotion is inferred. The ten-second/#1895 target remains open.

`benchmarks/results/pbe0-order5-pair-logs-20261005/` retains full raw samples,
reconstructable source, binary and scheduler receipts, work counts, actual
histories and an inventory-aware verifier. This completes the scoped
composition/device gates listed above; separate default-selection decisions
still belong to #1834.
