# Rejected promotion: retain amplitude layouts across DF CCSD Q tiles

Status: rejected
Date: 2026-10-10

## Mechanism

The matrix-lowered primal auxiliary graph repeats two pure amplitude transposes
per one-Q action and four per nonunit Q tile. A prototype used the generic
TensorIR purity/dependency proof to cut their union into the existing per-trial
preparation graph. Full content identities, not equal shapes or addresses,
deduplicated cuts. Preparation rebuilt every cut after each amplitude change;
there was no cross-iteration, cross-reference or changed-geometry cache.

All retained outputs entered the existing preparation arena query. Original
scalar/resource fallbacks and the independent expanded convergence replay were
preserved. No equations, contraction trees, Q reduction order, precision or
production oracle dependency changed. This is a plausible work reduction, but
it is not sufficient evidence for endpoint promotion.

## Correctness pitfall

TensorIR canonicalizes output keys alphabetically. The generated C++ preparation
aggregate declares `tau` before the layout fields. Returning `tuple(program.outputs)`
therefore assigned valid arrays to the wrong pointers: host graph tests passed,
but 15 native GPU oracle/trajectory gates failed. Explicitly emit `df_tau` first,
followed by the aggregate's declared field order. A generator-call regression and
all 34 native GPU gates pass after this fix. Never infer native aggregate ABI
order from canonical mathematical output-map order.

The master arena-planner refactor #2169 arrived during qualification. All three
corrected candidate generated artifacts remain byte-identical under that
refactor; only targeted compatibility checks were rerun. Do not repeat full
unrelated qualification just because master advances.

## Endpoint evidence and decision

Source-matched n2 Slurm 2795, one PRO 6000, ethane230 energy-only, FP64, Q tile 8,
DIIS 8, 64-GiB budget and fresh molecular state in each process. Retain all four
alternating/reversed observations in
`benchmarks/results/df-cc-amplitude-layouts-20261010/`.

Logical packing bytes fall 10.348% (5.578 to 5.001 TB); this is not physical DRAM
traffic. CCSD numeric capacity rises 31,648,768 bytes. CCSD medians change from
99.847 to 99.368 s, while the complete process wall changes from 195.954 to
195.630 s, only 0.166%. Native baseline spread is 0.566 s, greater than the
0.329-s native median difference. These observations do not establish a robust
complete-endpoint gain. Do not retain the production policy or open a weak-win
performance PR merely because copies or component time decrease.

Every completed observation passes independent total / separate-(T) energy
gates and original expanded physical replay. Work/trajectory counts match.
Raw failures, source archives, verified compiler-cache receipts and binaries
remain ignored local/Slurm-host artifacts; no Release or external backup was
created. Separate follow-up observations expose existing completed iteration,
replay and update timers before choosing the next optimization.

## Revisit when

Revisit if a representative shape has a materially larger complete-endpoint
benefit, or if eliminating layouts rather than retaining their union reduces
both work and complete resource use. Preserve per-amplitude invalidation,
exact admission, bounded fallback, the independent expanded audit and explicit
C++ aggregate ordering. Stronger endpoint evidence, not kernel-only timing,
must justify promotion.

## Follow-up

The separate original expanded-audit packing experiment is recorded in
`../implemented/performance/2026-10-10-df-cc-expanded-matrix-replay.md`. Its
conditional endpoint gain does not rehabilitate this rejected primary-layout
retention policy.
