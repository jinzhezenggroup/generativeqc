# Decision: share J source scaling and budget the retained primitive domain

Status: implemented, independently qualified
Date: 2026-10-20

## Problem

After #2630, the 96-atom r2SCAN-3c fixed-density J consumer still takes about
1.08 s against 0.63 s for the pinned GPU4PySCF MD target. The retained
[source audit](2026-10-19-gpu4pyscf-cold-source-audit.md), especially J27,
identifies a real arithmetic difference: upstream folds the common Coulomb
prefactor into Boys seeds, while native multiplied it into every output dot.
The native cumulative omission certificate also used the complete primitive
census even after a much smaller source/target prefix had been certified.

The parent of this experiment is `27b79cc8006ec65ff721a259986a7b9c23db1a34`.
Both arms include the intervening grid-default changes. Explicit version-1,
48x16x32 unpruned moving grids preserve the earlier matched-grid diagnostic;
these results are not a grid-converged production ranking.

## Decision

The compiler-owned radial lowering accepts an optional common prefactor. It
initializes the existing seed scale with that prefactor, and the native
consumer removes multiplication after each forward/reverse dot. Linearity
permits this movement, but FP64 rounding changes; the unscaled generated
entry point and independent auxiliary/80-digit references remain available.
Boys evaluation, radial states, density signs, symmetry and output reduction
are unchanged. No global compiler arithmetic flag changes.

The primitive recipe uses two device-only hierarchy passes. Define epsilon as
min(screening, 1e-12), N as the complete primitive census, m as the actual target
parent multiplicity, and M as the global maximum multiplicity.

1. Prefix target admission uses epsilon/(2*m*N); source admission uses
   epsilon/(2*M*N). All divisions round down, including the initial halving.
   Existing conservative norm bounds and maximum-source/output bounds remain.
2. Freeze all five prefixes. Their lengths sum to A, a conservative superset of
   sources across the entire system window. Rebuild thresholds and node minima
   using denominator min(N, 2*A), or N for an empty prefix. This pass never
   changes the prefixes and adds no host readback or resident allocation.
3. When 2*A < N, all excluded-prefix directed products satisfy a half-budget
   certificate, including products excluded because their target was removed.
   At most m*N such products contribute epsilon/2 per public entry. At most
   m*A omitted retained products contribute the other epsilon/2.
4. When 2*A >= N, use the original uniform epsilon/(m*N) allowance. The stricter
   prefix omissions also meet that allowance, so summing all omissions still
   fits epsilon. This avoids a more restrictive second-stage allowance when
   prefix removal is small.

The checked branch computes 2*A only when A <= N/2, avoiding integer overflow.
Inactive gaps and other systems only increase A and make the certificate more
conservative. Zero screening, exact zero density, unknown/infinite norms,
empty windows and captured density replay retain their conservative behavior.
The public-AO-mask reference recipe keeps its original admission policy.

## Rejected alternative

The first experiment emitted constant-address low-order contractions and kept
radial sources in thread-private registers for the five one-component-lane
classes on sm_120. It shared the existing recurrence and term inventory; its
old lookup tables were byte-identical. All 804 initial host checks and both
storage variants' 28 runtime checks passed. All native classes had zero local
bytes, but r2 J changed only 1.077241 -> 1.075189 s and PBE0
0.332046 -> 0.331062 s. Individual angular times were essentially unchanged.
The extra emitter and architecture branch were removed. The exact patch,
identities, matrices and traces are retained in the qualification receipt.

This differs from the earlier rejected nonzero-source census: the new second
pass spends a separately certified remaining budget over an already pruned
source superset, instead of merely discounting exact zeros in the full census.

## Evidence

The compressed, source-bound receipt is
[`qualification.json.xz`](../../../../benchmarks/results/md-j-prefactor-active-budget-20261020/qualification.json.xz).
Ignored raw artifacts and reproduction scripts are under
`.artifacts/j-registers-20261020/` in the acceptance checkout. Frozen snapshots
are named `/data/jzzeng/qc-j-local-{base,registers,scale,active,final}-20261020`
on n4, with the final GPU regression snapshot on n5.

The fixed-D protocol uses 32 waters/96 atoms, real spherical orbitals,
r2SCAN-3c/def2-mTZVPP (1088 AO) and PBE0/def2-SVP (768 AO), direct FP64 J,
screening 1e-12, and the same deterministic public-AO MINAO density. Preparation
is separate. Five clean warm samples include density upload, evaluation, AO
mapping and host matrix publication; profiles and work counters are separate.
GPU4PySCF uses MD commit `9af748a71ca8f4881c604332c2f0cc1632e033ac`
with released 1.8.1 frontend/helpers, not a completely pinned package.

| Complete J median, seconds | Parent | Seed prefactor only | Plus active budget |
| --- | ---: | ---: | ---: |
| r2SCAN-3c | 1.077241 | 1.006363 | 0.979726 |
| PBE0 | 0.332046 | 0.321160 | 0.311977 |

The final source is remeasured separately: r2 J 0.979261 s and PBE0 J
0.311897 s, reductions of 9.10% and 6.07% versus the parent. The same job's
upstream medians are 0.629452 s and 0.348645 s: native r2 remains 55.6%
slower, while native PBE0 is 10.5% faster. Complete endpoints and per-Fock
profiles are retained in the receipt. Matrix gates remain atol=1e-9, rtol=1e-10
against both parent and upstream. The source-only scaling arm changes native
matrices by at most 3.56e-14 and leaves semantic work unchanged.

| Work | r2 parent | r2 active budget | PBE0 parent | PBE0 active budget |
| --- | ---: | ---: | ---: | ---: |
| Radial roots | 2,003,901,808 | 1,940,227,124 | 811,922,712 | 787,382,540 |
| Directed contractions | 3,766,588,912 | 3,644,181,776 | 1,522,688,792 | 1,475,383,320 |
| Summands | 108,322,370,192 | 104,946,669,664 | 22,695,676,456 | 21,980,413,304 |
| Source tiles | 7,036,821 | 7,244,034 | 2,245,613 | 2,304,239 |
| Screened tiles | 3,165,722 | 3,493,149 | 1,036,964 | 1,131,484 |

The stricter prefix pass visits more tiles; the refined thresholds remove more
of them and reduce actual radial/contraction work. Owner storage is unchanged.
All 15 native classes retain zero local bytes per thread. Native/upstream
screening contracts differ, so matching matrices do not imply equal work.

Host qualification has 1,159 passes, including independent high-precision
arithmetic, diffuse/tight prefactor ranges, forward/reverse/self sources,
compiled runtime bounds, lifecycle/window replay and the combined omission
sum across both screening stages. The final GPU suite has 120 passes; one
unavailable completely pinned package case is deselected. The separate pinned
MD comparator is executed and matrix-gated. Memcheck, initcheck, racecheck and
synccheck report zero errors, with zero race warnings/hazards. Sanitizers cover
the two retained small scientific cases, not the full 1088-AO benchmark.

Complete endpoint gates remain energy <=1e-8 Eh and max force <=1e-7 Eh/bohr.
Cold includes preparation; no endpoint is divided by its iteration count.
The r2 cold observation is 127.048547 -> 124.913105 s, both 24 Fock builds;
changed geometry is 81.007942 -> 79.660804 s, both 13. Warm and changed-warm
have five repeats each. These are endpoint observations; the fixed-D median
and real per-Fock profiles separately establish J speed rather than crediting
solver-iteration changes. PBE0 cold is 61.178976 -> 60.795245 s, both
17 Fock builds; its five-repeat warm median is 10.343636 -> 10.264009 s.
The native r2 per-Fock J median is 1.090216 -> 0.988027 s (9.37% shorter),
with 24 builds in both profiles. Summed J is 26.274362 -> 23.834888 s;
XC medians remain 2.191527 -> 2.191675 s. These profiles are not divided
complete energy/force timings.

## Consequences and revisit conditions

The additional hierarchy pass adds up to five short launches per replay.
Its benefit is qualified on the declared 96-atom workloads, not asserted for
all small systems or architectures. Existing bounded owner admission and
fallbacks remain. The stronger cumulative screening contract is preserved;
no CPU/reference work enters production.

r2 J remains substantially slower than the measured upstream MD consumer.
Do not describe this step as per-iteration parity. Revisit remaining arithmetic,
Boys evaluation, source reuse and scheduling differences with separate ablations
and complete endpoint gates. The unhelpful register-local experiment should
only be revisited if a new layout or instruction profile changes its economics.
