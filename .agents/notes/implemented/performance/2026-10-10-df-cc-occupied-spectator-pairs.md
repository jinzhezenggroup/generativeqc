# Decision: fold occupied spectator pairs in the primal DF virtual ladder

Status: implemented
Date: 2026-10-10

Compiler/native-action and production owner/admission are qualified in the
worktree. Four matched complete energy-only endpoints pass numerical gates;
this is not shared formal/global performance or force promotion.

## Problem

The two leading virtual-ladder contractions contain `o^2 v^3` scalar summands
each and dominate the complete six-cut auxiliary action. Packing amplitude
layouts without reducing this work was previously rejected as endpoint noise;
see `../../rejected/2026-10-10-df-cc-amplitude-layout-epoch.md`.

The occupied `i,j` labels are spectators of the ladder. Retaining `i <= j`
would keep 45 instead of 81 dense virtual matrices for `o=9`, without adopting
the fully packed `(i,a)/(j,b)` coordinates used by DIIS. Diagonal occupied pairs
still require full virtual matrices. The missing `j,i` block is reconstructed
by transposing its virtual axes.

## Prototype decision

`python/generativeqc_compiler/cc/df_spectator_pairs.py` derives the fold from the
existing auxiliary DAG, replacing only `df_D05_vv_ladder`. Every other cut is
the original node, not a separately rewritten equation. The transform checks:

- Both free occupied labels travel through exactly one operand at every
  contraction. Neither may be reduced or shared with a factor operand.
- Their source is the existing occupied/occupied/virtual/virtual `df_tau`.
- The complete ladder, not just each spectator path, preserves simultaneous
  pair reflection assuming tau symmetry. This is an exact signed-rational
  polynomial proof using the shared CC `_expand` utility, dummy renaming,
  commutative scalar factors, and tau's simultaneous-pair involution.
- Polynomial expansion is bounded to 64 terms, six factors per term, and four
  dummy labels. Unsupported future inventories fail closed.

The last check matters: a one-sided virtual contraction has free spectators but
does **not** have a reflectable output. Negative dressings must remain signed;
ordinary `Counter` subtraction discards negative coefficients and is unsuitable
for the exact proof. Both mistakes have explicit negative tests.

`tools/generate_df_ccsd_spectator_pairs.py` emits separate raw, packed, and
Q-batched actions. CMake registers these optional actions; the production owner
enables them only after the admission and fallback gates below.
The existing matrix provider and explicit six-field output ABI are reused.
The fused consumer reflects only the ladder while preserving each ascending-Q
addition to the full retained cuts; it does not introduce a Q subtotal.

The common runtime-shape emitter recognizes only a full, named
`IndexSpace("occupied_pairs", "pair", ...)`, not arbitrary composite-pair
spaces. Host queries check triangular arithmetic with division before
multiplication. Device expressions use the same factorization after host
admission. Existing nonpair consumers do not emit new dimension declarations.
The shared ordered accumulator accepts a separately typed source only with an
explicit coordinate map when its dimensions differ from the target.

## Projection majorant

`build_ladder_pair_majorant` derives a positive scalar coefficient program
from the **original** ladder DAG. For full-tau projection error `delta`, the
real-arithmetic ladder error is bounded by

```text
delta * (coefficient_0 + coefficient_1 * current_t1_norm)
```

Coefficients are summed over every Q. Geometry factor norms are maximum
absolute row sums over the compiler-derived contracted axes; the current T1
norm is rebuilt from the current amplitudes, never an older DIIS state.
Current geometry bindings are Bvv/Bov summed over virtual axis 1, and T1
summed over occupied axis 0. Do not duplicate a hand-written ladder equation in
the native owner; consume the generated coefficient program and bindings.

The representative `(2,3)` schedule has input factors along tau's path, but
`(3,2)` reassociation also produces the compound factor `Bov^T T1`. Its row
norm must be derived recursively. All shared summed labels are assigned to
one operand; exclusive summed labels are assigned to their owning operand.
Splitting shared sums between two factors can underestimate the bound: for
identity matrices, `sum_xy A_xy B_xy = N` while the two split maximum row/column
norms are both one. This alternative is explicitly rejected.

This is a **real-arithmetic projection bound**, not a floating-point error
certificate for a complete solve. The current prototype lowers the same
positive scalar DAG into integer-exponent upper powers; independent physical
replay remains responsible for convergence and contraction/reconstruction
roundoff gates.

## Required production admission

No unconditional symmetry assumption is authorized. Original tau has no
declared simultaneous-pair symmetry, and physical solver trajectories can
contain rounding asymmetry. Before enabling this path, implement and qualify:

1. Require bitwise simultaneous-pair symmetry of supplied initial T2, including
   signed zeros, or disable the optimization for the complete solve. Do not
   project arbitrary supplied amplitudes to make them eligible.
2. Preserve full original tau, rebuild packed tau at every primary amplitude
   state, and use the existing overflow-safe pair mean. Bypass equal values
   and singleton orbits, preserving exact symmetric/subnormal values. For
   diagonal occupied blocks, canonicalize the virtual mean operands or share
   their computed representative: reversed floating-point mean evaluations
   must not be assumed bitwise identical, particularly near underflow.
3. Measure the actual projection error against every full tau coordinate.
   Nonfinite original tau must preserve sticky physical arithmetic failure;
   a nonfinite or unsupported *bound* must only reject the optional path.
4. Certify floating-point coefficient, geometry-sum, T1-norm, subtraction and
   final-bound rounding/ranges. A fixed epsilon threshold or an unproved
   multiplicative guard is not an admission proof. Exact-zero cases must not
   be confused with underflowed positive bounds. Directed outward arithmetic
   or a rigorously derived envelope with explicit range/operation limits are
   possible approaches. The integer-exponent implementation below is the
   selected implementation; its numerical scope is qualified below.
5. Relate the admitted projection contribution to the physical residual
   tolerance with explicit margin. Independent expanded replay and FP64
   numerical acceptance still cover contraction/reconstruction roundoff.
6. Admit optional packed tau, maximum scratch and all descriptors **after**
   the existing primary Q tile. Reuse the admitted provider. Resource refusal
   must retain the original graph and tile, not silently replan the primal to
   make the optional path fit. Preserve bounded allocation retries.
7. Add an ablation, actual executed work/byte counters and distinct refusal
   diagnostics. Keep the expanded replay and response/Lambda paths unchanged.

### Selected conservative certificate

`src/cc/df_pair_bound.hpp` represents exact zero, a positive upper power
`2^exponent`, and refusal separately. All admission products/sums are integer
exponent operations; exponent range or integer overflow refuses only the
optional optimization. No positive bound can underflow to the exact-zero
marker. Native scalar coefficients are derived from the same positive IR,
including exact rational constants, not from another CC equation.

Geometry row norms use checked integer sums. Their common quantum is the
largest entry's upper exponent minus 32; each entry's upper power is rounded
up to that quantum, with at least one unit for arbitrarily tiny contributions.
This is tighter than domain-size times largest entry, without floating-point
norm accumulation. The complete Q sum uses maximum coefficient bounds times
an upper power of Q. Current T1 uses the on-device maximum magnitude times
the compiler-derived summed-axis extent at **every** amplitude state.

`df_pair_projection.cuh` stores the compiler-owned pair mean in separate
occupied-pair storage, canonicalizing diagonal virtual operand order. It
measures both original partners against the actual stored value with CUDA
12.9.1 `__dsub_ru` positive subtraction. Integer max reductions preserve even
subnormal errors. Nonfinite metadata never writes the physical sticky status.
The current gain from the ladder cut to the physical residual is proved from
the original core DAG (one); changes to that path fail closed. Admission
compares the final upper exponent against `floor_log2(residual_tolerance)-3`,
so the projection contribution is at most tolerance/8 even below DBL_TRUE_MIN.
CuMetal refuses this optional path until it has directed-DP qualification.

Resource admission follows the already chosen primal tile and expanded replay.
An optional pair OOM is retried without pairs **before** replay, tile, or
provider fallbacks. The original full arena remains available for any
per-state bound refusal. Supplied initial T2 must be bitwise simultaneous-pair
symmetric, independently of the separate packed-DIIS option.

Twenty-six host scientific/metadata tests and six GPU projection tests pass.
The latter run in Slurm job 2802 on node2 with assigned visibility `0`;
normal, subnormal, diagonal, original-tau retention, nonfinite refusal, and
buffer canaries are independently checked. Host constructor-failure injection
also passes with the new pair-first OOM chain. Job 2802 passes 36 native solver
checks and projection memcheck (zero errors); job 2804 passes nine final-library
checks, including exact executed work deltas, budget/tail/refusal behavior and
the no-work-saving single-occupied-block bypass.

## Production endpoint evidence

Frozen parent `2daa0aceaf6d140389ac5a183ead8b0ac3d3cd4a`, merged replay PR
#2173. Master was fetched at implementation/qualification/publication
milestones and did not advance; no unrelated full tests were triggered by the
earlier master changes.

Slurm 2805 runs four fresh processes in ABBA order on node2's RTX PRO 6000,
ethane230 (o=9, v=221, Q=488), FP64, DIIS8, Q8, 64-GiB budget, **ordinary full
DIIS storage**. All 38 primary/trial states admit the bounded fold in each
candidate; none refuse. Independent total/(T) error and expanded physical
residual gates pass for every sample. Primary/evaluation/iteration counts and
the complete physical Q scope match; actual saved contraction summands equal
the compiler query times the actual admitted primary Q count.

- Complete process wall median: 187.495296 -> 165.657467 s (-11.65%).
- Complete CCSD: 91.362928 -> 69.422902 s (-24.01%).
- Primary timer: 81.548458 -> 59.540565 s; expanded replay remains ~9.67 s.
- Complete contraction summands: 44,063,663,429,680 -> 27,891,350,238,256.
- Numeric capacity: 4,381,900,502 -> 4,717,423,790 bytes. The retained full
  fallback and larger folded scratch are part of this explicit tradeoff;
  projected-tau/metadata payload alone is 17,582,784 bytes, not the whole delta.

The accepted compact bundle is
`benchmarks/results/df-cc-occupied-spectator-pairs-20261010/`. Two samples per
selection support scoped observations, not shared formal/global performance or
force promotion. All raw logs/source archives/binaries remain in ignored
scratch, without releases/assets. Numeric NVML indices did not work in one
Slurm allocation despite successful CUDA qualification; the ABBA driver now
records the assigned CUDA device UUID directly without visibility overrides.

The post-measurement CI correction uses typed Fraction-valued dictionaries
instead of integer-typed Counter containers for rational polynomials. All
three generated pair CPU/CUDA artifacts compare byte-for-byte with the frozen
measured library's artifacts, and all 26 focused host tests still pass; no
unrelated GPU/endpoint rerun is warranted. Current ownership shards classify
the projection header as runtime metadata/storage and the folded equation and
majorant as compiler-generated. The inherited matrix-provider migration ledger
records the additional read of the existing admission gate, not a new selector.

An exact-only current-tau gate is another bounded first experiment: copy a
bitwise-symmetric tau without projection and retain the original action for
every asymmetric state. Its projection error is zero, so a projection majorant
is unnecessary. Do not assume it saves meaningful endpoint time:
`SolverOptions::packed_diis` defaults to false, and the observed trajectories
can have generated rounding asymmetry. Measure initial/current eligibility
before spending a full matched endpoint series on this narrower alternative.

## Earlier native-action qualification

Frozen source parent: `3c7c2c775f836f39c5a3f6864c1b0083aae88138` (the expanded
replay PR plus its focused owner-probe CI fix), not current master. Master was
observed at `88cfa701753e91aa3abbe17877fb239ee3fd9ab4` (#2167 DF-guess admission).
This update does not alter these standalone ladder actions; no unrelated full
requalification was launched because master advanced.

- 20 focused host checks pass: pair reconstruction, batched lifting, exact
  negative proofs, derived asymmetric-tau majorants, independent determinant
  complete-core residual/energy, independent NumPy virtual residuals, native
  CPU actions, and compiler-only generation without runtime/reference imports.
- Five native CUDA action checks pass on n2's RTX PRO 6000. Actual Slurm node
  name is `node2`, partition `main`, resource `gpu:pro6000:1`, finite eight-minute
  allocation. The job's assigned `CUDA_VISIBLE_DEVICES` is preserved.
- Two compute-sanitizer memcheck cases (`o,v,Q,batch = 3,2,5,3` and `2,3,4,3`)
  report zero errors. They exercise batched tails of two and one respectively.
- Native checks compare every reconstructed cut, arena canaries, executed
  scalar-summand queries, and retained sticky arithmetic status.
- The 11 original conventional/DF/core/hoisted generated CPU/CUDA artifacts
  remain byte-identical after the common shape/consumer extension.
- `tools/check_compiler_structure.py`: 509 modules, zero dependency errors.
  Ruff, clang-format and diff-whitespace checks pass.

A subsequent nonpair-detection compatibility guard receives its own focused
host check. At that earlier snapshot, all three paired CPU/CUDA artifacts were
byte-identical to the files compiled in the successful GPU allocation; hashes
are retained in `qualified-generated.json`. This compiler-only hardening does
not require another GPU qualification of unchanged native code.

Complete six-cut auxiliary scalar summands, excluding prepare, core, replay,
DIIS and all endpoint work:

| Runtime shape | Original packed one-Q | Folded packed one-Q | Original Q8 | Folded Q8 |
| --- | ---: | ---: | ---: | ---: |
| o=9, v=221 | 1,985,182,225 | 1,113,077,329 | 15,881,457,800 | 8,904,618,632 |
| o=21, v=243 | 16,076,019,537 | 8,487,042,057 | 128,608,156,296 | 67,896,336,456 |

Ignored full artifacts are in `.artifacts/df-cc-spectator-pairs-20261010/` and
`n2:/data/jzzeng/qc-cc-spectator-pairs-20261010/`. Retain the frozen parent,
manifest supplement, prototype-v2 archives and checksums, allocation record,
logs, and cached native probes. Generation comparison hashes are retained in
`generation-before.json`; work counts are in `symbolic-work.json`. The manifest
supplement is necessary for test-oracle imports from a reduced source archive.

## Publication boundary

The matched production endpoints above satisfy this iteration's numerical
scope; theoretical summand reduction alone never authorizes a performance PR.
Monitor master, integrating only relevant changes; do not label reused
frozen-parent timing as a new current-master comparison. Further force,
hardware or formal performance promotion requires its own explicit gates.
