# Decision: narrow the complete stationary geometry pipeline, not only ERI derivatives

Status: implemented as a private qualification candidate; public Fast is unchanged
Date: 2026-10-13

## Problem

The [weighted-derivative candidate](2026-10-13-mixed-weighted-force-candidate.md)
lowered four-center work but left the dominant native geometry kernel strict.
At 192 AOs that kernel consumed approximately 80% of kernel time. Merely
describing this bottleneck does not deliver an application-speed improvement.

## Decision

The private, invocation-local `DirectForcePrecisionRequest` now independently
selects weighted ERI derivatives, geometry pullbacks and grid production. The
test bridge's versioned precision mask is 1/2/4, respectively; 7 selects all.
The method, SCF state token, screening domain, quadrature and exact J/K model
are unchanged. No density fitting or reference-oracle work enters production.

Shared compiler algebra supplies all typed variants:

- AO spatial jets lower the existing Gaussian DAG to FP32.
- Density projections lower the same TensorIR contraction with FP32 operands,
  compute and accumulation, through the shared prepared provider portfolio.
  Explicit casts and widening preserve the existing double-buffer ABI; TF32
  and reduced-mantissa tensor-core products are not substituted.
- Density features use the same bilinear graphs with FP32 local/tree sums.
- XC interior exchange and correlation both use FP32 elementary operations;
  exact equal-spin inputs use four rather than eight derivative channels.
- AO translation pullbacks, per-point atom sums, Becke local partials,
  normalized products and reverse pair traversal use FP32 arithmetic.
- Point force contributions widen before global FP64 accumulation. Translation
  is recovered from the same rounded atom contributions, not two independently
  rounded cancelling traversals. No post-hoc force mean subtraction is used.

The Becke scalar partials are generated with a typed `ScalarCEmitter`; shared
reverse traversal templates retain the strict and cooperative consumers.
The float serial schedule reuses the charged double scratch allocation. AO
point sums borrow the later Becke distance segment. Float distances end at
byte `56*natom`, followed by `12*natom` bytes of point adjoints within the
existing `72*natom`-byte lane. Integer scratch keeps its original alignment.

## Strict boundaries and fallbacks

This is **not literal all-FP32 computation**. SCF, geometry displacements before
narrowing, physical XC normalization, center-pair preparation, global force
sums, Boys moments, one-electron/Pulay derivatives and ERI orders above five
remain FP64. XC boundaries and unrepresentable intermediates retain the shared
strict equation. Unsupported XC compositions have no float native binding.

The first grid experiment failed on ordinary water because an FP32 density
underflowed before its Cartesian derivative. Passing this artificial vacuum to
the unchanged XC domain correctly failed. The fix recomputes affected feature
bilinears in FP64 from the current panels; it does not clip density, remove
points, loosen the XC validity contract or change an error gate. Negative
density candidates also receive the wider bilinear pass; unresolved failure
still fails the complete candidate and triggers strict retry.

Nonfinite geometry never publishes a partial gradient. The method resets all
sources and retries the complete gradient once at the same FP64 state token.
CUDA errors, stale tokens and invalid requests are not numerical retries.
Optional resident float panels and provider allowance are bounded by remaining
owner capacity; allocation rejection selects the existing strict grid. Work
receipts distinguish this fallback from executed FP32 grid points.

Neither nonfinite checks nor empirical gates certify arbitrary finite rounding
error. Public application-budget admission is deliberately not broadened by
this private experiment. Hessians, TS/IRC, NVE drift, diffuse/cancellation-heavy
systems and arbitrary user error budgets remain unqualified.

## Evidence and retained experiments

`benchmarks/pbe0_mixed_gradient.py --precision-mask 7` measures fresh construction,
unchanged FP64 SCF, the complete native moving-grid gradient, export and teardown.
It retains an excluded warmup pair and three alternating strict/mixed pairs,
plus an instrumented strict/mixed replay at one final-state token. Full
SCF work and geometry/grid work must match; empty mixed routes, unexpected
ordinary-case retries and failed independent force gates prohibit speed claims.

Independent oracles include Libxc exchange/correlation coefficients, Decimal
Becke directional derivatives, long-double density-feature sums and complete
PySCF moving-grid forces. Host tests execute generated C++ rather than merely
searching for `float` declarations. GPU tests use finite Slurm jobs only.

The first broad candidate, before FP32 point-local force sums, measured:

| AOs | Complete strict/mixed (s) | Gradient strict/mixed (s) | Max force error (Eh/bohr) |
| --- | --- | --- | --- |
| 192 | 11.466717 / 8.506387 | 5.613804 / 2.636161 | 8.30e-7 |
| 384 | 56.339326 / 34.987948 | 36.485320 / 15.167041 | 1.24e-6 |

Both had zero task/complete-gradient retries. Raw results, failed first-grid
test, build logs and source snapshots live under ignored
`.artifacts/pbe0-broad-fp32-20261013/` and the corresponding n1 campaign directory.
The before-local-sums source archive SHA-256 is
`54d5ee8bc0e7b4a7172668644007e4f4b4051d036e1f646b963e34923e4e99ac`;
library SHA-256 is
`aeb4cb2d35258473dcd1f9e2f38b23433a4cc95a680e689c8cf72f5d05971748`.

Final source-matched endpoint evidence belongs in
`benchmarks/results/pbe0-broad-fp32-20261013/`. Do not combine these native
endpoint times with older public Python-force timings or upstream's DF-r2SCAN
gradient-only speedup. A larger system helping this candidate is not a universal
size-based admission rule.

### Final local sums and recovery validation

FP32 point-local sums improved the timed candidate further: job 8137 measured
complete 1.3941x / 1.7083x and gradient 2.3360x / 2.7763x at 192 / 384 AOs.
Its library SHA-256 was
`cbb54b1229267a46b379359c6e9f63b309a566f8879095cc85f6438584950d43`, and source archive
SHA-256 was `bbbb0a51984aa62d816229fd85b184aab6c3b0f0824a724e4d561f8d1844c242`.

A follow-up repaired reset after an *early* source failure: a pending geometry
stream must be synchronized without re-reading its discarded numerical flag.
Otherwise that old flag can prevent the advertised complete strict retry before
reset clears it. `discard_geometry` preserves asynchronous driver errors.
The device probe deliberately overflows only the float geometry work, exercises
both uninspected and early-inspected sticky failures, then compares strict replay
with the clean strict source span. No production fault-injection flag is added.

The mixed point header was also added to installed assets, JIT header closure and
the stationary AOT contract dependencies. Generated scientific grid/native-PBE0
CUDA text is byte-identical across this follow-up, but the library/source
identities change; old timings were not relabeled. Job 8154 remeasured the final
library `3c998909ab3953245f1f55acd38e9606d08a32177c1ae1ff48cb8a9323368256`:

| AOs | Complete strict/mixed (s) | Gradient strict/mixed (s) | Complete/gradient speedup |
| --- | --- | --- | --- |
| 192 | 11.249630 / 8.039422 | 5.523108 / 2.310956 | 1.3993x / 2.3900x |
| 384 | 55.390198 / 32.428234 | 35.859010 / 12.918232 | 1.7081x / 2.7758x |

Maximum independent errors remain 8.28e-7 / 1.24e-6 Eh/bohr; the 384-AO candidate
must not be represented as passing a 1e-6 force target. The work receipts and
all raw samples are bound in the tracked results summary. This is a meaningful
FP32 improvement of this executor, **not yet a faster public Fast endpoint**.
The public force consumer uses a different schedule; native baseline ratios
alone do not justify replacing it. SCF/J/K and public gradient admission are
still unfinished parts of the requested broad Fast policy.

Additional Cartesian testing exposed an oracle-adapter bug: `reference()`
hardcoded spherical PySCF AOs despite the fixture specifying Cartesian d shells.
The unchanged native strict energy differed by 1.50e-3 Eh, so the gate correctly
failed before any mixed-precision acceptance. The oracle now honors the fixture
representation; no tolerance or native force equation changed. The failing log
is retained separately. Previous spherical-reference records are unaffected.

Public regression initially found stale/missing optional Python stationary AOT
artifacts. Building only `generativeqc` does not build the separate
`generativeqc_stationary_pbe0_rks_spd_manifest` and
`generativeqc_stationary_pbe0_rks_manifest` targets. Rebuild those before checking
the public facade after changing a shared contract; do not disable identity
validation or silently JIT over the mismatch. Native library SHA stays fixed.

Final validation: 83 targeted host numerical/asset tests passed (24 GPU skips),
109 host API/native-binding tests passed, and compiler ownership checked 537
modules with zero dependency errors. The earlier broad geometry/Becke suite
passed 3526 tests (234 skips). Job 8162 passed 22 public/native CUDA endpoint
tests, including ammonia, Cartesian water and the pending-geometry recovery
probe. Job 8160 passed six selected endpoint/recovery tests under memcheck with
zero errors. The failed Cartesian-oracle and stale/missing-AOT attempts remain
in the evidence directory; the corrected runs do not hide them.

## Revisit when

The later public admission and fair-comparator decision is recorded in
`2026-10-13-public-budget-fp32-gradients.md`. The timings and unpromoted status
above describe this historical private campaign, not the later public route.

Broader independent molecular coverage and an observable-error admission/reporting
policy justify public promotion. Remaining strict SCF/high-order ERI work and
double transport buffers require separate precision and complete-endpoint gates;
do not label them FP32 merely because this geometry candidate was selected.
