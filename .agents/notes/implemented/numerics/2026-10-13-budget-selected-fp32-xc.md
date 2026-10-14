# Decision: application-budget-selected FP32 XC warmup with a strict final state

Status: implemented
Date: 2026-10-13

## Problem

The user rejected both a duplicate-SCF audit called Fast and an ordinary exact
FP64 optimization relabeled Fast. Exact integral-source reuse belongs to every
ordinary FP64 caller (PR #2398). The requested distinction is real lower-precision
arithmetic selected from application allowances, with bounded strict fallback,
not manual kernel switches or unmeasured accuracy claims.

This supersedes the Fast alias in
[the exact-reuse decision](../performance/2026-10-13-small-ks-reuse-fast.md).
The rejected duplicate-solve design and its negative measurements in
[the audit rejection](../../rejected/2026-10-13-fast-endpoint-audit.md) remain valid.

## Decision and arithmetic boundaries

Reuse `TargetErrorBudget` rather than invent another energy/force budget type.
`PrecisionPolicy("fast", budget=...)` passes immutable per-owner allowances to
native KS preparation. Admission covers local all-electron restricted PBE/PBE0,
24–256 AOs through d, exact full-range J/K and qualified device-fused XC. Energy,
maximum force-component and optional RMS allowance floors are respectively
`1e-6 Eh`, `1e-5 Eh/bohr` and `3e-6 Eh/bohr`. These are conservative qualification
floors, not a conversion from residual norms to observable error bounds.

The provisional route uses actual FP32 AO production, restricted PBE correlation
arithmetic, tiled density/potential products and local point-total reductions.
Shared product tiles, products and inner sums are float, not double arithmetic
on rounded inputs. Global arrays, inter-tile scatter, point normalization and
PBE exchange remain double. Representability-gated tails and non-tiled products
fall back to the original double equation. The point gate does not certify
molecular accuracy. The common scientific algebra is parameterized; no second
independently maintained PBE equation is introduced.

J/K, eigensolution, DIIS, physical residuals and all analytic gradients stay
FP64. Warmup stops at max physical residual `1e-4` or twelve mixed iterations.
Strict refinement restores every schedule region and resets the energy epoch;
mixed energies never supply strict stopping evidence. Provisional DIIS history
may propose densities, but is cleared after six unsuccessful strict steps.
Nonfinite mixed work retries the same density strictly. Original convergence,
electron-count and final-state gates remain mandatory.

The exact resident J/K source survives this XC-only handoff: its geometry,
screening and arithmetic identity did not change. This is not permission to
share an actually mixed J/K epoch with a strict source. Existing memory,
geometry/lifetime, fitted/range and resource-ledger guards remain authoritative.

## Evidence semantics and fallbacks

An admitted Fast endpoint executes once and reports `qualified_strict_final`,
the requested budget, native work and strict checks. It reports no invented
observable differences. Scope is `qualification_not_per_call_observable_audit`:
empirical qualification plus a strict final endpoint is not a per-call error
certificate, stability/root test, or transferable MD/Hessian guarantee.

An explicit `TargetAccuracy` still requests a complete independent relaxed
FP64 comparison, even with Fast. Tight budgets and unqualified domains use the
bounded two/three-solve FP64 audit. This fallback may be slower; do not hide its
work or call it accelerated. The candidate owner is released before allocating
an audit owner. Admitted nonaudit Fast can retain its plan until `clear_cache()`.

Native operator counts describe submitted composite operators, not an
instruction-level FP32 percentage. Point fallback counts are not exported.
Gradients are not qualified FP32, and no all-FP32/2–7x claim follows from this
implementation. Large expected gradient gains require a separate derivative
implementation and independent force/energy-consistency gates.

## Transport and seed invariants

The existing public C KS descriptor remains unchanged. A private size-tagged
extension retains it as a prefix and appends three doubles. Python verifies
`generativeqc_application_precision_version_v1`, keeps ctypes pointees alive,
and native preparation copies/validates the allowances. Truncated extensions
are rejected. A future public extension must deliberately reconcile this private
layout; never infer it on an old library. Budget enters execution identity but
not scientific model identity. Process-global experimental controls are rejected
for application requests; two policy instances cannot change each other's mode.

Initially, AUTO admission silently selected Hcore where FP64 selected automatic
MINAO. The fix preserves native strict MINAO preparation at the Python query,
native method registry and preliminary-provider boundaries, only for the
application extension. Ordinary AUTO and HF/LDA preliminary solves retain their
old domain. Projection, metric admission and resource inventory are unchanged;
no PySCF/reference producer is used by production. Tests must cover both default
and explicit MINAO, not just the Hcore benchmark.

## Rejected alternatives and measured evidence

Point-only and point-plus-AO ablations were weak or regressed complete endpoints.
The strict tetramer profile spends only about 32 ms in point algebra; AO radial,
matrix contractions/scatter and strict derivative work dominate. Float matrix
products with bounded proposal-history reuse are the useful measured increment.
Keep failed ablations and failed transport/seed tests rather than deleting them.
An additional `1e-2`/`1e-3` early-handoff screen improved the Hcore octamer but
increased tetramer work to eighteen iterations (strict fourteen, current Fast
sixteen). Neither earlier threshold was promoted globally on that evidence.

Complete PBE0 energy-plus-force endpoints compare against already-optimized
ordinary FP64, not the old no-residency baseline. Every endpoint includes fresh
construction, SCF/refinement, all forces, export, release and CUDA synchronization.
First calls are retained separately from five alternating-order measured repeats.
Independent CPU PySCF SCF/moving-grid analytic forces gate all samples at
`1e-7 Eh` and `1e-6 Eh/bohr`; no oracle work supplies production inputs.

See `benchmarks/results/application-budget-fast-20261013/README.md` for final
source-matched samples, work counts, separate Hcore/automatic-seed timings,
numerical errors, device/build provenance and scheduled sanitizer evidence.
Kernel tests independently compare points with Libxc and indexed/partial-tile
products with NumPy, including float representability, FP64 inter-tile priors,
overflow rejection and real-device memory checks. Public CUDA tests cover
one-solve admission, explicit target audits, tight/domain fallbacks, per-instance
isolation, plan replay, changed geometry and release/rebuild.

## Revisit when

Extend precision to gradient integrals, exchange or further functionals only with
independent endpoint qualification and honest fallback/arithmetic census. Revisit
handoff/history policy across different molecules and initial densities rather
than tuning one water fixture. Validate NVE drift, continuity, sampling and force/
energy consistency separately; this static endpoint campaign does not cover them.
