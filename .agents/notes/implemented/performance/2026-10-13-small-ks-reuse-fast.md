# Decision: make Fast a single strict endpoint and reduce repeated KS source work

Status: implemented
Date: 2026-10-13

The Fast alias below is superseded by
[budget-selected FP32 XC warmup](../numerics/2026-10-13-budget-selected-fp32-xc.md).
The exact reuse optimization remains shared with ordinary FP64 (PR #2398);
its historical timing labels and rationale are preserved here.

## Problem

The rejected Fast implementation performed an AUTO solve followed by a full
FP64 SCF and force audit. Removing its misleading name did not remove its
runtime cost. AUTO itself was slower than strict PBE0 in the measured small
systems: only J recurrence was lowered, while K, XC and gradients stayed FP64.
The user explicitly requested an actual complete-endpoint speed improvement.

Profiling identifies repeated J/K source work, not gradient precision, as the
first actionable bottleneck. The existing exact phase-local resident source
was excluded for small screened s/p/d systems by thresholds inherited from
the generic f-shell domain. Forced reuse on the previous source reduced the
96-AO PBE0 tetramer endpoint from about 6.1 to 3.4 seconds. Bounded AUTO warmup
alone did not beat the ordinary FP64 endpoint. Preserve that exploratory
screen rather than presenting only its successful variant.

## Decision

Admit cold screened through-d hybrid KS sources from 24 public/24 Cartesian AOs with
at least eight permitted iterations. Keep the f-shell automatic thresholds
at 64/128 and the generic RHF allocation floor at 64. Pure-Coulomb KS retains
its original thresholds rather than inheriting the hybrid cost model. KS's source allowance
must use the same policy as its earlier admission check; otherwise water's
new eligibility would still be refused inside the lease. This changes
scheduling, not recurrence, screening, accumulation, mathematical model or
SCF stopping tests.

Fast without an explicit target performs one native FP64 endpoint. It does
not request AUTO, loosen solver settings, or run a second full SCF. Cap its
controls at ordinary strict defaults, preserving tighter settings. This
optimization belongs to ordinary Calculator too: do not withhold an exact
speed improvement to manufacture an advantage for a preset.

Separate strict execution from empirical budget evidence. Fast has no implicit
observable target, reports `strict_fp64`/`native_fp64`, and publishes no fabricated
error differences. An explicit TargetAccuracy, including on Fast, still selects
the full two/three-endpoint audit. Such a call is not advertised as accelerated.
Balanced, Strict and audit_auto retain their explicit audit contracts. This is
an exact-work-reduction implementation, not new FP32 kernel qualification.

## Invariants

- Keep the complete live numeric inventory, optional capacity/free-memory
  refusal, bounded Direct fallback, frozen Schwarz mask and phase-local lifetime.
- Keep warm/external-seed refusal, fitted/range/resource-ledger guards and the
  explicit source-disable control. No mixed epoch may borrow strict values.
- Native convergence, final-state checks and all requested force contributions
  remain mandatory. No CPU/PySCF oracle is added to production.
- The required observable audit is never silently dropped. A strict FP64
  result is not a certified physical error bound, root/stability proof or NVE
  qualification.
- Benchmark facade construction, SCF, analytic forces, export, release and CUDA
  synchronization. Report both the current FP64 default and the former
  small-system no-reuse schedule; never call the latter the new FP64 default.

## Evidence

See `benchmarks/results/pbe0-fast-20261013/` for source-matched complete endpoint
measurements, semantic work, independent PySCF gates and the earlier screening.
Host tests cover threshold/capacity boundaries and unchanged RHF/f-shell
admission. Native small-source tests cover independent signed RKS/UKS J/K,
full/partial residency, PBE0/B3LYP states, warm/frozen/external seeds, strict
refinement, chunk requests, screening guards and memory refusal. Fast controller
tests distinguish one strict solve from explicit error-budget verification.

## Rejected alternatives and revisit

Do not tune the FP32 warmup in isolation or remove precision qualification
guards merely to produce a speed ratio. Further FP32 work needs real expensive
operator lowering and new independent accuracy/performance evidence. A faster
strict endpoint is useful now but does not establish the requested mixed
gradient/SCF accelerator. Revisit the source thresholds if complete cold/warm
endpoints show regressions or the resource/cost model changes.

This supersedes the unavailable-Fast API decision in
`../../rejected/2026-10-13-fast-endpoint-audit.md`; its negative measurements and
rejection of duplicate-solve acceleration remain valid.
