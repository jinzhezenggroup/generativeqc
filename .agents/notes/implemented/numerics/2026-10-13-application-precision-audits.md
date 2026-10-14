# Decision: application precision budgets with bounded endpoint audits

Status: implemented
Date: 2026-10-13

The speed-oriented `fast` selection described below is subsequently withdrawn.
See [the rejection decision](../../rejected/2026-10-13-fast-endpoint-audit.md).
The machinery remains an explicit experimental audit; historical measurements
and their original Fast labels are unchanged.

The later [budget-selected FP32 XC decision](2026-10-13-budget-selected-fp32-xc.md)
introduces a distinct qualified one-solve Fast path, while preserving this audit
for explicit evidence requests and unsupported budgets/domains.

## Problem

Applications tolerate different observable errors, whereas native `precision`
selects arithmetic and `target_accuracy` alone attaches an evidence request.
Mixed SCF and mixed gradient integration have different risks. On master
`e0ad22036`, native AUTO already owns strict SCF refinement, but there is no
application-qualified FP32 gradient route to promote with a preset.

## Decision

Add opt-in `PrecisionCalculator`/`PrecisionPolicy` without changing existing
Calculator behavior. Fast may use admitted native CUDA AUTO. Balanced and
Strict retain FP64 SCF. All use existing gradient routes. Every result requires
independent norm agreement with a separately relaxed same-backend FP64 solve.
An over-budget candidate can fall back to FP64 only after a third, tighter
solve verifies that returned endpoint. Failure after this bounded work raises.

The report states observed agreement, not certification. Native work evidence
is preserved; requested AUTO does not imply actual mixed execution. Workspaces
are released between complete endpoints to avoid two simultaneous owners under
a budget sized for one. The cost is two or three full endpoints per call.

## Rejected alternatives

- Inferring force accuracy from an SCF energy/density tolerance mixes distinct
  observables and does not establish the requested force error.
- Promoting unqualified FP32 gradients from isolated speed/error anecdotes has
  no reproducible endpoint gate or coverage of derivative semantics.
- Reusing an audit at another geometry, or comparing only fixed-density
  operators, does not validate a relaxed endpoint.
- Treating the returned fallback reference as automatically accurate would
  bypass the user's budget when both candidate and initial reference are poor.
- Using `basis_metadata()['model_identity']` for audit equivalence rejects
  legitimate tightened references: that prepared-plan signature includes
  precision and solver controls. HF reuses ResolvedModel; KS hashes actual
  mathematical basis identities and resolved science, excluding scheduling.

## Evidence

`test_precision_policy.py` covers independent norm gates, relative/zero targets,
bounded fallback, invalid/nonfinite reference rejection, unchanged CUDA backend
selection, workspace release and truthful incomplete work counts.
`test_precision_policy_native.py` checks RHF/UHF against pinned independent
PySCF energy/force fixtures, KS model identity invariants, and PBE energy/force
against independent PySCF SCF energies on the same moving quadrature.
The initial live PBE reference was run with PySCF 2.11.0. Initial CUDA controller
checks used injected endpoints; native CPU and warnings-as-errors documentation
builds were the first executed build gates.

### PBE0 GPU follow-up

Slurm node4 job 804 now measures PBE0/def2-SVP E+forces for water (24 AOs) and
its hydrogen-bonded tetramer (96 AOs), with independent PySCF analytic
moving-grid forces. The source-matched raw samples, build configuration,
operator counts and negative performance evidence are retained in
`benchmarks/results/pbe0-precision-20261013/README.md`.

AUTO has only mixed J recurrence in these executions; K, XC matrix products
and derivatives remain FP64. Complete AUTO endpoints take 28%/45% longer
than ordinary FP64; audited Fast takes 2.32/2.53 times as long. This is not a
FP32-gradient implementation or a reason to reject more extensive mixed
precision. It demonstrates that the current conservative facade is validation
infrastructure, not the requested performance endpoint. Increasing FP32 coverage
must remove actual expensive work and be measured including refinement/audits,
not inferred from the AUTO name or a count of low-precision operators.

The first GPU attempt also exposed a constructor replay bug: resolved KS
options contain composition/functional outputs that discovered named methods
and MethodIR constructors reject as conflicting model owners. Preserve the
already-validated immutable KS constructor input instead; only file-backed
bases need replacement by their resolved snapshot. Independent audit model
identity checks remain mandatory. Retain the failed run separately, and protect
discovered PBE0/B3LYP and MethodIR construction with native CPU regressions.

## Revisit when

Add a selective-gradient candidate only with independent endpoint references,
explicit application error gates, source/device/method/basis/grid coverage and
complete timing/work counts including audit overhead. Skipping intermediate
audits needs separately qualified transferable error/conditioning evidence.
NVE additionally requires continuity, drift and force/energy consistency gates.
