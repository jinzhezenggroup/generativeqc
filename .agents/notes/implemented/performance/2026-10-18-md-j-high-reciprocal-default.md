# Decision: default angular-three reciprocal MD-J reuse

Status: implemented
Date: 2026-10-18

## Problem

The [initial implementation decision](2026-10-18-md-j-high-reciprocal.md)
retained the measured angular-three reuse as opt-in. After reviewing the actual
prepared-J and complete energy/force gains, the user explicitly requested
default-on behavior when beneficial. This supersedes only that selector's
default-off decision, not the rejected residual contractions or the broader
draft residual-tile qualification.

## Decision

At resident-owner preparation, unset/1 for
`GENERATIVEQC_MD_J_RECIPROCAL_HIGH` enables the existing generated reciprocal
consumer for uniform pair-angular `(0,3)`, `(1,3)`, `(2,3)` and `(3,3)` work.
Explicit `0` retains their oriented provider. The parent
`GENERATIVEQC_MD_J_RECIPROCAL=0` disables this extension even when high reuse is
explicitly requested. Invalid values still reject active MD preparation;
environment changes after preparation cannot alter replay policy.

The promotion changes selection only. It introduces no new recurrence,
precision, screening allowance, resident allocation or production CPU work.
Angular-four work remains oriented; optional-budget refusal and unsupported
bases still select normal J. Independent direction masks, exact residual AO
masks, nonsymmetric/UKS density behavior, K and force/response ownership remain
unchanged. The historical profile interval `oriented_seconds` still includes
angular-three reciprocal kernels, so kernel names remain necessary for timing
attribution.

## Evidence and limits

The [source-matched contraction trials](../../../../benchmarks/results/md-j-residual-20261018/contraction-trials.md)
record 4.121193 to 3.923415 seconds prepared J (4.80%), and one same-GPU,
same-binary complete endpoint pair: cold 270.418484 to 264.922447 seconds
(2.03%), four-phase sum 557.010250 to 547.387625 seconds (1.73%). Both arms
perform 24/1/13/1 Fock builds. Independent energy/force and raw-J gates pass;
uniform radial evaluations fall while all density consumers and residual
products remain conserved.

The user-authorized decision accepts that measured benefit with the explicit
opt-out. These small paired endpoint samples do not establish statistical or
cross-hardware benefit, and the existing draft PR's older residual-default
qualification limits remain independent. Default-path numerical, policy and
work-count regression tests accompany this selection change; additional raw
receipts must retain their own source/binary identities rather than relabeling
the historical opt-in measurements.

The [separate default-path qualification](../../../../benchmarks/results/md-j-residual-20261018/default-reuse.md)
does exactly that in Slurm 10136: unset default versus explicit `0` lowers cold
270.286147 to 265.244123 seconds (1.87%), and the four-phase sum 556.477525 to
548.431200 seconds (1.45%). Both arms retain 24/1/13/1 Fock builds and pass every
independent E/F gate. Default and explicit `1` have equal per-class radial work;
114 scheduled GPU tests and 400 focused host tests pass. These two endpoint
pairs support the limited measured decision, not a cross-hardware guarantee.

## Revisit when

Reassess the default on reproducible complete-endpoint regressions, a new
angular/device domain, changed numerical screening, or altered owner admission.
Preserve the fallback and independent scientific gates. Neither the neutral
register/unroll experiment nor the slower J-publication experiment becomes a
default merely because this different uniform-source experiment succeeds.
