# Decision: keep occupied XC experiments out of production KS

Status: implemented
Date: 2026-11-04

The production opt-in exclusion is superseded by the
[audited-proposal decision](2026-11-04-occupied-xc-audited-proposals.md).
Matrix defaults and exact-mode lower-level verification remain intact.

## Problem

The [exact-identity repair](../numerics/2026-11-04-occupied-xc-exact-identity.md)
protects downstream XC consumers by recomputing every original-D feature after
candidate projection. It publishes the candidate only if that candidate is
already bitwise equal to the recomputed vector. Otherwise it publishes the
original vector. Candidate work therefore cannot change the published result,
but it adds cost to an explicitly selectable production route.

At source `def31af3f6c9b8476e8ff9d892f7f74589d168c7`, the retained
[native endpoint capsule](../../../../benchmarks/results/occupied-xc-exact-identity-20261104/README.md)
compares ten alternating matrix/explicit pairs per RKS/UKS phase on the same
24-AO, default-grid fixture and library, with equal SCF/Fock/J/K work. Cold and
moved complete E+F medians cost about 22% more for RKS and 51–53% more for UKS;
warm UKS costs about 16% more. Cold includes fresh owner preparation in a shared
process, not process startup or cache-miss compilation. This is a measured
matrix-versus-experiment cost, not a before/after claim about an older version.
Keeping only the default safe does not resolve the explicit route's cost.

## Decision

CUDA KS always uses ordinary matrix XC contraction, including when
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` is set. Remove KS candidate-frame publication
and selector state rather than compute candidate work and discard it later.
Keep the zero occupied-frame trace counter as the observable fallback contract.
Retire the former control from the live default-promotion inventory; its old
`already-default` classification and automatic-route timing no longer apply.
The lower-level `CudaXcPlan` borrowed-frame API, generated candidate kernels,
exact stored-D certificate and their independent numerical/resource/lifetime
tests remain available for research. This changes production route selection,
not the original-D matrix arithmetic, precision, SCF or force acceptance gates.

## Rejected alternatives

- Retaining the opt-in production route knowingly retains its measured cost.
- Removing exact verification restores the unresolved numerical sensitivity
  problem; a nonzero feature-error prefilter is not an E/Vxc/force bound.
- A guessed size/rank crossover cannot establish a safe or profitable domain.
- Faster dense verification still adds candidate work and needs new evidence;
  it is not a proven narrow fix for the observed complete-endpoint cost.

## Validation boundary

Public native RKS/UKS control tests now require no admitted frame for unset,
`1`, `0` and invalid values across cold, warm and moved geometries. Independent
original/displaced E+F tests retain the exact 2e-9 Eh / 1e-7 Eh/Bohr gates and
matrix pairing. Lower-level experimental tests and historical receipts are
unchanged. Existing receipts remain bound to their recorded source; they are
not relabeled as current-head GPU runs. The scoped fallback reuses the already
qualified matrix route and makes no new speedup or broad performance claim.

## Revisit when

A cheaper certificate proves the applicable downstream numerical contract,
retains independent boundary regressions and bounded original-D fallback, and
has source-matched complete native E+F/resource/admission-inclusive cost evidence
showing a useful production domain without credible regressions.

Refs #2751 and PR #2971.
