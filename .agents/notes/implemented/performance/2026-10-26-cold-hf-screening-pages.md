# Decision: bounded cold HF proposals and larger reusable task pages

Status: implemented; explicit opt-in only
Date: 2026-10-26

## Problem

Large direct RHF cold execution spent most of its time evaluating accurate
integrals for densities still far from the stationary solution. Small reusable
task pages also repeated compaction and launch overhead. Warm timings alone
missed both costs.

## Decision

Containment after review: only exact `GENERATIVEQC_HF_COLD_SCREENING=1`
selects both the proposal and larger page. Unset/0/none/empty/invalid values
restore the incumbent 8M-task / 1.5-GiB / one-sixteenth-memory policy and
ordinary SCF operator. Physical-reference exports keep the incumbent page.
The historical capsule below is unchanged and describes the pre-containment
source, not the current default. Its failed default-force diagnostic remains
negative evidence; tighter requests do not turn it into a passing default.


Admit a screened FP64 proposal only for large unseeded single-system restricted
bounded quartet owners. At most eight steps at `8e-8` precede a fresh target
DIIS/energy epoch. The requested target screening and convergence criteria are
unchanged; both epochs consume one user iteration allowance. A local graph
owner prevents screened work from leaking into frozen warm or imported owners.
The policy's complete admission is documented in `hf_force_finalization.md`.

Use an independent reusable page budget capped at 32M tasks / 6 GiB on large
devices. A refused larger arena retries the old 8M-task page once. Rebuild all
layout offsets and queue bounds before uploads or graph capture, preserve
unrelated CUDA errors, and retain the original physical-reference page profile.

Count screened FP64 applications as FP64 work. The target tail is separately
reported by `refinement_iterations`; neither public iteration counts nor
`strict_stage_fock_builds` may omit the preliminary stage.

## Evidence and limits

The source-bound capsule at
`benchmarks/results/cold-hf-defaults-20261026/` retains three alternating
fresh-process default-grid HF96 endpoint pairs, cold plus three frozen warm
calls. Native cold median is 11.3787 s versus GPU4PySCF 11.9044 s, with
non-overlapping ranges. Complete FP64 work is eleven iterative builds. This is
an endpoint result, including analytic force, rather than a kernel comparison.

All populations pass the unchanged #2786 common observable gates. Default
native force error against a tighter independent reference is 3.02e-7 Eh/Bohr,
versus 5.10e-8 for the previous production path. An additional 1e-7 diagnostic
therefore fails at default tolerances; it is not an old baseline failure.
At explicitly tighter native targets, independent 1e-8-Eh / 1e-7-Eh/Bohr gates
pass for cold, frozen warm, moved/restored and checkpoint-imported execution.
The stricter run qualifies the algorithm's accuracy control; it does not replace
or relabel the default-tolerance measurements. Failed reference convergence at
an unattainable extensive energy decrement is also retained.

A real 4-GiB ledger refuses the optional large page and verifies successful
bounded fallback, independent observables, bounded peak and zero live bytes
after close. Short energy-only solves share the global iteration allowance.
Existing direct/DF complete-work CUDA gates and native policy tests pass.

## Rejected alternatives

The 16M-task page also wins but leaves less endpoint margin than 32M in the
paired experiment. All-fused streaming, separate J/K and Rys K experiments
regressed complete endpoints and were not promoted. One-electron nucleus
cooperation mostly exposed the next module's first launch cost for HF; it did
not produce the expected endpoint gain. Packed Schwarz thread remapping was
too small to justify combining it with this promotion.

The first default timing campaign ran near an independent GPU4 oracle startup
and showed unusually slow GPU4 cold times. Retain it as supplemental evidence;
use the second isolated alternating campaign for the performance claim.

## Revisit when

Revisit admission or page capacity after changes to generated task ABI, target
memory budgets, convergence policy, graph/checkpoint ownership or provider
selection. Require complete cold/warm work and independent observables, including
failure paths. Do not broaden measured performance claims from one water
geometry to all large molecules or use the tighter-target result to hide a
default-tolerance accuracy change.
