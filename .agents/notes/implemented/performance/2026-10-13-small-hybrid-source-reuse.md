# Decision: share exact small-hybrid source reuse with all eligible callers

Status: implemented
Date: 2026-10-13

## Problem and decision

Cold screened s/p/d hybrid KS repeatedly rebuilt geometry-invariant ERIs when
the generic f-shell source's 64-public/128-Cartesian AO thresholds excluded an
otherwise bounded resident inventory. Admit these hybrid sources from 24/24
AOs and eight permitted iterations. Pass the same policy to the capacity
allowance, whose previous 64-AO floor otherwise refused the smaller source.

This is a general exact FP64 scheduling optimization. It belongs to ordinary
Calculator, not a Fast precision preset. The experimental single-FP64 facade
used during measurement does not implement an application-budget-driven
mixed-precision Fast, and is deliberately excluded from this change.

## Invariants and alternatives

- Retain pure-Coulomb KS, generic RHF and f-shell automatic thresholds.
- Retain screened-mask, cold-seed, complete capacity and lifetime admission,
  explicit disable controls and bounded Direct fallback.
- Do not change kernels, arithmetic, convergence or force evaluation.
- Do not compare a future Fast profile only against the slower no-reuse
  control. Its comparator must include this shared optimization.
- Do not admit smaller pure-J sources using unmeasured hybrid profitability.

## Evidence

`benchmarks/results/pbe0-source-reuse-20261013/` preserves the original
endpoint records, reconstruction patch, independent oracle,
GPU tests and memcheck. Ordinary PBE0 FP64 E+F improves from 2.114368 to
1.758106 s (24 AO) and 6.120576 to 3.401016 s (96 AO). Five alternating-order
repetitions include construction, SCF, analytic forces, export, release and
CUDA synchronization. The reference gates are 1e-7 Eh and 1e-6 Eh/bohr.
No mixed arithmetic executes. Census records count canonical source values,
not primitive FLOPs or a complete count of baseline recomputation.

The three reconstructed native changes match this PR apart from formatting,
but the full reconstructed source fingerprint does not match the recorded binary.
The evidence README records both identities and the missing original build-input
inventory; complete measured-build provenance remains unverified. These timings
describe the retained binary, not a remeasurement of the integrated head.

Host boundary tests protect small-source eligibility, capacity refusal and
unchanged pure/RHF/f-shell policies. Submitted GPU cases cover signed RKS/UKS
J/K, partial residency, PBE0/B3LYP states, warm/external seeds, AUTO strict
refinement, chunk widths and complete gradients. Terse pytest receipts preserve
aggregate totals without independently verifiable source/job bindings; retained
zero-error/leak memcheck covers two ordinary-FP64 water calls.

## Revisit when

Larger cold/warm endpoints regress, resource economics change, or a measured
cost model supports additional domains. Precision lowering requires its own
independent error and complete-endpoint performance qualification.
