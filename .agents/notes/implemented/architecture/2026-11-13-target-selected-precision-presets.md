# Decision: select precision execution by resolved targets, not preset names

Status: implemented
Date: 2026-11-13

## Problem

The user clarified that a useful optimization must not belong to Fast merely
because only the relaxed target benefits. Standard/Fast should supply defaults;
the actual convergence/error target must choose the route. Shared constructors
were insufficient: Python rejected arithmetic budgets outside Fast, and native
candidate classification still received a `fast_profile` privilege flag.
The [uniform anchor-default trial](../../rejected/2026-11-13-shared-fp64-anchor-default.md)
also enabled reuse for strict goals without measurable benefit. Its rejected
performance decision remains historical and must not be rewritten into a pass.

## Decision

Resolve Fast's default `TargetErrorBudget` once. Accept an explicit allowance
with any precision preset. All subsequent Standard/Fast budget admission,
transport, scalar/batch candidate selection and classification use that resolved
allowance and actual caller SCF ceilings, not the preset name. The native
candidate context now contains `budget_requested`, `budget_native` and
`budget_fock`; its size/layout and versioned checked ABI remain unchanged, and
repository consumers migrate directly without obsolete field aliases.

Equal target contracts receive equal execution. A Standard preset with the
default Fast allowance can use every scientifically admitted budget route;
a Fast preset with a tighter allowance cannot silently inherit the relaxed
default. Standard without an allowance remains its optimized FP64 endpoint.
Explicit observable audits still execute independently for either preset.
Audit-oriented presets retain their explicit audit semantics; this does not
equate audit requests with unaudited qualification.

CBS retains its workflow-specific relaxed Standard default, but an explicit
Standard budget now overrides it just as a Fast budget does. It must not silently
discard a caller's tighter allowance after the shared policy accepts that value.
The strict Tight/FP64 defaults and independent preconvergence control remain.

Fixed-FP64 anchor reuse is a separate profitability decision. Its common
selector receives actual energy/density/screening goals rather than budget
flags. Automatic reuse requires goals at least `1e-6/1e-4/1e-10` and 192 AOs.
Both strict FP64 and budget arithmetic use the same predicate after scientific
admission. A tighter goal remains opt-in even under Fast; a relaxed admitted
goal may select reuse under Standard. Explicit zero rolls back and one overrides
only the profitability choice, never scientific/resource/lifetime guards.

## Invariants and rejected shortcuts

SCF tolerances are not observable-error certificates. Do not infer a force
allowance or permission for FP32 gradients from a loose scalar SCF threshold.
Keep independently authenticated model/provider/shape admission, actual work
inventories, physical final-state proofs, explicit solver ceilings, bounded
optional storage and failure/force-pressure retirement. Neither profile name
nor allowance presence alone establishes any of these facts.

Do not disable useful relaxed-target optimization to equalize mode labels,
force every optimization into strict-target defaults, or require strict-target
speedups to justify sharing a qualified relaxed-target route. Conversely, do not
advertise historical relaxed-target gains as newly measured Standard speedups.
The frozen earlier trial retains its original source patch, source identity,
96 endpoints and failed profitability decision; it is not current-source timing.

## Validation

Host regressions exercise equal Standard/Fast/FP64 target contracts with relaxed
and tight budgets, independent tight solver ceilings and both native transports.
Scalar and prepared-batch strict fallback tests cover both preset names. Native
candidate ABI/error/forgery tests migrate to budget facts. The anchor controller
exercises strict and relaxed targets for every admitted arithmetic combination,
both sides of the size floor, explicit overrides and malformed/nonfinite input.
Scheduled public CUDA tests compare both presets at equal tight/relaxed targets
against an independent PBE0 energy/force oracle, including same-owner warm calls.
The focused host selection passes **1,009 tests**, with 21 scheduler-gated GPU
skips; this is not the full repository suite. It includes CBS explicit tight
budgets under both presets and migrated incremental-controller consumers. The
compiled public-header/native candidate fixture also passes against the final
normal library.

Slurm RTX 5090 job **17341** passes all four PBE0 cases, eight cold/warm E+F
endpoints. Both relaxed presets execute mixed cold SCF; neither strict preset
does. Independent GPU4PySCF gates are energy/max-force/RMS-force
`1e-8/1e-7/3e-8` for tight targets and `1e-5/1e-4/3e-5` for relaxed targets.
The normal SM120 Release image has fast compilation and test hooks disabled.
Its native source identity is
`eadbbfa8070a69036d22eda37c09b5fc597c3f17e8e446022d689dfe8dcaac23`
over 2,137 inputs; library SHA-256 is
`be4535baba67654d18a1500c7a84d05d050110e1d4bb3fedd2fd9b4e0609d98d`.
The scheduled preflight matches local and deployed inputs, the loaded library's
identity and its exact binary digest before scientific execution. Incremental
normal builds use the verified existing ccache 4.5.1 launcher and retain before/
after statistics without clearing the shared cache.

Slurm job **17342** repeats all four PBE0 cases successfully and additionally
passes both 192-AO WB97M-V preset cases: unset/OFF/ON, each with cold and warm
E+F, twelve anchor-selection endpoints against independent GPU4PySCF. Tight
Standard keeps its strict goals; default Fast keeps its qualified relaxed goals.
The selector's equal-target behavior is separately covered by the compiled host
controller. These are scientific tests, not new timing/profitability samples.
Warnings-as-errors Sphinx, native API/default-promotion inventories and the
frozen-production-source pre-commit checks pass.

Retain failed attempts as well: the first remote copy omitted runtime manifests;
job 17340 then passed the two relaxed cases but failed both strict cases because
the separate packaged PBE0 stationary force artifact was missing. Deploying the
actual PBE0 artifact and manifest resolved that prerequisite; no other method's
artifact was renamed and no scientific gate changed. An initial extra host
command named a nonexistent test file and ran no tests; the corrected complete
selection above supersedes it. A first pre-commit run overlapped later code edits
and reported modified files; only the final frozen-source run qualifies.

Qualification receipts remain in ignored
`.artifacts/target-selected-precision-20261113/`, without new endpoint-performance
claims. The checked Libxc catalog identity is refreshed while its old measured
qualification remains explicitly stale.

## Revisit when

Expand admitted targets or scientific domains only with independent numerical,
complete-endpoint and bounded-resource qualification appropriate to that new
target—not because a preset was renamed or a budget field became nonzero.
