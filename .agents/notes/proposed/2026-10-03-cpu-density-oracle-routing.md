# Decision: keep fresh density-oracle exports on reference HF

Status: proposed (implemented only in separate local qualification integration)
Date: 2026-10-03

## Problem

The bounded scalar target prototype at1a0b5aa8 changes ordinary primary CPU HF
iteration/finalization. It protects native physical-reference exports by their
existing export flag, but the older NativeSource RHF/UHF density exports initiate
a fresh CPU solve without that flag. Their Python contract explicitly calls the
CPU route a small-system dense oracle used for post-HF canonicalization. These
are not getters of a previously computed primary state and must retain reference
iteration/finalization. The prototype and all its receipts remain frozen.

## Decision

Expose one named internal run_cpu_reference_fock_strategy entry through the
already-used mean_field.hpp interface. Both it and the existing primary entry
share one implementation of the original resolver/CPU guard, reference-capacity
preflight, plan construction and output path. Their only difference is eligibility
for the fixed scalar target callback. The helper is not a public option, environment
selector or ScfOptions policy field.

A private bridge helper copies each fresh density-export request's existing
controls and resolves the identical standard exact/DF RHF/UHF spec. Only the CPU
branches of the two density-oracle exports call the named reference entry.
Preserve default compute_forces=true and first-derivative preparation even though
the legacy API discards forces. Preserve exact nullptr versus DF auxiliary-source
selection, screening, metric threshold, all scalar controls and exception order.
Do not enable export_physical_reference as a solver selector or allocate its
canonical output objects.

Native physical-reference exports and MP2/RCCSD reference generation keep their
existing routes. NUM01 accuracy and SOL01 proposal/hooks are native solvers under
test against separate independent references, so they continue exercising the
primary target. Clarify SOL01's historical 'CPU reference solve' comment. Actual
primary density/snapshot getters keep the primary returned state. CUDA branches
remain unchanged; there is no blanket claim that every posthf-named API shares
one routing rule.

## Owner and resource invariants

PreparedFockPlan construction stays inside its SCF owner. The rejected first
proposal would include fock_prepared.hpp from posthf/bridge.cpp, introducing a new
post-HF→SCF edge forbidden by the repository's frozen architecture debt ceiling.
Do not relax that guard. The existing mean_field.hpp edge is sufficient.

The reference and primary entries share all old CPU admission and lifetime checks.
The density bridges construct fresh options with no caller-supplied resolved spec,
export request or seed. Identical standard-spec construction makes the original
exact wrapper's standard-exact check true by construction; fitted metric validity
still comes from the same resolver. Exact requests continue ignoring unused bad
metric thresholds, while DF requests reject them. No mathematical kernel, cap,
shared default, validation gate or resource estimate changes in this integration.

## Qualification required

Execute four exact/DF RHF/UHF oracle routes with actual reference-leaf observation
and density/scalar byte parity to clean main. Exercise malformed and nonconverged
requests with preserved output sentinels, including exact-invalid-metric versus
DF-invalid-metric behavior. Prove NUM01/SOL01 and ordinary primary route/data
invariance against frozen1a, and rerun native/resource/structural hooks without
new dependency allowances. Source-only approval does not replace these tests or
real-device CUDA qualification. No publication or default readiness is implied.

## References

- Prototype base8ac08f0a, frozen candidate1a0b5aa8, treea36ef61f
- Previous rationale: 2026-10-03-bounded-scalar-hf-target.md
- tools/generativeqc_posthf/sources.py and export.py: fresh oracle purpose
- tools/generativeqc_numerics/audit.py: NUM01 native solver under test
- tools/generativeqc_scf/solver.py and docs/developer/scf_proposals.md: SOL01
- tools/generate_scf_proposal_references.py: independent PySCF generator
