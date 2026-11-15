# Decision: separate CUDA AOT static audit from global default qualification

Status: implemented
Date: 2026-11-15

## Problem

Issue #3086 P1c/P1d requires an auditable answer for portable-disabled and
SM120-only consumers without inventing architecture qualification. A local
warm winner does not establish acceptable startup, changed-geometry, code-size
or full-endpoint behavior. Source-spelling ratchets also confuse a harmless
refactor with a new policy, while a bare `passed` flag can conceal missing work.

## Decision

Generate a versioned static audit from the existing compiler owners. Enumerate
extra Fock/force rows, changed shared Fock mappings, all declared mixed-Fock
consumers, exact derivative packaging, optional Rys/K-block/task/shared routes
and the unmatched-registry fallback. Retargeted target-plan admission is an
offline experiment, never a production manifest mutation. Preserve FP64 and
mixed/derivative consumer boundaries, optional packaging and native fallback.

For SM80/A100, SM90/H100 and SM120/RTX5090, distinguish executable static
legality from actual NVCC resources, independent device numerical acceptance
and all endpoint phases. Missing current-source evidence remains explicitly
not-run. Shared force declarations do not imply identical Fock algorithms or
binaries. The retained `fpps` force selection remains provisional under the
existing f-shell owner. Historical A100/H100/f-shell receipts stay identified
as historical, not fresh measurements of this implementation.

The generated table contains 72 per-consumer features plus six optional-route
groups and the common primary-force contract. Its count describes audit
coverage, not a count of qualified device implementations.

Retain the P1b stock manifest as an integrity-checked historical input. Compare
effective per-target schedules, recurrences, capabilities, preferences,
packaging and target facts, not source function bodies. A semantic stock default
change requires source-matched complete cold/warm/moved/moved-warm campaigns,
independent full energy/force references, actual work counts and nonregression
of resources, peak storage, first module use and package bytes. Check actual
raw cubin output with the existing resource parser, bind acquisition records to
the measured run, and verify retained input/reference/build artifacts.

Freeze effective semantic policies as well as the original manifest: resolving
an old manifest with a new compiler could apply a changed implicit choice to
both sides and conceal the default change. A new target starts from an empty
AOT inventory and explicit bounded native fallback, not an assumed portable
qualification. The snapshot is declarative policy, never generated source text.

The local HF warm gate and ordinary `auto` resolution remain unchanged.
Refreshing static audit JSON is deliberately incapable of accepting a
promotion. Control discovery still owns native control boundaries; the separate
AOT decision checker owns semantic default changes rather than another
hardcoded class whitelist. Inert flags and removal of an explicit preference
identical to the shared set are not scientific default changes.

## Rejected alternatives

- Enable all 22 extra Fock rows, mixed consumers or exact derivatives on every
  target because an IR can be built: misses numerical, resource and endpoint
  gates and conflates a target with a particular GPU card.
- Retain source text/AST fingerprints: prevents harmless compiler refactors
  without establishing measured semantics.
- Accept a local warm result or precomputed `passed` flag as global authority:
  excludes startup/moved behavior and can hide repeated semantic work.
- Require a new GPU performance campaign merely to preserve the existing
  defaults during the structural repair: delays the independent P1a correction
  and falsely suggests the retained historical choices were freshly certified.

## Evidence and boundaries

Executable checker tests use explicitly synthetic records, including negative
cold/moved, oracle, resource, integrity, work and provenance cases. They are
not device qualification. Real-device work uses finite Slurm allocations and
retains its own source/binary/device records. Available local nodes provide
RTX5090 and RTX PRO 6000 Blackwell GPUs; a finite n2 discovery job reports
compute capability 12.0 for the PRO 6000. That observation is not a performance
equivalence claim or an A100/H100 numerical run.

The focused implementation suite passes 494 tests, with one existing skip and
one full-native-library identity test deselected. This includes negative
qualification cases, the complete raw receipt chain, nonzero C ABI route masks
and the original build/profile resolver checks. Compiler structure, native API
contracts/boundaries, Libxc retained-evidence revalidation and control inventory
checks pass without changing scientific source identities or historical outcomes.
The additional regressions cover frozen baseline semantics, new explicit and
wildcard compatibility domains, and an opt-in named experiment that leaves
ordinary auto selection unchanged.
Raw per-kernel resource comparison prevents an unrelated maximum from hiding
a newly spilling/register-heavy consumer. Explicit frozen-density warm replay
keeps genuine zero SCF/Fock work; that exemption never covers cold or moved
convergence and does not permit artificial positive counters.

This work does not change methods, quadrature, thresholds or numerical
precision to manufacture a win. It does not dismiss #3084 review objections,
merge PRs, publish a release, or relabel earlier benchmark populations.

## Revisit when

Fresh source/device-matched raw evidence qualifies an explicitly proposed
default or new architecture. Keep local workload/device scope and global
production scope separate; architecture identity alone does not prove equal
performance on different cards. Cross-link foundational #8/#9 and ongoing
performance owner #2976 rather than duplicating their numerical work.
