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

## Additional source-matched compile observations

The P1b native source identity
`56fde0d6534cbfd3c003df9ca385c08e50f46c4ae0d3586d45eee31f76288eef`
compiled all 44 primary selections, audit-only, for each of SM80 and SM90
under finite Slurm job 5297 on n2. All 88 translation units compiled; raw
CUOBJDump observations are retained in the policy evidence directory. This
includes declared mixed/streaming primary companions but not optional
competitors or exact SR/LR derivative packages. The compile host's SM120
PRO 6000 GPU was not used to execute the candidates and cannot qualify
A100/H100 science or performance.

The current-source table may attach only covered compiler/resource observations;
on a source change they become historical automatically. Exact raw compiler
inputs and objects remain in ignored acquisition storage, with their identities
and deterministic source/emission contract retained. Input regeneration is an
explicit opt-in check rather than a permanent refactor blocker. This preserves
provenance without adding an entire generated-source bundle to the repository's
bounded evidence budget. The initial unavailable-GCC12-host-toolchain failure
remains separate from the corrected GCC11 acquisition. No default is promoted.

Two source-matched SM120 Release libraries now pass the actual native profile
test, including the formerly rejected forced-portable case, plus loaded-source
and invalid-descriptor ABI checks. Finite Slurm job 17464 retains 16 distinct
RHF processes on one RTX 5090, independent Cartesian PySCF references and all
four endpoint phases. SVP passes all 32 phase observations. TZVP cold/warm fails
the unchanged force gate in both profiles (about 6.76e-7); moved phases pass.
These failed stock outcomes are executable evidence, not overwritten metadata.
Separate explicitly nondefault SCF-convergence diagnostics reduce both profiles'
force errors to about 1.12e-10 while increasing cold iterations from 9 to 16.
That supports a convergence-limited discrepancy, not a portable-only force bug;
it does not justify a production tolerance change or replace stock acceptance.
The route-level smoke lacks per-shell execution and valid quartet counts and
therefore cannot become a global-default promotion or #3084 cohort substitute.

## Current inventory and second-card requalification

After master expanded the source inventory, the earlier acquisition became
historical rather than being relabeled. Fresh P1b commit
`cacd7164584993ee789b4d97c789ed840a962110` has native identity
`8e117f84a925090780f18408bc396236cc3e13cc39270e3cab276c1cdb698b9f`.
Finite n1 job 17468 rebuilds both Release profiles and passes both native
profile tests; job 17469 replays the full 16-process endpoint smoke. Finite n2
job 5298 recompiles all 88 SM80/SM90 audit units successfully. These are actual
new acquisitions, not metadata edits to earlier binaries or compiler reports.

Finite n2 job 5300 loads the exact same two ELF files on RTX PRO 6000 Blackwell,
also SM120, and repeats all 16 fresh processes. Both cards reproduce the same
stock TZVP cold/warm force failures in both profiles; both pass SVP and moved
TZVP phases. This excludes a 5090-only observation for that discrepancy, not
an SM120-versus-other-architecture cause or a card-only performance attribution.
The source-matched n1 diagnostic in job 17470 still reaches roughly 1.12e-10
with explicit tighter SCF requests and is not stock acceptance.

The first n2 load fails in job 5299: its system glibc 2.35 cannot satisfy the
n1 ELF's glibc 2.38 requirement. Preserve that environmental failure. Use a
private, checksum-recorded n1 glibc 2.39 runtime and explicit loader invocation
for Python and every native/reference child; never replace system libraries or
override scheduler visibility. Shared source/binary identity does not erase
different host/Python conditions, so these diagnostics establish no GPU-only
speed ranking. Available authorized n4/n5 resources are also RTX5090/SM120;
there remains no authorized A100/H100 execution in this campaign.

Master's additional evidence leaves insufficient room for duplicated receipts.
Losslessly compress the frozen manifest, check its original byte identity,
retain replay-consumed raw results, and deduplicate transferred build receipts
against their hashed n1 origin. Heavy unused raw command/resource text remains
in ignored acquisition storage with hashes. Earlier complete bundles remain at
published commit `3958635e7687537d50ae3f61689ba257416b872c`, not a new external
archive. Preserve all positive and negative outcomes without raising budgets,
dropping reference tests or manufacturing default qualification.

## Python-version-independent replay reductions

CI run 43702787870 rejects the fresh audit under Python 3.11 although Python
3.13 reproduces its bytes. Replaying the exact same retained arrays identifies
one translation residual: `1.4221263056057865e-14` versus
`1.422169673692686e-14`. Python 3.12 changed builtin floating-point `sum` to a
compensated algorithm. Do not round evidence, relax the stale-audit check or
relabel outcomes to conceal that interpreter dependency. Use explicit
`math.fsum` for both endpoint replay and the global translation gate, with
synthetic exact-cancellation regressions and both interpreters. No raw forces,
independent references, production kernels or scientific limits change.

## Revisit when

Fresh source/device-matched raw evidence qualifies an explicitly proposed
default or new architecture. Keep local workload/device scope and global
production scope separate; architecture identity alone does not prove equal
performance on different cards. Cross-link foundational #8/#9 and ongoing
performance owner #2976 rather than duplicating their numerical work.
