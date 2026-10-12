# Evidence: specialization-bounded Libxc alias object reuse

Status: proposed independent test/evidence phase for #1123 / #1118
Date: 2026-10-12

## Problem

Existing build-plan tests replace a registration name and provenance to check
deduplication. That is not proof that two real registrations execute through a
reused compiled artifact. Exact-source reuse also needs to remain bounded to
the actual spin, derivative-output, backend and work-domain specialization.

## Decision and Invariants

Use real `LDA_C_CHACHIYO` / `LDA_C_CHACHIYO_MOD` unpolarized E/vxc programs.
They have different import identities but identical emitted programs in this
specialization. Keep both registration records and their numerical/reference
checks separate. A canonical shared source path and a frozen, explicitly
asserted Linux GCC toolchain/header closure bracket one object-build request
through the verified compiler cache (a cache hit is permitted).
Publish it through the existing immutable object store, remove the original
object, and resolve the second registration through a fresh complete closure.
Link and execute the stored object against independent pinned Libxc energies
and potentials for both names. Closure collection still performs compiler
preprocessing/queries; this is not an endpoint or cache-hit speedup claim.

The same registrations have different polarized programs and must not reuse
that specialization's build task. Source equality in one spin/output variant
must never collapse scientific provenance, other variants, or numerical
qualification. No production formula, registry/admission rule or cache key is
changed by this phase. No scientific evidence is promoted by the build plan.

## Evidence and Boundaries

The executable integration and polarized negative control pass against the
current master-based source. Reuse goes through real dependency collection,
cached GCC object compilation, object-store publication/lookup and a real
shared-library execution, not a compiler stand-in or fake object.

The preceding owned-WIP discovery scans 544 registrations for only CPU,
unpolarized E/vxc: 221 emitted variants, 213 exact-source groups and eight
reused build tasks. Five groups contain multiple real names, including the
CHACHIYO pair. This is discovery evidence, not qualification of the PR source
or the full backend/spin/output Cartesian product. Its 323 graph blockers and
bounded 64-task plan are retained, not silently promoted. A fresh census of
the independent PR source is recorded separately below.

The independent source starts at master
`3e3e4b7481f26df20872225380b704bc3b29509d`. Its fresh census confirms 544
registrations, 221 emitted CPU unpolarized E/vxc variants, 213 source groups,
eight reused build tasks and 323 retained blockers. Per-registration emitted
source is 2,060,681 bytes versus 2,025,880 unique bytes. The bounded plan selects
64 tasks / 577,765 source bytes; it does not compile those tasks or promote
them. The two new executable/negative tests and the existing package, cache,
dependency, publication and real-import controls pass together: 167 cases.
Receipts and census are retained under
`.cache/1118-runtime/alias-artifact-phase-pr/`; they are not measurements of
another revision or of an installed/production endpoint.

This phase does not prove CUDA closure/reuse, installed bundle selection,
ordinary runtime alias resolution, broad compiled-object sizes, SCF, forces
or the combined Lane F monotone evidence catalog. Keep #1123 open until those
original acceptance gates have concrete current-source evidence.

## Revisit When

Production bundles consume the offline object store, broader specializations
become executable-equivalent, or a complete CUDA dependency collector exists.
Retain actual object execution and independent scientific references when
expanding reuse; metadata-only aliases are not a substitute.
