# Decision: Explicit offline CPU dependency collection

Status: implemented
Date: 2026-10-10

## Problem

Issue #1123's existing probe recipe does not hash downstream dependencies.
`CacheClosure` already defines conditional complete identity and the object store
already validates publication and retrieval; neither discovers actual headers.
Automatically labeling a depfile complete would omit driver children, libraries,
configuration and negative include dependencies.

## Decision

Add a separate, explicit Linux system-GCC C collector. Use real `-M` output,
including system headers; parse escaped paths and continuations conservatively;
hash observed inputs and repeat discovery/checks under one deadline. Preserve
source physical path, ordered flags, native target, specs, actual preprocessor
output and a scrubbed compiler environment as executable-build inputs. Binding
preprocessor output also covers non-including `__has_include` branches absent from
the depfile. Require an explicit caller-owned complete
toolchain snapshot covering the unenumerated inputs; default to incomplete.

The narrow support domain is `/usr/bin/gcc`'s resolved ELF driver, native x86_64 or
aarch64 Linux, `.c` inputs, plain optimization/PIC/C99 flags and absolute include
directories. Shell/alternate wrappers, plugins, response files, environment-based
overrides and other targets are outside the contract. Exact source bytes are
verified against `SourceVariant`; registration provenance remains separate.

## Rejected alternatives

- `-MM`: excludes system headers and cannot justify safe persistent reuse.
- Depfile means complete: ignores downstream toolchain and namespace state.
- Automatically trusting companion-program or `ldd` discovery: those are useful
  minimum evidence, not a full toolchain manifest.
- Retrofitting today's dependencies onto an existing probe object: cannot prove
  which bytes its compilation consumed.
- Modifying runtime JIT, CUDA, object storage or production packaging: expands
  ownership and scientific scope without resolving this collector's boundary.

## Invariants

Unknown coverage produces no reusable key. Observed failures or mutations leave
an incomplete closure or no closure. Never weaken the existing source/hash gates
to accommodate platform text newline conversion. Import must execute no compiler.
Object publication requires an immutable snapshot and matching fresh collection
keys before/after a cached compilation using the identified recipe/environment.
Two observations cannot prove absence of an ABA mutation; snapshot ownership is
the explicit producer trust boundary, not an inferred collector guarantee.

## Evidence

`tests/python/test_bulk_aot_closure.py` contains parser, unsupported-input,
missing-file, mutation, deadline and completeness controls. Linux integration
requires real GCC, emitted nested/system header paths, independently checked file
hashes, a verified cached object compile and existing-store reuse/invalidation.
It also injects a mutation after a real second dependency scan. Windows-local
execution skips real Linux compilation; source-matched repository CI is required
before claiming that gate passes. The existing cache/store tests and compiler
structure checker protect the handoff without changing those implementations.

## Consequences and revisit conditions

Cross-checkout source paths intentionally do not share keys. Manifest producers
must own immutable build inputs and unobserved namespaces, including libraries
and PCH; a hand-built speculative list must stay incomplete. Filesystem hashing
checks deadlines but cannot forcibly interrupt a blocked filesystem operation.
Revisit only with independently tested compiler-specific evidence for additional
drivers/targets, or an immutable filesystem snapshot/namespace manifest producer.

References: #1123; merged identity/store slices #1134/#1137;
`docs/developer/libxc_cpu_aot_closure.md`.
