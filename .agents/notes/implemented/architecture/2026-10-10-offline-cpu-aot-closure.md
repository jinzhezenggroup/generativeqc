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
source physical path, working directory, ordered flags, native target, specs,
actual preprocessor output and a scrubbed compiler environment as executable-build
inputs. Binding
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
- Generic shell/C backslash unescaping: GCC's `libcpp/mkdeps.cc` `munge` preserves
  ordinary backslashes and uses GNU make's 2N+1 quoting only before whitespace.
  Real integration includes a literal backslash followed by a space to preserve
  this compiler-emitted path contract; ambiguous trailing names fail closed.
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
The inherited working directory must be part of identity and must stay fixed:
an assembler `.incbin` can select different manifest-covered relative files
without changing source, headers or preprocessor output. A complete input-file
set alone does not bind that selection context.
Compiler-cache identity is also separate from the persistent store key. Return
a deterministic GCC `-frandom-seed` from the unsalted observed closure payload so
manifest/configuration/opaque assembler changes enter the actual compile command
identity. The final closure includes that flag without self-reference. A launcher
must honor it without sloppiness; a cache miss at one layer never proves a fresh
object at another. The producer's complete manifest still owns all opaque inputs.

## Evidence

`tests/python/test_bulk_aot_closure.py` contains parser, unsupported-input,
missing-file, mutation, deadline and completeness controls. Linux integration
requires real GCC, emitted nested/system header paths, independently checked file
hashes, a verified cached object compile and existing-store reuse/invalidation.
It also injects a mutation after a real second dependency scan.
The assembler-input regression compares object digests and embedded payload bytes
after both a cwd change and an opaque input change within the same cwd, using the
verified finite compiler-cache path rather than merely checking closure/store keys.
Latest-head review exposed two further depfile boundaries: a terminal filename
backslash can absorb the dependency separator, and blanket output stripping loses
escaped terminal spaces/tabs. A merged decoy file can make the first misparse look
valid while real header comment changes remain invisible to preprocessing. Parser
negative controls reproduced accepted merged/wrapped tokens and false continuations;
a collector control reproduced a dangling escape after terminal-space stripping.
Reject indistinguishable depfile spellings rather than use file existence as a
disambiguation oracle, and trim only CR/LF record terminators. Real-GCC regressions
retain both merged/wrapped decoys and require no reusable key, while unambiguous
terminal spaces/tabs must remain hashed and invalidate on header comment changes.
This intentionally rejects otherwise valid paths with the same ambiguous spelling;
loosening it requires an independent, non-depfile source of dependency identity.
The separator before a continuation must itself be unescaped: GCC can emit a
directory's escaped space/tab immediately before a literal backslash/newline.
Removing that pathname sequence can resolve to an existing collapsed decoy and
omit the actual header. Odd preceding escape runs now fail closed, with parser
negative controls and real-GCC space/tab directory-decoy regressions.
Windows-local execution skips real Linux compilation; source-matched repository CI is required
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
