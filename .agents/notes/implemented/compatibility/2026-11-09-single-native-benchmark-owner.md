# Decision: qualify the actual native process owner, not only a selected CDLL

Status: implemented
Date: 2026-11-09

## Problem and evidence

An unchanged host-only DF setup clone crashed in A100 LSF230607 before the real
pipeline candidate executed. The retained core mapped both an old GGA default
native library and the explicitly selected qualified normal library. CPU-only
LSF230618 reproduced dependency resolution without constructors: the Python
executable's inherited DT_RPATH resolved the clone's SONAME dependency to the
old image despite a correctly targeted stage symlink and supplied library path.
Python subsequently opened the intended image by absolute path. Hashing only
that selected CDLL incorrectly made the process look like the intended owner.

These observations establish mixed ownership, not an isolated native crash
cause or a performance result. The actual pipeline never executed. Preserve
the original failed experiment rather than replacing its receipts. Detailed
proposal/failure evidence remains in the resident-generation/whitening note;
the issue memory is #2976, comment6735476715.

## Decision and invariants

The shared README endpoint `binary_identity` checks Linux process mappings
before publishing selected-library metadata. All mapped native library segments
must have the selected file's device/inode identity. Repeated segments and
hard-link/symlink path aliases represent one owner. A second distinct native
image, including a deleted image, fails closed. A replaced or unmapped selected
file and unavailable Linux mappings also fail instead of producing a plausible
receipt. Successful receipts retain the actual paths and device/inode alongside
the existing selected-library hash and embedded source identity.

Non-Linux benchmarks keep the existing selected-library contract, with mapping
metadata explicitly unavailable. No GPU probe, solver/precision/grid change,
device-name/architecture selector, performance default or production science
dependency is introduced. Identity collection remains outside endpoint timing.
Unrelated profiling/host overlays and CUDA/provider libraries are not additional
full native owners. This is a narrow benchmark-integrity gate, not a general
proof that every experimental overlay is scientifically or ABI isolated.

## Validation and consequences

CPU-only tests cover segment/path aliases, distinct and deleted old images,
unrelated libraries, replaced selected files, unavailable mappings, non-Linux
behavior, refusal to issue mixed-owner receipts and successful retained metadata.
Existing default-endpoint/reduction/independent-qualification/publication tests
remain unchanged. A corrected experiment must still authenticate its complete
headers, explicit native dependency, scientific results and all unrelated work
before timing or promoting a candidate. This guard cannot rescue old evidence.
