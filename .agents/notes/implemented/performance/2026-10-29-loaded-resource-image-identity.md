# Decision: retain loaded image identity across resource-planning wrappers

Status: implemented
Date: 2026-10-29

## Problem

The D4 global-budget qualification exposed repeated hashing of a large native
CUDA library when complete scalar/fleet endpoints construct fresh calculators.
An attribute on each `ctypes.CDLL` wrapper avoids repeated reads within that
wrapper, but different wrappers commonly refer to the same loader handle.
A diagnostic full endpoint profile spent 10.50 seconds in library provenance
out of 27.01 seconds, before addressing the separate optional-XC regression.

## Decision

Retain a bounded process-local map of 32 loader handles to their actual CDLL
owners and original binary digests. Holding the owner protects the handle's
lifetime; keeping only an integer could confuse a later reused handle. A new
wrapper for that retained image shares its digest, while its profile identity
remains local. New images are independently hashed. Metadata-only adapters
without real loader lifetime guarantees retain ordinary file hashing.

This retains library wrappers, not scientific calculation handles, contexts or
buffers. Ordinary loaded-image lifetime applies: modifying mapped library bytes
in place is unsupported. Replacing the path must not relabel an already-loaded
image with the replacement file's digest. Eviction bounds additional retained
owners; existing wrappers keep their own identity.

## Rejected alternative

Do not cache on path/device/inode/size/mtime/ctime alone. A local executable test
changed equal-size bytes and restored mtime within one filesystem clock tick;
ctime was unchanged too, and a stat-keyed cache returned the old digest. The
retained loader image, with its lifetime protected, supplies the reuse identity.
No compiler-cache sloppiness or scientific/source identity override is involved.

## Evidence and limits

Host-only tests compile actual shared libraries through the verified compiler
cache. Their returned C symbols independently identify the loaded image while
wrappers and path replacements are exercised. Separate metadata adapters detect
same-size edits with preserved mtime. Reusing the image avoids a poisoned reread
without sharing profile diagnostics.

The original D4 CUDA budget campaign still failed the 1.05 complete-endpoint
gate (maximum 8.62888); CPU budget timing also failed (1.08248). Optional-XC
suppression changes kernel work independently of hashing. This fix alone does
not qualify global D4 performance or close #2227 R2/R6. The failed campaign's
production source is `16a1c897f49962f100aa7e61b71ad462421a9afb`; subsequent runs
must identify their own frontend source and native binary.
