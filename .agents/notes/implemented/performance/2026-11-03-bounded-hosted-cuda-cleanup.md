# Decision: avoid unnecessary and unbounded hosted CUDA disk cleanup

Status: implemented
Date: 2026-11-03

## Evidence and problem

PR #2957 job `125983531776` failed before CUDA installation or compilation.
Its initial filesystem report showed 86 GiB free, but deleting unrelated SDKs
still exhausted the five-minute cleanup step. The cancellation report showed
103 GiB free. This was not a numerical, compiler or CUDA-resource failure.

## Decision

Skip optional SDK deletion when at least 64 GiB is already available. This is
a conservative cleanup heuristic, not an asserted build-space requirement or
a resource-admission certificate. If space is lower or its numeric probe is
unavailable, retain the same Android/.NET/GHC deletion allowlist, bounded by
120 seconds plus a five-second forced-termination grace period. A timed-out
optional deletion warns and reports actual free space; other deletion errors
remain failures. Never clear source trees or compiler caches.
The timeout runs with the same privileges as deletion so it can signal the
complete owned deletion process group.

All configure, compile, resource, link, runtime and packaging gates remain
required. Real build-space exhaustion still fails the build. Executable shell
tests cover hosted-only ownership, high-space and threshold behavior, unknown
space, partial-cleanup timeout and ordinary deletion failure without deleting
any real SDK. This does not establish an end-to-end CI time-saving measurement.
