# Decision: bounded trusted-master CPU cache persistence

Status: implemented
Date: 2026-10-13

## Problem and evidence

The shared CPU cache was successfully saved by master run 38380230797, job
115246765231 at 10:25:20 UTC (493,325,495 encoded bytes), but the next master
run 38383915691, job 115253677140 missed both the exact key and the whole
`ccache-linux-cpu-v4-ubuntu24-x86_64-` prefix at 10:38:55. It rebuilt 1,204
compiler calls and saved 89,773,144 bytes at 11:11:14. The corresponding ccache
size reports were 26.12% and 4.79% of the 2 GB limit. Both used the same cache
action, path, runner image, cross-OS setting and compression. Server-side cache
metadata was unavailable, so eviction, deletion and service failure remain
unproven. Do not describe this repair as a diagnosed eviction fix.

PR native setup had 234 misses while a compared queue run had 186 hits and
48 misses. Their coverage modes differed, so these are an opportunity and a
persistence failure receipt, not a controlled whole-CI speedup measurement.

## Decision

Keep actions/cache first, with its existing v4 path/key and strict ccache
compiler/source/flag identities. Invoke the optional artifact fallback only
when the restore step succeeded and `cache-matched-key` is empty. A prefix hit
is useful primary data, even though `cache-hit` is false, and skips this work.
The four consumers are the master seed, CPU CTest matrix, Python matrix and
CodSpeed CPU job. CodSpeed already had Actions read access; only the other
three job tokens add `actions: read` alongside `contents: read`. No token gets
write access, no credentials are created, and no cleanup permission changes.

The master seed packages only after both native and Python seed builds succeed.
It includes regular cache data in ccache's hexadecimal bucket layout, excludes
configuration/statistics/locks/temp files, and does not trim, clear, or modify
the existing cache. An encoded gzip stream is capped while writing at 511 MiB,
leaving 1 MiB for the manifest and uncompressed upload ZIP wrapper. Decoded
cache data is capped at 2 GiB and 100,000 files. Oversized, empty or failed
packages are not uploaded. Packaging and upload each have an optional two-minute
ceiling. Artifacts expire after one day, with no active deletion job.

Consumers select at most three candidates from the latest 20 successful master
CI pushes. Every candidate must explicitly match the repository and head
repository, `.github/workflows/ci.yml`, push event, master branch, completed
status and successful **whole workflow** conclusion. They verify the exact
attempt-suffixed artifact name, ID, repository IDs, producer SHA/run, non-expiry,
512 MiB encoded artifact cap and available SHA-256 digest. The artifact service
may contain a seed from a workflow that later failed; it is never eligible.
Missing/inaccessible provenance remains a normal cold fallback.

The pinned download-artifact action receives the exact selected ID/run/repository
and `digest-mismatch: error`. A manifest separately binds source SHA, run/attempt,
repository/workflow/event/ref, ubuntu24/X64, ccache version, archive SHA-256,
encoded size and decoded inventory. Toolchain changes remain ordinary ccache
misses; the artifact is transport, not a substitute for compiler identities.

Download and extraction use a new run/attempt-scoped directory under `~/.cache`,
on the same filesystem as the primary cache. No fallback runs if the primary
cache already exists, including a symlink. The extraction parser accepts only
regular USTAR members matching the cache-data path grammar. It rejects path
traversal, absolute paths, duplicates, links, devices, PAX/longname extensions,
truncation, corrupt gzip footers, excess decoded data/counts and appended data.
Parsing ordinary headers directly avoids generic tar readers expanding large
extension metadata before the size check.

Only a successful download outcome can verify, and only a successful verification
outcome can activate. One atomic rename into the absent normal cache activates
all verified data; no CCACHE_DIR override or changed actions/cache save path is
needed. Interruption before that rename leaves the normal cache untouched.
Interruption after it leaves a completely verified cache, never a partial tree.
Cleanup is restricted to the fresh staging path for this consumer run/attempt;
it cannot remove the activated cache or any old attempt. If the whole composite
is cancelled before cleanup, its bounded staging dies with the hosted runner.

Each calling composite step has a two-minute total deadline and is non-gating.
Discovery additionally has a 25-second shell bound and eight-second API calls,
without request retries or pagination; verification has a 45-second shell bound.
The artifact action's own transfer/retry behavior remains inside the composite
budget. GitHub runner v2.337.0 supports composite outcome/continue-on-error and
inherits the parent's remaining timeout into embedded steps. Normal required
build/test steps and their time limits remain unchanged.

## Qualification and limits

Focused tests execute the helper subprocesses and actual composite shell commands,
with controlled API responses and injected failed/cancelled transfer outcomes.
They cover provenance, config exclusion, encode/decode/file caps, corrupt and
hostile archives, preservation of preexisting data, attempt isolation, and a real
shell timeout after verification but before successful-step completion. Workflow
condition tests distinguish primary exact/prefix hits from true misses, preserve
master-only production and outcome-gated activation, and reject negative policy
mutations. These tests do not simulate GitHub's actual artifact service or claim
measured network throughput. Observe the first naturally occurring successful
master publication and true-miss consumption before claiming live effectiveness.

## Cost and revisiting

At the observed encoded sizes, eight Python fallback consumers alone would
transfer approximately 0.72–3.95 GB collectively; CPU CTest, CodSpeed and master
seed consumers add more. At the hard cap, eight downloads total 4 GiB. Every
successful seed may publish up to 511 MiB plus ZIP overhead until one-day expiry,
so storage can grow with master push frequency. No silently expanded deletion or
retention authority is used. Repeated exact-key master seeds still package a
snapshot to keep availability independent of actions/cache; this trades bounded
producer/storage overhead for availability. Revisit cadence and caps with actual
artifact receipts rather than weakening the successful-master trust condition.
A cold or unavailable snapshot can add up to two minutes without any speedup.

## References

- [Successful first seed](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38380230797/job/115246765231)
- [Missing-prefix next seed](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38383915691/job/115253677140)
- [Runner composite schema](https://github.com/actions/runner/blob/v2.337.0/src/Runner.Worker/action_yaml.json)
- [Runner embedded timeout inheritance](https://github.com/actions/runner/blob/v2.337.0/src/Runner.Worker/ExecutionContext.cs)
