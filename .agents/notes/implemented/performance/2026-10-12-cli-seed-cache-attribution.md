# Decision: attribute native CLI cache statistics to each build stage

Status: implemented
Date: 2026-10-12

## Problem

The native CLI job resets ccache before its fast shared-library seed, then resets
it again before the monolithic CLI build. Previously no report preserved the
seed's counters. If the seed failed, the always-running CLI diagnostics could
also mislabel the surviving seed counters as CLI work.

In queue run [38291875563](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38291875563),
the seed lasted 12m56s and the following fast CLI build reported 519/528 hits.
Those post-seed hits do not establish that omitting the seed would save 13 minutes.
The seed and CLI share 490 object paths; the CLI adds 32 automatic-XC shards and
six CLI sources, while three seed test objects are absent from the CLI build.

Matching source paths and preset names do not establish compiler-key identity.
The preceding CUDA job selected scalar CPU linear algebra, while the CLI job's
installed OpenBLAS changed its emitted provider definitions and include paths.
The monolithic build also differs from the shared seed in packaging and, for a
release-tier job, CUDA compiler optimization. Existing strict preprocessed-cache
hits can still reuse equivalent results across some of these differences.

## Decision and invariants

- Preserve verbose seed statistics and a labeled summary before the CLI reset
- Record successful-reset boundaries independently for the seed and CLI; never
  attribute inherited counters to a stage whose reset did not succeed
- Keep reporting best-effort, including on build failure, without changing the
  configure/build exit status or making a reporting failure a new build gate
- Report selected CMake fields, emitted provider definitions and configured
  compiler identities without executing a compiler or dumping the environment
- Keep all native configure, compile, link, smoke, catalog and package commands,
  backend choices, flags, cache correctness, trust and transport unchanged

## Rejected shortcuts and future work

Removing the seed is not a demonstrated elapsed-time saving: it can move cold
compilation into the CLI and remove a distinct configuration's compile coverage.
Likewise, blindly preferring the discovered trusted-master artifact over the
CLI's current cache is not justified; the artifact may be older.

No cache handoff, extra artifact, new native build or performance claim is added.
Use naturally occurring per-stage counters and complete job timings to decide
whether a later bounded, same-run handoff is beneficial. Any such change must
retain strict compiler input validation and all existing build variants.

## Verification

Executable fake-tool tests check report/reset ordering, separate counters,
reset/configure/build failures, failed diagnostic commands and unavailable
summary destinations. Context tests distinguish an `auto` provider request from
emitted scalar/OpenBLAS definitions and report missing or mixed evidence honestly.
An integration-time parsed-workflow comparison confirms that all non-diagnostic
workflow settings and build/test/package commands are unchanged.
