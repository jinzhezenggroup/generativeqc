# Decision: reuse stationary lowering before binary artifact lookup

Status: implemented
Date: 2026-10-07

## Problem

Issue #2071 identified deterministic integral emission and Becke primal/AD/wrapper
construction ahead of the ordinary binary cache. Keeping an owner alive hides
this work, but a new process/owner still performs it. This is different from
prepared-owner replay, Becke device lowering, or extending packaged AOT methods.

## Decision

Keep the integral and method owners separate. The integral schedule owns a
request/domain/shard recipe and its source products. The stationary compiler
accepts that owner as a lazy source provider, invokes it only after strict CUDA
admission, and independently caches the wrapper recipe. Both reuse the common
logical source inventory; the wrapper additionally uses the existing MethodIR
and FunctionalSpec provenance rather than inventing an imported-XC identity.

Persistent entries are bounded UTF-8 sources plus a JSON recipe/byte-count/hash
manifest, not pickled IR or executable Python. A producer publishes a complete
directory by same-filesystem rename. A losing producer verifies the published
recipe and exact source bytes. Missing entries regenerate; corrupt or partial
entries reject before compilation. An explicit source-cache-disabled route
retains byte bounds and the ordinary binary cache.
Each product also validates its expected unit count from the demand/schedule:
removing a manifest record must reject even when the remaining files and their
individual hashes are intact.

Source recipes include generator/dependency hashes, target, strict FP64 policy,
exact primitive demand and ordering, shard width, or method/point-program
provenance, stationary plan and partition iteration count as applicable. A hit
still supplies real source bytes to the original source/header/toolchain/flags/
target/object/link/binary checks. No artifact identity is overridden.

## Rejected alternatives

- Retained-owner-only reuse: cannot address a fresh process or owner.
- Forcing global hybrids through a semilocal packaged artifact: changes the
  qualified method contract instead of fixing pre-cache work.
- Passing an arbitrary recipe directly to a generic binary cache: could conceal
  source or primitive-demand changes and weaken existing integrity checks.
- Importing integral lowering into the method package: violates the checked
  compiler dependency direction. A lazy integral-owned provider preserves it.
- Pickled IR: adds executable deserialization and version-sensitive object
  lifetimes without being necessary to bypass the measured emission/AD work.

## Invariants

- Numerical lowering, source bytes, ordered consumer weights, physical derivative
  ABI and complete-endpoint execution do not change.
- Small-system bounded fallback retains its full primitive inventory. Only the
  existing mandatory-native capability policy may project it to nuclear roots;
  no new native-provider qualification or CPU/oracle production work is introduced.
- AOT selection and source/hash/binary validation remain independent boundaries.
- Dependency verification occurs on every source lookup, not behind a process
  lifetime cache of filesystem hashes. Logical keys contain no checkout paths.
- Complete endpoint timers include source preparation. Source lookup, lowering,
  publication and recipe verification spans are exclusive; binary-cache time is
  inclusive and historical artifact compile time is never current process work.

## Evidence

Device-free regressions exercise independent processes/owners with actual PBE0
wrapper and primitive lowering, forbidding emission, Becke IR construction and
scalar differentiation on a hit. They also cover recipe/method/target/dependency/
ABI/demand invalidation, damaged/partial entries, concurrent producers and
nondeterministic publication, byte limits, disabled reuse, legacy explicit-source
callers, retained fallback roots and mandatory-native failure boundaries.
The source-only DFT-MP capacity qualifier's whole-endpoint source fingerprint is
re-audited for the lazy provider and new telemetry only. Its native requirements,
resource bounds, work windows, reserves and reductions do not change; mutation
regressions continue rejecting changes to those contracts. AOT branch tests now
require the lazy cache provider to remain exclusively in the JIT branch.

A separate CPU-only full-SPD source census has 362 requests, 23 primitive units
and 23,002,437 primitive source bytes, matching the issue's source inventory.
Its actual PBE0 wrapper is 79,906 bytes. Miss lowering takes 60.652 s primitive
and 13.857 s wrapper on this host; the independent hit process takes 0.844 s
including metadata planning and a **fake binary consumer**, with both generation
spans exactly zero and forbidden producers. Miss/hit composition SHA256 is
`804654b31ebcf8cbbe339acf50886e4b3ab465b86a919350094c6b94fa7ec9ee`.
These are source-work evidence, not GPU or complete molecular endpoint timings.

Local CPU receipts: `/tmp/qc2071-host-{miss,hit}.json` and
`/tmp/qc2071-host-cache.py`. Durable validation evidence is retained locally under
`/data/jzzeng/qc-issue2071-20261007/`; GPU population and revision receipts are
recorded there separately from this source-only census.

## Consequences

A first recipe miss still pays the original lowering and finite NVCC processes.
Fresh-process hits retain bounded source reads/hashing and all normal binary
validation. Cache disk usage grows with distinct recipes; entries can be removed
explicitly. Corruption is not repaired by silently trusting a partial product.

## Revisit when

Measured endpoint evidence shows source verification/read costs dominate, cache
growth needs an eviction policy, or a qualified binary cache exposes an equally
strong pre-emission dependency contract. Do not weaken source witnesses merely
to claim a faster cold hit.

## References

- #2071; parent #1895 and evidence owner #1965.
- #1123 imported-XC catalog identity; #663 retained-owner replay;
  #666 qualified semilocal AOT; #1894 Becke device lowering.
- `docs/developer/stationary_cuda_diagnostic.md`.
