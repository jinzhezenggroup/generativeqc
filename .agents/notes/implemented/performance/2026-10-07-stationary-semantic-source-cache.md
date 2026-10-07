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

### Clean cold endpoint population

Re-audited and rebuilt master
`2b0feff1d5577e58f3a44fbecbc1974be92df4e3` before collecting the population.
The native CUDA library SHA256 is
`06d43f6b61cabc6a26be2c328a51f4effd54b2112097b2706db4f279fc9333a0`.
The population uses source overlay `66bfbf3ef5cdd8467bb54432f1d6ebcb1bec103c`,
SHA256 `4d5bbc9eb5995ca3096a134efa044d0d4fd80782b4b8263b4af40db16f8f8c5b`.
Later bounded-snapshot hardening changes integrity validation, not source
mathematics; final-source qualification is recorded separately below. These
receipts do not qualify subsequent master revisions or their native binaries.

All 36 interleaved fresh-process samples passed: three reference, regenerated
(`GENERATIVEQC_STATIONARY_SOURCE_CACHE=0`) and cached samples at each size.
Every sample includes complete preparation and first energy/analytic-force
execution, excluding imports/context initialization/native Calculator
construction, with persistent binary caches reused. No preparation or source
work is moved outside that timer. The protocol remains full spherical def2-SVP,
PBE0, unpruned `(48,16,32)` moving grids, FP64, unchanged convergence/screening,
and host-return forces. Reference Fock builds suppress both `dm_last` and
`vhf_last`. Native/reference screening remains `1e-12`/`1e-14`; this is not an
equal-admitted-work comparison between engines. All real-GPU commands use finite
Slurm allocations on `n1` (`node1`, main partition, RTX 5090) and preserve
Slurm-assigned visibility.

Complete endpoint medians in seconds, including all three samples per cell:

| Atoms | Reference | Regenerated | Cached |
| ---: | ---: | ---: | ---: |
| 12 | 20.101 | 97.722 | 35.468 |
| 24 | 33.633 | 49.978 | 40.311 |
| 48 | 51.584 | 99.452 | 87.709 |
| 96 | 87.183 | 226.589 | 195.367 |

The first cached 12/24 samples are genuine source misses, taking 99.020/50.704 s;
they are retained, not discarded from the medians. Regenerated and cached
artifacts have identical keys/binary hashes, and all 24 native samples execute
zero compiler subprocesses. At 12 atoms the bounded fallback retains 362
requests, 23 units and 23,002,437 primitive bytes, plus the 79,906-byte wrapper.
Regeneration takes approximately 50.6 s primitive and 11.8 s wrapper; both
exclusive generation spans are exactly zero on hits. At 24/48/96 atoms the
existing qualified native derivative route leaves one nuclear primitive unit
(1,550 bytes) and a 73,597-byte wrapper. Its approximately 11.2 s wrapper
generation is likewise eliminated, without changing native-provider admission.

Only the 12-atom regenerated/cached SCF work is identical across the population:
23 iterations/Fock builds per sample. At 24 atoms the counts are `[19,19,19]`
versus `[19,19,20]`, at 48 `[23,23,26]` versus `[23,23,23]`, and at 96
`[24,28,30]` versus `[26,24,25]`. Reference 96-atom counts also vary. Therefore
the full endpoint differences, especially at 96 atoms, are not attributed solely
to source reuse. Histories, semantic work counts and all outliers remain in the
raw receipts; exclusive generation spans independently establish the eliminated
work. The 12-atom hit still spends approximately 18 s constructing the owner,
including approximately 17.5 s of existing inclusive binary-cache validation.
Toolchain/binary validation probes have not been eliminated.

Maximum cold errors are `7.7307e-11 Eh` and `3.0037e-11 Eh/bohr`, within the
independent `1e-8 Eh` and `1e-7 Eh/bohr` gates. Raw receipts, `cold.py`, `run.sh`,
`summary.py`, source/native/overlay identities and `results/SUMMARY.json` are
retained under `/data/jzzeng/qc-issue2071-20261007/`. The cold harness SHA256 is
`4e3deb309ab38941a237a5124885d784072cfa9feae28395894914ebfe95dc1e`.
The earlier interrupted population remains in
`results/initial-source-a6fbdf001/` and is not pooled with these samples.

### Warm, moved and final-source qualification

For all four sizes, independent reference/native benchmark runs pass all four
cold/warm/moved/moved-warm gates. The largest errors across these 16 native
endpoints are `7.4579e-11 Eh` and `3.0023e-11 Eh/bohr`. The existing PBE0/RKS/
FP64/separate GPU force test also passes retained-owner reuse, moved geometry
and independent finite differences. Receipts are `results/warm-native-*.json`,
`results/warm-reference-*.json`, `warm-matrix.log`, `warm-unit.log` and
`warm-unit-evidence/`.

Final source `0c0f1ffebae12efe80b6b17f8ed0c4a90239d189` includes the bounded,
single-snapshot cache integrity repair and the re-audited AOT/capacity source
guards. Its verified overlay is built against the pinned master above, not a
moving remote ref: SHA256
`7e5a8234fd291634b87fc33b991de946d4a8dcd7ecfe0af01c6f079ff8db366f`.
Every tracked file is checked against the final-source tree before supplemental
GPU qualification (`results/final-tree-check.log`). Two further independent
12-atom cache-hit processes pass and retain zero source generation (35.379 and
35.266 s). A final-source disabled-cache control also passes (98.342 s), with
matching artifacts and the same 23 iterations/Fock builds. These supplemental
samples are not pooled into the 36-sample table. `final-summary.py` verifies the
inventories, source generation, artifact equality and gates and records
`results/final-SUMMARY.json`. The small-system warm/moved/finite-difference GPU
test passes again. An initial
overlay assembled against a moving shared ref was not accepted as final-source
evidence; its receipts are retained separately under
`results/unverified-overlay-final-gates/`.

Local integrated validation passes 468 tests with one skip, including source
reuse, demand, AOT, capacity, mutation and fallback regressions. The final focused
cache rerun passes 31 tests; compiler structure checks 488 modules with zero
dependency errors, and Ruff lint/format and diff checks pass. Source-only
full-SPD miss/hit receipts on the final source preserve the composition hash
above and must not be presented as complete GPU endpoint timings.

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
