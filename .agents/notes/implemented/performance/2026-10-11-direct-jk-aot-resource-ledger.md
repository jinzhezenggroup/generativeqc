# Decision: fail-closed static cubin resource ledger for the direct J/K AOT shard

Status: implemented
Date: 2026-10-11

## Problem

Issue #2101 observed that a default-off J pair-density candidate changes the
static resources of **unselected** ordinary RHF J/K streaming kernels while
preserving their generated mathematical source. On the toolchain recorded by
the issue (CUDA/NVCC 12.9.1, ccache 4.5.1, Release/O3, SM120,
`--split-compile=8`, stable eight-shard AOT layout), the frozen candidate
allocated dsss/ddss/ddps 255/154/204-206 registers with 544 bytes of static
STACK where the matching master build allocated 114/114/158 registers and 0
bytes. Shard 6 (which contains these consumers) is byte-identical between the
two builds. The candidate's six-repeat complete-cold profit gate already
failed; no promotion is justified by that issue.

Probes 6585-6588 established **compilation-context-sensitive resource
allocation**:

- 6585: standalone control and restored-header builds both give 114/114/158.
  Restoring the four optional `GeneratedShellPairStream` members cannot be
  credited, because the unchanged-header control also differs from the
  historical candidate.
- 6586: the actual Ninja-derived command with the unchanged candidate header,
  on a genuine compiler-cache miss, reproduces the high footprint; an
  identical-command diagnostic recache reproduces it again. This is not a
  stale-cache artifact and grants no permission to clear caches.
- 6587: a fixed shadow-header path with fixed command/source/output compiles
  control/restored/control bodies to 114/114/158 with the exact candidate
  header hash.
- 6588: native-header versus identical-shadow-header preprocessing differs
  only in source line markers; removing those markers yields a byte-identical
  2,722,131-byte input (SHA256
  `2a0682c4a54a52fd7c767aeba646dbe54ce4a4cd489e8a674879dfd644a00ad6`).

The responsible NVCC pass is not isolated. The common-stream layout
hypothesis is insufficient, and the issue forbids speculative layout
rewrites. Every production shard's prelude includes
`src/scf/generated_shell_task.hpp` and `src/runtime/compensated_atomic.cuh`,
so a header perturbation recompiles unselected kernels and can flip NVCC
allocation without touching emitted math.

The issue's evidence root `qc-pbe0-j-pair-focus-20261008-ed8d21684` and its
durable rationale file `2026-10-08-direct-j-class-qualified-policy.md` are
not retained in the repository checkout at `637466c624` and were not found on
the local machine; the cluster gateway rejected non-interactive SSH during
this session. The issue text above is therefore the authoritative
reproduction record, and this note records that provenance explicitly.

## Decision

Deliver the fail-closed static resource identity qualification slice
(option 2 of #2101), not a speculative build/codegen change.

`tools/validate_direct_jk_aot_resources.py`:

1. Generates the production shards with the exact CMake generator command
   (`--shards 8 --unit-mode stable-shards --target-architecture sm_120`) into
   the production CMake source path
   `build/cuda-release-sm120/generated/production_shell_kernels`, the same
   directory the real release build compiles from.
2. Pins the preprocessed shard-6 translation unit, **line markers included**,
   so the qualified compilation context (including the production path) is
   part of the identity.
3. Compiles shard 6 with the exact production release flags (`-O3`, `sm_120`,
   `c++20`, `GENERATIVEQC_HAS_CUDA=1`) and the **normal release split
   setting** (`--split-compile` omitted, i.e. the CMake default of 1 thread:
   split compilation changes optimizer resource choices and the release build
   keeps 1). The compile goes through the verified sccache/ccache launcher,
   with a bounded identity-keyed object cache whose hits require matching
   metadata and object/log hashes. ccache `--show-stats` snapshots before and
   after the compile are retained in the cache receipt; sccache's private
   foreground server is never queried for shared statistics.
4. Dumps the per-kernel cuobjdump REG/STACK/SHARED/LOCAL table and requires
   exact equality with
   `manifests/maintenance/direct_jk_aot_resource_ledger.json`, including the
   toolchain versions, source/preprocessed/header/manifest identities, the
   complete typed flag schema, and the complete kernel inventory.

Any drift fails with per-kernel, per-field attribution, and a truncated or
shape-changed ledger is itself rejected before comparison. The gate runs in
the `cuda-resources` CI job (ubuntu-24.04, `cuda-nvcc-12-9`, ccache, no GPU),
the same environment every merge gate compiles in — required precisely
because the values are context-sensitive. `--record-output` measures a
candidate ledger for deliberate re-qualification; it never overwrites the
committed ledger and is not itself a gate. The committed ledger values were
measured in that canonical CI context on 2026-10-11 at the PR head that
introduced the gate.

No mathematical source, precision, AOT layout, or cache behavior changes;
the compile always uses the verified compiler-cache launcher and no cache is
cleared or sloppified.

## Rejected alternatives

- **Speculative layout/stream rewrite**: #2101 explicitly forbids landing a
  layout rewrite until the responsible compiler pass is isolated.
- **Pinning the node1 table (114/114/158/0) as the ledger values**: values
  are compilation-context sensitive; the node1 context differs from the CI
  context, and the gate must qualify the context it actually runs in. The
  node1 table remains the issue's reference evidence.
- **Qualifying with `--split-compile=8` (the issue's probe toolchain)**:
  the repository's release build keeps the CMake default of 1 split-compile
  thread, and `docs/developer/build.md` records that split compilation
  changes optimizer resource choices. A ledger compiled with split=8 would
  qualify a different cubin than production builds. The first measured
  split=8 ledger was deliberately retired; under the repaired gate it fails
  closed, which is itself a retained negative control.
- **Generating/compiling at a private scratch path**: because the qualified
  identity pins preprocessed line markers, a scratch path would qualify a
  different compilation context than the production object. The shards are
  generated and compiled at the CMake production source path instead.
- **Full eight-shard coverage in one CI job**: a release-flag compile of all
  eight shards exceeds the `cuda-resources` time budget alongside the
  existing validators. Shard 6 is the issue's evidence scope; the tool's
  qualified shard is a named constant that can be extended after separate
  qualification.
- **PTXAS `-v` logs instead of cuobjdump**: ptxas register lines can disagree
  with the embedded cubin; the existing f-shell gate already established the
  cuobjdump table as the static resource of record.

## Invariants

- The gate fails closed: a missing, malformed, or drifted ledger fails CI
  with attribution; unknown measurements never pass.
- The committed ledger is only changed deliberately, after a `--record-output`
  run in the canonical CI context and review of the full diff.
- Generated mathematical source and precision are unchanged by this gate; it
  measures and pins, it never re-emits or re-schedules.
- Compilation always goes through the verified compiler-cache launcher
  (sccache then ccache); caches are reused, never cleared.

## Evidence

- Host tests: 75 fail-closed unit tests (parser, ledger schema, per-kernel
  attribution for every drift class, complete typed flag schema, truncated
  ledger rejection, launcher-statistics receipts, production split default,
  outer-cache transitive header coverage, workflow failure artifacts,
  byte-deterministic generation of all eight shards) pass at the PR head.
- CI qualification (PR #2245, canonical `cuda-resources` context,
  ubuntu-24.04, `cuda-nvcc-12-9` V12.9.86/82, ccache 4.14): the measured
  ledger covers all 86 kernels of shard 6, generated and compiled at the
  production CMake source path with the release split setting (1). ccache
  `--show-stats` snapshots before and after the compile are retained in the
  cache receipt.
- Repeated controls across contexts (same generated bytes, same nvcc, two
  deliberately different qualified contexts):

  | Ordinary RHF streaming kernel | split=8, scratch path (CI) | split=1, production path (CI) | issue node1 matching-master (split=8) |
  | --- | ---: | ---: | ---: |
  | dsss REG | 114 | 254 | 114 |
  | ddss REG | 114 | 154 | 114 |
  | ddps REG | 158 | 190 | 158 |

  The split=8 scratch-path measurement reproduces the issue's node1
  matching-master register table exactly (114/114/158), while the production
  split=1 path lands the same source in the high-footprint basin
  (254/154/190, dsss STACK 0). This is an independent confirmation of the
  context sensitivity the issue established; no causal attribution to a
  single flag is claimed. The committed ledger pins the production context
  values (254/154/190), because that is the object the release build ships.
- Negative controls retained: the initial split=8 scratch-path ledger fails
  closed under the repaired gate (identity and flags drift), which is the
  intended requalification behavior. The issue evidence root and rationale
  file are absent from the checkout; the cluster gateway rejected
  non-interactive SSH on 2026-10-11 (vlab "connection closed" after banner);
  no favorable machine-code artifact was selected and no cache was cleared
  to obtain any number.
- Rebase control after upstream #2236: a fresh record run on the rebased
  head measured an identity byte-identical to the committed ledger
  (shard source, preprocessed identity, headers, flags, toolchain, and all
  86 kernel resource rows), so #2236's owner/grid changes did not alter
  shard-6 static resources and the ledger was retained unchanged.
- Windows-local caveat: the Windows checkout's CRLF line endings leak into
  locally generated shard bytes, so shard hashes generated on a Windows
  checkout are not comparable to the CI (LF) identity. Only CI-measured
  identities may be compared against the ledger; a local hash difference is
  not evidence of drift.

## Consequences

Any PR that perturbs the AOT header contract, the generated shard sources,
the preprocessed context, the toolchain, or any shard-6 static resource fails
`cuda-resources` until the ledger is deliberately re-qualified and reviewed.
NVCC patch releases can trigger a re-qualification; the failure message names
the drifting component. This is the intended qualification cost, mirroring
the existing f-shell/libxc resource-gate precedent.

## Revisit when

- The responsible NVCC pass is isolated and a control-proven build/codegen
  fix lands; the gate should then guard the fixed invariant and may be
  re-scoped.
- CI budget allows extending the ledger to the remaining seven shards, each
  qualified in the canonical CI context before admission.

## References

- Issue #2101 and its probes 6585-6588; PR #2245.
- `tools/validate_direct_jk_aot_resources.py`;
  `manifests/maintenance/direct_jk_aot_resource_ledger.json`;
  `tests/python/test_direct_jk_aot_resource_ledger.py`.
- Gate precedent: `tools/generativeqc_validation/f_shell.py` and
  `.agents/notes/implemented/performance/2026-09-25-libxc-aot-resource-gate.md`.

## Superseding source-contract repair

The validator now includes CMake's NVIDIA Release `NDEBUG` definition in
preprocessing, compilation, and the recorded flag identity. The previously
measured split-1 ledger above did not include this macro; it remains historical
evidence and is not relabeled. It failed the repaired flag schema as intended,
and an explicit production-context requalification then supplied the committed
ledger that includes `NDEBUG`; all 86 kernel resource rows were unchanged by
the macro, while the preprocessed identity changed as expected.

The configured production `CMakeCache.txt` must declare the qualified eight-way
shard partition, including preset and command-line overrides; missing, ambiguous,
or different counts fail before generation or compilation. Separately, hashes of
the validator and compiler adapter bind object-cache entries to the implementation
that constructs the command, without rewriting scientific ledger identity.

## CMake compile-context binding repair

Review of the rebased gate found three additional production-context gaps. The
`cuda-resources` CMake cache was configured with
`GENERATIVEQC_CUDA_FAST_COMPILE=ON`, even though the validator intentionally
compiled the normal `-O3` release recipe; the validator did not bind its
`--nvcc` executable to `CMAKE_CUDA_COMPILER`; and its ccache invocation used the
repository root plus an absolute shard path instead of CMake's build-directory
working directory, relative generated-source path, and
`CCACHE_BASEDIR=${CMAKE_BINARY_DIR}` contract. The last difference is material
because issue #2101 established sensitivity to source-location line markers.

The workflow now reconfigures the same `cuda-release-sm120` build directory to
normal Release compilation (`GENERATIVEQC_CUDA_FAST_COMPILE=OFF`) immediately
before qualification. The validator rejects a fast cache or a configured CUDA
compiler that differs from `--nvcc`, preprocesses from the CMake build directory,
and performs the cached compile there with the shard path relative to that
directory and `CCACHE_BASEDIR` set to the same directory. The bounded inner
cache records this execution context and will not replay metadata from the old
cwd/base contract.

This repair changed the qualified preprocessed/compile context, so the existing
ledger was not relabeled. Canonical CI record run 38141692779 at `a862d12ed`
compiled a new object through ccache 4.14 (cache miss, return code 0) with the
build-directory cwd/base contract and retained the cache statistics, compile
receipt, NVCC log, cuobjdump output, report, and candidate ledger. The candidate
kept the complete 86-kernel inventory and every REG/STACK/SHARED/LOCAL row
byte-for-byte equal to the prior ledger. Only the preprocessed identity changed
from `37433b5c...` to `e58a8a27...`, and the typed flag schema now records
`fast_compile=false`. The reviewed candidate is installed and gate mode is
restored; the subsequent gate run must still pass on the final exact head.

A later review identified two remaining provenance gaps. A non-default
`CMAKE_CUDA_HOST_COMPILER` would add NVCC `-ccbin` semantics that the qualified
recipe does not record, so the bounded gate now rejects such a cache and the
typed flag identity names the qualified `nvcc-default` host context. CMake also
records the exact compiler-cache executable it resolved. The validator verifies
that path and passes the resulting launcher object through the CUDA adapter,
instead of independently applying the normal sccache-first selection policy.
The launcher remains execution provenance and does not become scientific source
identity, while its implementation hash and retained receipt prevent an old
execution path from replaying silently. This typed-context change requires a
second canonical record run before gate mode can be restored again.
