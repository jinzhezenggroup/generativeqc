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
   (`--shards 8 --unit-mode stable-shards --target-architecture sm_120`).
2. Pins the preprocessed shard-6 translation unit, **line markers included**,
   so the qualified compilation context is part of the identity.
3. Compiles shard 6 with the exact production release flags (`-O3`, `sm_120`,
   `c++20`, `--split-compile=8`, `GENERATIVEQC_HAS_CUDA=1`) through the
   verified sccache/ccache launcher, with a bounded identity-keyed object
   cache whose hits require matching metadata and object/log hashes.
4. Dumps the per-kernel cuobjdump REG/STACK/SHARED/LOCAL table and requires
   exact equality with
   `manifests/maintenance/direct_jk_aot_resource_ledger.json`, including the
   toolchain versions, source/preprocessed/header/manifest identities, flags,
   and the complete kernel inventory.

Any drift fails with per-kernel, per-field attribution. The gate runs in the
`cuda-resources` CI job (ubuntu-24.04, `cuda-nvcc-12-9`, ccache, no GPU), the
same environment every merge gate compiles in — required precisely because
the values are context-sensitive. `--record-output` measures a candidate
ledger for deliberate re-qualification; it never overwrites the committed
ledger and is not itself a gate. The committed ledger values were measured in
that canonical CI context on 2026-10-11 at the PR head that introduced the
gate.

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

- Host tests: 29 fail-closed unit tests (parser, ledger schema, per-kernel
  attribution for every drift class, byte-deterministic generation of all
  eight shards) pass at the PR head.
- CI qualification (PR #2245, canonical `cuda-resources` context,
  ubuntu-24.04, `cuda-nvcc-12-9`, ccache 4.14): the measured ledger covers all
  86 kernels of shard 6 with nvcc `V12.9.86`, ptxas `V12.9.86`, cuobjdump
  `V12.9.82`, generator ABI 1, `-O3`/`sm_120`/`c++20`/`--split-compile=8`.
  The first release compile took 422 s on the CI runner; the gate re-measures
  the cuobjdump table and re-verifies object/log/identity hashes on every run.
- Comparison with the issue reference: the ordinary streaming kernels
  measured in the CI context are dsss `REG=114 STACK=0`, ddss
  `REG=114 STACK=0`, ddps `REG=158 STACK=112`. The register values match the
  issue's node1 matching-master table (114/114/158); the ddps STACK differs
  (112 versus 0). Whether that difference stems from the shard-source
  evolution between `ed8d21684` and `637466c62` (new work-streaming and
  rys-task variants share the translation unit) or from context sensitivity
  is not claimed; it is recorded as an observation. Current master's
  `rys_task_ddss` streaming kernel measures `REG=255 STACK=232` in the same
  context; this is likewise an observation of the current baseline, not a
  claim.
- Negative results retained: the issue evidence root and rationale file are
  absent from the checkout; the cluster gateway rejected non-interactive SSH
  on 2026-10-11 (vlab "connection closed" after banner); no favorable
  machine-code artifact was selected and no cache was cleared to obtain any
  number.

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
