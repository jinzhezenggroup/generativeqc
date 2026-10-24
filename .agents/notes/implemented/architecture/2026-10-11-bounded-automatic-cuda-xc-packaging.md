# Decision: explicit bounded automatic CUDA XC packages

Status: implemented (artifact selection only)
Date: 2026-10-11

## Problem

Default-off automatic CUDA AOT protects ordinary builds, but the opt-in full
structural catalog offers no intermediate package size. Artifact economics must
not become a named-functional scientific whitelist or imply endpoint promotion.

## Decision

Add `GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS` as an explicit semicolon-separated
artifact-content input. Require CUDA and automatic AOT, and reject CUDA compiler
detection that would otherwise fall back to a CPU-only build. Preserve the empty
selection's existing full-catalog opt-in and the default-off AOT option.

The offline generator validates every request before writing any output,
deduplicates names and selects in common structural registration order. Selected
programs preserve canonical mathematics, native codes, descriptor identities,
and ordinary/batched E/vxc launchers. All 32 transport shards are regenerated:
empty shards are inert and shrinking a package removes omitted dispatch cases.
Unchanged output preserves mtimes and existing compiler-cache reuse.

The native null-descriptor path remains the explicit missing-artifact boundary.
It must not invoke runtime NVCC/Maple/Libxc or silently stage CPU science. This
slice neither promotes public methods nor changes pointwise scientific policy.
Executable reuse across distinct provenance aliases and a trusted persistent
binary-cache/toolchain dependency closure remain separate requirements.

## Rejected Alternatives

- A production whitelist of representative names mixes packaging with admission.
- Default eager full-catalog compilation includes unrequested/unadmitted objects.
- Retaining omitted dispatch cases makes contents depend on build-directory history.
- Runtime compilation or CPU/reference fallback violates ordinary native execution.
- Treating the descriptor hash as a complete persistent-cache identity omits the
  toolchain and downstream dependency closure.

## Validation And Revisit Conditions

Host gates exercise deterministic selection, unchanged provider bytes/identities,
unknown and unsupported requests, no partial outputs, shrinking packages, mtime
stability, and the real CMake/Ninja forwarding/dependency and fail-closed guards.
The opt-in three-registration GPU bundle runs existing independent fixed-density
E/Fock/replay gates for LDA_C_VWN_4, GGA_X_APBE and MGGA_X_LTA, plus both-spin
omitted-provider controls. Each real-device run requires finite Slurm allocation
and an exact source/artifact manifest. Object counts and resource measurements
are retained with that snapshot, not generalized into a cold-build speedup claim.

Revisit the fixed shard transport only with complete compile/object/resource
evidence. Packaging can be adjusted independently of scientific admission;
neither interior fixed-density success nor an installed descriptor clears a
separate kinetic-boundary or public analytic-force gate.
