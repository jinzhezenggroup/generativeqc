# Decision: qualify the current upstream CuMetal pin before provider changes

Status: implemented
Date: 2026-10-06

## Problem and source review

The old `8a1434f` typed importer refused the first QC kernel's aligned 64-bit
load at byte offset 32 of a 312-byte by-value parameter. One bounded diagnostic
retained the exact 7,051-byte PTX and the CLI error, independently of the failing
scientific gate. Upstream main at review time is
`e87f368060cf09a45b148b4f8892470c74093ebd`, 123 commits ahead of that old pin.
Its typed importer explicitly combines two packed 32-bit words for this exact
aligned 64-bit load shape. Test current upstream before proposing custom fixes.

The reviewed CMake minimum/dependencies, CUDA registration ON with binary shim
OFF, fake-toolkit generation, native registration ABI, and CLI options used by
this lane remain compatible. The reported provider version changes from 0.5.0
to 0.6.0. The newer fake CUDA compiler also disables device strict aliasing to
preserve CUDA type-punning semantics; this makes replaying the old exact PTX
important when separating importer changes from regenerated compiler input.

## Decision

Update only the CUDA-test job's immutable provider commit. Existing toolchain,
CuMetal ccache, and GenerativeQC ccache keys include that commit; the immutable
toolchain cache has no restore fallback. Keep the qualified OS/Xcode/SDK and
all compiler identity checks. The separate benchmark job is unchanged.

Decode a checksum-verified retained PTX fixture and run the selected provider's
strict `cumetal-ir` / `fast48` CLI on those exact bytes, without any legacy
recapture. Preserve compiler identity, provider cleanliness, version output,
input hash, CLI stdout/stderr, and result in a separate artifact. The source
fixture uses base64 solely to preserve its original missing final newline while
retaining ordinary text-file hygiene. A mismatch fails closed.

The comparison runs only on the first attempt of PR 1997's next synchronize
event from the exact `48e7afd` head; it is not recurring merge-queue or scheduled
work. It has a 90-second total / 60-second compiler bound and is separate
from acceptance: the normal native smoke and real QC endpoint gate must still
run and pass. Compilation alone is not scientific or Apple-GPU evidence. The
previous one-shot capture is already complete and its old-before-SHA guard is
not re-armed.

## Precision and acceptance invariants

Keep `fast48`, the production endpoint selectors, independent CPU references,
numerical/convergence checks, per-case GPU provenance, and existing gate/job
budgets. Upstream added some correctly rounded transcendental implementations
for typed `ieee64`, but that is not the selected mode. `fast48` still has the
documented paired-FP32 core arithmetic and binary32 transcendental/libdevice
limitations. This update makes no full-FP64 claim and changes no public QC
defaults or provider source.

## Evidence and follow-through

The old exact input SHA256 is
`4247e547d34e4764cd51ae38ebc82619da07646c53b59e2e0620e9370f755037`.
Its source is the diagnostic artifact from PR head `48e7afd`, tested merge
`5a7368744e4d24fb911d04188662925e51dac0c7`. Local policy/identity tests do not
prove the upstream repair works. Read both the exact-input comparison and normal
real-device CI results before claiming recovery; preserve any next rejection.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/37407047053/artifacts/11387834509
- https://github.com/Lulzx/cuda-metal/blob/e87f368060cf09a45b148b4f8892470c74093ebd/compiler/ir/src/ptx_importer.cpp#L4243-L4260
- https://github.com/Lulzx/cuda-metal/blob/e87f368060cf09a45b148b4f8892470c74093ebd/docs/fp64-policy.md
- https://github.com/Lulzx/cuda-metal/compare/8a1434ffc1a4f7a17d2f8a41e157afbac22af171...e87f368060cf09a45b148b4f8892470c74093ebd
