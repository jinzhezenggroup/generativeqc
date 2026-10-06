# Decision: align the CuMetal QC runner with its AIR deployment target

Status: implemented
Date: 2026-10-06

## Problem

The first real RHF endpoint in CuMetal CI failed before GPU dispatch. The pinned
CuMetal PTX lowerer emitted AIR 2.8 (`air64_v28`), but the macOS 15 runner's default
Xcode 16.4 Metal tools accepted AIR 2.7. The native smoke passed because its
source-first `cumetalc` path did not exercise this registered-PTX JIT boundary.

## Decision

Run the CUDA-test job on ARM64 macOS 26 with the installed Xcode 26.3 and
macOS SDK 26.2. The pinned lowerer's default target is
`air64_v28-apple-macosx26.0.0`; changing only Xcode on a macOS 15 host would not
establish runtime deployment compatibility. Fail closed on a missing Xcode or
unexpected OS, Xcode, or SDK, and log Metal's version before building.

Namespace the immutable CuMetal source/build/install cache by the host/Xcode/SDK
identity with no restore-prefix fallback. Compiler ccache keys also record that
identity; historical ccache entries remain safe fallback inputs because ccache
validates the real compiler, flags, and headers. Do not force compiler identities,
clear caches, or change sloppiness. Reuse a separate toolchain-scoped JIT cache
across the selected endpoints. The separate source-first benchmark job is
unchanged.

## Rejected alternatives and invariants

- Do not assume Xcode 26 alone makes a macOS 26 deployment target valid on macOS 15
- Do not rewrite AIR metadata to impersonate an older compiler ABI
- Do not accept CuMetal's experimental non-executable container as GPU evidence
- Keep all endpoint selectors, numerical checks, per-case dispatch provenance,
  and routine/full time budgets unchanged
- A successful native smoke does not qualify scientific CUDA endpoints

## Evidence

Run 37399815691, head 97453174efa064f6defc0bb0db5ed3bda1502bc9, used image
macos-15-arm64/20260907.0337. The retained endpoint XML identified
`build_shell_primitive_pair_cache_kernel` as the first failing JIT kernel.
`air-opt` rejected `air64_v28`; final packaging reported AIR version 2.8.0 where
2.7 was expected. CuMetal correctly rejected its fallback experimental container.
The original image's official software manifest lists Xcode 16.4 as the default.
The official macos-26-arm64/20260907.0351 manifest lists Xcode 26.3 (17C529),
`/Applications/Xcode_26.3.app`, and macOS SDK 26.2. GitHub identifies `macos-26`
as an ARM64 label. Pinned CuMetal's AIR ABI notes record AIR 2.8 with Metal
language 4.0, and its lowering options specify macOS 26.0. Real-device endpoint CI remains the acceptance
gate; local policy tests alone do not establish GPU success.

## Revisit when

The CuMetal pin changes its AIR output contract, GitHub removes the selected
Xcode, or the host build toolchain is intentionally upgraded. A host build
upgrade must also account for immutable cached CuMetal build-tree identity.

## References

- https://github.com/jinzhezenggroup/generativeqc/actions/runs/37399815691
- https://github.com/jinzhezenggroup/generativeqc/actions/runs/37399815691/artifacts/11385037626
- https://github.com/actions/runner-images/blob/macos-15-arm64/20260907.0337/images/macos/macos-15-arm64-Readme.md
- https://github.com/actions/runner-images/blob/macos-26-arm64/20260907.0351/images/macos/macos-26-arm64-Readme.md
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/compiler/ptx/include/cumetal/ptx/lower_to_llvm.h
- https://github.com/Lulzx/cuda-metal/blob/8a1434ffc1a4f7a17d2f8a41e157afbac22af171/docs/air-abi.md
