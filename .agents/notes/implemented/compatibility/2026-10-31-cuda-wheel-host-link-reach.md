# Decision: keep multi-architecture CUDA payloads beyond host sections

Status: implemented in draft; complete wheel qualification pending
Date: 2026-10-31

## Problem

The six-target wheel's first cold attempt, Actions run `41161686415` at source
`7805c3d6b5221e60f5bc22f3bb8de3fdce3bdb81`, reached 810/936 steps before its
330-minute build ceiling. It retained a 1,440,288,920-byte compressed compiler
cache. The cache-assisted continuation `41165470958`, source
`14ad82a764c8177d046ae6215aff36d9f86aaee8`, reached the final shared-library
link at 934/936 after an 84m06s build step, then failed with x86-64
`R_X86_64_REX_GOTPCRELX`, `R_X86_64_TLSGD` and `R_X86_64_PC32` overflows.
Neither attempt produced a qualified wheel or ran the required payload checks.

Large opaque device-image sections can separate host code/unwind tables from
GOT/TLS/data by more than the small code model's signed 32-bit reach. Wheel
import discovery deliberately uses the CXX final host link, so it does not get
NVCC's host-linker-script handling automatically. CUDA's `nvcc --help` documents
`--host-linker-script=gen-lcs` for this class of relocation truncation.

## Decision

For Linux x86-64 provider-free wheel targets, use an augmenting linker script to
place `.nv_fatbin` and `__nv_relfatbin` after `.bss`. Keep the small
`.nvFatBinSegment` wrappers immediately after `.data`, with their full-width
pointers to the payload. Page alignment preserves read-only payload mappings.
Preserve every real/PTX/relocatable image and every registration wrapper.

Probe the selected host compiler/linker for `INSERT` support at configuration;
unsupported linkers fail early. Use CMake `HOST_LINK` so nvlink never receives
the ELF layout option. Track the script as a relink dependency and source
identity input. Other platforms and ordinary native SDK links are unchanged.

## Rejected alternatives

- `--no-relax` alone does not extend PC32/TLSGD reach.
- Compression is not a correctness fix. Its default remains OFF because the
  retained [cold endpoint gates failed](../performance/2026-10-28-cuda-image-storage.md).
- Archive pruning remains useful but cannot guarantee a six-target image fits
  below 2 GiB. The failed wheel environment had no `nvprune`; that supported
  fallback must still link without discarding `__nv_relfatbin` or target images.
- Increasing timeout or retrying unchanged inputs cannot fix a relocation error.
- Shrinking the architecture matrix before addressing the link layout would
  conflate binary compatibility with the separate recurring-cost decision.

## Evidence and limits

Executable host tests force each device-image section beyond 2 GiB using NOBITS,
avoiding gigabytes of physical memory or file I/O. A middle-placement control
fails with the expected relocation diagnostic; the actual wheel CMake/Implib
path links and loads. TLS isolation, exception unwinding, wrapper pointers,
image extents, read-only mappings, retained PROGBITS bytes and script-triggered
relinking are exercised. Generated CUDA commands retain their device-link path
without the host layout option.

A separate CUDA 12.4.99 compile-only probe retained SM 80 and SM 90 cubins and
compute-90 PTX byte-for-byte across default and adjusted links. This does not
qualify CUDA 12.9's complete six-target wheel or GPU runtime execution. Keep
the PR draft until a complete wheel receipt and the existing cold/warm cost
acceptance are available; neither test scale nor a cache-assisted build proves
that cold recurring builds fit the budget.
