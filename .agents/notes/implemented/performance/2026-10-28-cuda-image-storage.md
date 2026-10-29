# Decision: separate stationary device ownership from installed image storage

Status: implemented
Date: 2026-10-28

## Problem

After Direct/GFN2 host-archive pruning, native stationary and standalone
stationary AOT libraries still retained resolved relocatable device inputs in
addition to their executable device images. DF cubins also occupied substantial
space, but common algebra alone did not establish a shareable execution ABI.

## Decision

Apply the existing `nvprune --no-relocatable-elf` host-copy operation to native
stationary's already closed static archive. Give each standalone stationary DSO
a private static device-link owner, then wholly retain its selected host archive
in the public shared library. Keep public filenames, exports, manifests and
original device-link archives. Unsupported backends retain direct shared-library
construction; pruning OFF or unavailable `nvprune` keeps the complete archive.

Compile options, includes and concurrency pools belong to the returned source
owner. Wheel import synthesis belongs to the final shared-library host link:
putting its linker launcher on the private static owner cannot resolve the final
ELF's provider symbols. The owner consumes the provider-free header/interface
requirements separately. The compiled wheel fixture exercises this distinction.

Add `GENERATIVEQC_COMPRESS_CUDA_FATBINS`, **default OFF**, for the legacy
`-Xfatbin=-compress-all` compressor at compilation and device-link time. This is
an explicit installed-size/first-load tradeoff. The original compression trial
failed the declared 1.02 cold-endpoint ratio in DF FP64, DF Fast and PBE-DF at
96 AOs (1.0367, 1.0470 and 1.0239). A three-arm follow-up reproduced compression
costs against pruning alone (1.0414, 1.0367 and 1.0252). Warm behavior and unchanged
instructions do not override these cold failures.

## Rejected alternatives

- Marking DF `contract_shell_task` noinline increased sampled object sizes; do
  not infer profitable ABI sharing from repeated mathematical expressions.
- Modern `--compress-mode=size` compressed samples further but adds the driver
  format requirement documented for CUDA 12.4+. Legacy compression avoids
  introducing that additional requirement.
- `--compress-mode=speed` and `balance` did not improve sampled DF storage.
- Blindly stripping sections from shared ELF does not establish correct CUDA
  registration. Use the toolkit's archive transformation after device closure.
- Promoting compression despite a cold regression is not justified by installed
  size or by point-kernel/steady-state timing.

## Invariants

Preserve requested real/virtual architecture images, device closure, host ABI,
registration, original archives, incremental rebuilds, and compiler-cache use.
Keep device-specific products distinct from generated-source identity. No
scientific lowering, precision policy or runtime launch inventory changes here.

## Evidence and limits

The [qualification receipt](../../../../benchmarks/results/cuda-image-storage-20261028/README.md)
separates the original source campaign from the mainline integration after
#2835/#2852. It retains all endpoint observations, independent PySCF references,
work counters, cold/warm/moved gates, failed trials, binary identities, compact
instruction multiset attestations, actual compiler/cache commands and replay
recipes. Full decoded inventories remain ignored qualification artifacts; their
identities and collector recipe are retained, rather than executable payloads.

Real/PTX/combined/multi-SM/multi-config compiled contracts include forced PTX JIT,
shared-library exports, provider-free wheel linking, incremental rebuilds and
fallbacks. All 32 historical standalone artifacts pass manifest and CUDA
registration audits. Their load witness is not numerical qualification of every
legacy wrapper. No controlled whole-project compile-time speedup is claimed.

## Revisit when

Reconsider default compression only after qualifying complete cold and warm
endpoints on the intended architecture/driver/workload. Reconsider mathematical
sharing when a concrete common ABI and complete-endpoint benefit are established,
including register pressure and occupancy. Reconsider the modern format when the
supported driver floor explicitly permits it.
