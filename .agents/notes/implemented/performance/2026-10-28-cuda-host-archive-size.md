# Decision: retain device-link inputs separately from host-link archives

Status: implemented
Date: 2026-10-28

## Problem

The single-SM120 XC-sharing validation library still occupied 699.24 MB. ELF
inspection attributed 92% to CUDA images: 328.15 MB to 80 DF shell classes,
189.28 MB to Direct native inputs/output, and 112.33 MB overall to relocatable
device sections. Release symbols/strings explained only 5.89 MB. This was neither
a debug build nor a multi-architecture bundle.

## Decision

Private Direct and GFN2 static archives already own complete device links.
On Linux NVIDIA builds, default `GENERATIVEQC_PRUNE_CUDA_ARCHIVES=ON` creates
separate host-link copies with the compiler toolkit's `nvprune
--no-relocatable-elf`. Preserve original archives for device-link consumers and
incremental builds. Preserve every target's real/virtual architecture intent.
Missing nvprune or another backend retains the original archive; OFF is explicit
rollback. A broken selected tool or an unclosed device archive is an error.

The imported host archive retains CUDA link-language and transitive provider
requirements. GFN2 retains whole-archive registration and the monolithic CLI's
rescan group, now naming the actual selected archive. Temporary output plus
rename avoids publishing a partial host archive after a failed tool invocation.

DF's existing cost policy excludes mixed arithmetic for s/p bra classes. A normal
conditional still instantiated both templated launch expressions, so NVCC emitted
unreachable mixed kernels despite later host dead-code elimination. An
`if constexpr` now discards that expression before instantiation. No admission,
precision, mathematical formula, launch ABI, schedule or angular domain changes.
Private probes can still instantiate every supported mixed class directly.

## Rejected alternatives and invariants

- A linker-script `/DISCARD/` for `__nv_relfatbin` fails: CUDA registration
  wrappers still reference that section. Do not suppress those errors or blindly
  remove allocated sections with objcopy. nvprune updates their payloads safely.
- Do not prune the source archives in place, or feed these copies to a consumer
  that requires unresolved device symbols. The helper requires an explicit
  `CUDA_RESOLVE_DEVICE_SYMBOLS` owner and only private final-host consumers use it.
- Do not treat the imported archive as host-only C++ metadata. Virtual-only
  linkage needs CUDA's implicit runtime libraries; the initial probe exposed
  this missing closure. Keep CUDA link-interface language and evaluate nested
  link-only provider expressions in the original target's context.
- PTX-only archives can have no relocatable ELF to remove. Equal size is valid;
  preserving executable coverage and successful forced-JIT execution is the gate.
- Do not merge distinct high-angular polynomial/Rys or strict/mixed programs just
  because the class count is large. This change exploits proven unreachable
  instantiations; broader mathematical sharing still needs its own ABI/resource
  and complete-endpoint qualification.

## Evidence and consequences

The [retained campaign](../../../../benchmarks/results/cuda-library-size-20261028/README.md)
records matched libraries at 699,244,888 and 537,659,848 bytes: 23.1% smaller.
The independent integrated build is 537,665,248 bytes. Host-archive pruning saves
112.32 MB without changing decoded executable cubins. DF objects save 49.39 MB;
348 kernels become 174 in the affected 20 classes, with every surviving machine
code section identical. These are opt-in historical witnesses, not source locks.

300 paired complete E+F calls retain semantic work and independent references;
all declared phase ratios pass 1.02, maximum 1.008844. Native DF layout/fallback,
HF-DF strict/Fast, Direct-UHF, spdf FCI and GFN2 recovery tests pass. Real-device
archive probes cover multi-SM, PTX-only/forced-JIT, incremental and multi-config
contracts. The receipts preserve initial failed experiments and missing test
dependencies rather than treating them as successful qualification.

Extra host-link copies cost build-directory storage and a short pruning pass;
installed binaries contain only the selected copies' contents. Compilation-time
claims remain bounded: eliminating 174 instantiations removes compiler work, but
this campaign does not establish a controlled full-project cold-build speedup.

Revisit when new targets export device symbols across archive boundaries, CUDA
changes registration/image formats, or another backend receives equivalent
packaging support. Qualify decoded images, device execution and complete endpoint
behavior before extending the supported pruning domain.
