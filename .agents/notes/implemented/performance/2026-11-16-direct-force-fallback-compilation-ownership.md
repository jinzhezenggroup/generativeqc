# Decision: share bounded/class-domain force compilation within the RDC archive

Status: implemented; matching manylinux wheel acceptance remains separate
Date: 2026-11-16

## Problem and decision

The two existing force launch owners retain hundreds of duplicate high-order
gradient callees. A dedicated external arithmetic module eliminated too little
work; combining unrelated Fock owners also increased observed compile cost.
The complete investigation and rejected alternatives remain in
`../../proposed/2026-11-16-direct-force-fallback-compilation-ownership.md`.

Within the existing non-Apple NVIDIA-RDC conditional, generate a private
`direct_force_fallback_kernels.cu` including the unchanged class-domain and
bounded-fallback owners. Remove their independent entries from that archive's
actual source list. Keep both original files and independent non-RDC/CuMetal
build ownership. Preserve the separately compiled angular-force WPO owner:
its launch-bound propagation contract must not be folded into device linking.

No mathematics, public launch ABI, angular class, architecture, screening,
precision mode, range/omega admission, source identity/lifetime, queue bound,
counter, retry or bounded fallback is removed. Ninja records both original
included `.cu` files as real dependencies. Executable CMake graph tests cover
the consolidated and independent paths, real/virtual targets and global RDC.

## Complete affected-group evidence

Unchanged `80;90-real;120-real`, compute80 PTX, Release flags:

| Metric | Original two owners | Include-wrapper |
| --- | ---: | ---: |
| Unlinked object bytes | 178,165,680 | 143,452,816 |
| Identically extracted PTX dump bytes | 177,680,559 | 152,347,547 |
| Function copies, sm80 | 960 | 680 |
| Function copies, sm90 | 957 | 678 |
| Function copies, sm120 | 1,058 | 778 |

Objects shrink **19.48%**, PTX dumps **14.26%**. No function identity is added
or removed; all retained resource records match. Every candidate SASS section
matches an original copy. The original sm120 `__internal_accurate_pow` copies
already differ, and the wrapper matches the bounded-fallback copy; do not
misrepresent this as equality of every original duplicate.

Original observed requests are 2,635.695 and 547.543 seconds; candidate
2,867.417 seconds, about 9.92% below their sum. Each has three hits/five misses
and requests overlap. These are **not controlled cold-build or whole-wheel
speedup measurements**. Reaped-worker maximum RSS is 5,710,904 / 1,751,424 KiB
for the originals and 6,068,284 KiB for the wrapper. Preserve serial wheel
compilation and the 12-GiB isolation cap; a worker maximum is not a measurement
of aggregate container memory. Device link already coalesced duplicate bodies,
so smaller input objects do not establish a smaller final ELF or wheel.

## Actual native integration and independent endpoints

The actual sm120 Release library/CLI builds and device-links with portable
automatic XC AOT. All **778** retained production functions, resource records
and SASS sections match the qualified full-matrix prototype. The incremental
native build observes 1,865 seconds, zero hits/six misses including two host
objects, and a 6,636,468-KiB reaped-worker maximum. Do not substitute this
single-architecture incremental observation for a cold-wheel comparison.

Patch-stage library SHA256:
`ff2ea8bffc58a1b94faf47b392f98d2052983c51b1d84bdaf3865dc91b0503d1`.
Native source identity:
`15ee03fc52ef55e7a116a4b32e9b3b9313933fc01478076f58132bd7d08350d5`.
The isolated baseline is clean tensor/Fock code `7b727b9f9`, library SHA256
`0bd9fac2f766e080f78b0621d7999f5accfcc4b5e6d553bd17cf4beed0eba489`.

Slurm job **17547** passes **64** ABBA complete independent PySCF endpoints:
RHF/UHF, Cartesian/spherical, cold/warm/moved/restored. Every iteration,
Fock-build and precision-work count agrees. Maximum independent energy/force
errors are about **6.31e-14 / 8.04e-11**; maximum baseline/candidate differences
are **4.44e-15 / 2.44e-15**. Diagnostic execute-time median/maximum ratios are
**0.9998 / 1.0543**; no runtime speedup is claimed.

The same allocated job passes **53** native profile/safety and bounded high-l
tests, including RHF/UHF forced-bounded versus fixed work-domain equality and
an independent PySCF energy/force oracle. The host/CMake/compiler/catalog/CI
suite passes **925 tests**, including all seven actual CUDA graph configurations.
Current audit writers change only current source identities; all scientific,
historical and acceptance fields remain unchanged. The full changed-file
pre-commit run passes. Clean committed-source replay is a separate pre-publication
gate; this paragraph records the patch-stage artifact, not that later replay.

## Clean committed-source replay

Code commit **feaddd7aa46297d057130b98f2c11e0fdefd9db5** is checked out in the
independent native snapshot, clean before and after. The 65-second configure/
incremental replay makes **zero compiler requests**, retaining the exact library
SHA256 and native source identity above. This authenticates the committed
inputs/artifact and native dependency graph; it is not a fresh cold compile.
The actual owner still has identical function/resource/SASS records to the
full-matrix prototype.

Slurm job **17548** repeats all **64** complete independent ABBA endpoints and
all **53** native profile/safety/forced-bounded tests, with no skips or failures.
Maximum oracle energy/force errors are about **6.40e-14 / 8.04e-11**, and maximum
baseline/candidate differences **5.33e-15 / 1.89e-15**. All iteration, Fock and
precision-work counts agree. Diagnostic median ratio is **0.9996**; retain the
worst **1.2262** ratio (RHF spherical moved endpoint, **94.088 -> 115.372 ms**)
rather than claiming uniform timing parity. No runtime or cold-wheel speedup
is established. The follow-up evidence commit changes only this note, not the
qualified compiler/native inputs. Clean receipts are `force-clean-native/` and
`force-clean-qualification/` under the evidence directory.

## Evidence and remaining gates

`/tmp/gqc-wheel-weighted-owners/` retains the full-matrix comparison, frozen
inputs, original/candidate objects and resources, production commands and Ninja
dependencies, patch-stage native library, independent endpoint/work reports,
Slurm receipts, host tests and audit-preservation check. The first GPU transport
attempt fails before execution because its copied source tree has no test
directory; the retained retry creates that directory and completes the gates.

Keep the PR Draft pending matching full manylinux compile/link/repair and
installed-wheel acceptance. Do not delete numerical/architecture/provider or
lifecycle coverage to shorten that gate. The separate f-shell sharing proposal
records why smaller source alone does not justify a new lowering when COSX
objects/registers/stack regress.
