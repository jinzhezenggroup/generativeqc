# Decision: Keep wheel Hessian links provider-free

Status: implemented
Date: 2026-11-21

## Problem

Enabling `GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES` in wheel mode failed at
CMake generation: the Hessian archive unconditionally requested `CUDA::cudart`,
but a provider-free wheel configures from compiler headers without that target.
The native Hessian CLI also performs its own CUDA buffer and stream operations;
the wheel core's hidden import trampolines cannot satisfy executable references.

## Decision

In wheel mode the static Hessian archive propagates the existing provider-free
interface. Each final binary that references CUDA symbols owns its own strict
host link and discovered lazy imports. A native SDK continues to link the
toolkit DSOs. Neither build changes the integral inventory, the accepted SCF
state, the FP64 stationary derivative, nor the runtime policy against JIT and
CPU numerical fallback.

Do not create a fake `CUDA::cudart` target, link the builder's CUDA DSOs into
the wheel, export the core's private trampoline symbols, or silently drop
the compiled Hessian from the CLI. Keep the opt-in until package size and
build time have been qualified for the shipped architecture matrix.

## Evidence

The initial configure failure and repaired SM120 wheel generation are
retained under `build/hessian-wheel-sm120/`. Both generated final link
commands include the default full factorized Hessian archive and strict
Implib resolution, not a CUDA provider DSO or static library. The SDK link
retains its toolkit dependency. The host linker and NVIDIA API regressions
pass (37 passed, one skipped). The installed-artifact checks are separate
from these configure and linkage checks.

The 705-MiB local SM120 wheel (SHA-256
`c96ee1e09fb56bbf5389453f98e0e8aaf852832af3e0ccf4530b7615638c6750`)
was installed into a separate environment without `GENERATIVEQC_LIBRARY`.
The native executable, launcher and core have no direct CUDA-provider ELF
dependencies. With the SDK builder's non-CUDA OpenSSL/LibArchive dependency
closure staged for this unrepaired Linux wheel, 20 real-GPU fitted-LR,
metric-cutoff and ECP CLI/Python tests passed (Slurm 1694). The separate
RKS and UKS f/g matrices (6 and 9 columns) matched the independent-qualified
SDK results. This first wheel precedes the physical CsH confirmation change;
it is not a test of the default release build, manylinux repair, or other
architectures.

After integrating bounded ECP confirmation, the rebuilt SM120 wheel
(SHA-256 `bdbba62a2fc7d778e97943c360219c2d7a372764f26ed4849193909f1c8b8fd8`)
passed 21 isolated GPU tests (Slurm 1698), including physical CsH with the
compiled CLI and public Python API. The rebuilt binaries still have no
direct CUDA-provider ELF dependencies. Local non-CUDA dependencies remain
test-environment inputs, not evidence of a repaired distribution wheel.

The public capability query now checks the loaded library's linked Hessian
entry point and ABI version, using the same check as the executor. An opt-out
build must not advertise CUDA second derivatives just because the method is
qualified. This is a package-availability gate, not a replacement for molecule
preparation and its per-leaf inventory checks. Host tests cover missing
symbols, a disabled build, a wrong ABI and a matching ABI without starting SCF.

`auditwheel show` on the CsH qualification wheel confirmed a `manylinux_2_35`
ABI floor on this builder (`GLIBC_2.35` in `libm`, among other host library
requirements); retagging or repairing it cannot satisfy the project's
`manylinux_2_28` image target. The original audit is in
`build/hessian-wheel-sm120/wheel-ecp-auditwheel-show.log`.

The subsequent pinned CUDA 12.9/glibc 2.28 image build enabled the Hessian
inventory alongside the release packaging options, targeting `120-real`
only. `auditwheel repair` and `auditwheel show` confirmed the
`manylinux_2_28_x86_64` tag. The repaired artifact is 942,908,298 bytes
(SHA-256 `06bf78af4d4b1f3cf3a828ba10d29b2313f83d3a6ce6449806e8b202d0c0ca0d`).
The installed launcher, CLI and core have no CUDA provider `DT_NEEDED`,
and the archive contains no bundled CUDA provider libraries.

Slurm job `1720` on RTX 5090 passed 16 isolated installed-artifact tests,
covering the r²SCAN f-shell path and rVV10, AUTO SCF routing and fitted
long-range WB97M-V. The independent HF/cc-pVTZ HVP took 51.70 s complete
wall time, produced 8,103,728 exact primitive components and matched the
independent result to `4.33e-7 Ha/Bohr²`. Build, repair, digest and linked
GPU logs are in `build/hessian-wheel-sm120/manylinux-qualification/`.
This establishes the repaired SM120 package, not the `80;90-real;120-real`
release architecture matrix or its size/build-cost bounds. No release was
published.

### Standalone release-matrix archive

Building only `generativeqc_hessian_aot` from a fresh build directory exposed
an implicit dependency on the core build: the stationary Hessian translation
unit includes generated D3 tables, dispersion scalar derivatives and D4 method
parameters. A full build could happen to generate these first. The archive
now depends directly on their generators, not on compiling the core library.

With CUDA 12.9.86 and sccache 0.16.0, the corrected standalone build finished
the default orbital-f/auxiliary-g inventory for `80;90-real;120-real`. The
648,646,790-byte archive had 139 CUDA objects, each with SM80, SM90 and SM120
cubins and SM80 PTX. The initial build stopped on the missing D3 header after
10 min 28 s; the incremental retry generated the three headers and completed
in 4 min 13 s. These are diagnostic times, not a cold-build measurement.
Logs, compiler commands, cache statistics and the device-image audit are under
the directory recorded in `build/hessian-aot-wheel-arch.path`. This proves
archive compilation, not final wheel linkage, repair or real-device execution
on SM80/SM90.

The pinned manylinux build also produced its wheel-mode Hessian archive
(648,391,966 bytes). Its generated inventory JSON is byte-identical to the
standalone build, and it has the same image counts across all 139 CUDA objects.
See `wheel-hessian-archive-audit.log` in the directory recorded in
`build/hessian-manylinux-full.path`. This checks the actual wheel build input,
not the final installed binary.
