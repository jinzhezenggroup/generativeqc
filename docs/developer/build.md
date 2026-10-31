# Building GenerativeQC

This page collects source-build, CUDA toolchain, generated-AOT, and native SDK
details. For ordinary Python use, start with the [installation guide](../user/installation.md).

## Requirements

Source builds require CMake 3.24+, a C++20 compiler, and Python 3.10+ because
repository code generation currently runs at build time. CUDA builds additionally
require the supported CUDA toolchain; the current production configuration uses
CUDA 12.9.

An installed native SDK/runtime does not require Python. Ordinary built-in
calculations are AOT-first. Explicit runtime JIT/autotuning requires the
documented NVCC/PTXAS developer toolchain.

## Python build from source

`scikit-build-core` drives CMake and bundles the native library into the
installed package:

```bash
python -m pip install .
```

Force a CPU-only Python build with:

```bash
GENERATIVEQC_ENABLE_CUDA=OFF python -m pip install .
```

For a CUDA build, set `CUDACXX` to the desired NVCC and select the target
architecture, for example `GENERATIVEQC_CUDA_ARCHITECTURES=89` for Ada. Targets
without a tuned or explicitly compatible shell profile automatically use the
portable CUDA implementation. `CMAKE_ARGS="-DGENERATIVEQC_AOT_PROFILE=portable"`
forces that implementation even when a tuned profile is available.

The source-tree Python interface finds `build/libgenerativeqc.so`
automatically. An installed wheel loads its bundled library first;
`GENERATIVEQC_LIBRARY` remains the explicit override for a different
development or benchmark build.

Library discovery follows that precedence lazily on every load; an existing
explicit override avoids scanning lower-priority wheel and build locations.
Linux CUDA provider discovery skips directory scans only when every curated
SONAME group has a retained successful handle. Missing or unloadable groups
are retried on later calls. Native ABI binding, CUDA device facts and local
profile selection are still evaluated per load.

Linux CUDA wheels keep NVIDIA user-space provider DSOs outside
`libgenerativeqc.so`, but declare the reviewed CUDA 12 `nvidia-*` packages as
runtime dependencies because NVCC registration runs when the CUDA-bearing native
library is loaded. The NVIDIA kernel driver remains system-owned. See the
[CUDA wheel decision note](../../.agents/notes/implemented/architecture/2026-09-18-provider-free-cuda-wheels.md)
for the retained provider boundary and fallback rationale.

## Native CMake builds

For native development and benchmark builds, configure CMake directly:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
```

For CPU-only development:

```bash
cmake -S . -B build -G Ninja -DGENERATIVEQC_ENABLE_CUDA=OFF
cmake --build build -j10
```

## CUDA build profiles

The generic `cuda-dev-fast` and `cuda-release` presets require a caller-selected
numeric CUDA target and use portable kernels. The development preset uses
compile-oriented NVCC settings. The release preset retains production
optimization settings. Both leave process-level CUDA compile parallelism to
the build tool. For example, build for an Ada GPU with:

```bash
cmake --preset cuda-dev-fast -DCMAKE_CUDA_ARCHITECTURES=89-real
cmake --build --preset cuda-dev-fast
```

Use `cuda-release` with the same target option for performance measurements.
`cuda-release-sm120` and `cuda-release-cli-sm120` are explicitly specialized
Blackwell builds with the measured `sm_120` shell profile. To use that profile
with `cuda-dev-fast`, pass `-DCMAKE_CUDA_ARCHITECTURES=120-real
-DGENERATIVEQC_AOT_PROFILE=sm_120`.

By default, `GENERATIVEQC_CUDA_COMPILE_JOBS` is empty, so native CUDA and
generated AOT compilation follow the build tool's normal parallelism (for
example, `ninja -jN`). Set it only when a memory-constrained host needs a
separate CUDA compiler cap. When that native limit is set, generated AOT work
shares it by default. Set `GENERATIVEQC_AOT_COMPILE_JOBS` only when an
independent AOT cap is desired; explicit limits should match the build host's
available memory.

The fast preset uses bounded NVCC split compilation for native and generated AOT
kernels. The release preset keeps split compilation disabled by default.

Linux NVIDIA builds default to `GENERATIVEQC_PRUNE_CUDA_ARCHIVES=ON`.
The private Direct, GFN2 and native stationary archives first complete their own
device links. Standalone stationary AOT libraries use the same device-closed
archive ownership before their final shared-library link, retaining their public
filenames, entry points and artifact manifests.
`nvprune` then creates separate host-link copies without the resolved relocatable
device inputs, preserving all requested executable cubins, PTX and registration
wrappers. Original archives remain available for device linking and incremental
builds. This changes packaging, not mathematical kernels or architecture coverage.
Set the option to `OFF` to retain the complete archives in final binaries.
Other backends, or toolkits without `nvprune`, retain the original archives.

Linux x86-64 provider-free CUDA wheels place device-image sections after host
code and data using `cmake/cuda_wheel_payload.ld`. This keeps host PC-relative,
GOT and TLS references within the small code model's range even when the image
matrix exceeds 2 GiB. Small registration wrappers remain near host data; all
executable images, PTX and retained relocatable images remain present. The script
applies only to the final host link, with no compression or device-link change.
Configuration requires a linker with `INSERT` script support (GNU ld or lld).

The distributable `cuda-release-cli-sm120` preset **always includes the
entire structurally representable automatic CUDA XC catalog**. It explicitly
sets `GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON` and leaves
`GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS` empty (full catalog, no name
whitelist). Release CI compares generated dispatch cases against the live
structural registry before packaging; a partial bundle must fail the build.
This affects compilation and binary size, not scientific admission: unsupported
KS/force endpoints still fail explicitly. Development builds can opt into
smaller experimental bundles with a nonempty functional selection.


`GENERATIVEQC_ENABLE_CXX_PCH=ON` is an opt-in clean-build experiment that
precompiles only stable standard-library headers for host C++ sources; CUDA
translation units remain outside that PCH. Keep it off with the default compiler
cache workflow unless the cache has been explicitly configured for PCH support.

## CUDA architecture and AOT profiles

NVIDIA builds can opt into `GENERATIVEQC_COMPRESS_CUDA_FATBINS=ON` (default `OFF`). The legacy
fatbinary compressor stores cubin/PTX images compactly during compilation and
device linking, including optional AOT libraries. It preserves the architecture
inventory and executable code. Set the option to `OFF` to restore the toolchain's
storage defaults. This reduces installed size, not mathematical compilation
work; image decompression contributes to first-load latency, so enable it only
after qualifying cold as well as warm endpoints for the intended workload. The newer
`--compress-mode=size` format is not selected, since it requires drivers from
CUDA 12.4 or later independently of CUDA minor-version compatibility.

CUDA 12.9 can build portable generic binaries for `80`, `86`, `89`, `90`, and
`100`. Only `sm_120` currently has a measured generated-shell profile. The
`auto` mode selects an exact or explicitly compatible profile first, then the
portable CUDA implementation, including generated value-only Rys J/K coverage.
The fallback retains portable, untuned metadata; see the
[shell-codegen profile contract](shell_codegen.md) for coverage and fallbacks.
Explicit incompatible profile requests and invalid or ambiguous manifests
remain errors.

The shared native direct-integral archive disables ptxas optimizer constants to
keep its device-linked recurrence graph within the compiler constant-bank limit.
This is scoped to that archive; independently compiled CUDA kernels retain their
normal compilation policy.

A distributable fat binary automatically uses portable kernels for untuned
targets and the measured profile for `sm_120`:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DGENERATIVEQC_CUDA_ARCHITECTURES="80;90;120"
```

Use `-DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF` to omit generated shell bundles
entirely. Use `-DGENERATIVEQC_AOT_PROFILE=portable` to explicitly select the
generic CUDA path for every target, including measured architectures.

Derivative shell registries follow the same target/profile mapping, including
custom manifests. Portable targets and builds with shell AOT disabled retain
the bounded native derivative route. They do not inherit `sm_120` package
metadata or schedules.

## Compiler cache, generated shards, and compile units

For calculation-time module triggers and the distinction between packaged AOT,
Python artifacts and compiler caches, see the
[runtime compilation inventory](../maintainer/runtime_compilation.md).

Builds automatically prefer `sccache` and fall back to `ccache` when either
is on `PATH`. Runtime JIT compilation uses the same precedence on artifact
cache misses and refuses to start an uncached compiler process. `sccache` is a
base Python dependency so a normal `pip install generativeqc`
provides the preferred launcher on supported wheel platforms; a system `ccache`
remains the fallback. An already valid JIT artifact can replay without either
launcher because no compilation is performed.

Runtime sccache requires version 0.16.0 or later and uses a private foreground
server for each finite compiler invocation. Server startup, compilation and
cleanup share the invocation deadline; its compiler children are terminated
without stopping other invocations or the user's shared sccache server. These
private workers lease separate persistent JIT cache slots under
`$XDG_CACHE_HOME/generativeqc/sccache-jit` (default `~/.cache` for the cache home).
`GENERATIVEQC_JIT_SCCACHE_ROOT` can relocate this exclusively owned pool; keep it
outside the shared sccache cache tree. Sequential calls reuse the lowest free
slot, while concurrent calls receive independent stores. Disk-cache environment
options such as `SCCACHE_CACHE_SIZE` apply per slot; the shared configuration
file's `[cache.disk]` policy is not inherited by this separate JIT disk pool.
Configured remote cache backends remain available. Distributed sccache compilation
is refused because remote compiler processes cannot be owned by the local timeout.
An older or unusable sccache executable falls back to verified ccache.

Override native CMake builds with `-DGENERATIVEQC_COMPILER_CACHE=off` or an
explicit executable. That CMake-only escape hatch does not weaken the runtime
JIT requirement.

Generated CUDA is split into eight stable shards by default. Tune this with
`-DGENERATIVEQC_AOT_SHARDS=N` when local compile parallelism or memory is
limited. Stable virtual slots preserve the historical 1/2/4/8-shard assignment
while allowing larger counts such as 12 or 16 to use additional compile buckets.
The base shard map is versioned and based on measured shell-class compile cost,
so manifest insertion/removal does not move unrelated classes. Identical
generated bytes retain their timestamps and compiler-cache keys.

Development builds may set `-DGENERATIVEQC_AOT_UNIT_MODE=class` to expose one
object target per manifest shell class; release presets retain
`stable-shards`.

## Device linking and split compilation

NVIDIA builds device-link a small native direct-integral archive so its launch
owners share retained numerical functions. The angular-force owner retains
whole-program compilation to preserve its launch-bound register ceilings.
Generated kernels and unrelated SCF/DF owners keep independent compilation.

Set `-DGENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=OFF` to compare standalone direct
modules. CuMetal and non-NVIDIA compilers use that standalone path
automatically. Whole-library separable compilation remains optional through
`-DGENERATIVEQC_CUDA_SEPARABLE_COMPILATION=ON`. The requested
`120-real`/`120-virtual` suffix is retained for NVCC while profile directories
continue to use the canonical `sm_120` identity.
Requesting both suffixes retains both SASS and PTX in that target; list order
does not select which image survives. An unsuffixed numeric architecture already
requests both kinds.

Separate stationary AOT build directories can share generated scientific sources
by setting `-DGENERATIVEQC_STATIONARY_AOT_SOURCE_CACHE=/absolute/cache/path` to the
same directory. The default is empty (no persistent AOT source cache). This uses
the [bounded semantic source cache](stationary_cuda_diagnostic.md), separately
from sccache/ccache and from architecture-specific build objects. It covers the
shared s/p primitives, s/p/d shards, legacy wrappers and native scientific
wrappers. It does not cache native binding metadata or installed manifests.
Cache hits retain generator/asset dependency checks and source integrity checks;
corruption fails the build. CMake tracks the full recipe dependency closure even
when a hit avoids importing a lowering module.

Within one multiarch build, Stationary and DF shell sources are already generated
once and compiled for the requested architectures. Direct-shell profiles may
resolve different schedules per target, including a Fock companion selected
during force lowering, so their target-dependent source generation remains
separate. Cross-target reuse requires identical resolved lowering inputs.

For an offline census of Stationary geometry contracts and DF shell mathematics,
run `python tools/audit_aot_reuse.py --output /tmp/aot-reuse.json`. This reports
source/math equivalences and layout differences; it does not qualify shared
kernel dispatch or imply a runtime speedup.

For compile-only CUDA experiments,
`-DGENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS=N` enables NVCC split compilation of
the native direct kernel owners, while
`-DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=N` does the same for generated AOT
owners. Both default to `1` because split compilation can change optimizer
resource choices; use the normal settings for performance and release binaries.

## Prebuilt CUDA wheel coverage

Wheel CI builds native images for `80`, `86`, `89`, `90`, `100`, and `120`, with
PTX for `120` for forward compatibility. It selects portable shell profiles
for the other targets and the measured profile for `120`. This is binary
coverage, not a claim of measured performance parity across those GPUs. Existing
archives explicitly named `sm120` remain architecture-specific; source builds
must select the architecture of the deployment GPU.

## Install a native SDK/runtime

Install a configured native build into a relocatable prefix:

```bash
cmake --install build --prefix /opt/generativeqc
/opt/generativeqc/bin/generativeqc methods
/opt/generativeqc/bin/generativeqc run molecule.xyz --method gfn2-xtb --forces
```

The native executable links the installed `libgenerativeqc` through a
relocatable install RPATH. The first native `run` endpoint is GFN2-xTB, whose
intrinsic basis lets the command consume XYZ directly; Gaussian-basis native CLI
resolution remains a separate capability.
