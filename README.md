<!--
IMPORTANT: Keep this README concise and user-facing. It should contain only
the project identity, current supported methods, essential capabilities,
installation, and minimal examples. Put implementation history, kernel
details, benchmark analysis, and extended roadmaps in docs/ or
benchmarks/results/ instead of expanding this file.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/generativeqc-logo-dark.svg">
    <img src="assets/generativeqc-logo.svg" width="640" alt="GenerativeQC — AI-assisted creation and compiler-generated quantum chemistry">
  </picture>
</p>

<h1 align="center">GenerativeQC</h1>

<p align="center">
  <a href="https://codecov.io/gh/jinzhezenggroup/generativeqc"><img src="https://codecov.io/gh/jinzhezenggroup/generativeqc/graph/badge.svg" alt="Codecov"></a>
</p>

<p align="center">
  GPU-native, batched quantum chemistry with analytic forces.
</p>

GenerativeQC uses compiler and code generation workflows to build specialized
CPU and CUDA electronic-structure kernels. Humans set scientific goals,
constraints, and review standards; coding agents help develop the code, tests,
benchmarks, and documentation. Numerical results are checked against independent
references, and performance claims require reproducible gates.

## Features

- **Compiler-generated quantum chemistry.** MethodIR, ProgramIR, and IntegralIR
  turn declarative scientific definitions into validated execution graphs,
  derivative/response programs, schedules, and specialized CPU/CUDA kernels.
  Built-in paths are AOT-first; JIT and autotuning are explicit opt-in workflows.
- **Hartree–Fock and correlated methods.** RHF/UHF provide energies and analytic
  nuclear forces with direct and density-fitted execution paths. Qualified public
  post-HF domains include MP2, RCCSD, and RCCSD(T), behind the same method registry
  and prepared-execution contracts.
- **Broad Kohn–Sham DFT.** RKS/UKS execution covers semilocal LDA/GGA/meta-GGA,
  global-hybrid, range-separated, and nonlocal-correlation compositions. MethodIR
  drives named and automatically imported Libxc selectors; qualified examples
  include PBE, PBE0, B3LYP, r2SCAN, WB97M-V, generic D4 compositions, and
  r2SCAN-3c.
- **Analytic derivatives with fail-closed capability gates.** Supported
  method/backend/property combinations expose native analytic forces, including
  CUDA HF, global-hybrid DFT, WB97M-V, composite r2SCAN-3c, and qualified
  correlated-method paths. Unsupported derivative domains fail explicitly rather
  than silently changing methods or backends.
- **GPU-native prepared execution.** Reusable CPU/CUDA plans retain topology and
  numerical state across calls, with density warm starts, ragged batches,
  per-system failure isolation, CUDA-resident execution, and resource/precision
  diagnostics.
- **GFN2-xTB.** Molecular energies and analytic forces are available on CPU and
  native CUDA SDK builds, including charged and standard restricted open-shell
  states for H-Rn with the method's intrinsic minimal basis.
- **Gaussian basis and ECP support.** Contracted Cartesian and real-spherical
  bases reach `s` through `g` on CPU and `s` through `f` on CUDA. Bundled
  basis data, [offline local/custom basis input](docs/user/external_basis.md) with
  H–Og identities, and [scalar Gaussian ECPs](docs/user/ecp.md) share explicit
  provenance and capability checks.
- **Multiple interfaces, one scientific core.** Python, C, C++, and a Python-free
  native CLI share the native runtime; PyTorch and JAX bindings reuse native
  analytic derivatives for custom backward paths.


## Build and install

Requirements for source builds: CMake 3.24+, a C++20 compiler, Python 3.10+
(build-time code generation), and optionally CUDA 12.9 for the GPU backend.
An installed native SDK/runtime does not require Python.

For a Python installation from source, `scikit-build-core` drives CMake and
bundles the native library into the installed package:

```bash
python -m pip install .
```

Force a CPU-only Python build with `GENERATIVEQC_ENABLE_CUDA=OFF`. For a CUDA build,
set `CUDACXX` to the desired NVCC and select the target architecture, for
example `GENERATIVEQC_CUDA_ARCHITECTURES=120`.

For native development and benchmark builds, configure CMake directly:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

Two CMake presets make the build-time tradeoff explicit: `cuda-dev-fast`
targets one real device image, keeps native CUDA compilation bounded, and uses
a wider generated-AOT compile pool, while `cuda-release-sm120` keeps the
production optimization settings and full AOT manifest.  The presets are starting points; set
`GENERATIVEQC_CUDA_COMPILE_JOBS` to bound CUDA compilation. Generated AOT work
shares that pool by default, preserving the same total compiler bound. Set
`GENERATIVEQC_AOT_COMPILE_JOBS` only when an independent AOT pool is desired; when
set, both limits should match the memory available on the build host. The fast
preset also uses bounded NVCC split compilation for native and generated AOT
kernels; the release preset keeps split compilation disabled by default.
`GENERATIVEQC_ENABLE_CXX_PCH=ON` is an opt-in clean-build experiment that
precompiles only stable standard-library headers for host C++ sources; CUDA
translation units remain outside that PCH. Keep it off with the default ccache
workflow unless ccache has been explicitly configured for PCH support.

```bash
cmake --preset cuda-dev-fast
cmake --build --preset cuda-dev-fast
```

CUDA 12.9 can also build portable generic binaries for `80`, `86`, `89`, and
`90`. Only `sm_120` currently has a measured generated-shell profile. `auto` is
fail-closed: a target without a tuned or explicitly compatible profile is a
configuration error rather than an implicit generic fallback. A distributable
fat binary can opt into portable kernels for untuned targets while overriding
`sm_120` with its measured profile:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DGENERATIVEQC_CUDA_ARCHITECTURES="80;90;120" \
  -DGENERATIVEQC_AOT_PROFILE=portable \
  -DGENERATIVEQC_AOT_PROFILES="sm_120"
```

Use `-DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF` to omit generated shell bundles entirely.
Use `-DGENERATIVEQC_AOT_PROFILE=portable` only when the validated generic CUDA path is
an intentional build choice.
Builds automatically use `sccache` or `ccache` when either is on `PATH`;
override this with `-DGENERATIVEQC_COMPILER_CACHE=off` or an explicit executable.
Generated CUDA is split into eight stable shards by default. Tune this with
`-DGENERATIVEQC_AOT_SHARDS=N` when local compile parallelism or memory is limited.
The stable virtual slots preserve the historical 1/2/4/8-shard assignment while
allowing larger counts such as 12 or 16 to use additional compile buckets.
The base shard map is versioned and based on measured shell-class compile cost,
so manifest insertion/removal does not move unrelated classes. Identical
generated bytes retain their timestamps and compiler-cache keys.
Development builds may set `-DGENERATIVEQC_AOT_UNIT_MODE=class` to expose one object
target per manifest shell class; release presets retain `stable-shards`.

NVIDIA builds device-link a small native direct-integral archive so its launch
owners share retained numerical functions. The angular-force owner retains
whole-program compilation to preserve its launch-bound register ceilings.
Generated kernels and unrelated SCF/DF owners keep independent compilation. Set
`-DGENERATIVEQC_CUDA_DIRECT_DEVICE_LINK=OFF` to compare standalone direct modules;
CuMetal and non-NVIDIA compilers use that standalone path automatically.
Whole-library separable compilation remains optional through
`-DGENERATIVEQC_CUDA_SEPARABLE_COMPILATION=ON`. The requested
`120-real`/`120-virtual` suffix is retained for NVCC while profile directories
continue to use the canonical `sm_120` identity.

For compile-only CUDA experiments,
`-DGENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS=N` enables NVCC split compilation of the
native direct kernel owners, while `-DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=N`
does the same for generated AOT owners. Both default to `1` because split
compilation can change optimizer resource choices; use the normal settings for
performance and release binaries.

For CPU only, configure with:

```bash
cmake -S . -B build -G Ninja -DGENERATIVEQC_ENABLE_CUDA=OFF
```

Then build:

```bash
cmake --build build -j10
```

Install a native SDK/runtime prefix to use GenerativeQC without Python on the target
machine:

```bash
cmake --install build --prefix /opt/generativeqc
/opt/generativeqc/bin/generativeqc methods
/opt/generativeqc/bin/generativeqc run molecule.xyz --method gfn2-xtb --forces
```

The native executable links the installed `libgenerativeqc` through a relocatable
install RPATH. The first native `run` endpoint is GFN2-xTB, whose intrinsic
basis lets the command consume XYZ directly; Gaussian-basis native CLI
resolution is tracked separately. Python is still required when building from
source because repository code generation currently uses Python.

The source-tree Python interface finds `build/libgenerativeqc.so` automatically. An
installed wheel loads its bundled library first; `GENERATIVEQC_LIBRARY` remains the
explicit override for a different development or benchmark build. Linux CUDA
wheels keep NVIDIA user-space provider DSOs outside `libgenerativeqc.so`, but declare
the reviewed CUDA 12 `nvidia-*` packages as runtime dependencies because NVCC
registration runs when the CUDA-bearing native library is loaded. The historical
`generativeqc[cuda12]` extra remains an empty compatibility alias. The NVIDIA kernel
driver remains system-owned. Runtime JIT/autotuning still requires the documented
NVCC/PTXAS developer toolchain. The provider boundary
and fallback rationale are recorded in the
[CUDA wheel decision note](.agents/notes/implemented/architecture/2026-09-18-provider-free-cuda-wheels.md).

## Methods

Stable native ABI IDs, providers and compatibility selectors are generated from
`manifests/public_methods.json`; see the
[native ABI registry](docs/public_methods.md). DFT scientific names and
compositions are discovered from the compiler MethodIR catalog, including
generated metadata from the pinned Libxc sources. `Calculator(method=...)`
accepts qualified `<method>-rks` / `<method>-uks` selectors without requiring
one ABI-manifest row per functional.

Run the Python frontend (`python -m generativeqc methods`) for the current MethodIR-aware\ndiscovery set. MethodIR representation is
not by itself an execution promise: missing primitive lowerers, unsupported
backends/models, or method-specific requirements such as an explicit hybrid
grid fail closed. The Python API also accepts the composite selectors
`r2scan-3c`, `r2scan-3c-rks`, and `r2scan-3c-uks`.

```bash
python -m generativeqc methods
python -m generativeqc methods --json
```

## Python API

Coordinates are in Bohr, energies in Hartree, and forces in Hartree/Bohr.

```python
from generativeqc import Calculator

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
result = calc.singlepoint(
    [
        ("H", (0.0, 0.0, -0.7)),
        ("H", (0.0, 0.0, 0.7)),
    ]
)

print(result.energy)
print(result.forces)
```

Prepared batches retain reusable topology and density state:

```python
from generativeqc import Calculator

systems = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("He", (0.0, 0.0, 0.0))],
]

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
with calc.prepare_batch(systems, warm_start=True) as batch:
    first = batch.execute(strict=True)
    second = batch.execute(strict=True)  # reuses compatible densities

print(first.energies)
```

## Benchmarks

| Method | Performance |
| --- | --- |
| HF (direct / DF) | <a href="benchmarks/results/df-one-step-warm-20260926/hf.svg"><img src="benchmarks/results/df-one-step-warm-20260926/hf.svg" width="900" alt="GenerativeQC versus GPU4PySCF: direct and DF RHF energy-plus-force latency"></a> |

RTX 5090, spherical def2-SVP: complete warm RHF energy + forces, five repeats.
[Protocol and results](benchmarks/results/df-one-step-warm-20260926/README.md).

## Documentation

Choose the path that matches what you are trying to do:

- [Learn quantum chemistry](docs/learn/index.md) — the minimum background needed to use GenerativeQC correctly.
- [User Guide](docs/user/index.md) — install GenerativeQC and run calculations.
- [Reference](docs/reference/index.md) — methods, capabilities, units, and lookup material.
- [Developer Guide](docs/developer/index.md) — architecture, implementation, and extension points.
- [Maintainer Guide](docs/maintainer/index.md) — validation, performance qualification, generated artifacts, and project operations.
- [Agent Guide](docs/agent/index.md) — workflow for coding agents; normative repository rules remain in [AGENTS.md](AGENTS.md).

## License

GenerativeQC is licensed under [GPL-3.0-or-later](LICENSE).
