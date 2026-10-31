# Installation

Source builds require CMake 3.24+, a C++20 compiler, and Python 3.10+. CUDA
builds additionally require a supported CUDA toolchain.

For a Python installation from source:

```bash
python -m pip install .
```

Force a CPU-only build with:

```bash
GENERATIVEQC_ENABLE_CUDA=OFF python -m pip install .
```

For CUDA source builds, set `CUDACXX` to the desired NVCC and select the
target architecture. For example, an Ada GPU (compute capability 8.9) uses
the portable CUDA profile automatically:

```bash
CUDACXX=/path/to/cuda/bin/nvcc \
GENERATIVEQC_CUDA_ARCHITECTURES=89 \
python -m pip install .
```

Use `80`/`86` for Ampere, `90` for Hopper, or `100` for data-center Blackwell.
For an RTX 5090, use `120` to select its measured shell profile. The default
`auto` selection uses a matching tuned or compatible profile when available,
and otherwise uses portable CUDA with untuned profile metadata. Explicitly
requesting an incompatible tuned profile remains an error.
Architecture-specific prebuilt `sm120` archives
cannot execute on older GPUs; see the [build guide](../developer/build.md)
for multi-architecture builds.

For relocated CUDA runtimes, set `CUDA_HOME`, `CUDA_PATH`, or `CUDAToolkit_ROOT`
to the toolkit prefix, or supply library directories through `LD_LIBRARY_PATH`.
Provider discovery also checks the active Python environment's `lib` directory
and NVIDIA's separately installed Python packages.

Verify method discovery after installation:

```bash
generativeqc methods
```

An installed native SDK/runtime can run without Python. See
[Native CLI without Python](native_cli.md) for the command-line workflow.

Developers and maintainers who need CMake presets, portable CUDA profiles,
compiler-cache controls, generated-AOT shard tuning, device linking, split
compilation, or native SDK installation should use the
[build and CUDA configuration guide](../developer/build.md).

Continue with the [quick start](quickstart.md).
