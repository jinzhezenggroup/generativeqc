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
target architecture, for example:

```bash
CUDACXX=/path/to/cuda/bin/nvcc \
GENERATIVEQC_CUDA_ARCHITECTURES=120 \
python -m pip install .
```

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
