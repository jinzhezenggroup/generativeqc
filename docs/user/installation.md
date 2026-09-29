# Installation

For a Python installation from source:

```bash
python -m pip install .
```

Force a CPU-only build with:

```bash
GENERATIVEQC_ENABLE_CUDA=OFF python -m pip install .
```

Native development, benchmark builds, CUDA architecture selection, and build profiles are documented in the repository [README](../../README.md).

Verify method discovery after installation:

```bash
generativeqc methods
```

Continue with the [quick start](quickstart.md).
