# Generated CUDA host ABI for provider-free wheels

The wheel intentionally builds **without NVIDIA cuBLAS/cuSOLVER developer
libraries** and loads CUDA 12 providers lazily using the existing Implib
linker discovery in `tools/link_cuda_implib.py`. The runtime ABI must
therefore carry the function declarations needed by NVCC.

## Ownership

- `src/runtime/nvidia_host_api/generativeqc_nvidia_host_api.h` owns only
  opaque handles, ABI enums and stable value/type definitions.
- `src/runtime/nvidia_host_api/generativeqc_nvidia_host_functions.h` is
  **generated**. It owns all cuBLAS/cuSOLVER function prototypes and
  NVIDIA `_v2` aliases used in the main native CUDA wheel.
- `tools/generate_nvidia_host_api.py` discovers in-tree CUDA calls and
  reads NVIDIA SDK declarations through Clang's AST, not ad hoc regex
  reconstruction of pointer constness or argument types.
- The separate xTB-native host ABI and CUDA driver functions remain
  independently owned; they are **not** migrated by this change.

## Fast pre-merge check (no toolkit or GPU needed)

```sh
python tools/generate_nvidia_host_api.py --check-usage
```

This is run in the ordinary CPU CI and again before the manylinux wheel build.
Missing functions, including newly introduced batched cuBLAS calls, fail early
instead of waiting for a costly NVCC compile.

## Automatic regeneration from the NVIDIA SDK

Use the CUDA toolkit's runtime headers alongside the installed **matching**
cuBLAS and cuSOLVER development include directories. Only the headers are used;
the generated artifact is committed, not rebuilt or downloaded when installing
or building a wheel.

```sh
python tools/generate_nvidia_host_api.py \
  --sdk-include /usr/local/cuda/include \
  --sdk-include /path/to/nvidia/cublas/include \
  --sdk-include /path/to/nvidia/cusolver/include \
  --write
python tools/generate_nvidia_host_api.py \
  --sdk-include /usr/local/cuda/include \
  --sdk-include /path/to/nvidia/cublas/include \
  --sdk-include /path/to/nvidia/cusolver/include \
  --check-sdk
python tools/generate_nvidia_host_api.py --check-usage
```

The generator selects existing used declarations and any new API calls; it
preserves versioned exported symbols and emits the same output given identical
headers and source. If a symbol or declaration is unavailable, or its parameter
shape is too complex to represent safely, generation stops. A CUDA SDK upgrade
requires review of generated diffs, enum/type additions if any, and the existing
host ABI, Implib, wheel payload and real CUDA smoke tests.

The initial checked-in inventory was bootstrapped from the previously reviewed
handwritten declarations plus the independently reviewed
`cublasDgemmBatched` prototype in #2266. A full real-SDK regeneration and
manylinux wheel qualification are required before declaring the entire SDK
snapshot validated; synthetic Clang tests do not establish GPU numeric behavior.

No CUDA provider shared libraries may become a wheel `DT_NEEDED` edge.
The existing lazy-import mechanism is unchanged, and no numerical kernel,
precision, execution/provider selection or runtime JIT behavior changes.
