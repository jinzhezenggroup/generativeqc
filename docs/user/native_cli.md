# Native CLI without Python

GenerativeQC's computational runtime is a native C/C++ shared library. A native SDK
install also provides a `generativeqc` executable, so the target machine does not need
a Python interpreter to discover methods or run the native CLI endpoints.

Python is still a **build-time** dependency for source builds because repository
code generation is currently implemented in Python. This is separate from the
installed runtime dependency.

## Install a native runtime

For a CPU build:

```bash
cmake -S . -B build -G Ninja -DGENERATIVEQC_ENABLE_CUDA=OFF
cmake --build build
cmake --install build --prefix /opt/generativeqc
```

The installed executable uses a relocatable RPATH to find the adjacent
`libgenerativeqc`, so an ordinary prefix install does not require Python or a
package-specific `LD_LIBRARY_PATH`.

## Discover native methods

```bash
/opt/generativeqc/bin/generativeqc methods
/opt/generativeqc/bin/generativeqc methods --json
/opt/generativeqc/bin/generativeqc --version
```

Method discovery comes from the same generated native method registry used by
the C and C++ APIs.

## Run GFN2-xTB from XYZ

The first direct native CLI calculation path is GFN2-xTB. It is a useful
Python-free product boundary because GFN2-xTB owns its intrinsic minimal basis
and therefore does not need the Python named-Gaussian-basis resolver.

```bash
/opt/generativeqc/bin/generativeqc run molecule.xyz \
  --method gfn2-xtb \
  --backend cpu \
  --charge 0 \
  --multiplicity 1 \
  --forces
```

XYZ coordinates default to Angstrom. Use `--units bohr` for Bohr input.
Energies are Hartree and forces are Hartree/Bohr. Add `--json` for
machine-readable output. Native CUDA SDK builds may select `--backend cuda`;
unsupported build/device combinations fail closed.

The native `run` command currently accepts `gfn2-xtb` (and its `gfn2`
alias). Gaussian-basis HF/DFT calculations already have native C/C++ execution
APIs, but the CLI still needs a native named-basis/data resolver before those
methods can accept user-friendly XYZ input without Python.


## Manage local profile activation

The native executable also owns the profile-cache operations that do not need
the Python compiler or autotuner:

```bash
generativeqc profile show
generativeqc profile clear
generativeqc autotune --show-profile
generativeqc autotune --clear-profile
```

`profile show` reports the resolved cache root and active profile index without
probing a GPU. `profile clear` deactivates profiles while retaining immutable
bundle directories that may still be used by live processes. The cache root
uses `GENERATIVEQC_PROFILE_CACHE` first, then `XDG_CACHE_HOME`, then
`~/.cache/generativeqc/profiles`, matching the Python frontend contract.

This migration only covers cache administration. The native `run` path still
uses the library selected by its native installation/linkage and does not yet
discover or switch to a cached local profile library. Native local-profile
selection is a separate runtime migration.

## Stable subcommand namespace

The top-level CLI keeps one stable subcommand layout while implementations move
from Python to the native runtime. The native executable currently owns
`methods`, `run`, and the non-compiling `profile show/clear` operations.
`resources`, `profile install/export/diagnose`, and the actual `autotune`
search remain in the Python frontend because they still depend on Python-side
basis/resource/compiler or bundle-validation machinery.

Invoking one of those remaining operations from the native executable returns a
clear unsupported message. The native runtime does **not** discover, spawn, or
silently fall back to a Python interpreter.
