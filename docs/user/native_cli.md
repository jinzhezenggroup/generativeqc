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

MethodIR compositions have a separate build-time generated catalog. Query the
full list, including backend qualification and unsupported-graph reasons:

```bash
generativeqc methods --compositions
generativeqc methods --compositions --json
```

## Discover bundled Gaussian bases

The native CLI basis catalog is generated at build time from the same bundled
Basis Set Exchange subset used by the Python frontend:

```bash
/opt/generativeqc/bin/generativeqc basis list
/opt/generativeqc/bin/generativeqc basis list --json
```

The installed executable contains generated C++ constants and does not read
`basis_pack.json` or start Python at runtime. Native shell expansion reuses
that generated table. The upstream BSD-3-Clause license is installed under
`share/generativeqc/licenses/` in the native prefix.

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

## Run RHF/UHF with bundled Gaussian bases

The native CLI can also expand the generated bundled basis catalog directly
into the public C/C++ system descriptor:

```bash
/opt/generativeqc/bin/generativeqc run molecule.xyz \
  --method rhf \
  --basis def2-svp \
  --representation spherical \
  --backend cpu \
  --forces
```

`rhf` and `uhf` use the same exact bundled decimal basis records as the
Python frontend. The currently bundled names are reported by
`generativeqc basis list`. GFN2-xTB continues to own its intrinsic basis, so
passing `--basis` or `--representation` with GFN2 is rejected instead of
being silently ignored.

## Select HF density fitting

RHF/UHF can select the same native density-fitting modes exposed by the public
method descriptor:

```bash
generativeqc run molecule.xyz \
  --method rhf \
  --basis sto-3g \
  --density-fitting cpu \
  --auxiliary-basis def2-svp
```

`--density-fitting` accepts `none`, `cpu`, `cuda`, or `auto`. When no
auxiliary basis is named, the native method contract reuses the orbital system
as the auxiliary basis. An explicit `--auxiliary-basis` is expanded from the
same generated bundled catalog and must accompany an enabled density-fitting
mode. This CLI layer does not change the existing metric threshold or planner
policy.

## Run native DFT energies

Native DFT methods already present in the generated method manifest can use the
same bundled Gaussian-basis resolver:

```bash
generativeqc run molecule.xyz \
  --method pbe-rks \
  --basis def2-svp \
  --representation spherical \
  --backend cpu
```

The selector is resolved from the native method manifest rather than a second
CLI-specific DFT list. Native DFT **energy** can use the same exact
`--density-fitting none|cpu|cuda|auto` and `--auxiliary-basis` options as
HF; the backend choice must match the selected fitting provider:

```bash
generativeqc run molecule.xyz \
  --method pbe-rks \
  --basis sto-3g \
  --backend cpu \
  --density-fitting cpu \
  --auxiliary-basis def2-svp \
  --json
```

With no explicit auxiliary basis, the orbital basis remains the fitting basis.
Choose a scientifically appropriate auxiliary basis for production.

### Compiler MethodIR compositions without Python

The CLI method catalog is generated **at build time** from
`generativeqc_compiler.method.spec.METHOD_CATALOG`, its upstream aliases, and
`compile_ks_execution_plan`, not from handwritten method-name branches. The
generated C++ constants carry the exact semilocal components, coefficients,
full-range exchange contributions, SCF domain, spin and MethodIR identity.
The native method manifest remains the authority for stable C ABI IDs; these
composition names are not new ABI IDs. C++ `KsComposition` and the CLI use the
same native prepared-calculation owner.

```bash
generativeqc run molecule.xyz --method pbe0-rks --basis sto-3g \
  --backend cpu --units bohr \
  --grid-radial-points 64 --grid-polar-points 12 \
  --grid-azimuth-points 24 --json

generativeqc run molecule.xyz --method b3lyp-rks \
  --basis def2-svp --backend cpu --json
```

Built-in curated LDA/PBE/r²SCAN compositions, supported global hybrids (such
as PBE0 and B3LYP), and the generated CUDA-only split-hybrid compositions
(M06-2X/MN15) use the same generic code path when their individual native
backend/basis/SCF gates succeed. RKS and UKS selectors are derived from the
same MethodIR, but support still depends on the selected backend and system.
Run `generativeqc methods --compositions --json` to see `cpu`, `cuda` and
`reason` for each canonical name or generated alias; `--backend` mismatches
fail **before** molecular execution.

The default version-1 small-grid quadrature is a **reference** prescription
(64 radial x 12 polar x 24 azimuth points per atom), not a converged
production-grid policy. Positive `--grid-radial-points`,
`--grid-polar-points` and `--grid-azimuth-points` are available only for
compiler-generated compositions; the existing stable native-manifest method
paths preserve their established controls.

**Representation is not execution qualification.** Methods requiring
unrepresented range-separated exchange, VV10/rVV10 nonlocal correlation,
or a D3/D4/gCP/basis correction (such as WB97M-V or r²SCAN-3c) are currently
listed as unavailable instead of silently dropping operators or returning a
partial energy. Newly generated MethodIR rows do not automatically gain
native capability: preparation remains authoritative.

### Context-qualified native DFT analytic forces

A narrow **CPU, density-fitted, all-electron RKS PBE** force path now composes
the same prepared electronic state as the energy route with the native one-
and two-electron response providers, generated XC point derivatives, the
moving molecular grid/Becke partition derivative and nuclear repulsion:

```bash
generativeqc run molecule.xyz --method pbe-rks --basis sto-3g \
  --backend cpu --density-fitting cpu --forces --json
```

The generated **PBE0-RKS** composition can use this route only when its
prepared full-range exact-exchange **and** Coulomb terms both admit the same
CPU density-fitted derivative provider. The installed C++ SDK queries the
actual prepared context through
`generativeqc_calculation_get_supported_properties_v1`; the generic method
registry remains energy-only because it cannot promise forces for every
backend, grid, spin or approximation. Use `Calculation::supported_properties()`
rather than the global method manifest to test this narrow admission.

This is not support for native CUDA DFT forces, CPU Direct DFT forces, UKS,
r²SCAN, ωB97M-V or correction-bearing MethodIR. Unsupported `--forces`
requests reject before execution and do not return a partial derivative.
Prepared DFT *batch* forces remain a separate qualification task under #2151.
This slice requires complete independent numeric/CI acceptance before being
considered production-qualified. See [C++ SDK usage](native_cpp.md).

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
probing a GPU. Invalid indexes fail closed rather than emitting malformed JSON.
On POSIX systems, `profile clear` deactivates profiles while retaining immutable
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
