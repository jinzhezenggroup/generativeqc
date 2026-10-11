# Native CLI without Python

GenerativeQC's computational runtime is a native C/C++ shared library. A native SDK
install also provides a `generativeqc` executable, so the target machine does not need
a Python interpreter to discover methods or run the native CLI endpoints.

Python is still a **build-time** dependency for source builds because repository
code generation is currently implemented in Python. This is separate from the
installed runtime dependency.

Installed Python console scripts forward runtime commands to their packaged
native executable. `GENERATIVEQC_NATIVE_CLI` can select an explicit executable;
an invalid override fails instead of falling back. Source-only compiler installs
retain their Python administration compatibility interface. The native executable
never discovers, starts, or silently falls back to a Python interpreter.

## Install a native runtime

For a CPU build:

Native CLI builds also require nlohmann-json 3.7+, OpenSSL Crypto and libarchive
development packages. On Debian/Ubuntu these are `nlohmann-json3-dev libssl-dev
libarchive-dev`; on Homebrew they are `nlohmann-json openssl@3 libarchive` (add
their prefixes to `CMAKE_PREFIX_PATH`). Library-only builds can set
`GENERATIVEQC_BUILD_CLI=OFF`.

Wheel builds can obtain the pinned JSON headers from their isolated build
dependencies when no suitable system CMake package exists; that header package
is not an installed Python runtime dependency.

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
/opt/generativeqc/bin/generativeqc methods --libxc --json
/opt/generativeqc/bin/generativeqc --version
```

Method discovery comes from the same generated native method registry used by
the C and C++ APIs.

`methods --libxc` also exposes automatic pure-semilocal Libxc selectors using
the compiler's structural admission and explicit blacklist at build time.
`run --method libxc:GGA_X_APBE` (or `libxc-rks:GGA_X_APBE`) selects RKS;
`libxc-uks:GGA_X_APBE` selects UKS. These programs execute through the existing
native CPU KS lowerer, without importing Python at runtime. CUDA and mixed
automatic component graphs remain unavailable rather than changing the method.

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

### SCF convergence controls

The Python-free `run` command accepts explicit `--max-iterations`,
`--energy-tolerance`, `--density-tolerance`, and `--screening-tolerance`
overrides. They are passed directly to the existing native method descriptor;
the scientific equations and default convergence policies do not change.
Iteration counts must be positive integers and tolerances must be positive,
finite numbers. Invalid values (including NaN, infinity, zero and trailing
text) fail before a calculation is prepared. Integral screening applies to
Gaussian HF/DFT methods, not GFN2-xTB.

```bash
generativeqc run molecule.xyz --method pbe0-rks --basis def2-svp \
  --backend cpu --density-fitting cpu --forces \
  --max-iterations 200 --energy-tolerance 1e-12 \
  --density-tolerance 1e-10 --screening-tolerance 1e-14 --json
```

The unchanged defaults are 100 SCF iterations, energy tolerance `1e-12`
Hartree for Gaussian methods (`1e-10` for GFN2-xTB), density tolerance
`1e-10` for Gaussian methods (`1e-8` for GFN2-xTB), and screening
tolerance `1e-14` for Gaussian methods. A larger iteration limit does not
relax strict convergence or guarantee success.

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

### Offline canonical local basis and ECP input

`--basis FILE.json` and `--auxiliary-basis FILE.json` accept the checked
[canonical local basis schema](external_basis.md). C++ verifies checksum,
provenance, units, normalization, contraction dimensions and FP64 conversion;
there is no BSE/PySCF/network lookup. Records retain their representation, and
an explicitly conflicting `--representation` is rejected. Paths preserve case
and underscores. Orbital ECP core counts and residual terms go through the
native ECP system constructor and its existing method/backend/property gates.
Auxiliary fitting records cannot contain ECP potentials. BSE-to-canonical
conversion remains an explicit build-time import step.

```bash
generativeqc run molecule.xyz --method rhf --basis local-basis.json --forces --json
```

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
CPU density-fitted derivative provider. Its H2/STO-3G native CLI analytic
force is independently checked against central differences of fully
reconverged molecular energies at displaced geometries, while the C++ SDK
test checks the exact prepared force-capability contract and translation
invariance. These tests are a narrow admission gate, not evidence of general
PBE0 gradients across bases, spins or backends. The installed C++ SDK queries the
actual prepared context through
`generativeqc_calculation_get_supported_properties_v1`; the generic method
registry remains energy-only because it cannot promise forces for every
backend, grid, spin or approximation. Use `Calculation::supported_properties()`
rather than the global method manifest to test this narrow admission.

This is not support for native CUDA DFT forces, CPU Direct DFT forces, UKS,
r²SCAN, ωB97M-V or correction-bearing MethodIR. Unsupported `--forces`
requests reject before execution and do not return a partial derivative.
Native **prepared batches** also reuse that CPU density-fitted PBE/PBE0
RKS force consumer only when every member has a qualified prepared
context; unsupported members remain fail-closed. The Python `PreparedBatch` now queries real per-item
native force capabilities and reuses this exact batch consumer when every
context qualifies. Legacy libraries or unqualified native domains retain
the previously qualified Python force path, with no duplicated PBE method
whitelist in Python. General native CUDA/UKS/Direct DFT batch forces
remain work under #2151.
This slice requires complete independent numeric/CI acceptance before being
considered production-qualified. See [C++ SDK usage](native_cpp.md).

## Manage local profile activation

The native executable also owns the profile-cache operations that do not need
the Python compiler or autotuner:

```bash
generativeqc profile show
generativeqc profile clear
generativeqc profile validate accepted-bundle/
generativeqc profile export accepted-bundle/ profile.zip
generativeqc profile diagnose
generativeqc autotune --show-profile
generativeqc autotune --clear-profile
```

`profile show` reports the resolved cache root and active profile index without
probing a GPU. Invalid indexes fail closed rather than emitting malformed JSON.
On POSIX systems, `profile clear` deactivates profiles while retaining immutable
bundle directories that may still be used by live processes. The cache root
uses `GENERATIVEQC_PROFILE_CACHE` first, then `XDG_CACHE_HOME`, then
`~/.cache/generativeqc/profiles`, matching the Python frontend contract.

`validate` and `export` verify artifact hashes, executable schedules, promotion
gates and matching numerical/endpoint evidence without a GPU or compiler.
Exports contain exactly the four existing bundle files. `install` rejects
traversal, duplicates, links, oversized and corrupt ZIP entries, verifies the
visible device/source/ABI and optional installed toolchain identities, and
checks the binary before atomic activation. Failed installation preserves the
previous active index. `install` and `diagnose` probe the visible GPU and must
run inside a scheduler allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:02:00 generativeqc profile install profile.zip
```

On Linux, CUDA `run` honors `GENERATIVEQC_PROFILE=off`, an explicit bundle
directory, or the compatible active index. Complete kernel libraries are
selected through native re-execution/loader interposition, with no Python or
compilation. Device visibility is preserved and the binary is revalidated on
replay. Missing/corrupt/incompatible profiles retain the linked baseline.
Loader selection currently requires paths without spaces or colons. CPU runs
never select profiles. DFT schedule-only winners are validated and reported,
but not automatically applied: their workload-specific KS admission cannot be
replaced by a binary-only shortcut.

## Run correlated methods

All registered native selectors reach the native preparation owner, including
MP2/UMP2, RCCSD and RCCSD(T). Canonical correlated references default to
unscreened integrals; positive screening overrides are rejected. Existing spin,
basis, backend, precision and derivative gates remain authoritative. `--help`
lists native `--precision`, DF threshold/memory, correlation memory/denominator
and `--ccsd-*` convergence, DIIS, damping and level-shift controls.

```bash
generativeqc run molecule.xyz --method mp2 --basis sto-3g --forces --json
generativeqc run molecule.xyz --method 'rccsd(t)' --basis def2-svp \
  --ccsd-max-iterations 150 --correlation-memory-bytes 1073741824 --json
```

## Metadata-only resource dry runs

```bash
generativeqc resources molecule.xyz --method rhf --basis def2-svp \
  --batch 4 --host-bytes 1073741824 --headroom-fraction 0.1
```

Resource planning resolves compact basis/spin metadata in C++ and uses native
allocator-owned scalar layout queries. It never prepares a calculation,
computes integrals, allocates scientific tensors, probes a device, selects a
profile or starts Python. Fleet retained capacities are summed; serialized
workspaces are charged once. Byte arithmetic has signed-64-bit overflow checks;
reserve/headroom use exact integer rounding. Zero caps permit no allocation.
`--diis-history 0` resolves to the native default of eight history vectors for
both execution and planning; it does not disable DIIS or erase its capacities.

Inventories cover CPU RHF/UHF including DF, CPU energy-only LDA/PBE RKS/UKS,
small conventional CUDA HF, small CUDA DF resident/source/packed alternatives,
and conventional CUDA KS energy arenas. ECP workspace uses the same generated
two-grid/radial-batching policy as native execution. Existing ECP-DF, orbital/
auxiliary-size and other provider admission limits remain fail-closed.
Unqualified correlated/tau-dependent/other inventories return JSON
`status: unsupported` and exit 2; budget failures return `infeasible` and exit 2;
feasible plans exit 0. Output includes phase capacities, fleet peaks, resident
storage, effective limits, controls, topology/source identities and exclusions.
`scope: native-cli-allocation-inventory-v1` is not a Python `ResourcePlan`
checkpoint, process-RSS cap or free-memory guarantee.

## Stable subcommand namespace

The top-level CLI keeps one stable subcommand layout while implementations move
from Python to the native runtime. The native executable owns method/basis
discovery, calculations, resource dry runs and profile administration,
validation, installation and kernel-library selection. Actual `autotune`
search remains an explicit **build-time compiler workflow**: it generates and
compiles new kernels and collects independent reference gates. Use
`python -m generativeqc autotune INPUT.xyz` for that source/toolchain operation;
the SDK executable explains this boundary instead of starting an interpreter.
