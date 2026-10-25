# Native CLI

`generativeqc` is a compiled C++ executable. Running it does not require
Python; building it from source still uses Python for code generation.
An SDK install places the executable in `bin/`. Python wheels include
the same native executable and a small native launcher.

## Build and install

Native CLI builds require CMake, a C++20 compiler, nlohmann-json 3.7+,
OpenSSL Crypto and libarchive. On Debian/Ubuntu the extra packages are
`nlohmann-json3-dev`, `libssl-dev` and `libarchive-dev`.
On macOS with Homebrew, install `nlohmann-json`, `openssl@3` and `libarchive`.

For a CPU build:

```bash
cmake -S . -B build -G Ninja -DGENERATIVEQC_ENABLE_CUDA=OFF
cmake --build build
cmake --install build --prefix /opt/generativeqc
```

The executable finds the adjacent GenerativeQC shared library in a normal
SDK installation. You do not need a Python interpreter or a
GenerativeQC-specific `LD_LIBRARY_PATH`.

## Precompiled CUDA CLI

Successful `master` CI runs provide a Linux x86_64 CUDA 12.9 / sm_120
(Blackwell) executable through
[GitHub Actions artifacts](https://github.com/jinzhezenggroup/generativeqc/actions/workflows/ci.yml).
Download the archive for the desired commit, verify its checksum and run:

```bash
tar -xzf generativeqc-v0.1.0-linux-x86_64-cuda12.9-sm120.tar.gz
./generativeqc-v0.1.0-linux-x86_64-cuda12.9-sm120/bin/generativeqc --version
```

The filename is an example; `master` artifacts also contain a commit SHA.
This package embeds the GenerativeQC core but still needs the appropriate
NVIDIA driver, CUDA libraries and host dependencies. It targets sm_120 only;
other GPUs require another build. No Python process is launched by the CLI.

The monolithic binary does not support replacing its linked core with a
locally activated binary profile. Use the shared-library SDK to activate
those profiles; CLI-only scheduling diagnostics still work.

For the release-mode build preset, archive tooling, library-only builds and
package CI checks, see [build and CUDA configuration](../developer/build.md).
Tagged releases, when published, are listed on
[GitHub Releases](https://github.com/jinzhezenggroup/generativeqc/releases).

## Compiler-AD WB97M-V Hessian and HVP

The C++ executable exposes native `hessian` and `hvp` commands for canonical
`wb97m-v-rks` (the default) and `wb97m-v-uks`, in strict FP64 on CUDA. Build with
`GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON`. With no explicit inventory, CMake
generates the complete factorized through-f orbital / through-g auxiliary
integral-component closure at configure time. An existing cache that explicitly
selected a partial inventory must clear `GENERATIVEQC_HESSIAN_AOT_INVENTORY`
to use this default. Missing linked
closure is an explicit error, never a request to compile or fall back at runtime.
This opt-in path replaces the old Python/CuPy CLI worker, including the response
solver and stationary assembly, rather than merely precompiling isolated leaves.
`--compiler-python`, `--compiler-library`, `--cache` and `--timeout-seconds` are
obsolete and rejected. CPU CLI Hessians are not offered by this native endpoint;
the separate Python API retains its existing CPU contract.

For an independently retained inventory, the equivalent complete selection can
also be generated before configuring CMake:

```bash
python tools/complete_hessian_aot_inventory.py \
  --inventory manifests/hessian_aot_development.json \
  --output .artifacts/wb97mv-f-gaux.json \
  --families overlap kinetic nuclear_attraction eri three_center_eri coulomb_metric \
  --max-angular 3 --max-auxiliary-angular 4 --integral-schedule factorized
```

Set `GENERATIVEQC_HESSIAN_AOT_INVENTORY` to that file's absolute path alongside
the CUDA, CLI and Hessian-AOT build options. Smaller explicit development
packages can lower the angular
bounds. Enumeration alone is not numerical or performance qualification; see
the [AOT package contract](../developer/dft_hessian.md#native-aot-package-selection)
for compiler ownership, physical selectors and independent gates. All generation
and compilation happen at build time; the executable never expands the inventory
on demand. The public Python CUDA Hessian frontend uses the same linked native
owner, without CuPy numerical work or a runtime compiler.

For example, the following **small-grid smoke test is not a grid-converged
scientific calculation**:

```bash
generativeqc hessian h2.xyz --method wb97m-v-rks --basis sto-3g \
  --representation spherical --backend cuda --units bohr \
  --grid-radial-points 8 --grid-polar-points 4 --grid-azimuth-points 8 --json

generativeqc hvp h2.xyz --method wb97m-v-uks --charge 1 --multiplicity 2 \
  --basis sto-3g --representation spherical --backend cuda \
  --density-fitting cuda --auxiliary-basis auxiliary.json --units bohr \
  --direction direction.json \
  --grid-radial-points 8 --grid-polar-points 4 --grid-azimuth-points 8 --json
```

Run real GPU commands through the site scheduler. `direction.json` contains a
finite real `natoms`-by-3 JSON array in atom/xyz order, for example
`[[0,0,0.1],[0,0,-0.1]]` for H2. XYZ units follow `--units` (Angstrom by default);
the direction is a Cartesian vector in the Bohr-coordinate Hessian space and is
not rescaled with the XYZ input. Results use `hartree/bohr^2`, with `matrix` or
`value`, identity and work/residency diagnostics in JSON. The matrix is raw, not
post-hoc symmetrized. Canonical local orbital/ECP and independent auxiliary
records use the same offline basis schema as `run`.

The C++ frontend prepares one native SCF state and binds its actual immutable
grid, canonical orbital frame and exact/fitted Hamiltonian sources. Generated
CUDA computes reference W, nuclear RHS, coupled CPKS and all moving-density
stationary contractions. A complete matrix is privately staged and checked for
raw symmetry before publication; it is never repaired. No displaced SCF,
interpreter, runtime compiler or reference engine executes in this path.
Numerical intermediates remain device-resident; native input handling, bounded
status receipts and final publication are not claimed to be GPU work.
See [the Hessian contract](../developer/dft_hessian.md) for admitted basis,
full-rank DF, ECP quadrature and response gates.

Controls include `--device-budget-bytes` (2 GiB aggregate independently owned
response arenas plus CLI output/direction storage; borrowed SCF storage is
separate), `--integral-budget-bytes` (64 MiB per source), `--grid-budget-bytes` (128 MiB per grid owner),
`--output-budget-bytes` (64 MiB for the full matrix),
`--response-budget-bytes` (64 MiB), and `--max-nonlocal-pairs` (16,777,216 ordered
pairs per action). The pair cap is independent of memory; raise it explicitly
when qualifying a larger grid. Response controls are `--response-rtol` (`1e-10`),
`--response-atol` (`0`), `--response-restart` (`30`) and
`--response-max-iterations` (`200`). Use a finite scheduler time limit for real
GPU runs. Missing closure, failed solves, exceeded limits and nonfinite output
fail with no partial JSON result. Diagnostics distinguish native execution time
from an external complete-process wall measurement and expose source/response
work counts; these are not GPU4PySCF speed-comparison evidence.

The extension admits FP64 only and rejects CUDA with an explicitly CPU DF
provider. Native SCF convergence, screening, fitting and explicit grid controls
are forwarded, not ignored. Unrelated force, correlation and global resource
planning flags are rejected by these commands rather than silently discarded.

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

For ordinary Gaussian HF/KS methods, `run` and geometry `optimize` now use
the same SCF stopping defaults as Python `Calculator`: 100 iterations,
energy change `1e-10` Hartree and density RMS `1e-8`. The `resources`
dry-run reports these same effective convergence controls. Explicit
`--energy-tolerance` and `--density-tolerance` options take precedence
in all three commands.

Canonical correlated methods retain their stricter SCF reference-orbital
defaults (`1e-12` Hartree / `1e-10` density). GFN2-xTB keeps its SCC
defaults (`1e-10` / `1e-8`); the dedicated Hessian/HVP path retains
its own stricter SCF controls. Gaussian CLI integral screening is unchanged
at `1e-14` (distinct from the Python `Calculator` default of `1e-12`).
These are solver stopping thresholds, not proven absolute energy/force errors.
A larger iteration limit does not relax convergence or guarantee success.

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
full-/short-/long-range exchange, nonlocal correlation, supported D3/D4 corrections,
SCF domain, spin and MethodIR identity.
The native method manifest remains the authority for stable C ABI IDs; these
composition names are not new ABI IDs. CLI singlepoints and optimization use
the [named C/C++ preparation API](native_cpp.md#complete-named-ks-programs), which
owns complete method assembly and execution.

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

KS defaults to level-3 Treutler/Lebedev + NWChem quadrature; see
[grid options](ks_options.md) for its partition and accuracy contract.
Compiler-generated compositions accept `--grid-level 0..9`. Their existing
`--grid-radial-points`, `--grid-polar-points` and `--grid-azimuth-points`
select an explicit legacy version-1 tensor grid and cannot be combined with
`--grid-level`. Native-manifest methods use the common native default.

Canonical `wb97m-v-rks` and `wb97m-v-uks` execute their complete semilocal,
short-/long-range exchange and self-consistent VV10 energy. Native D3(BJ),
D3(BJ)-ATM, D3(0), and functional-specific D4-EEQ compositions use the generated
parameter records and the same native KS owner. For example:

```bash
generativeqc run molecule.xyz --method wb97m-v-rks --basis def2-svp --json
generativeqc run molecule.xyz --method 'pbe0-d3(bj)-rks' --basis def2-svp --forces --json
generativeqc run molecule.xyz --method 'b3lyp-d4(bj-eeq-atm)-rks' --basis def2-svp --json
```

`--nonlocal-memory-bytes` and `--dispersion-memory-bytes` each default to
256 MiB and bound their respective native owners. Zero or insufficient capacity
fails the applicable calculation. Dispersion compositions require FP64. An
energy-only request skips correction gradients; forces subtract the correction's
`dE/dR` exactly once after the electronic stationary force. Corrected methods
advertise forces only when their prepared electronic context qualifies. The
resource dry-run currently rejects these complete compositions because its
inventory does not include their additional owners.

Canonical `r2scan-3c-rks` and `r2scan-3c-uks` run complete CPU/CUDA energies with the
method-owned spherical H–Ar `def2-mTZVPP` basis, special D4-EEQ profile and gCP.
Omit `--basis` or name `def2-mTZVPP`; the C API expands its compiled snapshot.
Explicit local basis files are accepted only when their actual normalized data
match that snapshot. A different basis or Cartesian representation fails.

```bash
generativeqc run molecule.xyz --method r2scan-3c-rks --backend cpu --json
```

Native r²SCAN-3c also admits complete analytic forces on qualified CPU Direct/DF
and CUDA Direct contexts. CUDA evaluates gCP on device, alongside native D4;
there is no host gCP numerical fallback in this named route. See its
[capability contract](r2scan3c.md) for basis and derivative bounds.
Canonical WB97M-V admits complete native Direct forces on CPU and CUDA in the
bounded domain described below.
Exact backend/basis/property preparation remains authoritative. Independent
small-grid PySCF/correction references and reconverged energy differences check
implementation parity, not grid convergence. Real-device named CLI qualification
runs `tests/native/test_cli_cuda_named.py` inside a finite Slurm GPU allocation.

### Context-qualified native DFT analytic forces

Native force requests use the same converged prepared KS owner as energy.
It verifies the final-state token and the complete one-electron, Pulay,
Coulomb/exchange, XC, moving-grid and nuclear source inventory before publishing
forces. WB97M-V also includes its independent range-exchange and VV10 responses;
named D3/D4 and r²SCAN-3c programs add their native correction derivatives once.

```bash
generativeqc run molecule.xyz --method b3lyp-uks --basis sto-3g \
  --charge 1 --multiplicity 2 --backend cpu --density-fitting cpu --forces --json
```

The [C++ SDK derivative contract](native_cpp.md#density-fitting-and-derivative-boundary)
is authoritative for method, backend, angular-momentum, ECP and precision
admission. CPU and linked CUDA profiles support RKS/UKS and Direct/DF forces
for their admitted named programs. CUDA DF consumes resident final D/W and
reduces sources on device. It does not export density matrices for host force
science. Imported CPU semilocal programs reuse the same point evaluator as SCF.

Preparation checks actual derivative providers and checked workspace sizes.
The historical 8-atom/64-AO/128-primitive and 8192-nonlocal-point restrictions
are removed. This does not relax final-state convergence, actual integral/AO
capabilities, integer overflow checks, explicit memory budgets, or correction
provider limits. Mixed-precision qualification remains separate from FP64.

Query `Calculation::supported_properties()` or
`generativeqc_calculation_get_supported_properties_v1` for the actual prepared
context. The method-global registry remains conservative. Unsupported requests
fail before publication; failed items cannot publish partial forces.
Python `PreparedBatch` checks each item's native capability after model-identity
validation. Admitted items use native execution, including in a ragged batch;
older libraries and unqualified contexts retain their qualified compatibility
path. Native execution requires no Python or runtime compiler.

`tests/python/test_r1_native_forces.py` checks independent energies, analytic
forces, final D/W, cold/warm/moved replay and failed-neighbor recovery. CUDA
checks require a real scheduler allocation. Installed CLI tests use an empty
runtime Python/compiler search path; CPU-only packaging CI cannot establish
numerical CUDA qualification.

## Manage local profile activation

The native executable also owns the profile-cache operations that do not need
the Python compiler or autotuner:

```bash
generativeqc profile show
generativeqc profile clear
generativeqc profile validate accepted-bundle/
generativeqc profile export accepted-bundle/ profile.zip
generativeqc profile diagnose
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

For shared-core SDK and wheel commands on Linux, CUDA `run` honors
`GENERATIVEQC_PROFILE=off`, an explicit bundle
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

Inventories cover CPU RHF/UHF including DF, CPU LDA/PBE RKS/UKS energy plans,
small conventional CUDA HF, small CUDA DF resident/source/packed alternatives,
and conventional CUDA KS energy arenas. ECP workspace uses the same generated
two-grid/radial-batching policy as native execution. Existing ECP-DF, orbital/
auxiliary-size and other provider admission limits remain fail-closed.
Unqualified correlated, tau-dependent, force, density-fitted KS and other
inventories return JSON `status: unsupported` and exit 2; budget failures return
`infeasible` and exit 2; feasible plans exit 0. The Python
`Calculator.estimate_resources` estimator is the entry point that plans a
qualified force context, because its per-context capability record knows which
workspace the force consumer reserves. Output includes phase capacities, fleet
peaks, resident storage, effective limits, controls, topology/source identities
and exclusions.
`scope: native-cli-allocation-inventory-v1` is not a Python `ResourcePlan`
checkpoint, process-RSS cap or free-memory guarantee.

## Stable subcommand namespace

The top-level CLI keeps one stable subcommand layout while implementations move
from Python to the native runtime. The native executable owns method/basis
discovery, calculations, resource dry runs and profile administration,
validation, installation and kernel-library selection. Autotuning that generates and compiles new kernels is an explicit
**build-time/compiler tool**: `generativeqc-tune --quick INPUT.xyz` (or
`python -m generativeqc.tune_cli --quick INPUT.xyz`). It is not a
`generativeqc` subcommand, and its former profile aliases have been retired.
Runtime use of already accepted kernels requires neither Python nor a compiler.
