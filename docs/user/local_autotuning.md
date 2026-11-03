# User-local CUDA autotuning

Builds for untuned GPUs use `portable_cuda` automatically. Profile diagnostics
identify this fallback as portable and untuned; they do not claim measured
performance or reuse a tuned schedule from an incompatible architecture.
Autotuning is optional and never runs during installation. A local profile
contains a validated native library and its evidence, so later calculations can
use the generated kernels without rebuilding them.

## Run a workload-first search

Install the optional reference dependency with `pip install -e '.[autotune]'`.
Tuning requires NVCC, PTXAS, CUOBJDump, CMake, Ninja, a C++ compiler, and the
matching GenerativeQC source checkout. The baseline native library must be a Release
build with `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`. Rebuild it after changing native code,
generator code, or Python runtime policy; the command checks its source identity
before starting a search. A wheel installation can pass `--source-dir` pointing
to the matching source checkout. CPU use and non-probing profile operations
(`show`, `clear`, `validate`, `export`) need no GPU allocation. Profile install
and diagnostics require the allocated visible GPU; accepted binaries can replay
without CUDA compiler tools. See [native CLI ownership](native_cli.md).

XYZ coordinates default to Angstrom and are converted to Bohr. Use `--units
bohr` for a Bohr XYZ file. Charge, multiplicity, HF/KS method, Cartesian or
spherical representation, and batch size are explicit options. The command uses
logical CUDA device 0 within the allocated visibility by default. Use
`--device-id N` to select another visible ordinal. Device probing, candidate
timing, independent numerical gates and final endpoint validation all use that
same ordinal; physical GPU indices outside the allocation are not selected.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=02:00:00 env GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  generativeqc-tune --quick water.xyz --basis def2-svp

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=04:00:00 env GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  generativeqc-tune --full water.xyz --basis def2-tzvp --batch 4 \
  --budget-seconds 14000 --export local-profile.zip
```

These searches deliberately use a separate Python compiler frontend exposed as
`generativeqc-tune`; the C++ runtime command never imports or spawns it. On systems without
Slurm, run inside the allocation provided by the local scheduler. The tuner and
its child processes preserve `CUDA_VISIBLE_DEVICES`; they do not request a
second allocation. Set `--nvcc`, `CUDACXX`, or a toolkit root (`CUDAToolkit_ROOT`,
`CUDA_PATH`, `CUDA_HOME`) to select a toolkit. An empty visibility mask remains
empty in every child process.

Quick mode profiles the actual final-density shell tasks and ranks classes by
measured active primitive work. It targets 97% coverage, at most eight classes,
and at most four schedules per consumer, plus an official comparison baseline
when required. Full mode targets all measured classes and up to sixteen
schedules per consumer. Neither mode compiles classes with zero measured work.
Use `--coverage`, `--max-classes`, `--max-candidates`, `--compile-jobs`, and
`--budget-seconds` to bound the search. Coverage and the remaining uncovered
fraction are recorded even when a limit ends the search.

Very small workloads can use cached ERIs instead of direct shell-class kernels.
When that execution path supplies no direct-work counters, tuning records the
reason, compiles no classes, and preserves the current configuration.

Exact official consumers are retained by default. `--retune` explicitly includes
them in the search. `--portable-baseline` starts from the generic CUDA path even
on an officially tuned device, which is useful for validating the untuned-device
workflow on available hardware. It does not mislabel the actual GPU architecture.

Release compilation of the native fallback can dominate a first local build.
Subsequent proposals reuse the same build directory and existing compiler cache.
The default two-hour tuning budget is a limit, not an expected runtime. A budget
exhaustion, interrupt, or failed gate leaves the previously installed profile
active. Candidate logs and a report remain in the printed run directory.

## Acceptance and failure handling

The workflow reuses the developer schedule tuner and its target, compilation,
resource, and repeated on-device timing gates. It additionally runs the shared
libcint fixtures against all four direct/persistent RHF/UHF wrappers of each
proposed consumer. Fixtures cover asymmetric and coincident centers, Cartesian
and spherical conventions, pair reversals, and shell/atom permutations. Forces
must also satisfy translation invariance. Numerical gates use the established
combined absolute/relative floors rather than raw relative errors near zero.
After building each candidate library, the same all-spin gate runs against the
exact native object, preserving its actual source hash, resources, and launch
behavior before endpoint promotion.

Each proposed consumer is built into a candidate native library and compared
with the currently accepted build in fresh processes. Balanced ABBA ordering
collects at least four repeats per side (six by default). Process startup, cold
SCF, and warmup are excluded; the measured replay includes complete energy and
analytic forces from a frozen converged density. All samples are retained and
SCF iteration counts must match. Promotion requires at least 1.02x median
endpoint speedup and a positive 95% bootstrap lower speedup bound above 1.0,
alongside energy, force, and translation parity. A faster isolated kernel alone
cannot enable a local profile. When a compatible local profile is already active,
the final proposal must also beat that incumbent before replacing it.

Fock and force proposals are gated independently. An older official row with an
implicit Fock mapping cannot accept a force-only replacement until an explicit
Fock schedule has been accepted, because its value mapping otherwise changes
with the force schedule. This rejection is recorded. A newly tuned Fock mapping
does not enable an unvalidated mixed-precision companion; those calculations
retain the generic mixed-Fock route.

No winner is a normal outcome. Failed compilation, exhausted budgets, numerical
errors, noisy timings, and slower endpoints preserve the working configuration.
Only complete accepted bundles are activated through an atomic index update.
Unfinished staging directories are never read by runtime selection.

## Tune a DFT workload

The same frontend accepts `lda-rks`, `pbe-rks`, `pbe0-rks`, `b3lyp-rks`,
`r2scan-rks`, and their `-uks` counterparts. DFT searches the point tile of the
existing native **device-fused XC** implementation, not HF shell-class kernels.
The HF-specific profiling flags are never passed to KS batches. This is an
execution-schedule search: the functional, Treutler grid, screening, FP64
precision, charge, spin and analytic-force implementation remain fixed.
The XC search does not tune DFT J/K algorithms, generated kernel variants,
independent force tiles, or density fitting, and never promotes a CPU/host-XC path.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 env GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  generativeqc-tune --quick water.xyz --method pbe-rks --basis def2-svp \
  --grid-level 3 --xc-tiles 128 256 512 1024 --repeats 6
```

`--grid-level` selects the scientific quadrature **before** the search; it is
never optimized for speed. The default is level 3. Without `--xc-tiles`, quick
mode tries 128/256/512/1024 points, while full mode adds 64/1536/2048/4096.
Duplicate tiles and the actual incumbent tile are removed; `--max-candidates`
and the shared wall-clock budget bound the remaining search. DFT uses neither
`--coverage` nor `--max-classes`, which belong to HF shell-class selection.
The baseline keeps native automatic tiling unless an exact local winner is
already active. Ordinary CUDA semilocal XC, including global hybrids, now
defaults to 1024-point SCF and force tiles without a profile. CPU and explicit
resource-plan defaults remain bounded at 256; nonlocal CUDA retains its own
policy. Explicit user tiles and qualified profile overrides still take
precedence, and automatic allocation failures retain bounded smaller retries.
Use `--retune` to reconsider a local winner.
Autotuned winners additionally match the full basis data, representation, batch
size, MethodIR and SCF controls. A grid-only match cannot reuse endpoint timing
from a different AO or solver workload. Explicit initial guesses and fitted
calculations retain native policy rather than consuming direct-auto evidence.
The search fixes `max_iterations=100`, `diis_history=8`, energy tolerance
`1e-12`, density tolerance `1e-10`, and screening tolerance `1e-14`; runtime
selection requires those same controls. Explicit SCF tiles also prescribe the
inherited force bound under native policy; the search does not vary an
independent `force_tile_points` parameter.
An explicitly host-unfused incumbent is left unchanged rather than timed as if
it were device-fused; this resident-only search does not replace host-XC winners.

DFT requires at least five paired repeats. Fresh processes alternate baseline
and candidate complete energy-plus-force replays, excluding initialization,
cold convergence and warmup. Native KS diagnostics retain the resolved tile,
grid points, Fock builds and available J/K work census. Every candidate replay
must have a measured native memory ledger within its admitted capacity, with
no rejected allocation. These runtime-only candidates reuse the same compiled
library: their source binding is the verified native source inventory, and the
bundle hashes the exact library. No new PTXAS/occupancy evidence is claimed.

A separate, untimed CPU PySCF/libxc solve supplies independent energies and
analytic forces with molecular-grid response. The oracle bounds are `1e-5`
Hartree/atom for energy, `1e-4` Hartree/Bohr maximum force error and `3e-5`
Hartree/Bohr force RMS error, matching existing native KS qualification floors.
Every paired sample must additionally pass the much tighter schedule-parity,
translation, identical-iteration and statistically significant 1.02x endpoint
improvement gates. The reference is a tuning dependency only, never a runtime
production dependency.

A successful DFT search retains the incumbent native library, accepted HF
kernels and unrelated DFT workload winners when extending the bundle. With
`--portable-baseline`, a proposal must also beat the actual unmasked incumbent
before activation. Missing diagnostics, reference failure, insufficient memory,
timeout, noise or no improvement leave the active profile unchanged.

## Tune native DFT J/K policies

`--component jk` searches existing native full-range Coulomb and hybrid-exchange
lowerings and task schedules. It preserves MethodIR exchange coefficients,
spin, quadrature, FP64, screening and analytic derivatives. Semilocal workloads
do not propose absent exchange work. This is not generated-kernel autotuning or
a change to exact versus fitted integrals.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 env GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  python -m generativeqc.tune_cli water.xyz --method pbe0-rks \
  --basis def2-svp --component jk --repeats 5 --max-candidates 6
```

Each trial uses an isolated child environment and complete warm E+F replay.
Only allowlisted native execution knobs are changed; ambient GenerativeQC
debug/experimental controls are excluded without modifying the parent or Slurm
device visibility. The search compares observed grid/Fock/J/K work and resolved
XC schedules, checks every sample against the independent oracle and native
allocation ledger, then requalifies a selected winner against the original
incumbent using fresh holdout pairs. Native capability and capacity fallbacks
remain authoritative; requested lowerings are not claims of kernel coverage.

An accepted search returns `qualified_recipe`, **not** an installed global
profile. Native environment controls are process-scoped: applying them around
arbitrary concurrent Calculator calls would be unsafe and could transfer
performance evidence to a different workload. Explicitly replay the recorded
geometry/basis/batch/model with:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python -m generativeqc.execution_tune \
  /path/to/jk-run/recipe.json /path/to/replay.json
```

Replay rechecks native-library bytes, device/source identity, the exact workload
and policy, and all raw holdout numerical/performance/resource/work gates. It
needs no compiler or CPU reference solve; the independent reference is retained
in the adjacent report. No winner, failed holdout, timeout or missing diagnostics
publishes a recipe or changes the active profile index. `--xc-tiles`,
`--portable-baseline` and `--export` belong to other search contracts and are
rejected for J/K recipes. All timing claims exclude preparation/cold convergence
and warmup; they are not cold-start throughput claims.

Resident-session benchmark receipts are not installed profile bundles and are
not discovered automatically by Calculator. Apply their execution controls
explicitly, or use the normal qualified profile-installation workflow. An
installed XC winner still requires an exact scientific/source/device match and
never overrides an explicit `KsOptions.tile_points` choice. Native J/K and DF
recipes remain isolated replays, not global environment/default changes.

## Tune density-fitting execution

`--component df --method rhf` searches native FP64 density-fitting
storage/contraction/derivative mappings under a **fixed** auxiliary basis,
metric threshold and memory hint. It does not switch direct and fitted
Hamiltonians, tune the metric cutoff, or enable single-precision storage.
The domain is full-rank RHF: either at most 16 orbital and 128 auxiliary AOs,
or explicitly budgeted generated-source execution with at most 768 orbital
and 768 auxiliary AOs. The larger route retains bounded ordinary AO/metric
eigensolver workspaces, separately budgeted response tiles and singleton retry
storage. It does not depend on Direct-HF's small-eigensolver inventory.
Larger dimensions, unbudgeted large resident execution and composed KS/DF
resource admission remain refused rather than run without the allocation gate.
This extension applies to Calculator and the Python tuning/replay frontend;
the Python-free native CLI retains its separate small-DF preflight limits.
The memory hint is still checked by the native phase planners: a dimension
within this envelope does not imply that a small hint is feasible. In particular,
large force execution receives half of that hint and must also fit its fixed
metric, DIIS and source reservations. Insufficient budgets fail before execution.
UHF also remains refused until its native DF owner publishes complete observed
Fock work; an iteration-derived estimate is not a substitute for that census.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 env GENERATIVEQC_LIBRARY="$PWD/build/libgenerativeqc.so" \
  python -m generativeqc.tune_cli water.xyz --method rhf --component df \
  --basis sto-3g --auxiliary-basis def2-svp --representation spherical \
  --df-memory-mib 20 --repeats 5 --max-candidates 6
```

The auxiliary basis defaults to the orbital basis, **not** an implicit JK-fit
table. The fixed relative metric threshold is `1e-10`. The memory hint defaults
to 64 MiB and feeds the existing native phase-aware DF planner; it is not a
total-process-memory promise. The composed ResourceBudget and observed ledger
gate the complete endpoint, including forces. Resident/streamed and packed/dense
fallbacks remain native decisions under that hint.

An untimed independent PySCF/libcint fitted-HF oracle includes auxiliary-basis
response in analytic forces and rebuilds integrals from the exact immutable
Gaussian input tables, not potentially differently rounded named-basis aliases.
Near-singular/truncated metrics are refused until
matching pseudoinverse/derivative semantics have independent qualification.
Every candidate retains the same independently verified full metric rank and
observed Fock work. Absolute energy error must be at most `1e-9` Hartree,
maximum force error `1e-8` and force RMS error `3e-9` Hartree/Bohr; the looser
XC-grid discretization floors do not apply. All other recipe/holdout/replay
requirements are shared with J/K tuning. Replay uses
`python -m generativeqc.execution_tune recipe.json replay.json` under Slurm and
never performs a production CPU reference solve or changes the global profile.

## DFT Profile Acceptance

DFT grid/XC schedule tuning reuses the same bundle and activation index. Its
optional `dft_schedules` winners have a separate scientific workload identity
(architecture, functional/ingredients and jet outputs, grid/screening model,
FP64 precision, spin/observable, density route, and source identity) plus a
schedule hash. Before a DFT winner can be stored, legality and measured resource
bounds, an independent numerical reference, and at least five synchronized,
interleaved, matched complete energy-plus-analytic-force endpoint samples must
all pass. Schedule JSON alone is not acceptance evidence. A slower/noisy DFT
candidate simply leaves no winner and keeps the unfused or current path usable.

The prepared grid/XC layer and the native CUDA KS owner expose the same two
execution choices. `device_fused` keeps supported semilocal XC/Vxc resident on
CUDA; `host_unfused` downloads the current density for the audited CPU
semilocal integrator and uploads Vxc back into the unchanged CUDA SCF loop. The
native KS options ABI carries this execution policy as a v3 suffix, so v1/v2
callers retain the historical device-fused default.

Runtime DFT profile selection is batch-local and fail-closed. The calculator
derives the exact scientific workload from the actual geometry, GridSpec,
functional, spin, source and target architecture without materializing another
quadrature. A validated local bundle may replace the portable schedule only
when every batch item has an exact workload match and all matched items select
the same resolved schedule, including its point tile. A changed geometry or
mixed batch with any miss keeps the portable default. `GENERATIVEQC_PROFILE=off`
continues to provide the tuning/A-B bypass.
Explicit `KsOptions.tile_points` overrides are never replaced by local DFT
profiles. Calculators with an explicit `ResourceBudget` retain their bounded
native/user tile policy rather than enlarging it from a local schedule winner.

Real-device component qualification has exercised this boundary on an NVIDIA
GeForce RTX 4090 with CUDA 12.9.86. At revision `cc3fc40d`, a complete native
CUDA build and seven focused schedule/fallback tests passed. A separate
fixed-density PBE ablation used seven alternating-order warm pairs at three
scales: H2 (2 AO/32 points), water (7 AO/48 points), and spherical-f (16 AO/32
points). The device-fused median wall time was 1.84x, 1.23x, and 1.58x faster
than the host-unfused path respectively, with every execution still passing the
independent stored energy/potential gate. These measurements establish that
both lowerings really execute and that fusion can remove this prepared-boundary
cost.

A follow-up native-KS qualification at source
`39c31ff8bfd9e532aee7ac02638b4ea64089d31a` used the public
`energy+forces` endpoint on an RTX 4090. After independent compile-cold
records, seven synchronized alternating-order warm pairs were retained for H2,
water, and a two-system H2 batch. Taking `host_unfused` as the baseline,
`device_fused` passed the complete DFT endpoint gate in all three cases:
median speedups were 1.116x, 1.141x, and 1.146x, with 95% bootstrap lower
bounds 1.085x, 1.134x, and 1.140x respectively. SCF iteration branches matched;
the maximum measured energy/force discrepancies remained below
`2e-13`/`2e-14`, and changed-geometry replay preserved the same numerical
agreement. The raw evidence remains a benchmark record rather than an
installable local profile bundle; profile installation still requires the
bundle validator's matching legality/resource/independent-numerical/source
records and resolved schedule identity.

## Reuse and diagnostics

Profiles live under `$XDG_CACHE_HOME/generativeqc/profiles`, defaulting to
`~/.cache/generativeqc/profiles`. `GENERATIVEQC_PROFILE_CACHE` can select another directory.
Python CUDA calculators use this precedence:

1. A compatible explicit `GENERATIVEQC_PROFILE=/path/to/accepted/bundle`.
2. A compatible validated local profile selected by its exact identity.
3. The baseline library's exact official profile or portable CUDA fallback.

`GENERATIVEQC_PROFILE=off` selects the baseline library directly. `GENERATIVEQC_LIBRARY`
locates that baseline and does not disable local profile reuse. CPU calculators
do not consult CUDA profiles or probe hardware.

The identity includes GPU name/compute capability and relevant hardware limits,
CUDA runtime/toolkit/driver versions, host architecture/libc, native ABI, source
and codegen identity, schema, and FP64 consumer policy. Each tuned kernel has
source and schedule hashes, all promotion stages, and matching numerical and
endpoint evidence. Binary, manifest, and evidence files are hashed. An available
NVCC/PTXAS must match the recorded compiler identities; reuse of an immutable
accepted binary does not require those compilers to be installed.

```python
from generativeqc import Calculator

calculator = Calculator(method="rhf", basis="def2-svp", device="cuda")
print(calculator.profile_diagnostics)
```

Diagnostics report `official`, `local`, or `portable`, the selected identity,
tuned consumers, optional validated DFT schedule winners, and incompatible-cache
rejection reasons. `generativeqc profile
diagnose` probes the allocated GPU and prints the same selection. Static
management commands require no GPU:

```bash
generativeqc profile show
generativeqc profile clear
```

Clearing profiles deactivates them while preserving immutable binaries already
loaded by live processes. Those processes retain their selected library.

## Homogeneous cluster reuse

Export/import carries the binary, manifest, and evidence together:

```bash
generativeqc profile export /path/to/accepted/bundle local-profile.zip
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:02:00 generativeqc profile install local-profile.zip
```

Installation checks the same hardware, source, ABI, toolchain, and artifact
contracts as runtime selection. It probes the imported binary as well as its
metadata before activation. UUID and PCI address are excluded so identical GPUs
on different nodes can share a profile. A schedule JSON by itself cannot be
installed because it contains no executable kernels or matching evidence.

Local results describe the user's measured workload. Official repository
profiles remain the source for published benchmark claims. Native C/C++ clients
can load the exported `libgenerativeqc.so` explicitly; automatic cache selection is
provided by the Python CUDA calculator interface.
