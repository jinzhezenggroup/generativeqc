# Reproducible validation gates (CG01)

Storage, publication, exceptions and external archives follow the [evidence retention policy](evidence_retention.md).

This protocol supports integral, tensor, and correlated-method development
without requiring a GPU to write or test CPU code. It adds fixtures and evidence
registration. It does not itself declare which method, backend, basis, property,
or resource context is currently qualified: that is an execution-context
question answered by `Calculator.capabilities` for a fully resolved request and
described in [Capability sources and interpretation](../reference/capabilities.md).
The generated [public method table](../public_methods.md) lists every public
selector identity; a listed selector is not a support matrix.

What this page *does* fix is how acceptance evidence is produced and retained:

- reference and regression fixtures are committed, so ordinary tests consume
  recorded data and require neither PySCF nor a CUDA toolchain;
- PySCF/libcint/Libxc appear only in saved reference-generation scripts and in
  explicitly allocated GPU/CPU qualification jobs, never on a production path;
- each acceptance claim names its source revision, exact numerical gates, and
  the backend that actually executed it; and
- a skipped device test is not evidence.

## Current qualification surface

Use these owners rather than a snapshot table in this procedure page:

- [Capability sources](../reference/capabilities.md) — how discovery,
  registry, and contextual admission relate.
- [Methods and capability discovery](../user/methods.md) — public selector
  entry points and how to query a concrete context.
- [Performance engineering](performance_engineering.md) and
  [Evidence retention](evidence_retention.md) — how complete-endpoint
  measurements are qualified and stored.


## CUDA benchmark execution profiles

CUDA validation and benchmark tools share one scheduler-resource profile rather
than embedding a cluster/GPU selector in each script. The portable development
default is `partition=main`, `gres=gpu:1`, one node and one task, with a finite
per-tool time limit. A concrete GPU model such as `gpu:5090:1` is an explicit
cluster/reproduction choice, not the implicit meaning of CUDA.

Set `GENERATIVEQC_BENCHMARK_PARTITION`, `GENERATIVEQC_BENCHMARK_GRES`,
`GENERATIVEQC_BENCHMARK_NODES`, `GENERATIVEQC_BENCHMARK_NTASKS`,
`GENERATIVEQC_BENCHMARK_TIME` or `GENERATIVEQC_BENCHMARK_SRUN` to select another
scheduler/resource layout without editing source. Set
`GENERATIVEQC_BENCHMARK_LOCAL=1` for tools that support local execution. Explicit
CLI/function arguments take precedence over environment settings, which take
precedence over the project defaults. Empty environment values for partition,
GRES or time omit that optional Slurm flag. Current benchmark executables consume
one result stream and share one artifact path: nonlocal launches therefore require
exactly one node and one task. Larger requests are rejected, not silently clamped
or presented as distributed measurements.

Commands recorded in historical evidence and examples may still show
`main/gpu:5090:1`; those identify the cluster allocation used for that
measurement and are not portable eligibility checks.

### CuMetal QC quarantine

The CuMetal workflow still requires the native production-library build, CUDA
runtime CTest, and completed Apple-GPU smoke provenance. Its real QC endpoint
groups are temporarily quarantined for the exact provider/backend/precision
contract and node list in
[`manifests/cumetal_qc_quarantine.json`](../../manifests/cumetal_qc_quarantine.json).
Both routine and scheduled/manual lanes emit explicit skipped-group JUnit records
and a **QC NOT QUALIFIED** job summary. A green runtime job establishes only the
build/runtime contracts it actually executed, not QC numerical qualification.

The quarantine is opt-in through `CUMETAL_CUDA_QUARANTINE` in those two workflow
steps. A provider pin, backend, or precision mismatch fails closed and requires
review. Newly selected groups outside the exact allowlist execute normally; an
unexpected pytest skip, empty result, failure, missing per-case GPU provenance,
or exceeded budget still fails. NVIDIA and other test lanes are unaffected.

To qualify an upstream fix, run `.github/scripts/run_cumetal_cuda_pytests.py` with
`CUMETAL_CUDA_QUARANTINE` unset in both `CUMETAL_CUDA_TEST_MODE=gate` and `full`,
using the real Apple GPU and the workflow's remaining environment. Preserve every
endpoint's unchanged numerical/oracle checks, per-case `device=apple_gpu` and
`launch_success=true` output, and the existing time budgets. Only complete endpoint
success with no skips justifies removing the workflow opt-in. Compiler-only
success or the source-first runtime smoke cannot establish that result. See the
[compatibility decision](../../.agents/notes/implemented/compatibility/2026-10-06-cumetal-qc-quarantine.md)
for the shared-owner evidence and restoration criteria.

## Existing tests and oracle independence

The initial fixed-density XC contract is documented in [XC integration](../developer/xc_integration.md).
`test_xc_integration.py` checks saved identical-grid PySCF/Libxc energy and AO
matrix references, trace variations, spin/weight factors and stale-grid failures.
Its exporter and evidence runner preserve the existing FP64 element gate and
record all finite-difference steps.

`generativeqc_dft_tests` directly exercises native `GridSpec v1` ordering,
fixed-density LDA/PBE variational response, `lda-tail-v1` zero/tiny/invalid
domains, strict PBE vacuum/interior rejection, and the
[`semilocal-scaled-v1`](../developer/xc_scf_domain.md) production tail policy.
Its PBE H2 case also checks energy, electron
integral and AO potential elements against the hash-checked
`tests/reference_data/xc_integration/h2.npz` PySCF/Libxc fixture.
`generativeqc_dft_api_tests` exercises the exported CPU energy-only LDA/PBE RKS and
UKS contracts, closed-shell and integer spin-population validation,
force/batch/DF/CUDA rejection and nonconvergence diagnostics. UKS regressions
also cover separate alpha/beta densities, total-density Coulomb construction,
joined physical residual/DIIS handling, equal-spin reduction and finite
complete-polarization tails. The pinned small-system energies in that suite
are implementation regressions; the independent PBE H2 endpoint is recorded
separately. A separate PySCF/Libxc consumer runs LDA SCF on the same
materialized grid for H2 and He, and PBE SCF on the H2 matched grid; these
agree within the recorded `1e-8 Eh` threshold. The PBE endpoint details, source
identities and remote environment are recorded in
`benchmarks/results/xc-integration-162/pbe-rks-endpoint-20260912.md`. These records
establish only the small closed-shell endpoint cases. The independent CPU UKS
record in `benchmarks/results/xc-integration-162/pr305-corrections-20260913.md`
covers H2-, fully polarized H2+ and the OH doublet with LDA/PBE on the same
GridSpec-v1 prescription. None of these records establish quadrature
convergence, gradients, batching or GPU execution. Such gradient and GPU
claims are owned by the force and device pages, not by an energy fixture.

Every existing native and Python test is retained, including its tolerances.
The new small-fixture tolerances do not supersede stricter existing tests.

| Existing tests | Oracle and independence | What a pass establishes |
| --- | --- | --- |
| `tests/native/test_cartesian_integrals.cpp`, `test_density_fitting.cpp`, `test_spherical.cpp` | Handwritten CPU integral/derivative implementation in `src/integrals/s_integrals.cpp`, separate from generated CUDA; spherical transforms share production normalization code | CPU integrals, symmetries, derivative/DF contracts, and selected pinned libcint comparisons; shared transforms are not an independent transform oracle |
| `test_rhf.cpp`, `test_uhf.cpp`, `test_batch.cpp`, `test_cpp_batch.cpp`; Python calculator/batch/torch tests | Native HF plus pinned external energies/forces and invariants | Method/API endpoints, batching, error handling; self-comparisons alone do not establish independent accuracy |
| `tests/python/test_codegen_*.py`, `test_expr.py`, `test_integral_contracts.py`, `test_production_recurrence.py`, `test_rys_table_extraction.py`, `test_ppps_resident_rys.py` | Generator DAG, emitted-source checks, Rys versus symbolic recurrence | Structural/algebraic consistency; common geometry/Boys/normalization code can share bugs |
| Native mixed-precision/eigensolver/AOT tests and CUDA portions of Python tests | Generic CUDA versus handwritten CPU or pinned references; generic CUDA is separate from generated schedules but shares native dispatch/normalization | Only actually executed CUDA numerical/resource cases; skipped CUDA tests are not evidence |
| Existing benchmark/source tests (`test_benchmarks.py`, CUDA DF source and resident benchmark tests) | Pure runner/schema/source assertions; CUDA benchmarks explicitly invoked separately | Runner correctness and emission, not GPU execution |
| `tests/python/test_validation.py` | New saved PySCF/libcint values and analytic derivatives, compared with the generator host evaluator and native CPU HF | Independent small-fixture numerical acceptance, sign/order/version corruption detection, reproducible protocol behavior |

The host adapter in `tools/generativeqc_validation/integrals.py` is not another oracle:
it contracts the existing generator evaluator and compares it with libcint.
Its primitive normalization is explicit and tested against a different library.
The spherical f fixture is pinned for downstream consumers; this adapter only
executes Cartesian tensors. Existing native spherical tests remain in force.

Generate the capability table using the existing catalog:

```bash
python -m tools.generativeqc_validation.capabilities --output /tmp/capabilities.json
```

Each of the 55 canonical shell classes has separate **representation, source,
compilation, numerical, endpoint, and production** fields. The 34 f-containing
rows reuse the shared class matrix. Source support is attributed to the committed
emitter catalog; compilation and numerical stages await attached measurements.
Manifest selection is reported as historical state, not inferred acceptance.
FPPS is explicitly provisional until equivalent acceptance evidence exists.
This protocol never edits the production manifest or compiles the full f
matrix in ordinary PR CI. The JSON table is a projection of the catalog, not a
second independent matrix that needs manual maintenance.

### Native CPU canonical-hybrid curvature controls

The linked CPU execution contract and resource semantics are owned by
[DFT Hessian architecture](../developer/dft_hessian.md#linked-native-cpu-b3lyp-endpoint).
The opt-in `tests/python/test_native_canonical_heavy_hessian.py` checks actual
native RKS/UKS electronic curvature on the canonical external Ni/I/Cs/H records.
Set `GENERATIVEQC_RUN_NATIVE_CANONICAL_HEAVY=1` and select an explicit case:
`cs-rks`, `cs-uks`, `i-rks`, `i-uks`, `ni-uks`, or `ni-i-stationary-rks`.
Use a fresh artifact directory
and an explicit finite outer timeout; do not overwrite a previous attempt.
Set `GENERATIVEQC_NATIVE_CANONICAL_ENDPOINT=public` to run fresh complete
`Calculator.hessian` and `Calculator.hessian_vector_product` calls, including
their own SCFs. The default `batch` mode measures retained-batch derivatives;
it never claims public endpoint qualification. Installed-wheel qualification
requires a separate isolated prefix and actual installed calls, not a source
test with a wheel's shared library.
The manifest `tests/data/global_hybrid_ecp/native_curvature_manifest.json` owns
the geometry, charge/multiplicity, SCF/work allowances and numerical gates.
Changing an allowance creates a new qualification attempt, not an amendment to
a failed receipt or a relaxation of its numerical tolerances.
The heavy harness passes that case's explicit SCF iteration allowance to the
native solve and, unless the reference policy specifies its own finite allowance,
each independently cold-converged reference. The frozen
`tests/data/global_hybrid_ecp/independent_scf_policy.json` separately owns the
reference solver policy: both Ni cases use cold independent DIIS SCF with an orbital
gradient tolerance of `1e-11` and no extra nonlinear diagonalization. Its
actual final physical orbital-gradient norm is explicitly recomputed and
must be below `1e-11`; the solver's convergence flag is never overwritten.
The mixed stationary case explicitly selects common error-Gram normalization
and a 240-cycle independent allowance; its native allowance remains 600 cycles.
This opt-in extrapolation does not alter physical Focks, occupations or actual
convergence certification. Other cases retain upstream DIIS, and non-Ni cases
retain the default extra diagonalization and `1e-10`. The energy
tolerance and all physical acceptance gates remain unchanged. This is only an
independent electronic-state solver, not a production or geometry optimizer.
The same policy selects the independent gradient stencil: both Ni cases use the
four-point fourth-order formula, while other cases retain two-point second
order. The three displacement scales and raw-matrix acceptance threshold stay
unchanged; improving reference truncation accuracy never relaxes that threshold.
Every stencil point must independently converge and pass state/root checks.
The mixed stationary case additionally checks the complete Cartesian gradient,
physical Ni-58/H-1/I-127 masses, negative-mode count, and unscaled frequencies
and individual mode characters against an independently implemented mass-weighted
SVD null-space solve for every reference matrix. Its three internal frequencies
are nondegenerate; degenerate future fixtures need subspace comparisons instead.
Raw rotational residuals remain visible before the explicit supplied-H geometric
projection. This convention does not repair a raw matrix or certify exact
continuum rotational invariance. The case is a small mixed Ni/iodine stationary
control, not a representative reaction TS, public ECP saddle admission or practical
25--40-atom endpoint qualification. The retained rationale is in
[the independent-DIIS note](../../.agents/notes/implemented/numerics/2026-11-14-independent-diis-common-gram-normalization.md).
If a separate audit reuses retained cold point states, report reused and newly
solved states separately; loading point data is not SCF warm-start seeding or
acceptance of a previously failed full matrix.
Oracle nonconvergence
raises with the actual solver state available for a failed-state receipt;
neither a converged central state nor more native iterations makes an
unconverged displaced reference acceptable.

The driver matches an independently cold-converged central state, evaluates the
full raw native matrix and two fresh HVPs, and compares three independently
displaced-gradient stencils. Each displaced oracle retains its density,
occupations, coefficients, orbital energies, overlap, energy terms, gradients,
SCF iterations and orbital-gradient norm. Occupied-subspace singular values
use the cross-geometry AO metric separately for each spin; neither a convergence
flag nor energy agreement alone establishes root continuity. Native failed SCFs
retain their physical iteration diagnostics before the test raises.

State matrices must first share an independently checked AO coordinate system.
In particular, the SiH3 oracle groups interleaved native SP shells into PySCF's
stable atom/angular order. Its test verifies mapped AO values/first jets and
overlap matrices before permuting both density axes. This alignment does not
change the production basis or relax the electronic-state density gate.

When an isolated installed producer retains its full Hessian, two fresh HVPs
and supplemental central orbitals while independent references are still
running, keep its receipt unaccepted and explicitly pending. Once the matching
source qualification is accepted, audit the retained outputs separately:

```bash
python tools/validate_installed_cpu_hessian.py \
  --producer /path/to/installed-ni-uks.json \
  --reference-directory /path/to/accepted-source-case \
  --manifest tests/data/global_hybrid_ecp/native_curvature_manifest.json \
  --case ni-uks --output /path/to/new-installed-validation.json
```

The validator checks the frozen manifest identity, physical inputs, independent
root acceptance, installed central spin densities/occupations, raw symmetry and
translation, both HVPs, all three independent matrices and native execution
diagnostics. It records hashes of every consumed artifact and the actual
installed library. It does not import production/oracle engines, run new SCFs,
modify the producer, symmetrize outputs or qualify a TS/scale endpoint. Its
output is exclusive: an existing receipt is never overwritten. Synthetic unit
tests of this audit are not additional scientific endpoint evidence.

Raw derivative covariance qualifications must respect the actual discrete
molecular grid. A laboratory-oriented angular grid need not preserve arbitrary rotations;
use an explicitly grid-preserving rotation and transform both Cartesian axes of
the raw Hessian, together with atom ordering and HVP directions. Even for a
continuum rotationally invariant energy, a nonstationary Hessian obeys
`H (omega cross R) = omega cross gradient`, not a zero-rotation identity.
`tests/python/test_native_cpu_hessian.py` includes mixed-element RKS/UKS
discrete-grid covariance controls with and without two-body D3(BJ). These compact
controls do not qualify heavy ECPs, molecular saddles or endpoint scaling.

An internal retained-batch result is not a public Calculator endpoint. Source
curvature acceptance alone does not qualify a different public/installed
endpoint, an ECP TS or practical 25--40-atom complete electronic execution. Keep
those receipts separate. Similarly, the explicitly selected `sto-3g` SiH3
case in `test_molecular_saddle_sih3_reference.py` is a light-element UKS saddle
control, not a def2-SVP result, a Ni TS, or a lower-level preoptimization for
the requested chemistry. Both its basis choices are opt-in through
`GENERATIVEQC_RUN_SIH3_UKS_SADDLE_ORACLE=1`; select the intended parameter
explicitly rather than resuming an earlier timed-out attempt inadvertently.

## Numerical conventions and fixture layers

`tools/generativeqc_validation/fixtures.py::CONVENTIONS` is the machine-checked
convention record included in every fixture:

- Coordinates are Bohr; energies are Hartree; gradients/forces are Hartree/Bohr.
  Forces are **minus** nuclear-coordinate gradients. libcint `int2e_ip1`
  differentiates the electron coordinate, so moving-center derivatives negate
  it. Each quartet center is differentiated through an explicit permutation;
  coincident coordinates retain distinct shell-center slots.
- Cartesian AOs use descending x power, then descending y power. Thus d is
  `xx, xy, xz, yy, yz, zz`. Each contracted component has unit self-overlap.
  Input coefficients multiply individually normalized primitives; each
  contraction is normalized afterward. libcint Cartesian arrays are converted
  using their overlap diagonal, with the scale saved in the fixture.
- Real spherical AOs follow libcint phases: p is `x,y,z`, and higher l follows
  `m=-l,...,+l`. They have unit self-overlap. Never infer the loaded angular
  momenta from an AO count or a basis name: `angular_momenta_loaded` is inspected
  with `bas_angular` and checked against every explicit requested shell.
- ERIs use chemists' `(ij|kl)`, stored in C order with l fastest. Physicists'
  `<ij|kl>` is chemists' `(ik|jl)`. Derivatives have leading `(center, xyz)` axes.
- RHF uses `P=2 Cocc Cocc.T`, `F=h+J[P]-K[P]/2`. UHF keeps separate alpha/beta
  densities with occupation factor one. The data contain real spatial orbitals;
  any spin-orbital expansion orders all alpha orbitals before all beta orbitals.
- The Hamiltonian is all-electron nonrelativistic Coulomb, with frozen core zero
  and explicit nuclear charges. Auxiliary-center identities and parent-atom
  bindings must be saved independently of orbital-center indices; the present
  four-center fixtures contain `auxiliary_centers=[]`.
- Canonical orbitals are ordered by energy within each spin. Their largest
  absolute AO coefficient is positive, with the first index breaking ties.
  No rotation inside degenerate subspaces is implied. Save actual coefficients
  and occupations; phase alignment alone does not resolve degeneracy.

The small layer uses PCG64 seed 138, asymmetric centers, contraction lengths
one through three, exponents between 0.4 and 2.0, and distinct, coincident, and
all-same-center cases. Same-center fixtures are **raw integral** cases; they do
not define an SCF molecule with overlapping charged nuclei. Exact near-zero
and translation-invariance checks accompany combined absolute/relative gates.

The molecular layer includes H2, He, H2O, NH3, CH4, and open-shell HF+ (UHF
doublet). NH3 includes CCSD amplitudes and a nontrivial perturbative triples
correction. All use the exact bundled STO-3G coefficients, copied into the
fixture; PySCF's rounded built-in basis tables are not substituted. These six
small HF endpoints run on CPU in normal tests. Larger def2-SVP/def2-TZVP
molecules remain in `benchmarks/_cases.py` and `real_molecule_gate.py` for manual
GPU endpoints. An f-shell endpoint must actually contain l=3 shells.

## Reproducing and checking references

With an already available PySCF/NumPy/threadpoolctl environment (no automatic installation):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/generate_validation_references.py \
  --output /tmp/validation-generation-1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/generate_validation_references.py \
  --output /tmp/validation-generation-2 --compare /tmp/validation-generation-1
```

Both commands expose `--help`. Each generation saves the script/definition,
basis-pack and libcint library hashes, Python/PySCF/NumPy versions, NumPy's BLAS
configuration, actual loaded BLAS/OpenMP versions and thread counts, SciPy/h5py
versions, one-thread policy, geometry, basis coefficients, SCF thresholds,
occupation/phase conventions, data hashes, and timestamps. The manifest pins
each complete reference record. `stability.json` traces both generations and
reports per-block differences; generation fails if the stability gate fails.
Mathematical identity includes array ordering and conventions, but excludes
fixture names, basis aliases, timestamps, and RNG history once the explicit
numbers have been saved. Reordering dictionary keys does not change a hash.

The committed two-generation stability result is measured on one dependency
stack, not a claim of cross-BLAS bit reproducibility. Regenerate and inspect
stability before upgrading references. Independent CC residual evaluation is
a downstream requirement: the saved PySCF amplitude-update diagnostic is
explicitly labeled as such and cannot substitute for an independent residual.

Initial gates for new, scale-controlled fixtures are:

| Quantity | Proposed acceptance |
| --- | --- |
| Small FP64 raw integrals/derivatives | `atol=1e-11`, `rtol=1e-10`, per element |
| Same-Hamiltonian CC correlation energy | absolute error <= `1e-8 Eh` |
| Independently evaluated CC residual | <= `1e-9` |
| Gradient against independent analytic reference | max error <= `1e-6 Eh/bohr`, target `1e-7` |

The small HF endpoint test uses `1e-10 Eh` and `1e-7 Eh/bohr`; the existing He
and molecular gates retain their stricter historical tolerances. These are
requirements, not a statement that GenerativeQC implements CC or CC gradients.

`finite_difference` records at least three positive distinct steps and every
error; it does not promote the most favorable step. The HF example uses 0.01,
0.003, and 0.001 Bohr, showing the truncation-error curve against independent
analytic gradients. Hold method, approximation, screening and frozen-core
policy fixed. Tighten SCF, CC and Lambda thresholds before differencing (the
reference SCF energy/gradient thresholds are `1e-13`/`1e-11`, CC energy/update
thresholds `1e-13`/`1e-11`; downstream CC gradients must also record Lambda
convergence at `1e-11` or tighter). A plateau or unstable reference requires
investigation, not a relaxed acceptance threshold.

## Execution tiers

Continue using the standard commands:

```bash
cmake -S . -B build -G Ninja -DGENERATIVEQC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
GENERATIVEQC_LIBRARY=$PWD/build/libgenerativeqc.so python -m pytest tests/python -q
```

DF shell derivatives compile as 64 stable class units under
`generated/df_shells`, using the existing two-job CUDA pool. Their production
policy is compiled only into the light dispatcher. Policy-only edits should
leave numerical class objects cached; a class-local math edit should not
invalidate unrelated classes. Shared primitive or execution-template edits
still rebuild their real dependents. The numerical units retain normal CUDA
optimization even when compile-fast mode is enabled. See the
[compilation-boundary decision](../../.agents/notes/implemented/performance/2026-09-19-df-shell-compilation-units.md)
for the invariants and cold/incremental validation distinction.

The existing CPU/Python CI jobs discover the new tests automatically. CUDA CI
continues its separate compilation job. Its `GENERATIVEQC_CUDA_FAST_COMPILE=ON` setting
is a compilation smoke check only; never use that build for resource/performance
acceptance. `validation_gate.py run` wraps these existing commands and archives
their outputs instead of replacing ctest, pytest, autotune, or the f-shell runner:

```bash
python benchmarks/validation_gate.py run --tier cuda-compile --subject release-build \
  --output /tmp/compile.json -- cmake --build --preset cuda-release --parallel 2

python benchmarks/validation_gate.py run --tier gpu-numerical --subject f-matrix \
  --unavailable 'GPU job has not been allocated' --output /tmp/gpu-not-run.json

GENERATIVEQC_LIBRARY=$PWD/build/libgenerativeqc.so python benchmarks/validation_gate.py hf \
  --case h2 --device cpu --repeats 5 --finite-difference --output /tmp/hf-cpu.json
```

For actual GPU work on this machine, preserve Slurm's assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_LIBRARY=$PWD/build/cuda-release/libgenerativeqc.so \
  python benchmarks/validation_gate.py hf --case h2 --device cuda --repeats 5 \
  --output /tmp/hf-cuda.json
```

Use `run --tier gpu-numerical` inside the same allocation to wrap an existing
autotune command with `--local`, or let autotune request its own finite Slurm
allocation with `--partition main --gres gpu:5090:1`. Use `--attach PATH` to
register its original versioned output (including f-shell class evidence) by hash.
The existing large endpoint runner is wrapped identically with `--tier endpoint`.
No full f-shell matrix is added to ordinary CI. Missing hardware/tools become
`not-run` with reasons. An unallocated GPU invocation also records `not-run`.
A wrapped numerical process exiting zero is recorded as command success only:
it might have skipped tests, so inspect/attach its actual numerical evidence.

## Shared results and performance protocol

`python/generativeqc_compiler/common/evidence.py` defines the `generativeqc.validation` version-1
envelope. `new_evidence`, `block_error`, `attach_artifact`, and `write_evidence`
are the registration API for downstream tasks. Every record includes revision,
equation/IR/source/schedule identities, device/toolchain, actual selected backend,
settings, raw timings, allocated/peak bytes, compilation cost, residuals,
per-block errors, stages, and failure/not-run reasons. Unknown versions, nonfinite
numbers, missing-GPU success, and performance passes lacking provenance are
rejected. Existing autotune/endpoint artifacts retain their original schemas.

`measure_interleaved` reuses the existing AOT gate's ABBA order and timing
summary. It requires at least five measurements **per side**, synchronizes before
and after each sample, and hashes identical inputs. Optional `prepare` runs
outside timing. Separate cold native plan construction, retained unchanged
geometry, changed geometry, energy-only, and energy-plus-force workloads. State
clearing, process startup, imports and JIT compilation must be explicitly scoped
by the caller; the HF example's cold interval includes native plan creation and
destruction, not Python process startup or imports.

Kernel comparisons require an explicit fixed density/amplitude hash. Solver
comparisons require every iteration's energy and residual, keyed by timed sample,
plus independently evaluated final residuals and their numerical gate. Raw
samples are never discarded. Runtime assessment reports median, min/max, raw
samples and relative median absolute deviation (MAD). Improvement must exceed
2% and twice the sum of the relative MADs. Differences below this floor remain
inconclusive; this descriptive rule is not a statistical confidence interval.

Promotion also requires measured memory and compilation cost within explicit
`settings.promotion_limits.peak_bytes` and `.compile_seconds` budgets. A runtime
win alone cannot pass. Compilation, independent numerical and endpoint stages
must already pass; no automatic manifest edits are performed.
GPU promotion additionally requires explicit `settings.fast_compile=false`;
the HF example records the selected library hash and adjacent CMake build
settings when available, including any compilation-smoke-only flags.

The HF example compares identical native implementations as an A/B protocol
control. It validates all samples, freezes post-cold warm-start density, and
checks changed geometry against native one-shot execution (explicitly a replay
consistency check, not an independent displaced libcint reference). The public
HF API exposes neither full iteration histories, final orbitals/density, nor
allocator peaks. Explicit energy-only execution is
available through the C ABI's NULL/zero force output and Python's
``properties=("energy",)`` selection; the precision-boundary benchmark
measures that endpoint directly rather than subtracting force time. The example
therefore demonstrates numerical and endpoint evidence plus timing collection
while **prohibiting performance or production promotion**. It is not a speedup
claim or CC implementation.

The precision-boundary runner keeps its fixed-density Fock diagnostic separate from
clean endpoint timing. It records synchronized wall time, GPU global-timer shell-class
time, and the retained FP64/FP32 quartet count for every class actually routed
through the generated streaming workers. Run it only in an allocated GPU job:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash -lc \
  'GENERATIVEQC_LIBRARY=$PWD/build/libgenerativeqc.so PYTHONPATH=$PWD/python:$PWD \
   python benchmarks/issue174_precision_boundaries.py \
     --output .artifacts/benchmarks/issue174-precision-boundaries.json'
```

Each accepted diagnostic sample must use the frozen warm density, report no
cold fallback, and contain exactly one Fock evaluation. Diagnostic completion
retains a nonconverged status because it does not run an SCF solve; the fleet
recognizes that completion and omits its ordinary cold-start retry.

The broad experimental threshold is evidence plumbing, not an automatic-policy
promotion. Its JSON limitations explicitly reserve controller, strict
refinement, and promotion claims for later evaluation.
