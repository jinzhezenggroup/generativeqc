# Native progressive geometry optimization

Use `generativeqc optimize` for Cartesian **local-minimum** searches with the
Python-free C++ runtime. At every trial and accepted geometry the optimizer
requests the **same native method's complete analytic energy and forces**.
It does not take finite differences of energies or silently change the target
functional, correlated Hamiltonian, basis, precision or spin state.

## Automatic method ladder

```console
# Automatic GFN2-xTB (CPU) -> PBE -> conventional CCSD(T) (selected backend).
generativeqc optimize water.xyz --method 'ccsd(t)' --backend cuda \
  --basis def2-svp --output water-ccsdt.xyz --json

# Also works for expensive DFT; use a cheaper Gaussian basis for early PBE.
generativeqc optimize water.xyz --method pbe0-rks --backend cuda \
  --basis def2-tzvp --preopt-basis def2-svp --output water-pbe0.xyz

# DF-CCSD(T) keeps its required CUDA, auxiliary basis and FP64 target contract.
generativeqc optimize water.xyz --method 'df-ccsd(t)' --backend cuda \
  --basis def2-svp --density-fitting cuda --auxiliary-basis def2-svp \
  --correlation-memory-bytes 8589934592 --output water-df-ccsdt.xyz
```

Automatic stages are, in order: GFN2-xTB, an optional spin-aware PBE step for
hybrids/expensive DFT or correlated targets, and the exact requested final
method. RHF, UHF, LDA and PBE targets omit the PBE middle stage. The GFN2
preliminary stage uses CPU even when the final target is CUDA. Automatically
chosen *preliminary* stages can be skipped only if native execution reports
**NOT_IMPLEMENTED**. A failed SCF/CC convergence, nonfinite forces, resource
failure, or failed optimization **never** authorizes silently proceeding.
Explicitly requested stages cannot be skipped at all.

Use a hand-picked progression or inspect the stages without executing:

```console
generativeqc optimize water.xyz --method 'ccsd(t)' --backend cuda \
  --stages 'gfn2-xtb,pbe-rks,ccsd(t)' --basis cc-pvdz --plan-only

generativeqc optimize water.xyz --method 'ccsd(t)' --backend cuda \
  --stages 'pbe-rks,ccsd(t)' --basis cc-pvdz --opt-fmax 0.0003 \
  --opt-max-steps 80 --opt-max-step 0.35 --json
```

The **last stage must match `--method`**. Method aliases are canonicalized
(`ccsd(t)` is `rccsd(t)`; `df-ccsd(t)` is `df-rccsd(t)`).
`--preopt-basis` applies only to preliminary Gaussian methods; by default
they use the target basis. GFN2 always has its intrinsic basis.
The final step always recomputes and independently converges the *target*
method's gradient at the actual accepted final coordinates.

The optimizer uses Cartesian L-BFGS and backtracking Armijo with a maximum
per-atom displacement; curvature history is reset when the energy surface
changes. Input XYZ defaults to Angstrom (`--units bohr` is also supported);
`--opt-fmax` is the maximum **atomic force norm** in Hartree/Bohr (default
`3e-4`) and `--opt-max-step` is in Bohr (default `0.35`). Preliminary
stages use looser tolerances. Running out of steps or failing a line search
is **not** convergence. No CCSD(T) analytic Hessian is needed for this search.

Successful `--json` output includes final energy, maximum force, Bohr
coordinates, and per-stage status, iterations and evaluation counts.
`--output` creates a separate Angstrom XYZ file after final convergence only,
and refuses to overwrite an existing file. XYZ uses atomic numbers, supported
by the native CLI reader. Without an output path, the final coordinates are
available on stdout.

## Limits and accuracy

This is not a transition-state, periodic-cell or global-conformer optimizer.
An optimized stationary point is not automatically a true minimum: characterize
the structure by vibrational/Hessian analysis when necessary. Currently there
is no automatic cross-geometry SCF/CC amplitude warm start or persistent
checkpoint; expensive CCSD(T) optimizations may require many cold native
energy-plus-force endpoints.

Conventional CCSD(T) forces retain their [qualified AO / closed-shell domain]
(../developer/rccsd_t_api.md). DF-CCSD(T) has a separate
[CUDA/FP64 response model](../developer/df_ccsdt_gradient.md).
Unrestricted CCSD(T) energy availability does *not* imply unrestricted
analytic nuclear forces.

## Native saddle callback API

The installed C SDK separately exports `generativeqc_saddle_correct_v1` from
`generativeqc/saddle.h`. This bounded local eigenvector-following owner accepts
caller-supplied energy/gradient and raw-Hessian callbacks; the corrector itself
does not establish chemistry or extend the method domains of `generativeqc optimize`.
The caller must bind one qualified physical model, basis, grid and electronic
state policy across all geometries. No external optimizer, reference engine,
lower-level preliminary method or displaced production derivative is launched
by the corrector.

Initialize both descriptors with `sizeof(type)` and `GENERATIVEQC_ABI_VERSION`.
All options are explicit. The reference convergence controls are energy
change `1e-6` Hartree, gradient component RMS `3e-4` and maximum `4.5e-4`
Hartree/Bohr, displacement component RMS `1.2e-3` and maximum `1.8e-3`
Angstrom. Both RMS denominators are `3N`; displacements use unaligned Cartesian
coordinates. Coordinates and trust radii are Bohr. Every accepted or rejected
trial consumes one of at most 100 attempts. Initial preparation and fresh
final energy/gradient/Hessian verification count separately as derivative work.

Set `project_rigid_motions=1` to remove geometry-derived translations and
rotations before curvature analysis and stepping. Linear molecules have five
rigid modes, nonlinear molecules six; a rank change fails explicitly.
No small vibrational eigenvalue is discarded. The returned curvatures are
unweighted Cartesian values, not harmonic frequencies; the raw final Hessian
remains unprojected. Near-zero active curvatures block first-order classification.

Inspect both the C status and the optimizer receipt. A successful C call can
report failed optimization, another stationary index, or ambiguous curvature.
Only `GENERATIVEQC_SADDLE_FIRST_ORDER` with `stationary_verified=1` reports a
verified mathematical first-order saddle. The receipt and caller-owned outputs
preserve the last fully accepted state on derivative, tracking, trust-radius
or attempt failure. Failed initial preparation has no valid state. The optional
progress callback observes accepted-state geometry even after a rejected trial;
returning nonzero requests cancellation before more derivatives. Cancellation
clears stationary verification, including at the final observation.

`maximum_numeric_bytes` bounds the native optimizer's numerical storage, not
callback/provider memory or process RSS. Budget, dimension, ABI and control
rejections occur before derivative callbacks and preserve output buffers.
See the {ref}`native contract reference <native-c-generativeqc-saddle-correct-v1>`
for callback lifetimes, output capacities and failure semantics. This callback
API is algorithm infrastructure, not independent molecular or Ni/ECP TS
qualification.

(native-saddle-cli)=
## Native molecular saddle command

`generativeqc saddle` is a separate Python-free molecular task over the same
native corrector. Unlike `optimize`, it never introduces a cheaper preliminary
method, basis or grid. It requires a negative target mode in a nonstationary
starting geometry; it is not a minimum-escape or reaction-path search.

```bash
generativeqc saddle near-saddle.xyz --method 'b3lyp-d3(bj)-rks' \
  --basis sto-3g --representation spherical --backend cpu \
  --masses masses.json --max-attempts 100 --progress-json --json
```

`masses.json` must contain one positive finite physical/isotopic mass in dalton
per atom, in XYZ order. There is no implicit isotope assignment. Coordinates
follow `--units` (Angstrom by default). Optional `--initial-mode FILE.json`
contains a nonzero finite `natoms`-by-3 Cartesian target direction. Molecular
rigid projection is mandatory; it never removes small vibrational eigenvalues.

The current domain is the linked canonical Gaussian/VWN-RPA B3LYP RKS/UKS
CPU direct-FP64 domain, optionally with uncut two-body D3(BJ), ATM off.
The command defaults to bare `b3lyp-rks`; choose UKS and the supplied
multiplicity explicitly. ECP, DF, CUDA, semilocal-only and range/nonlocal
molecular saddle paths refuse. Prepared native force admission is checked
before SCF, and each raw Hessian consumes that geometry's actual retained
native batch. No Python, runtime compiler, reference engine, displaced SCF
or external optimizer runs. Occupied-root continuity is checked separately
for each spin against the last accepted geometry, not a rejected trial.

All five convergence controls remain independent:

| Flag | Default | Unit |
| --- | --- | --- |
| `--energy-change` | `1e-6` | Hartree |
| `--gradient-rms` | `3e-4` | Hartree/Bohr |
| `--gradient-max` | `4.5e-4` | Hartree/Bohr |
| `--displacement-rms` | `1.2e-3` | Angstrom |
| `--displacement-max` | `1.8e-3` | Angstrom |

`--max-attempts` defaults to 100 and cannot exceed 100; accepted **and rejected**
steps consume that budget. Initial preparation and fresh-final verification
are counted separately as derivative work. Trust-radius controls are
`--initial-radius 0.1`, `--minimum-radius 1e-8`, and `--maximum-radius 0.5`
(Bohr). Other controls are `--curvature-tolerance 1e-7`,
`--raw-symmetry-tolerance 1e-10` (Hartree/Bohr²),
`--minimum-mode-overlap 0.5`, `--minimum-occupied-overlap 0.7`, and
`--acceptance-ratio 0.1`. SCF defaults are 240 iterations,
`--energy-tolerance 1e-13`, and `--density-tolerance 1e-11`; explicit values
are forwarded without relaxing the native accepted-state verification gate.
Derivative work/response controls are the same as
[the native CPU Hessian command](native_cli.md#native-cpu-b3lyp-hessian-and-hvp).

`--saddle-budget-bytes` (128 MiB) bounds the corrector and, separately, final
harmonic analysis. `--root-budget-bytes` (128 MiB) bounds retained orbital
panels and root-tracking matrices, including the current orbital export
reservation. These are owner-specific numeric bounds, not an aggregate RSS
promise; primal/provider scratch, cross-overlap integral scratch, allocation
metadata, JSON and process overhead are excluded. `--output-budget-bytes`
also admits the task's dense result panels; `--max-coordinates` defaults to
300 and cannot exceed 300. Bounded storage does not imply practical electronic
work at the largest admitted dimension.

With `--json`, stdout contains one final `generativeqc.native-molecular-saddle-result/v1`
record. `--progress-json` streams derivative-stage and last-accepted optimizer
observations as JSON lines to stderr. The result separates optimizer attempts,
SCF iterations/Fock builds and completed response/integral/D3 work. Counters
do not invent unknown work inside a failed derivative stage. Each native
energy/gradient evaluation additionally reports `system_prepare`, `batch_prepare`
and `scf_forces` phase starts and completions (or failures), with coordinates
and elapsed time; terminal phase records include `phase_seconds`. A phase start
is not evidence of convergence or completed work. If the process is interrupted
before a terminal record, use the external wall timer rather than the last
record's timestamp as the total duration.

For deeper CPU SCF diagnostics, set `GENERATIVEQC_DF_PROGRESS_TRACE` to a fresh
JSONL path. The separate append-only journal records RKS/UKS SCF scopes, basis
and grid sizes, physical-evaluation durations and completed iteration/Fock-build
observations, plus terminal counts and convergence when the call returns.
A physical evaluation or SCF scope returning is not itself convergence; a missing
scope end denotes interrupted/incomplete evidence. This diagnostic does not
change tolerances or count in-flight work as completed in the final result.

The last fully accepted coordinates, energy, gradient and **raw** Hessian survive
algorithm failures. An initial derivative failure has no accepted state and
publishes null derivative panels plus the original coordinates. A verified
stationary result additionally publishes mass-weighted and Cartesian normal
modes and signed, **unscaled** frequencies in cm⁻¹. Harmonic analysis checks
raw symmetry before roundoff averaging its own copy; the published raw Hessian
is unchanged. Cartesian curvatures are explicitly separate from frequencies.
Harmonic rank/index disagreement or mode-publication failure retains the
verified geometry but prevents a successful task result.

Exit zero requires a freshly verified first-order saddle **and** successful
harmonic publication. Other indices, nonnegative targets, attempt/resource
limits, derivative failures and cancellation are not success. SIGINT/SIGTERM
request cancellation after the current synchronous derivative stage completes;
they do not interrupt native source cleanup. Input/closure/argument refusals
before the task starts produce no result. Small all-electron controls do not
qualify Ni/ECP transition states or practical 25–40-atom complete electronic work.

## Python saddle interfaces

`generativeqc.saddle.correct_saddle` binds that same native corrector to Python
energy/gradient and raw-Hessian callbacks. It does not reimplement the optimizer.
`generativeqc.saddle.optimize_saddle` supplies the molecular adapter, using an
existing `Calculator` without switching the method, basis or grid. These APIs do
not change `generativeqc optimize`, which remains a Python-free minimum search.
The separate {ref}`native saddle command <native-saddle-cli>` binds the same
derivative owners without Python.

### Small all-electron control

The following near-planar ammonia input is a deliberately small local-saddle
control, **not** the requested Ni/ECP/D3 workflow. It uses the same physical
method at every geometry. An arbitrary XYZ guess need not have a negative
initial target mode or converge to the desired saddle.

```python
import math

from generativeqc import Calculator
from generativeqc.response_solver import GMRESOptions
from generativeqc.saddle import SaddleOptions, optimize_saddle
from generativeqc_compiler.method import resolve_method

atoms = [
    ("N", (0.0, 0.0, 0.02)),
    ("H", (1.9, 0.0, 0.0)),
    ("H", (-0.95, 1.9 * math.sqrt(3) / 2, 0.0)),
    ("H", (-0.95, -1.9 * math.sqrt(3) / 2, 0.0)),
]
calculator = Calculator(
    method=resolve_method("B3LYP"), device="cpu", basis="sto-3g",
    basis_representation="spherical", energy_tolerance=1e-13,
    density_tolerance=1e-11, max_iterations=240,
)
try:
    result = optimize_saddle(
        calculator, atoms,
        masses=[14.00307400443, 1.00782503223, 1.00782503223, 1.00782503223],
        options=SaddleOptions(project_rigid_motions=True, raw_symmetry_tolerance=2e-8),
        grid_budget_bytes=2 << 30,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
        progress=lambda state: print(dict(state.receipt)),
    )
    print(result.diagnostics["final_status"], result.optimizer.coordinates)
    if result.converged:
        print(result.harmonics.frequencies_cm1)
    else:
        print(result.optimizer.failures, result.diagnostics["harmonic_failure"])
finally:
    calculator.clear_cache()
```

The 100-attempt bound is not a bound on SCF or response work. Inspect the
separate native evaluation counts and molecular diagnostics. The complete
endpoint includes fresh final derivatives, not just the last accepted step.

(python-saddle-backends)=
### Admission and execution

The molecular adapter requires an already qualified public Hessian and forces,
CPU FP64 direct all-electron canonical B3LYP RKS/UKS, and one fixed charge
and multiplicity. Restricted calculations require multiplicity one; UKS must
already have public Hessian admission. Semilocal-only, range/nonlocal, ECP, DF
and CUDA calculators are not admitted by this adapter. A qualified uncut two-body
D3(BJ) correction remains part of the same total MethodIR. This narrower adapter
does not promote an unqualified derivative domain. In particular, it does not
deliver the requested Ni/I/Cs transition-state endpoint.

One prepared native SCF supplies native energy/forces and the raw Hessian at each
geometry. The Hessian callback consumes that retained batch through the same
native CPU bridge as `Calculator.hessian`; coupled response and exactly-once
D3 curvature are composed in C++. Python does not reconstruct AO/grid owners,
solve electronic response or compile derivative code at runtime. Missing native
force or second-order closure fails explicitly rather than falling back to the
development Python numerical path. Canonical C and integer occupations cross a
separate bounded bridge only for root tracking, not to implement derivatives.
The obsolete molecular `cache` option is removed; no compiler cache is needed
by this precompiled endpoint. No reference engine, finite differences, external
optimizer or preliminary lower-level calculation runs in production.

`max_eri_primitives` optionally overrides the native nuclear-integral work cap
for each complete Hessian request; its default and scope are defined in the
[native CPU derivative contract](../developer/dft_hessian.md#linked-native-cpu-b3lyp-endpoint).
It is not a bound on time, induced-Fock work or the complete optimization, and
does not promote an unqualified molecular domain.

Independent cold SCFs must preserve
the occupied subspace relative to the **last accepted** state: the smallest
principal cross-geometry AO overlap in **each occupied spin channel** must meet
`minimum_occupied_overlap` (default `0.9`). Each channel retains its integer
population; an empty channel must remain empty. Independent occupied-orbital
gauge rotations do not change this test, and one continuous spin cannot mask
a root change in the other. Rejected trial roots never replace the accepted reference.
For this comparison alone, the current AO centres are translated to the last
accepted geometry's mass-weighted centre. This removes arbitrary whole-molecule
translation, which otherwise attenuates compact core-orbital overlaps even for
an unchanged state. AO coefficients do not change under this translation.
Rotations are not registered because the finite angular grid has a fixed
orientation. Production coordinates, derivatives, steps and the unaligned
convergence RMS conventions are unchanged. Evaluation records report the
comparison frame and its translation in Bohr.
This fail-closed local policy is not an electronic stability proof, a guarantee
of the global ground state or an unrestricted root-selection algorithm.

(python-saddle-values)=
### Inputs and outputs

Atoms and callback coordinates use Bohr in fixed nucleus order, with at most
300 Cartesian components. Molecular calls require one positive physical or
isotopic mass per atom in dalton. Derivative callbacks return Hartree energy,
gradient (not forces) of shape `(N, 3)` in Hartree/Bohr and raw Hessian of shape
`(3N, 3N)` in Hartree/Bohr². Inputs must be real and finite; complex, boolean
and string arrays are not silently converted. The native reference controls,
attempt counting and unaligned `3N` RMS conventions are defined above.

`SaddleOptions` holds the independent convergence, trust, symmetry, tracking and
native numerical-capacity controls. Molecular calls require rigid projection,
which their default options enable. `SaddleProgress` describes the accepted
geometry and receipt, even after rejection. Returning true cancels; None/false
continues. `evaluation_progress` additionally receives start/complete/failed
derivative stages, coordinates, elapsed seconds and completed SCF/response work.
Those observer calls are not optimizer iterations.

`SaddleResult` separates native execution from verified convergence. Inspect
`converged`, `receipt`, `detail` and `failures`; zero C execution status alone is
not convergence. `MolecularSaddleResult.optimizer` retains that native outcome;
`atoms` is the last accepted geometry. Only fresh stationary verification
produces `harmonics`, with unscaled frequencies and **all** vibrational modes
retained. Its native curvatures are unweighted Cartesian values, not frequencies.
The harmonic analyzer explicitly averages a raw matrix only after the declared
symmetry gate; the optimizer result always retains the unmodified raw Hessian.

(python-saddle-errors)=
### Failure publication

Invalid controls, shapes, units and masses raise ValueError or TypeError.
Unqualified molecular domains raise NotImplementedError before SCF; native
numeric-capacity and molecular output preflight refusals raise MemoryError.
Library and ABI errors raise RuntimeError. Exceptions from derivative, progress
and cancellation callbacks, including Python interrupts, never unwind through
the C ABI: they produce explicit failed/cancelled results with exception type
and message. Failed initial preparation has no valid derivative arrays and uses
None for receipt metrics. Later failures preserve the last fully accepted
geometry, gradient and raw Hessian, never a failed trial as a success result.

Harmonic-publication failure preserves the native result but makes molecular
`converged` false, with `diagnostics['final_status']='harmonic_analysis_failed'`.
Publication also fails if harmonic rigid-mode rank differs from the native
optimizer, or if a verified native saddle's negative-mode count changes after
mass weighting. The latter check includes soft negative modes, not only
significant imaginary frequencies; no mode is deleted to obtain agreement.
Completed SCF/response counts do not invent partially completed failed work;
`work_counters_complete` and `work_counters_scope` make that limitation explicit.
The attempt cap, a nonnegative target, lost tracking, ambiguous curvature and
cancellation are never reported as successful index-one convergence.

(python-saddle-ownership)=
### Ownership and budgets

Calls are synchronous. Serialize calls sharing a calculator or callback resources.
Progress coordinates and returned arrays/mappings are detached read-only data,
usable after callback return. No callback retains a borrowed C pointer, and no
prepared electronic owner survives the molecular endpoint, including failures.
Changing the calculator's physical model during a callback fails explicitly.

`maximum_numeric_bytes` bounds the native optimizer only; overlap, integral,
grid and output controls bound their named providers independently. The root
overlap budget separately admits canonical-frame export (verified frame,
staged C and caller C) and the rectangular cross-geometry overlap matrix;
it excludes producer eigensolver scratch and detached Python copies. Neither
these bounds nor their sum is a whole-process peak-RSS claim. Complete endpoint
seconds include preparation, derivative work, native optimization and final
mode analysis. Qualification must independently measure process peak memory
and inspect both completed and failed stage receipts.
