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
caller-supplied energy/gradient and raw-Hessian callbacks; it does not provide
a molecular TS task or extend the method domains of `generativeqc optimize`.
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

## Python saddle interfaces

`generativeqc.saddle.correct_saddle` binds that same native corrector to Python
energy/gradient and raw-Hessian callbacks. It does not reimplement the optimizer.
`generativeqc.saddle.optimize_saddle` supplies the molecular adapter, using an
existing `Calculator` without switching the method, basis or grid. These APIs do
not change `generativeqc optimize`, which remains a Python-free minimum search;
there is not yet a molecular TS CLI task.

(python-saddle-backends)=
### Admission and execution

The molecular adapter requires an already qualified public Hessian and forces,
CPU FP64 direct all-electron unpolarized full-range hybrid RKS, and one fixed
charge with multiplicity one. Semilocal-only, range/nonlocal, UKS, ECP, DF and
CUDA calculators are not admitted by this adapter. A qualified uncut two-body
D3(BJ) correction remains part of the same total MethodIR. This narrower adapter
does not promote an unqualified derivative domain. In particular, it does not
deliver the requested Ni/I/Cs transition-state endpoint.

One prepared native SCF supplies energy/forces and the raw Hessian at each
geometry. All electronic owners close before additive D3 curvature publication.
No reference engine, finite differences, external optimizer or preliminary
lower-level calculation runs in production. Independent cold SCFs must preserve
the occupied subspace relative to the **last accepted** state: the smallest
principal cross-geometry AO overlap must meet `minimum_occupied_overlap`
(default `0.9`). Rejected trial roots never replace the accepted reference.
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
grid and output controls bound their named providers independently. Neither
these bounds nor their sum is a whole-process peak-RSS claim. Complete endpoint
seconds include preparation, derivative work, native optimization and final
mode analysis. Qualification must independently measure process peak memory
and inspect both completed and failed stage receipts.
