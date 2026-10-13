# Application precision and explicit audits

`PrecisionCalculator(policy="fast")` selects FP32 arithmetic from an application
error budget, without asking users to select kernels. All-electron CUDA
PBE0/RKS with exact full-range J/K, 24–384 AOs through d shells and 2–48 atoms
admits mixed stationary gradients at the FP64-converged density. The separate
provisional XC warmup covers local restricted PBE-family calculations with
24–256 AOs. Outside both domains, the explicit FP64 audit fallback remains.
This is not a global all-FP32 switch.

The AO limit is an admission boundary, not a performance crossover: larger
systems do not automatically use more FP32. Outside this domain, Fast can cost
two complete FP64 energy/force endpoints because of the audit. Use `policy="fp64"`
for one strict endpoint when an observable-budget audit is not required; inspect
`precision_validation.solves` and native precision work rather than inferring
acceleration from the preset name.

Fast narrows AO evaluation, PBE exchange and correlation point algebra, supported density
and potential matrix products and local point-total sums. Matrix inner sums
use FP32, not FP32 inputs multiplied in FP64. Global arrays and inter-tile sums
stay FP64. Non-admitted point tails and small/non-tiled contractions retain
FP64. Exact SCF J/K, eigensolution, DIIS, physical residuals and final SCF refinement
remain FP64. Fast does not change the basis, grid,
screening, initial guess or final convergence criteria.

Admitted gradients use FP32 weighted ERI derivative algebra and primitive sums
for orders 0–5, AO jets, density projection/features, interior PBE X/C, AO
pullbacks and point-local Becke forward/reverse sums. Global force reductions,
one-electron/Pulay sources, Boys evaluation, normalization, higher orders and
unsafe tails retain FP64. Nonfinite weighted tasks retry before scatter; a
numerical grid/geometry failure retries the complete gradient once in FP64 at
the same authenticated final density. Optional FP32 grid storage has a bounded
FP64 fallback. No density clipping or CPU reference work enters production.

Automatic initial guesses follow ordinary FP64's native Modified Huckel policy
(MINAO on older compatible libraries). Explicit MINAO and Huckel retain their
strict FP64 preparation and resource admission; `initial_guess=None` keeps Hcore. Ordinary
`Calculator(precision="auto")` does not inherit this application-only MINAO
admission. No reference density or extra reference SCF prepares Fast's seed.

```python
from generativeqc import (
    GridSpec, KsOptions, PrecisionCalculator, PrecisionPolicy, TargetErrorBudget,
)

calculator = PrecisionCalculator(
    method="pbe0-rks", basis="def2-svp", device="cuda",
    ks_options=KsOptions(grid=GridSpec(48, 16, 32)),
    initial_guess=None,
    policy=PrecisionPolicy("fast", budget=TargetErrorBudget(
        energy_abs=1e-5, force_max_abs=1e-4, force_rms=3e-5,
    )),
)
```

`budget` reuses the adaptive numerical policy's energy/force allowances; users
do not select kernels. The native admission floors are `1e-6` Eh energy,
`1e-5` Eh/bohr maximum force component, and `3e-6` Eh/bohr RMS if requested.
Tighter allowances or unsupported domains do not silently inherit ordinary
AUTO: they select FP64 and a measured fallback audit. Fast uses the published
native FP64 stopping criteria even when the application budget is looser.

An admitted mixed-force result reports `qualified_mixed_gradient` and
`gradient_admission="pbe0-rks-budget-fp32-v1"`. Warmup-only results and complete
FP64 gradient retries report `qualified_strict_final`. Each includes its requested
budget, executed precision work, and FP64 final-SCF checks. This is empirical
qualification plus strict final refinement, **not a measured per-call
observable error, mathematical bound, electronic-root or stability test**.
It never reports `observed_met` without an actual reference endpoint.
KS single-point plans are released after each call; repeated calls start fresh
SCF solves.
Approximate work and retained proposal history are bounded; failed mixed work
retries in FP64, and all final physical convergence checks use FP64 operators.

`result.precision["gradient"]` distinguishes attempted FP32 grid/geometry points,
weighted derivative dispatch, complete-gradient retries and returned arithmetic.
Per-task weighted retry counts are unavailable on clean timing calls, not zero;
they require separate intrusive qualification. A float64 result array does not
mean that its intermediate arithmetic was FP64. For a **measured per-call**
energy/force allowance, supply `target=TargetAccuracy(...)`: the full FP64
comparison and, if necessary, verified fallback are included in its cost.

`PrecisionCalculator(policy="fp64")` runs **one strict FP64 endpoint**, without
an unconditional second SCF/force solve. The CUDA KS scheduler can retain
geometry-invariant J/K integrals and reuse them across SCF iterations, within
the existing memory/lifetime guards. This exact optimization also benefits
ordinary `Calculator`: FP64 is not faster arithmetic withheld from FP64 users.
The native force executor and cooperative scheduling also benefit ordinary FP64
PBE0/RKS through 32 atoms in the qualified shape domain. Larger d-shell strict
calculations retain their faster phased executor. Fast comparisons must use this
already-optimized strict baseline, not withhold scheduling improvements from it.
It does not imply FP32 execution or promise a speedup for every workload.

FP64 without `target` reports `strict_fp64`, with `target=None` and no measured
errors. Native convergence/final-state checks are retained, but **no independent
observable-budget audit is claimed**. Its solver controls are at least as strict
as `1e-10` energy, `1e-8` density and `1e-12` screening; stricter caller settings
are preserved. These controls are not a bound on force error.

Supplying an explicit `TargetAccuracy`, including with Fast or FP64, requests a complete
same-backend comparison. Balanced, Strict and `audit_auto` also request these
experimental audits and cost two or three complete endpoints. A custom budget
is never silently dropped to improve a timing. `Calculator.target_accuracy`
behavior remains unchanged.

```python
from generativeqc import PrecisionCalculator

calculator = PrecisionCalculator(
    method="pbe0-rks", basis="def2-svp", device="cuda", policy="fp64",
)
```

```python
from generativeqc import (
    ObservableTarget, PrecisionCalculator, PrecisionPolicy, TargetAccuracy,
)

policy = PrecisionPolicy(
    "balanced",
    target=TargetAccuracy((
        ObservableTarget("energy", "absolute", "Eh", absolute=1e-7),
        ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-5),
        ObservableTarget("forces", "rms", "Eh/bohr", absolute=3e-6),
    )),
)
calculator = PrecisionCalculator(method="pbe-rks", basis="sto-3g", policy=policy)
result = calculator.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])
print(result.precision_validation.to_dict())
```

The preset names are case-insensitive. Balanced is the facade default.
Providing `target` replaces all preset audit requirements, so energy-only audits
use a target containing just energy. `properties` must include every explicit
target observable and energy; by default these are requested automatically.
Without a target, Fast defaults to energy and forces; requesting only energy
does not create a force-error measurement.

(python-precision-values)=
## Allowances, norms and evidence

| Preset | Energy allowance (Eh) | Max force-component allowance (Eh/bohr) | RMS force-component allowance (Eh/bohr) | Candidate SCF |
| --- | ---: | ---: | ---: | --- |
| Fast | `1e-5` | `1e-4` | `3e-5` | Qualified FP32 XC warmup plus strict FP64 final state; FP64 audit fallback |
| FP64 | No implicit budget audit | No implicit budget audit | No implicit budget audit | FP64, one endpoint unless an explicit target is supplied |
| `audit_auto` | `1e-5` | `1e-4` | `3e-5` | Native AUTO on admitted CUDA models; FP64 otherwise |
| Balanced | `1e-7` | `1e-5` | `3e-6` | FP64 |
| Strict | `1e-9` | `1e-7` | `3e-8` | FP64 |

These are selectable **allowances**, not measured accuracy promises or
universal chemistry thresholds. Screening or coarse optimization may tolerate
larger errors; labels for AI training must use their own accuracy requirements.
Set force-error allowances well below the actual optimizer stopping thresholds.
Maximum absolute Cartesian component and RMS over all `3*Natom` components are
different from maximum atomic gradient length and RMS of atomic lengths.

When an audit is requested, every norm must pass independently. A requirement resolves to
`absolute + relative * reference_norm`; a zero reference norm supplies no
relative allowance. Values and force differences must be finite, and force
arrays must have identical `(Natom, 3)` shapes. Coordinates are Bohr, energy is
Hartree, and forces are negative energy gradients in Hartree/Bohr.

`Result.precision_validation` records the profile, selection budget, requested target, actual
scientific-model identity, measured errors and allowances, rejected-candidate
errors, fallback reasons, solve roles, requested precision, backend, solver
controls, iterations, force evaluations and complete `singlepoint` wall time.
Fock-build counts are included only when execution reports valid counters;
missing counts remain `None`. `Result.precision` retains the native arithmetic
and operator-work evidence for the returned endpoint. Requesting AUTO is not
evidence that FP32 actually ran.

An audited success has status `observed_met`. It measures agreement with the stricter
reference, whose own uncertainty remains unresolved. It does not certify an
absolute error, identify the intended electronic root, or test stability.
The identity fixes geometry, basis/representation, electron/spin state,
functional/composition, quadrature/domain, auxiliary basis and fitted metric.
Changing these cannot count as a precision audit of the same target.

## Verification and cost

Each audited call executes a complete candidate and an independent FP64 reference,
including every requested force contribution. Reference energy, density and
screening controls are capped at `1e-12`, `1e-10` and `1e-14` respectively;
tighter caller controls remain in effect. These controls are a reference
recipe, not a conversion from force error to SCF residual.

If any norm exceeds its allowance, the candidate is discarded. The FP64
reference can be returned only after agreement with one additional FP64 solve
with controls tightened by another factor of ten. There are at most three
completed endpoints. The report retains the rejected comparison as well as
the successful returned-endpoint comparison.

An explicit-target or audit-profile call costs two full
endpoints, or three after rejection. Its wall time includes audit/fallback
construction, model checks and resource release, but excludes facade
construction and Python import. Audit profiles make no acceleration claim. Reusing error
estimates across geometries or skipping intermediate audits requires separately
qualified error/conditioning evidence; this facade does neither.

(python-precision-backends)=
## Arithmetic and supported endpoints

The facade supports all-electron HF and KS single points within the underlying
calculator's admitted property domain. CPU is supported; CUDA calls and FP64
audits remain on the selected CUDA device. Unsupported AUTO constructor domains
or requested properties fall back to the same model's FP64 calculator. An
energy-only request can still use AUTO when its force consumer is unavailable.
`audit_auto` uses the existing native
SCF AUTO policy with its FP64 refinement. It is distinct from Fast's
budget-selected XC warmup. FP64, Balanced and Strict keep SCF in FP64.

For CUDA PBE0, ordinary AUTO is **not all FP32**. Its schedule admits FP32 computation
with FP64 accumulation for Coulomb J, and for density contractions only where
the selected layout is qualified. Exact exchange K, XC point algebra and
reductions remain FP64; strict refinement restores FP64 throughout the schedule.
The mixed J implementation narrows ERI recurrence arithmetic, not all of J:
Gaussian preparation, density products and primitive sums also retain FP64.
Inspect execution-owned operator records rather than interpreting the AUTO
name or effective-bits summary as a uniform arithmetic type. The native SCF
census is not an instruction-level census of the analytic gradient.

Only the bounded PBE0/RKS domain above currently admits application-selected
FP32 gradients. `gradient_admission` and native work records distinguish it
from warmup-only and audited FP64 paths. Larger bases, other functionals and
spin domains do not acquire a gradient route merely from a looser allowance.

The facade does not expose prepared batches, correlated methods, ECP, Hessians,
response or an optimizer/MD driver. A successful single-point audit does not
qualify TS/IRC branches, small-curvature Hessian modes or NVE trajectories.
NVE admission additionally needs energy-drift, force/energy consistency,
continuity and sampling tests. The underlying native geometry optimizer is
unchanged by this Python facade.

## Reproducing PBE0 comparisons

`benchmarks/pbe0_precision.py` compares ordinary FP64, native AUTO and explicit
`audit_auto` at fixed PBE0/def2-SVP spherical models and moving quadrature. It requires
an independent PySCF reference and retains first calls separately from
alternating measured repeats. All runs start from the same Hcore guess; warm
timings reuse process caches, not converged densities. The endpoint includes
calculator construction, SCF, analytic forces, host export and cache release.
Audit timings include the complete audit/fallback work. The current benchmark
schema is version 2; older measurements retain their original names and source
identities and are not relabeled as results of the current revision.

With the source tree on `PYTHONPATH` and the matching native library selected,
generate the CPU reference first (PySCF is an optional benchmark dependency):

```bash
python -m benchmarks.pbe0_precision --engine reference \
  --output .artifacts/pbe0-precision/reference.json
```

On an available, permitted Slurm GPU node, run the native comparison through
the scheduler, preserving its assigned device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:30:00 \
  python -m benchmarks.pbe0_precision --engine native \
  --reference .artifacts/pbe0-precision/reference.json \
  --output .artifacts/pbe0-precision/native.json
```

Speed ratios are emitted only after independent energy/force gates pass.
The report preserves native operator evidence and explicitly flags incomplete
precision inventories instead of inferring FP32 execution from the request.

For Fast against already-optimized FP64, run `python -m benchmarks.pbe0_fast` inside the
same finite Slurm allocation, with `--reference`, `--output` and `--repeats 5`.
This reports the current ordinary FP64 default and budget-selected Fast, including
facade construction, strict refinement, analytic forces and any fallback work.
Use `--initial-guess automatic` to measure the default native Modified Huckel seed policy,
or `--initial-guess minao` for the explicit policy; the default benchmark uses
Hcore. Both timed modes always receive the same seed policy, and the independent
oracle's Hcore protocol remains separately recorded. It checks the same
scientific target; speed ratios are not kernel ratios.
Historical no-reuse/FP64 facade and audit measurements
remain bound to their original sources and are not relabeled.

(python-precision-errors)=
## Failure behavior

Invalid policy types, requirements, properties or molecular inputs raise
`TypeError`/`ValueError`; unsupported scientific domains raise
`NotImplementedError`. Native nonconvergence, allocation/execution failures,
nonfinite observables and mismatched reference models propagate without a
successful result. They are not reclassified as an in-budget endpoint.
FP64 propagates the same native failures; it does not convert a failed solve
into a successful precision report. Invalid/nonfinite solver controls are
rejected before applying FP64's convergence ceilings.

When a converged FP64 fallback still disagrees with its tighter verification,
`PrecisionBudgetError` carries `.validation` with `observed_unmet`, measured
norms and all completed work. It does not carry a publishable partial result.
Extremely small caller solver controls that underflow during tightening are
rejected. `precision` and `target_accuracy` constructor arguments are rejected
here: use the policy target to express the application requirement.

(python-precision-ownership)=
## Ownership and concurrency

Policies and report records are frozen. Targets own immutable requirements;
`to_dict()` returns detached data. Calculator inputs are snapshotted through the
native facade, including resolved local basis and KS settings. Calls on one
instance serialize; use independent instances for concurrent execution.

Endpoint workspaces are released before starting the next audit owner, so a
retained candidate does not double the resident resource budget. Returned
arrays retain the ordinary `Result` ownership contract and outlive those
workspaces. `clear_cache()` remains available and serializes with active calls.
Frozen `Result` fields do not make force arrays or native evidence dictionaries
recursively immutable.

