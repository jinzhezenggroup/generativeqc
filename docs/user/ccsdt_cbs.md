# MP2-assisted CCSD(T)/CBS energy and forces

`CCSDTCBSCalculator` composes native HF, MP2 and CCSD(T) calculations. It uses
two larger bases for MP2 extrapolation and one smaller basis for the
CCSD(T)−MP2 correction. It does **not** run CCSD(T) in both larger bases, or
use a reference package or finite differences as a production force backend.

## Run the workflow

Supply checked local [canonical basis records](external_basis.md), with explicit
cardinal numbers from one consistent family. For example, TZ/QZ MP2 plus a DZ
correction:

```python
from generativeqc import CBSBasis, CCSDTCBSCalculator, load_basis

workflow = CCSDTCBSCalculator(
    (CBSBasis(load_basis("cc-pvtz.json"), 3),
     CBSBasis(load_basis("cc-pvqz.json"), 4)),
    correction_basis=load_basis("cc-pvdz.json"),
    device="cuda",
    calculator_options={"correlation_memory_budget_bytes": 2 << 30},
)
result = workflow.singlepoint(
    [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
    properties=("energy", "forces"),
)
print(result.energy, result.forces)
print(result.hf_energy, result.mp2_correlation_energy, result.delta_ccsdt_energy)
for component in result.components:
    print(component.roles, component.elapsed_seconds, component.result.correlation)
```

The example presupposes that those local records exist and their native
operator/derivative domains are admitted; names do not download basis data or
promise an executable GPU domain. Use `import_bse` for an explicit local BSE
conversion. The bundled named-basis subset is not a complete CBS basis library.
If `correction_basis` is omitted, the lower MP2 basis is also the correction
basis and its MP2 result is reused. The same reuse applies when the explicitly
supplied correction record has either MP2 basis's full identity.

Set `properties=("energy",)` to omit all analytic forces. The default requests
both energy and forces. Ionic charge is passed identically to every component;
only a closed-shell singlet is currently admitted.

## Fast convergence

Pass `policy="fast"` to `CCSDTCBSCalculator` to relax SCF, CCSD, Lambda and
orbital-response convergence. `policy="standard"` (the default) and `"fp64"`
retain the original strict controls. Fast also enables bounded FP32 auxiliary
preconvergence for CUDA DF-CCSD by default. Final energy/residual checks,
Lambda, triples and analytic force response use FP64. The selected Hamiltonian,
basis groups and complete analytic-force work are preserved. Relaxed convergence
applies to conventional CPU/CUDA CBS and both CUDA DF reference models;
CPU and conventional CCSD keep FP64 arithmetic.

```python
from generativeqc import PrecisionPolicy, TargetErrorBudget

workflow = CCSDTCBSCalculator(
    (CBSBasis(load_basis("cc-pvdz.json"), 2),
     CBSBasis(load_basis("cc-pvtz.json"), 3)),
    auxiliary_bases=(load_basis("cc-pvdz-jkfit.json"),
                     load_basis("cc-pvtz-jkfit.json")),
    hf_reference="df",
    device="cuda",
    policy="fast",
    calculator_options={"correlation_memory_budget_bytes": 32 << 30},
)
result = workflow.singlepoint(atoms)
print(result.convergence)
```

For a smaller allowance, replace `policy="fast"` with
`policy=PrecisionPolicy("fast", budget=TargetErrorBudget(5e-6, 5e-5, 1.5e-5))`.

The default composite allowance is `1e-5 Eh` energy, `1e-4 Eh/Bohr` maximum
force error and `3e-5 Eh/Bohr` RMS force error. The workflow first combines
coefficients of repeated method/basis components, then divides each allowance
by the sum of absolute combined coefficients. This accounts for extrapolation
amplification without assuming cancellation. For default DZ/TZ extrapolation
with a DZ correction that sum is approximately `4.196866`.

The component allowance selects a scale `s`: the minimum of one and its
energy/max-force/RMS ratios to `1e-5`, `1e-4`, `3e-5`, respectively (omit the RMS
ratio when unspecified). The following ceilings apply:

| Control | Standard CBS default | Fast ceiling |
| --- | ---: | ---: |
| Reference SCF energy change | `1e-12` | `max(1e-11, s * 1e-8)` |
| Reference SCF density change | `1e-11` | `max(1e-11, s * 1e-9)` |
| CCSD energy change | `1e-12` | `1e-8` |
| CCSD residual maximum | `1e-10` | `max(1e-9, s * 1e-6)` |
| Lambda residual | `1e-9` | `max(1e-9, s * 1e-7)` |
| Response absolute solve tolerance | `1e-12` | `max(1e-12, s * 1e-8)` |
| Independent CC orbital residual | `1e-10` | `max(1e-10, s * 1e-7)` |
| Complete CC orbital stationarity | `1e-8` | `max(1e-8, s * 1e-6)` |

Explicit tighter SCF/CC values in `calculator_options` take precedence. Below
`s=1e-3`, Fast retains standard convergence and reports
`component_budget_requires_strict`. An older native library without correlated
convergence support rejects an admitted Fast request before execution. DF Fast
also requires native precision telemetry so the executed arithmetic is observable.

Fresh physical residual replays still gate publication using the selected
limits. Reference canonical-frame checks, denominator/gap safeguards,
nonfinite rejection, iteration limits and resource checks remain active.
The more conservative SCF density ceiling compared with DFT Fast preserves
the post-HF canonical-reference contract. Integral screening remains unchanged.
DF-CCSD warmup switches to FP64 at a residual of at least 100 times the selected
CC residual ceiling, or after at most 24 warmup evaluations. It clears the mixed
DIIS history before FP64 iterations. Resource or provider refusal retains the
bounded FP64 path; numerical failures remain visible.

`result.convergence` records the requested policy, selected route, composite
and component allowances, coefficient sum and configured limits. Its
`arithmetic_request` records the FP32 preconvergence request; `arithmetic`
reports `"mixed"` only when FP32 GEMMs actually executed, otherwise `"fp64"`.
Component `cc_performance` records FP32 admission, refusal, evaluation, GEMM and
cast counts. Native diagnostics retain measured residuals and iteration counts. The
allowance-to-tolerance mapping is a stopping policy: it is not a per-call
observable certificate, a mathematical error bound, or a basis-set error
estimate. No strict duplicate endpoint is run to estimate the error. Validate
representative molecules against Standard or an independent reference when
choosing a budget. Explicit `PrecisionPolicy.target` audits and other precision
presets are not implemented for CBS; private correlated-budget transport also
rejects prepared batches and legacy RI-MP2.

## Correlation-only DF-CBS on CUDA

Supply `auxiliary_bases=(low_fit, high_fit)` to select **DF-MP2 correlation**
extrapolation and a **DF-CCSD(T)−DF-MP2** correction. The reference is still
conventional exact RHF in every orbital basis; this is not a DF-HF composite.
When the CUDA reference uses nonresident direct integrals, preliminary FP64
iterations use Schwarz and density-weighted J/K screening. The same plan then
clears its operator-dependent DIIS history and reconverges with unscreened FP64
integrals before publishing the reference. Final convergence and physical-frame
checks are unchanged; this does not introduce a screened reference into the
correlation energy or analytic forces. Resident-integral routes retain their
existing execution policy.

For a distinct correction orbital basis, also supply
`correction_auxiliary_basis`. For example, with checked local records:

```python
workflow = CCSDTCBSCalculator(
    (CBSBasis(load_basis("cc-pvdz.json"), 2),
     CBSBasis(load_basis("cc-pvtz.json"), 3)),
    auxiliary_bases=(load_basis("cc-pvdz-ri.json"),
                     load_basis("cc-pvtz-ri.json")),
    device="cuda",
    calculator_options={"correlation_memory_budget_bytes": 2 << 30},
)
result = workflow.singlepoint(atoms, properties=("energy", "forces"))
```

The lower basis is the default correction basis, so this example executes two
native groups, not six standalone calculations. Each group owns one converged
RHF reference. The correction group retains one original whitened DF source,
metric and MO frame through both correlation consumers. Its MP2 companion
does not repeat SCF, three-center integrals, metric factorization or MO source
construction, and does not solve CC or Lambda. HF forces use the same stationary
reference. Method-specific orbital response solves remain separate.

Equal orbital basis identities must share the same auxiliary identity. These
identities are snapshots, not mutable file names. Different orbital bases do
not share orbitals or integral factors. Native DF source, metric-rank,
derivative, convergence and budget checks still apply. Analytic forces include
auxiliary-center and metric derivatives as well as the exact RHF response.
This correlation source/nuclear sink admits orbital shells through f and
auxiliary shells through g, including spherical fitting bases used with TZ.
This scoped admission does not widen generic CUDA DF exports or orbital g.
There is no inherited conventional 12/56-AO force cutoff on this DF owner, but
this is not an unlimited-size admission promise: the source retains
`O(nAO² naux)` values and MP2's generated energy/response scratch is
`O(nocc² nvir²)`. A representative QM7 molecule needs actual basis/auxiliary
counts and endpoint qualification; atom count alone does not establish fit.

Standalone same-reference DF-MP2 is selected with
`Calculator(method="mp2", density_fitting="correlation_cuda", device="cuda",
auxiliary_basis=..., correlation_memory_budget_bytes=2 << 30)` and an explicit
`singlepoint(..., properties=("energy", "forces"))` request. This DF owner
requires a positive explicit correlation budget; standalone DF-MP2 defaults
to energy-only. The older `density_fitting="cuda"` RI-MP2 selector
keeps its DF-HF reference and energy-only CUDA contract; it is not silently
reinterpreted. Correlation-only DF-MP2 currently supports single-point/groups,
not prepared batches or CPU fallback.

## Full DF-HF/RI-JK CBS on CUDA

Add `hf_reference="df"` to the workflow with explicit `auxiliary_bases` to
use a stationary RI-JK reference throughout, not merely a DF initial guess.
The same RI operator supplies the orbital-response actions and final residual;
analytic forces include the three-center, auxiliary-metric, nuclear-center,
orbital and Pulay derivatives. No four-center RHF audit is substituted into
this different Hamiltonian. Allocation failures do not silently change it.

Each supplied auxiliary record currently serves **both JK and correlation**.
Choose and validate a fitting basis suitable for both purposes; using an
MP2-RI basis for HF is a separate fitting approximation, not automatically
equivalent to a JK-fit basis. Separate JK-fit and correlation-fit records
within one orbital-basis group are not yet supported. The old default
`hf_reference="exact"` preserves the correlation-only model and its results.
The returned `basis_metadata["hf_reference"]` distinguishes the models.

The native SCF RI-JK plan, its metric and factors stay alive through all
same-basis HF/MP2/CCSD(T) companion responses. Correlation retains its own
MO-source owner shared by the correlation consumers. Its construction does
not yet borrow SCF's AO factors; this is not a claim of a single factor build
across SCF and correlation. Different orbital-basis groups remain sequential.

Standalone full-DF analytic E+F uses
`Calculator(method="mp2", density_fitting="full_cuda", device="cuda",
auxiliary_basis=..., correlation_memory_budget_bytes=2 << 30)`;
`method="df-ccsd(t)"` accepts the same selector. This is distinct from legacy
`density_fitting="cuda"` RI-MP2. Full DF currently supports single points and
same-basis groups, not prepared batches, CPU fallback, or open-shell variants.

The positive `correlation_memory_budget_bytes` bounds complete numeric
capacity, including simultaneous host/device owners and a conservative retained
RI-JK allowance. It is not a measured GPU-memory peak. A budget refusal does
not authorize switching to exact RHF, omitting Lambda/response, or publishing
energy-only work as complete E+F. Choose an explicit allowance appropriate to
both the endpoint and available device memory.

(python-cbs-values)=
## Scientific model and values

For cardinals $X<Y$ and small basis $S$, the model is

```{math}
E = E_{\mathrm{HF}}^{\mathrm{limit}}
  + E_{\mathrm{corr,MP2}}^{\mathrm{limit}}
  + E_{\mathrm{CCSD(T)}}^S-E_{\mathrm{MP2}}^S.
```

MP2 **correlation** energy, not total MP2 energy, uses
$E_{\mathrm{corr}}(X)=E_{\mathrm{corr}}^{\mathrm{limit}}+A X^{-p}$.
The default is $p=3$. Its low/high weights are

```{math}
(w_X,w_Y)=\left(-\frac{X^p}{Y^p-X^p},\frac{Y^p}{Y^p-X^p}\right).
```

HF is handled separately. The default `hf_extrapolation="exponential"` uses
$E_{\mathrm{HF}}(X)=E_{\mathrm{HF}}^{\mathrm{limit}}+B e^{-\alpha X}$,
with fixed `hf_exponent=1.63`. Writing $r=e^{-\alpha(Y-X)}$, its weights are
$(-r/(1-r),1/(1-r))$. `hf_extrapolation="highest"` instead uses HF in basis
$Y$, with weights $(0,1)$; it is explicitly a highest-basis approximation,
not an independently extrapolated HF limit. `correlation_exponent` and
`hf_exponent` are caller-selected model inputs, not geometry-fitted parameters.
These are conventional extrapolation forms, not universal error guarantees;
see Helgaker et al., *J. Chem. Phys.* **106**, 9639 (1997), and the
[Psi4 CBS scheme definitions](https://psicode.org/psi4manual/master/cbs.html).

Every component uses the same geometry, charge, spin, nonrelativistic
all-electron reference and full correlated occupied space. The default uses
exact correlation integrals; explicit auxiliaries select correlation DF, while
`hf_reference` selects exact or RI-JK reference response. Final physical checks
and response use FP64; [Fast](#fast-convergence) can use FP32 DF-CCSD warmup.
Frozen-core, ECP and unrestricted
variants are not admitted. Cardinals must be increasing integers, the larger
bases must have distinct identities, and all bases must have the same AO
representation. Cardinal numbers are explicit declarations; an arbitrary
record is not certified as a systematic CBS family merely by labeling it TZ.

Because the weights are geometry independent, the analytic force is

```{math}
\mathbf F=\mathbf F_{\mathrm{HF}}^{\mathrm{limit}}
  +\sum_{K=X,Y}w_K(\mathbf F_{\mathrm{MP2}}^K-\mathbf F_{\mathrm{HF}}^K)
  +\mathbf F_{\mathrm{CCSD(T)}}^S-\mathbf F_{\mathrm{MP2}}^S.
```

Do not use extrapolated energy with only the larger MP2 force. Do not refit
exponents at each geometry without their derivatives. Same-basis components
execute as one native group: MP2 supplies an HF companion, and CCSD(T) supplies
MP2 and, when needed, HF companions. Each group shares its converged canonical
RHF reference and orbitals. In the conventional model they also share the exact
AO interaction owner. The correction-basis
MP2 energy and initial force adjoint also borrow the already transformed CC
OVOV block. Its remaining orbital-RHS, relaxed-weight and response-operator
reads borrow a complete six-block CC ERI symmetry cover when admitted, rather
than repeating AO-to-MO transformations. Equal method/basis identities still
contribute only once per call.

The other conventional CUDA MP2 E+F groups can build the same six-block symmetry cover once
from their own exact device-readable RHF source. This optional cache is admitted
only when it uses at most one quarter of the remaining correlation phase budget,
its construction fits, and the full force consumer fits alongside it. Otherwise
the existing bounded force path reconstructs tiles. Only memory failures trigger
that fallback; numerical errors propagate. Cache production runs on CUDA, but
its returned MO blocks and some response orchestration retain host storage.

Coordinates use Bohr, energy Hartree, and force Hartree/Bohr with shape
`(natoms, 3)` and sign `force = -gradient`. `CBSResult` exposes the three energy
and force contributions, fixed weights, actual completed native components,
basis identities/cardinals, native `execution_groups` and endpoint timing. Each component records its
combined coefficient and native diagnostics; the weighted total components
equal the composite. All force fields are `None` for energy-only calls.
Component timings identify their containing group and are **not additive**;
use `execution_groups` for non-overlapping group times. Companion convergence
fields describe the completed group, not an additional standalone method solve.
Convergence of every component is necessary but does not certify CBS model
accuracy. No basis-error certificate is manufactured from SCF convergence.

(python-cbs-errors)=
## Errors and publication

Malformed cardinals, representations, extrapolation parameters, occupations,
properties and unsupported calculator options fail with validation errors.
Unsupported Hamiltonians/spin domains fail explicitly. Local basis reads retain
their filesystem/checksum errors. Native method, basis, derivative, memory,
stationarity, response and convergence failures propagate without a partial
`CBSResult`; a successful subset is not a successful composite. Nonfinite
energies/forces or a component executing on the wrong backend also reject.
Previously returned results remain independent of a later failed call.

`calculator_options` supplies common native SCF/CC convergence, screening,
initial-guess and workspace/resource controls. Defaults tighten SCF energy and
density tolerances to `1e-12`/`1e-10`, CC energy/residual tolerances to
`1e-12`/`1e-10`, with 200 SCF and 150 CC iterations. It cannot override component
methods/bases, precision, frozen core or density fitting. Native admissibility
checks still apply to these controls. `Calculator.singlepoint_components` also
exposes same-reference groups directly for conventional, full and correlation-only
DF MP2/CCSD(T).

(python-cbs-ownership)=
## Ownership and concurrency

`CBSBasis` snapshots local records at construction. Modifying a file or the
caller-owned options dictionary does not change an existing workflow. Keep the
workflow's configuration fixed during use. One workflow serializes its calls;
use separate instances for concurrent execution.

Native basis groups execute sequentially and release their retained single-point
storage in `finally`, including failed calls. Only one native endpoint owner is
retained at a time. The Python workflow retains owned basis data and completed
result/force diagnostics, not the several bases' CUDA executables. Published
arrays outlive cleanup and are separate from the composite force storage.
Frozen dataclasses do not recursively freeze their NumPy arrays or mappings.

Within a group, CC-only blocks/amplitudes are retired after its complete E+F
replay. Only an admitted six-block MO cache, reference/source and published primary outputs stay
live beside the companions. Every companion admission charges these live
reservations. A memory-only fallback can release optional MO/source/plan
residency and reconstruct bounded tiles; numerical failures are never retried
through a different scientific model. Different-basis orbitals/integrals are
not interchangeable. MP2 and CC retain separate method-specific adjoint/Lambda
and orbital-response solves; this is not a shared multi-RHS response solver.

Native budgets apply independently to each **group**, not to total process RSS,
Python storage or a scheduler-wide device reservation. Planned/measured peaks
are available in each native result and must not be added or confused with a
measured whole-workflow allocation peak. This policy trades retained prepared-
plan warm reuse for bounded sequential ownership; it does not claim a fully
resident multi-basis GPU workflow.

(python-cbs-backends)=
## Backends and qualification

Select `device="cpu"` or `device="cuda"` explicitly; there is no CPU fallback
when CUDA is requested. `executed_backend` retains the native backend spelling
(`cpu_reference` or `cuda`). Chemistry, force-domain and resource support are
inherited from each actual native component, not broadened by composition.
In particular, legacy DF-HF CUDA RI-MP2 forces remain unsupported, and the small
CCSD(T) force calculation must fit its actual native domain. The explicit
correlation-only DF owner is separate from conventional force admission. This orchestration
does not remove remaining host staging or improve the asymptotic MP2 response
algorithm. Optional MO caching reduces repeated transformations, not the
method-specific response work or its domain limits.

Independent reference data live under `tests/reference_data/cbs/`. H2 uses the
physical spherical cc-pVDZ/cc-pVTZ sequence. A separate water **synthetic pair**
tests the composition with nonzero triples; it is not evidence of predictive
CBS accuracy for water. Both reference composites are checked against
independently reconverged displaced energies. The public native acceptance
gates are `1e-9` Hartree for energy, `2e-6` Hartree/Bohr for force and composite
directional derivatives, and every constituent endpoint is checked separately.

Run CPU contracts with `pytest tests/python/test_ccsdt_cbs*.py`. CUDA qualification
requires `GENERATIVEQC_CBS_CUDA_TEST=1` and a real Slurm allocation. The
physical H2 DZ/TZ **CPU force** replay is opt-in through
`GENERATIVEQC_CBS_LARGE_TEST=1`; routine CPU CI checks its energy and the water
algebra E+F case, while allocated CUDA qualification checks both full E+F cases.
The endpoint
benchmark `benchmarks/ccsdt_cbs_endpoint.py` times the **whole** workflow,
including component construction, SCF, correlation, analytic forces and owner
cleanup, and optionally records an untimed semantic-work replay. Timed replay
samples are not advertised as retained prepared-batch warm calls.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 python benchmarks/ccsdt_cbs_endpoint.py \
  --device cuda --case h2-dz-tz --oracle --work --repeats 1 \
  --output .artifacts/ccsdt-cbs-h2.json
```
