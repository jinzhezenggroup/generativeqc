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
all-electron Coulomb Hamiltonian, exact integrals and full correlated occupied
space. The workflow is FP64. Frozen-core, ECP, density-fitting and unrestricted
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
RHF reference, orbitals and exact AO interaction owner. The correction-basis
MP2 energy and initial force adjoint also borrow the already transformed CC
OVOV block. Its remaining orbital-RHS, relaxed-weight and response-operator
reads borrow a complete six-block CC ERI symmetry cover when admitted, rather
than repeating AO-to-MO transformations. Equal method/basis identities still
contribute only once per call.

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
exposes same-reference groups directly for conventional MP2 and CCSD(T).

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
In particular, CUDA RI-MP2 forces remain unsupported, and the small CCSD(T)
force calculation must fit its native qualified domain. This orchestration
does not remove remaining host staging or improve asymptotic MP2 response work.

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
