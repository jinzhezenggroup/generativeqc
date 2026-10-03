# Methods and capability discovery

GenerativeQC exposes public method selectors through several coordinated sources:
stable native ABI registrations, compiler-discovered DFT selectors, public
composite methods, and automatically imported Libxc semilocal MethodIR entries.

Use the runtime discovery command for the current public selector set:

```bash
python -m generativeqc methods
python -m generativeqc methods --json
```

The documentation view is generated from the same sources in the
[public method catalog](../public_methods.md). Do not treat a handwritten list in
another page as a support matrix.

## How to interpret a listed method

A listed selector means that GenerativeQC can identify that public method. It does
**not** mean that every backend, basis, spin state, precision, density-fitting
mode, ECP, grid, batch shape, or requested property is qualified.

After those execution choices are fixed, `Calculator.capabilities` is the
authoritative public view for that calculation context. Prepared batches expose
the corresponding contextual capability record. Unsupported combinations fail
closed rather than silently changing the requested method, backend, or property.
Per-system geometry, electron-count, convergence, and resource checks still apply
during preparation and execution.

For the ownership of each capability source and why GenerativeQC does not maintain
a second handwritten CPU/GPU matrix, see
[Capability sources and interpretation](../reference/capabilities.md).

A Python-free native SDK install has a separate `generativeqc methods` command.
It reports the stable C/C++ ABI/provider registry and intentionally does not
import the Python compiler catalog.

## Second-order derivatives

The Python `Calculator` exposes analytic Cartesian Hessian-vector products and
full Hessians only when the selected execution context advertises second-order
support. For example, the qualified CPU direct all-electron strict-FP64
closed-shell LDA/PBE RKS domain can be queried and used as follows:

```python
from generativeqc import Calculator, GridSpec, KsOptions

atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
calc = Calculator(
    method="pbe-rks",
    basis="sto-3g",
    device="cpu",
    precision="fp64",
    ks_options=KsOptions(grid=GridSpec()),
)

if calc.capabilities.supported_second_order:
    hvp = calc.hessian_vector_product(
        atoms,
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        integral_budget_bytes=64 << 20,
    )
    hessian = calc.hessian(
        atoms,
        integral_budget_bytes=64 << 20,
        output_budget_bytes=64 << 20,
    )
```

The full Hessian is the raw analytic matrix. Unsupported second-order domains
remain fail-closed; callers should use the contextual capability record rather
than infer support from the method name alone. Implementation and response
details belong in the [Hessian developer documentation](../developer/hessian.md).

## Repeated GFN2 calculations

Reuse a `Calculator(method="gfn2-xtb")` for repeated molecular energy/force
calls. It retains one native workspace and starts fresh SCC on every call,
including changed coordinates. Changes to the molecule or scientific controls
are checked before reuse. Calls on the same calculator are serialized.

Call `calc.clear_cache()` when you want to release its resident CPU/GPU storage;
the next `singlepoint()` rebuilds the workspace. Collection of the calculator
also releases this storage. Separate calculators own separate workspaces.

## DFT forces

Analytic DFT forces are requested through the ordinary property interface when
the selected method/backend context advertises force support:

```python
from generativeqc import Calculator, GridSpec, KsOptions

calc = Calculator(
    method="b3lyp-rks",
    device="cuda",
    precision="fp64",
    basis="sto-3g",
    ks_options=KsOptions(
        grid=GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20),
        xc_schedule="device_fused",
    ),
)
result = calc.singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    properties=("energy", "forces"),
)
```

Admitted direct CUDA KS paths can use component-wise `precision="auto"`:
qualified Direct Coulomb J and density contractions may use reduced compute
precision while exact K and final-state audits remain FP64. For global-hybrid
forces, AUTO admission is limited to PBE0/B3LYP-style routes; generated split
hybrids retain strict FP64. See the
[CUDA hybrid acceptance contract](../maintainer/hybrid_cuda_acceptance.md).

Generated CUDA consumers may require a discoverable CUDA toolkit during first
use and may reuse compiler artifacts on later calls. Exact backend, basis,
precision, exchange, nonlocal-correlation, ECP, and resource admission remains
context-specific. See [KS options](ks_options.md), the method-specific user
guides, and `Calculator.capabilities` instead of copying those gates here.

## HF force convergence

Direct CUDA HF force publication applies an additional physical commutator check
and final physical-state validation beyond the ordinary density/energy stopping
criteria. This can add finalization work while preserving the requested
iteration limit and fail-closed convergence behavior. The implementation
semantics, Pulay construction, work counters, and diagnostic ownership are
documented in [Direct CUDA HF force finalization](../developer/hf_force_finalization.md).

## Planned work is not current support

The [implementation roadmap](../maintainer/roadmap.md) describes development
directions only. A roadmap entry, compiler representation, generated kernel, or
low-level capability record is not a public execution promise until the
execution-context capability gate admits it.
