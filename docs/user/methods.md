# Methods and supported calculations

List the available method names with the native CLI:

```bash
generativeqc methods
generativeqc methods --libxc --json
generativeqc methods --compositions --json
```

The first command lists native methods; `--libxc` includes imported semilocal
functionals, and `--compositions` lists compiler-generated compositions.
The [method table](../public_methods.md) is generated from the method registry.

A method name is not a guarantee that every basis, backend, spin state or
property is supported. Check the settings you intend to use:

```python
from generativeqc import Calculator

calc = Calculator(method="pbe0-rks", basis="def2-svp", device="cuda")
print(calc.capabilities.supported_properties)
print(calc.capabilities.supported_second_order)
```

`Calculator.capabilities` describes the chosen calculation context.
Unsupported combinations raise an error instead of silently selecting
a different method or backend. See [capability reference](../reference/capabilities.md)
for details.

(conventional-cpu-ump2-energy-and-analytic-forces)=
## Unrestricted MP2 on CPU

`ump2` computes unrestricted MP2 energy from a converged UHF reference.
Its current implementation uses all-electron, unscreened conventional
Gaussian integrals and FP64 on CPU. It does not support CUDA, RI/DF,
frozen core, ECP or mixed precision.

```python
calc = Calculator(method="ump2", basis="sto-3g", device="cpu")
result = calc.singlepoint([("Li", (0, 0, 0))], multiplicity=2)
print(result.energy)
print(result.correlation)
```

Energy is the default output. Request analytic forces explicitly:

```python
result = calc.singlepoint(
    [("O", (0, 0, 0)), ("H", (0.2, 0, 1.8))],
    multiplicity=2,
    properties=("energy", "forces"),
)
print(result.forces)
```

The force includes orbital response, Pulay terms and same-/opposite-spin
correlation. Spin orbitals remain separate even with equal occupations.
The physical UHF state matters: an SCF-converged broken-symmetry solution
is not necessarily the state you intended. A failed response or insufficient
numeric budget raises rather than returning incomplete forces.
`result.correlation` includes spin-resolved contributions, residuals and
workspace diagnostics. The workspace estimate is not measured process RSS.

Prepared UMP2 calculations can restore compatible two-spin HF density
checkpoints as starting guesses; the target UHF calculation must still
converge. See [checkpoint/restart](checkpoint.md) and the
[MP2 implementation](../developer/mp2.md).

## Restricted correlated methods

`mp2`, `rccsd`, `rccsd(t)`/`ccsd(t)`, and CUDA density-fitted
`df-rccsd(t)`/`df-ccsd(t)` use a restricted HF reference.
Backend, batch, memory and force support differ by selector.
See [restricted correlated methods](correlated_methods.md) and the
[CCSD(T)/CBS workflow](ccsdt_cbs.md).

## Second-order derivatives

Hessians and Hessian-vector products are available only for supported
calculation contexts. Check `calc.capabilities.supported_second_order`
before calling `calc.hessian(...)` or
`calc.hessian_vector_product(...)`. Results are analytic Cartesian
derivatives, not finite differences. The supported methods, resource
arguments and result shapes are in [Hessians and HVPs](hessians.md).

## Repeated GFN2-xTB calculations

Reuse a `Calculator(method="gfn2-xtb")` for repeated energy and force calls.
It keeps its workspace but starts a fresh SCC solve on each call, even when
coordinates change. Calls on the same calculator are serialized.
`calc.clear_cache()` releases the retained workspace.

## DFT forces

Request analytic forces using the ordinary property interface:

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

Available DFT forces depend on the functional, basis and backend.
Some direct CUDA calculations support `precision="auto"` for selected
SCF operations; exact exchange and final-state checks may remain FP64.
PBE0/B3LYP-style global hybrids and generated split hybrids have different
precision restrictions. See [KS options](ks_options.md),
[dispersion](dispersion.md) and
[CUDA hybrid validation](../maintainer/hybrid_cuda_acceptance.md).

## HF force convergence

Direct CUDA HF forces require a final physical-state check beyond the usual
energy and density stopping tests. It may add work at the end of SCF.
See [HF force finalization](../developer/hf_force_finalization.md).

The [roadmap](../maintainer/roadmap.md) lists work in progress; check
`Calculator.capabilities` for current support.
