# Automatic Libxc semilocal functionals

GenerativeQC exposes structurally supported imported Libxc semilocal registrations
without a positive method allow-list.

## Select a functional

Use the exact Libxc registration name:

```python
from generativeqc import Calculator

calculator = Calculator(
    method="libxc:GGA_X_APBE",
    basis="sto-3g",
    device="cpu",
)
result = calculator.singlepoint(
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    properties=("energy",),
)
```

`libxc:` defaults to restricted KS. The explicit spellings are:

- `libxc-rks:NAME` for unpolarized RKS;
- `libxc-uks:NAME` for polarized UKS.

List the currently admitted automatic selectors with:

```console
generativeqc methods --libxc
```

Add `--json` for machine-readable output.

## Admission policy

Automatic semilocal admission is structural and default-allow. An imported
registration is available when its Graph can be represented with the currently
supported `rho`, `sigma`, and `tau` ingredients. Registrations that require
another ingredient are rejected structurally. Adding another representable
Libxc registration does not require adding it to a positive GenerativeQC method table.

Zero-spin channels, zero gradients, density tails, and zero tau are not encoded
as functional-name exclusions. The compiler applies the pinned Libxc work-domain
policy generically before evaluating the imported interior Graph: density/sigma/
tau work floors, cross-spin sigma bounds, and meta-GGA FHC handling are shared
runtime semantics. The explicit exception hook is reserved for a genuinely
functional-specific defect and is currently empty.

The imported Graph uses its physical spin-density inputs directly for Maple
density-screening predicates. This avoids a rounded inverse `rs`/spin-polarization
conversion accidentally activating an empty-spin branch at the density floor.
Screening remains part of the canonical Graph before symbolic differentiation;
no backend-specific correction is applied to the resulting potential.

## Current execution scope

Automatic CPU and CUDA energy/SCF use the common compiler-owned point
mathematics. Explicit analytic forces execute through the prepared native
stationary owner, with the selected rho/sigma/tau differential, moving-grid
partition response, AO-center derivatives, Pulay terms and Coulomb response.
The installed package must contain the requested provider and derivative artifacts.
Official CUDA wheels and release CLI presets package the complete structurally
representable automatic Libxc catalog for their selected CUDA architecture(s),
not just SM120. Development builds may opt out of automatic CUDA AOT.
Packaging is not molecular SCF/force qualification for every registration.
Energy remains the default in one-shot and prepared calls:

```python
result = calculator.singlepoint(atoms, properties=("energy", "forces"))
```

CUDA automatic forces currently require strict FP64, device-fused XC,
all-electron s/p bases, Direct J, and three partition iterations. Prepared replay
rechecks the current state and resources; a failed force request cannot become a
successful energy-only result. CPU force admission also follows the current
native owner and its resource estimate. The historical small Python stationary
adapter's fixed atom/AO limits do not describe this native path.

Backend-neutral `method_capabilities()` remains conservatively energy-only.
`Calculator.capabilities` and the installed native preparation boundary describe
the selected context. Automatic CUDA DF/ECP forces, Fast/mixed force arithmetic,
and automatic response/Hessian/HVP coverage are not implied by semilocal energy
or first-force success. In particular, the physical empty-spin particle-birth
guard remains active; pointwise boundary agreement does not qualify a
particle-changing molecular continuation.

The generated [native qualification report](../reference/libxc_qualification.md)
records the measured family/backend/spin/product counts and exact blockers.
Its scopes include the executed binary, precision, quadrature, geometries,
independent reference gates and cold/replay/moved lifecycle. These are optional
qualification claims, separate from structural selector admission. A compiled
registration without a molecular receipt is not counted as molecule-qualified.

Recheck the retained evidence against the current checkout without running a
calculation:

```console
PYTHONPATH=python:. python tools/check_libxc_native_catalog.py
```

The [qualification procedure](../developer/libxc_bulk_capabilities.md#native-endpoint-catalog)
describes fresh CPU/CUDA campaigns and evidence updates. Independent PySCF
2.14.0 / Libxc 7.0.0 references are used only by offline qualification tools.

The native method ID used internally is only an ingredient/spin execution
provider. The scientific identity, component name, Graph, compiler work policy, SCF domain
and final state retain the selected Libxc registration. Work-coordinate floors
and clamps are not differentiated through; E/vxc is evaluated at work coordinates
and the energy uses the original physical total density.

## Registration names are literal

A Libxc registration can represent only one component. For example,
`GGA_X_APBE` is an exchange component; selecting it does not silently add a
correlation functional. Use the intended Libxc XC registration when one exists,
or an explicit composed MethodIR when composition support is available. GenerativeQC
does not infer missing X/C partners from a name.
