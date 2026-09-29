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

## Current execution scope

The automatic path is currently CPU, FP64, semilocal energy/SCF only. CUDA,
analytic forces, exact exchange, range separation, and nonlocal correlation do
not become available merely because a semilocal registration is importable.

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
