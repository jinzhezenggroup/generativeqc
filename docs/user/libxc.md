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

The automatic path supports CPU FP64 semilocal energy/SCF. Direct all-electron
s/p/d contexts also support explicitly requested analytic forces through the
shared stationary gradient consumer, including UKS and tau meta-GGA. Energy
remains the default in both one-shot and prepared-batch calls:

```python
result = calculator.singlepoint(atoms, properties=("energy", "forces"))
```

Forces require the exact selected native point provider and its rho/sigma/tau
derivatives, moving-grid/partition response, AO-center derivatives, Pulay terms
and J-only integral response. No functional-specific force formula or reference
callback is used. The existing bounded CPU force contract applies: at most eight
atoms, sixteen AOs, 128 primitives, one million grid points, and a 256 MiB
additional numeric host-workspace cap. Force requests are revalidated and
rebudgeted before prepared replay; a failed derivative never becomes a successful
energy-only result.

Backend-neutral `method_capabilities()` remains conservatively energy-only;
`Calculator.capabilities` describes the selected context. CUDA, ECP and
density-fitted automatic forces, Hessians/HVPs, exact exchange, range separation,
and nonlocal correlation remain outside this automatic force contract.

Independent qualification can be reproduced with the reference-test dependencies
and a CPU native build:

```console
PYTHONPATH=python python tools/qualify_libxc_forces.py \
  --functional GGA_X_APBE --output .cache/libxc-forces/GGA_X_APBE.json
```

The tool reconverges native and pinned PySCF 2.14.0 / Libxc 7.0.0 energies at every
finite-difference coordinate on the same native quadrature, in both spin layouts
and all cold/replay/changed-geometry phases. Machine-readable `forces` evidence
binds the functional/source, binary, method, plan, spin, grid and solver identities;
incomplete or failing matrices cannot claim endpoint coverage. These receipts are
qualification data, not a positive public method list. See the
[generic CPU force decision](../../.agents/notes/implemented/architecture/2026-10-11-automatic-libxc-cpu-forces.md)
for retained rationale and boundaries.

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
