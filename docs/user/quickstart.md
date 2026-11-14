# Quick start

After [installation](installation.md), run a small hydrogen-molecule calculation,
then choose the workflow you need. This example uses RHF with the STO-3G basis
and explicitly requests energy and analytic forces.

## Calculate energy and forces

The example selects CUDA. For a CPU-only installation, set `device="cpu"`.
The Python API takes coordinates in **Bohr**, not Ångström; this H₂ geometry
has a bond length of 1.4 Bohr. See [units](../reference/units.md) before supplying
a geometry from another program.

```python
from generativeqc import Calculator

atoms = [
    ("H", (0.0, 0.0, -0.7)),
    ("H", (0.0, 0.0,  0.7)),
]
calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
result = calc.singlepoint(
    atoms,
    charge=0,
    multiplicity=1,
    properties=("energy", "forces"),
)

print(result.energy)  # scalar, Hartree
print(result.forces)  # shape (2, 3), Hartree/Bohr, in input atom order
```

`charge=0` and `multiplicity=1` specify a neutral singlet. For open-shell
systems, choose a compatible unrestricted method and electron state; see
[charge and spin](../learn/molecules-charge-spin.md).

Forces are minus the energy gradient. A failed or unconverged `singlepoint`
call raises an exception rather than returning a partial successful result.
The [calculation contracts](../reference/python_execution_contracts.md)
define units, failures and result ownership.

## Request only what you need

To skip force evaluation, use:

```python
energy_result = calc.singlepoint(atoms, properties=("energy",))
print(energy_result.energy)
assert energy_result.forces is None
```

Property defaults vary by method and execution context. Explicit requests make
application intent clear; an unsupported force request is not silently replaced
with energy-only execution.

## Choose a method and workflow

Discover registered methods with:

```bash
generativeqc methods
generativeqc methods --json
```

Registration is not a promise that every property, basis or backend combination
is supported. See [method discovery and support](methods.md) and the generated
[public method table](../public_methods.md) for canonical identities and declared
capabilities; check `calc.capabilities` for the selected calculator.

- Repeated HF calculations or several molecules: [prepared batches](batched_hf.md).
- Geometry changes and saved state: [checkpoint/restart](checkpoint.md).
- Dipoles, atomic charges and density/Cube export: [electronic properties](electronic_properties.md).
- A Python-free interface: [native CLI](native_cli.md).
- Exact Python signatures and options: [API reference](../reference/api.md).
