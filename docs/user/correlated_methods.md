# Restricted correlated methods: MP2, RCCSD and RCCSD(T)

Use this guide for conventional closed-shell correlated calculations through
`Calculator`. Every selector below reuses one canonical RHF reference and
publishes native analytic forces wherever the declared domain is admitted; none
of them differentiates energies numerically or calls an external quantum-chemistry
program at runtime.

Check the exact domain before relying on a property:

```python
from generativeqc import Calculator

calc = Calculator(method="rccsd(t)", basis="def2-svp", device="cpu")
print(calc.capabilities.supported_properties)
print(calc.capabilities.supports_batch)
```

Selector discovery alone never proves that a basis, backend, spin state or
property is qualified. See [Capability sources and interpretation](../reference/capabilities.md).

## Choose a selector

| Selector | Reference and correlation | Admitted backends | Default properties | Prepared batch |
| --- | --- | --- | --- | --- |
| `mp2` | Canonical RHF-MP2, conventional four-center integrals | CPU and CUDA | Energy and forces | Ragged |
| `rccsd` | Conventional RCCSD in the RHF orbital frame | CPU and CUDA | Energy only | Homogeneous |
| `rccsd(t)` / `ccsd(t)` | Conventional RCCSD plus standard non-iterative `(T)` | CPU and CUDA | Energy and forces | Homogeneous |
| `df-rccsd(t)` / `df-ccsd(t)` | Conventional RHF reference, density-fitted correlation Hamiltonian | CUDA only, FP64, explicit auxiliary basis | Energy and forces | Unsupported |

Density-fitted MP2 is a separate family of reference contracts with their own
capabilities, including one CUDA route that remains energy-only. The
[CBS workflow](ccsdt_cbs.md) documents those selectors and the
[MP2 developer contract](../developer/mp2.md) owns the exact admission table.

All of these are real FP64, all-electron methods restricted to closed-shell
canonical RHF references. Frozen core, ECP, open-shell references and
screened or approximate correlation are rejected explicitly rather than
silently substituted. Density-fitted and conventional selectors are distinct
Hamiltonians; choosing one never reinterprets the other.

## Run an energy and its forces

```python
from generativeqc import Calculator

atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.43, 1.11)),
         ("H", (0.0, -1.43, 1.11))]

calc = Calculator(
    method="rccsd(t)",
    basis="sto-3g",
    device="cpu",
    correlation_memory_budget_bytes=1 << 30,
)
result = calc.singlepoint(atoms, properties=("energy", "forces"))

print(result.energy)                        # Hartree, RHF plus correlation
print(result.forces)                        # Hartree/Bohr, force = -gradient
print(result.correlation.ccsd_correlation_energy)
print(result.correlation.ccsd_t_triples_energy)
```

`rccsd` keeps its energy-only default because the force endpoint is bounded by
resource and numerical admission; request `properties=("energy", "forces")`
explicitly. `df-rccsd(t)` requires both CUDA and an explicit auxiliary basis:

```python
df = Calculator(
    method="df-ccsd(t)",
    basis="def2-svp",
    device="cuda",
    auxiliary_basis="def2-universal-jkfit",
    correlation_memory_budget_bytes=8 << 30,
)
```

Density-fitted MP2 has several reference contracts with different capabilities.
The [CBS workflow](ccsdt_cbs.md) documents correlation-only and full DF-HF
variants; the [MP2 developer contract](../developer/mp2.md) owns the exact
admission table.

## Control cost, memory and convergence

`correlation_memory_budget_bytes` bounds the simultaneous numeric storage of the
correlated phase, defaulting to 256 MiB when omitted. It is a capacity plan for
scientific buffers, not a process RSS or free-VRAM guarantee. A refusal is an
explicit failure: it never authorizes dropping forces, switching to a cheaper
method, or publishing partial work.

The `ccsd_*` options control the iterative solve and apply to restricted and
unrestricted CC alike:

```python
calc = Calculator(
    method="rccsd",
    basis="sto-3g",
    ccsd_max_iterations=150,
    ccsd_energy_tolerance=1e-11,
    ccsd_residual_tolerance=1e-9,
    ccsd_denominator_threshold=1e-10,
    ccsd_diis_history=6,
    ccsd_damping=0.0,
    ccsd_level_shift=0.0,
)
```

`ccsd_level_shift` modifies only iterative preconditioning; the physical
residual replay and the `(T)` correction always use unshifted energies.
Convergence controls are solver conditions, not observable error bounds. For
requested accuracy and precision policies, see
[accuracy and precision](application_precision.md).

## Batches and shared references

Conventional MP2 accepts ragged prepared batches. Restricted CC requires one
homogeneous `(nocc, nvir)` shape per batch and rejects ragged groups; split them
into separate batches instead. Every item owns its amplitudes, DIIS history and
status, and a failed item does not corrupt its neighbours. Cross-geometry
amplitude reuse is never implied by matching dimensions.

`singlepoint_components` shares one RHF reference among a group:

```python
components_calc = Calculator(method="rccsd(t)", basis="sto-3g", device="cpu")
group = components_calc.singlepoint_components(
    atoms, companions=("rhf", "mp2"), properties=("energy", "forces")
)
```

MP2 can supply the `rhf` companion, and CCSD(T) can supply `rhf` and `mp2`.
Method-specific response solves remain separate.

## Failure behavior

A nonconverged correlated solve, non-finite value, small denominator, failed
response solve, or insufficient budget raises without publishing a partial
energy or stale correlation diagnostics. CUDA failures never fall back to a
host computation under a CUDA label, and a force request is never answered by an
RHF or MP2 derivative alone. When you need the diagnostic rather than an
exception, inspect the per-item status of a non-strict prepared batch.

## Where the details live

- [MP2 energy and forces](../developer/mp2.md) — conventional and RI contracts.
- [RCCSD equations](../developer/rccsd.md) and [RCCSD GPU solver](../developer/rccsd_gpu.md).
- [RCCSD(T) energy and gradients](../developer/rccsd_t.md), with the internal
  facade in [RCCSD(T) Python API](../developer/rccsd_t_api.md).
- [DF-CCSD(T)](../developer/df_ccsdt.md) and its
  [analytic force owner](../developer/df_ccsdt_gradient.md).
- [Unrestricted coupled cluster](unrestricted_cc.md) for `uccsd`/`uccsd(t)`.
- [MP2-assisted CCSD(T)/CBS](ccsdt_cbs.md) for composite extrapolation.
