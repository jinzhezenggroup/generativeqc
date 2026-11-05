# Restricted correlated methods: MP2, RCCSD and RCCSD(T)

Use `Calculator` for closed-shell MP2, RCCSD and RCCSD(T). These methods use a
converged RHF reference; analytic-force support varies by method and backend.
Check the available properties for your calculation:

```python
from generativeqc import Calculator

calc = Calculator(method="rccsd(t)", basis="def2-svp", device="cpu")
print(calc.capabilities.supported_properties)
print(calc.capabilities.supports_batch)
```

See [method capabilities](../reference/capabilities.md) for backend and
property restrictions.

## Choose a selector

| Selector | Reference and correlation | Backend | Default properties | Prepared batch |
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

## Control cost, memory and convergence

`correlation_memory_budget_bytes` bounds the simultaneous numeric storage of the
correlated phase, defaulting to 256 MiB when omitted. It is a capacity plan for
scientific buffers, not a process RSS or free-VRAM guarantee. A refusal is an
explicit failure: it never authorizes dropping forces, switching to a cheaper
method, or publishing partial work.

Unrestricted CC primal execution, fitted source/metric preparation and generated
CC response also use a separate finite scalar-work policy. The default limit is
`100000000000` estimated scalar-work units per phase (cumulative within the CC
response runtime). Set `GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK` to a positive
decimal integer before preparation/execution to explicitly select a different
finite limit. Empty, zero, signed, nonnumeric and overflowing values are errors;
there is no unlimited setting. This process-wide policy does not change the
Hamiltonian, generated artifact identity or numeric memory allowance.

Work-policy refusal returns `GENERATIVEQC_STATUS_RESOURCE_LIMIT` (10), not
`OUT_OF_MEMORY`. Its diagnostic reports the phase, AO/spin occupied/virtual
counts, iteration ceiling when applicable, estimated/allowed work and numeric
storage estimate. These estimates are neither measured FLOPs nor runtime
predictions. Raising the limit does not qualify a molecule, guarantee convergence
or make a singleton-heavy execution path practical. Allocation, checked shape
arithmetic, finite-value, convergence and complete-output gates remain active.

Native DF-HF reference failures preserve the backend status and any available
provider detail. Diagnostics distinguish SCF nonconvergence from missing physical
reference export and include iteration, energy-change, density and physical
residual observations; an unavailable residual is not a convergence certificate.
The host-orchestrated CUDA DF-RHF retry uses a bounded damped startup followed by
undamped DIIS when the compact device solve fails. This changes the iteration
schedule, not the fitted Hamiltonian or final physical-reference acceptance gates.

Unrestricted fitted interactions cache spin-labelled three-index factors in the
accepted physical orbital frame. UMP2 analytic forces contract factor adjoints
directly, including both orbital legs, the fixed-rank metric response and nuclear
response; they do not reconstruct full AO fourth-order ERIs or their cotangents.
Initial same-spin and opposite-spin weights use occupied-pair/virtual-pair blocks,
and same-spin exchange reuses the ordered block. Device-resident factors are
optional: if their admitted allocation is unavailable, bounded selected-panel
uploads still execute the contraction on CUDA. The frame and backend must match
the cache; stale coefficients are rejected, not silently reused.

This removes the fitted-source AO fourth-order storage barrier, not all
fourth-order method storage or high-order correlated work. UCC still admits its
generated MO tensors and ordered triples work separately. Correlation-only DF
retains its explicit conventional-reference derivative consumer; full DF uses
the fitted reference consistently. Host FP64 metric conditioning and detached
factor/cotangent transport remain explicit parts of the native hybrid contract,
not an automatic CPU/PySCF correlation fallback.

For diagnostic qualification, set `GENERATIVEQC_DF_PROGRESS_TRACE` to a fresh
JSONL path. The journal distinguishes cached factor-block reuse, source scans,
transfers, semantic work and UMP2/UCC response phases. Timings include tracing
overhead and require matching trace-off repeats; a returned scope alone does not
certify convergence. Resource estimates remain numeric-capacity bounds rather
than process-RSS or observed device-memory measurements.

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

Nonconvergence, small denominators, response failures and insufficient
memory raise errors rather than returning partial energies or forces.
CUDA requests do not silently switch to CPU. To inspect per-molecule
failures, run a prepared batch in non-strict mode.

## Where the details live

- [MP2 energy and forces](../developer/mp2.md) — conventional and RI contracts.
- [RCCSD equations](../developer/rccsd.md) and [RCCSD GPU solver](../developer/rccsd_gpu.md).
- [RCCSD(T) energy and gradients](../developer/rccsd_t.md), with the internal
  facade in [RCCSD(T) Python API](../developer/rccsd_t_api.md).
- [DF-CCSD(T)](../developer/df_ccsdt.md) and its
  [analytic force owner](../developer/df_ccsdt_gradient.md).
- [Unrestricted coupled cluster](unrestricted_cc.md) for `uccsd`/`uccsd(t)`.
- [MP2-assisted CCSD(T)/CBS](ccsdt_cbs.md) for composite extrapolation.
