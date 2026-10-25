# Dispersion corrections

GenerativeQC adds geometry-only dispersion (D3 and D4) as a separate correction
component. The electronic method stays the same Kohn-Sham model, and the
correction is added exactly once when the calculation is prepared. Nothing here
reads an external `dftd3`/`dftd4` executable or adds an optional runtime
dependency.

## Choose a selector form

The composed model can be named in three ways. They differ only in how it is
spelled, not in the science:

| Form | Example | Notes |
| --- | --- | --- |
| Parameterized D4 selector | `pbe0-d4-rks`, `b3lyp-d4-uks` | Exists only when the underlying electronic method passes the native lowerer gates |
| Explicit MethodIR as `method` | `resolve_method("PBE-D3(BJ)", spin="unpolarized")` | The only form for D3; the spin is part of the identity |
| Named compatibility endpoint | `pbe-d4-rks` | Executes native PBE RKS plus D4(BJ)-EEQ-ATM |

For D4, the public selector set is generated from the intersection of the
electronic method catalog and the pinned D4 parameter records, so it is not a
hand-maintained list. Query it instead of guessing:

```python
from generativeqc.ks import public_dft_selectors

print([name for name in public_dft_selectors() if "-d4-" in name])
```

D3 has no `-d3-rks` selector spelling; a D3 request is always an explicit
MethodIR. A native KS selector can also carry the full graph through
`KsOptions(composition=...)`, for example `method="pbe-rks"` plus the same
PBE-D3(BJ) graph. In that form `Calculator.method_ir` reports the composed
graph while `Calculator.ks_options.method_ir` reports the electronic graph that
is lowered into the SCF owner, so no lower-level path can accept a D3 node and
silently omit it. Without a composition that contains the correction, the
electronic selector stays uncorrected.

## Run a corrected energy

```python
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc_compiler.method import resolve_method

method = resolve_method("PBE-D3(BJ)", spin="unpolarized")
calc = Calculator(
    method=method,
    basis="def2-svp",
    device="cpu",
    ks_options=KsOptions(grid=GridSpec()),
)
result = calc.singlepoint(atoms, properties=("energy",))

print(result.energy)              # total: electronic + dispersion
print(result.dispersion.energy)   # correction component only
```

`result.energy` is always the total. For a composed MethodIR the correction
component is separately reported in `result.dispersion`, and prepared batches
expose the retained plan through `PreparedBatch.dispersion_diagnostic`.

The named `pbe-d4-rks` endpoint follows the native exact-once rule instead: its
native electronic energy already includes the D4 term, so the correction
component is materialized only when forces are requested and it is never added
twice. Read `result.dispersion` when it is present; treat its absence on an
energy-only `pbe-d4-rks` call as "not separately reported", not as "no
correction applied". A parameterized D4 selector is used the same way:

```python
calc = Calculator(
    method="pbe0-d4-rks",
    basis="def2-svp",
    device="cpu",
    ks_options=KsOptions(grid=GridSpec()),
)
result = calc.singlepoint(atoms, properties=("energy",))
```

A dispersion correction never changes backend admission: the electronic carrier
keeps its usual CPU/CUDA domain. The named `pbe-d4-rks` carrier additionally
requires strict `precision="fp64"`; component-wise `precision="auto"` is
rejected rather than applied to only part of the composite. The D4 correction
itself needs the molecular charge, which `singlepoint` already takes; batch
items use their per-system charge.

## Forces

A dispersion correction never widens the electronic force domain. Request
`properties=("energy", "forces")` and check `Calculator.capabilities`: forces
are published only when the underlying electronic method, basis, backend and
precision context already advertise analytic forces. In particular a hybrid
such as `PBE0-D3(BJ)` has a complete D3 gradient but does not acquire public
total forces merely because that gradient exists.

When total forces are available, the composition is

```text
E_total = E_electronic + E_dispersion
F_total = F_electronic - dE_dispersion/dR
```

The subtraction is only a sign convention: the correction owner publishes
`gradient = dE/dR`, while the public endpoint publishes forces.

## Budget and resource planning

`dispersion_memory_budget_bytes` bounds the retained correction owner
independently of the electronic SCF state; the default is 256 MiB. It is a
capacity plan, not a process-RSS or free-VRAM guarantee.

`ResourceBudget` and `estimate_resources()` aggregate the separate D3(BJ) owner.
Composite D4 and gCP planning remain fail-closed: `estimate_resources()` for a
D4-corrected calculator raises rather than reporting an electronic-only plan as
the complete composite cost, and passing a global `resource_budget` to an
explicit D4 MethodIR calculation is rejected at construction.

## Evaluate a correction without an SCF

Both corrections are also usable standalone, for example to compare against an
external implementation or to post-process coordinates:

```python
from generativeqc import evaluate_d3_correction, evaluate_d4_correction

d3 = evaluate_d3_correction("PBE-D3(BJ)", atomic_numbers, coordinates)
d4 = evaluate_d4_correction(
    "PBE0-D4(BJ-EEQ-ATM)", atomic_numbers, coordinates, total_charge=0
)
```

`D3CorrectionBatch` and `D4CorrectionBatch` prepare a ragged fleet once and then
replay changed geometries per item without reallocating. Preparation fixes each
system's atom count and atomic numbers; `execute()` accepts new coordinates per
item and returns results in batch order.

## Failure behavior

Assembled parameter data is validated against the compiled tables before
execution, so a mismatched correction identity fails at preparation. Unsupported
damping variants, unknown compositions, malformed coordinates, a non-finite
value, or an exhausted budget raise without publishing a partial total energy or
a correction energy from an earlier call.

A per-item correction failure converts only that item to failure; successful
neighbours keep their results. CUDA failures never fall back to a host
correction under a CUDA label.

## Where the details live

- [D3 production correction runtime](../developer/dft_d3.md) — qualified
  variants, MethodIR composition, generated ragged candidate and parameter
  provenance.
- [D4 numerical qualification and EEQ integration](../reference/d4_reference.md)
  — the D4 scientific contract, EEQ charge response, memory bounds and
  reproduction commands.
- [r²SCAN-3c](r2scan3c.md) — the composite that binds one exact basis to
  r²SCAN plus D4(BJ)-EEQ-ATM plus gCP.
- [KS options](ks_options.md) — grid, XC schedule and other execution controls
  that accompany the dispersion model.
