# Prepared semilocal KS model options

`Calculator(method="lda-rks" | "pbe-rks" | "lda-uks" | "pbe-uks",
ks_options=...)` snapshots a `KsOptions` object for CPU and CUDA. Its default
grid and tile reproduce the original method behavior. The existing compiler
`FunctionalSpec` and `GridSpec` are reused, with an explicit native SCF
tail/spin domain; a second functional catalog is not introduced.

```python
from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget

options = KsOptions(
    grid=GridSpec(
        radial_points=64,
        angular_polar=20,
        angular_azimuth=40,
        element_radii=((1, 1.3), (8, 0.9)),  # Bohr, indexed by element
    ),
    tile_points=128,
)
calculator = Calculator(
    method="pbe-rks",
    device="cuda",
    ks_options=options,
    resource_budget=ResourceBudget(host_bytes=1 << 30, device_bytes=1 << 30),
)
print(calculator.ks_options.to_payload())
```

An omitted functional resolves to `LDA_X+LDA_C_PW` or
`GGA_X_PBE+GGA_C_PBE`, with unit coefficients. RKS resolves an unpolarized
composition; UKS resolves a polarized one. An explicit `FunctionalSpec` must
match those components, coefficients and spin convention. Different names
for an identical composition are descriptive; changed coefficients, exact
exchange, range separation or unsupported domain policies are rejected before
native loading. These energy methods do not advertise arbitrary functional
mixtures or hybrid support.

The effective native domain is
`semilocal-scaled-v1/pbe-spin-c2-1e-18`, documented in
[the SCF domain contract](../developer/xc_scf_domain.md). `FunctionalSpec` retains its
independent interior-reference provenance; that provenance does not silently
replace the native SCF boundary policy. The resolved options record both.
The functional's required ingredients determine AO order: LDA needs values,
GGA needs values and first spatial derivatives. SCF requires scalar energy
and its first derivative for the potential, even for an energy-only output.
Neither route computes tau or higher AO jets.

Completed results expose the resolved grid and physical iteration records in
[`ks_diagnostic`](../developer/ks_diagnostics.md).

`GridSpec` selects the radial/polar/azimuth counts, partition iterations,
coincident-center tolerance and per-element radial scales. The native grid
uses the same rational-Legendre radial rule, Legendre-trapezoid angular rule
and equal-radius Becke partition as the reference. Element radii scale radial
points and radial Jacobians; they do not add a heteronuclear partition
correction. Pruning, screening, rules and units retain the version-1 contract.
Changing the grid changes the discrete energy. Changing `tile_points` changes
the schedule, capacity and FP64 reduction order. With local-AO maps, larger
block unions can also admit additional AO tail work; complete numerical gates
remain necessary even though the grid and cutoff are unchanged.

## SCF and force tile policies

`KsOptions.tile_points` configures SCF AO/grid/XC panels. The ordinary public
CUDA analytic-force executor has a separate fixed 256-point policy; composite
forces instead use their own budget-aware planner. Increasing the SCF tile does
not request a larger Becke force workspace.

For an independently qualified larger-tile workload, request the size explicitly:

```python
options = KsOptions(
    grid=GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32),
    tile_points=512,
)
```

Larger tiles reduce submission/map counts but can include more local AO
summands. They are not universally faster, and map reservation bytes are not
complete endpoint peak memory. The default remains 256: there is no implicit
512-point promotion or automatic replacement of an explicit tile request.
Resource budgets and the existing dense/local-AO admission remain authoritative.
Use a new prepared owner when changing the tile, and inspect the actual
`ks_diagnostic.tile_points` rather than assuming the requested route ran.

See [complete tile qualification](../maintainer/pbe0_xc_tile_qualification.md)
for paired E+F validation and the limits of the retained configuration evidence.

## Prepared identity and ABI

### CUDA Direct integral forces

Full-range exact-Direct FP64 force workers use generated relative-coordinate
responses by default for angular orders one through five, including separate
and combined J/K and RKS/UKS sources. Unset, empty, `1` and `relative` select
this arithmetic; `GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES=0` or an unknown
nonempty value retains the nuclear-coordinate rollback. `ssss`, FP32 and
long-range roots are unchanged. See the
[compiler and finite-fallback contract](../developer/weighted_eri.md#relative-coordinate-force-responses).

Independently, qualified full-range exact-Direct restricted FP64 force producers
with at least 384 public AOs automatically request relative-response order-six
reuse and guarded pure-J density-first contraction in orders six/seven. The
method authenticates the actual J/K source and resident final-density token,
not a functional name. Pure J permits independent D4/gCP terms; nonzero
exchange, bilinear seeds and partial AO masks retain component contraction.
HF, UKS, fitted, ECP, RSH, nonlocal and D4-hybrid consumers keep their previous
order-six/seven defaults. All resource and derivative-capability guards remain.

`GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS=0` forces the incumbent schedule.
Explicit `=1` or `=spatial-reuse` selects the earlier spatial consumer;
`=relative-reuse` shares relative responses without density-first contraction,
and `=relative-density` also requests the guarded factorization. No selector
bypasses admission. Unset or empty selects the automatic policy; other values
retain the incumbent. Both independent preferences are captured when preparing
the force owner, so changing either requires a new prepared owner. Screening,
source/atom accumulation, order-eight and mixed-f fallback semantics remain.

See the [order-six/seven default decision](../../.agents/notes/implemented/performance/2026-10-17-direct-force-default-promotion.md)
and [low-order default decision](../../.agents/notes/implemented/performance/2026-10-17-weighted-relative-force-responses.md)
for qualification boundaries and retained alternatives.

### CUDA occupied-frame XC features

Strict-FP64 native CUDA KS prefers an authenticated occupied orbital frame when
evaluating physical four-jet density/gradient/tau features. This reuses the
accepted density's factorization; it does not change the functional, grid,
potential assembly, Direct J/K operators or convergence gates. Both RKS and UKS
retain their original occupations and spin conventions.

`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=0` selects the matrix contraction. Unset,
empty or `1` requests occupied reuse; other nonempty values also select the
matrix route. The preference is captured when preparing the native owner:
recreate the prepared owner after changing it. Admission still requires the
live accepted density/frame identity, strict FP64 and a smaller occupied rank
than the tile's active AO span. Imported, held or mixed densities, response,
nonlocal and chunk paths retain matrix evaluation when no valid lease exists.
Unrepresentable projected panels fall back to the original stored-density
calculation, rather than changing density storage semantics.

See the [generated arithmetic and lifetime contract](../developer/xc_native_cuda.md#occupied-frame-density-features)
and [independent representability decision](../../.agents/notes/implemented/numerics/2026-10-17-occupied-xc-representability.md).

### CUDA final-state validation

Native CUDA KS automatically requests resident final-state matrix products
for these strict-FP64 restricted producers:

- Full-range exact-direct global hybrids with at least 384 AOs. The signed
  exchange coefficient and Direct J/K operators, not a functional name such as
  PBE0 or B3LYP, determine eligibility.
- All-electron full-range Coulomb-only models using exact-direct or
  density-fitted J with at least **544 AOs**.
  This includes eligible r²SCAN-3c owners; their separate D4/gCP corrections
  remain part of the complete endpoint. The stationary-energy repair and its
  source-matched complete-endpoint qualification replace the temporary
  1088-AO exact-direct hold; the historical failed cohort remains rejected.

Resource and live-generation checks still require the matrix-library handle,
charged packets and no previously published stationary W/total-D lease. Smaller
pure-J owners, UKS, fitted-exchange, RSH, ECP, nonlocal, mixed and incremental
consumers are not admitted by the new pure-J policy.

`GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=0` forces the reference route.
Explicit `=1` requests the resident route outside the automatic scope, but does
not bypass resource, identity, lease or numerical gates. Unset selects the
automatic policy; other values are errors. This changes only execution of the
shared strict proof: scientific acceptance gates, dense host exports and the
orbital-energy expression for KS weighted density remain unchanged.

Ordinary strict-FP64 all-electron exact-direct pure-J CUDA KS can hold an
already stationary density while rebuilding the complete physical F[D] to
confirm a failed energy gate. Density change and both physical residual norms
must already pass their original thresholds. The fresh energy must still pass;
this neither replays a cached energy nor changes convergence or iteration limits.

See the [default-selection decision](../../.agents/notes/implemented/performance/2026-10-11-default-pbe0-resident-final-validation.md)
and [stationary pure-J energy confirmation](../../.agents/notes/implemented/numerics/2026-10-16-stationary-pure-j-energy-confirmation.md)
for their evidence and limits.

### Experimental incremental Direct-J/K and resource budgets

The benchmark-only `GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK=1` (or `on`)
selector is not supported by public KS resource planning. The current inventory
does not account for its additional anchor/delta storage and final-closure
history. `estimate_ks_resources` and `Calculator.estimate_resources` return an
unsupported plan; `require_feasible()`, budgeted execution, and preparation with
an ordinary precomputed plan reject it before native preparation. The legacy
`GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK` selector has the same planning limit.
Both switches must be unset, `0`, or `off` for ordinary resource planning;
setting the generic switch to `off` does not override an enabled legacy switch.
Other spellings are invalid, matching native validation.

Explicit experimental execution without a `resource_budget` or `resource_plan`
remains available under the native strict-FP64 exact-direct CUDA eligibility
checks. It has no whole-calculation capacity guarantee. CPU, mixed-precision,
and density-fitted execution remain excluded from incremental mode, and the
legacy selector retains its PBE0 RKS restriction. No default or numerical/
performance qualification changes with this planning guard.

Incremental solves return to full-density J/K builds once the density-change and
physical-residual gates are satisfied, even if the energy-change gate is not yet
satisfied. This full-density energy refinement keeps the original DIIS history
and convergence tolerances; it prevents differing full/ΔD screening omissions
from blocking the energy gate indefinitely. Strict full-density physical
finalization remains mandatory. Energy convergence requires two consecutive
full-density builds. For ordinary RKS, the qualifying full-density build is
already the final physical audit; UKS and ECP retain their separate corrective
closure. A fresh solve resets this refinement phase.

### Model binding

The complete options are included in resource identity and Python prepared
model identity. Resource estimates use the actual grid dimensions and tile,
including native radius-table storage. A replaced functional/grid/tile model
requires a new prepared object before a warm density, DIIS history, or cached
physical result can run. Compatible coordinate changes keep the existing
native rebuild and spin-normalization path.

The C ABI appends a nullable `ks_options` pointer to
`generativeqc_method_descriptor`. A missing tail field or NULL preserves defaults.
`generativeqc_ks_options_version()` reports support without creating a context.
Version 1 is the original grid/tile prefix, version 2 adds explicit semilocal
scales/full-range exchange, and version 3 adds the XC execution schedule. Version
4 appends the compiler-resolved self-consistent execution selector: spin-channel
count and semilocal primitive family. The complete v3 layout, including its
trailing padding, is preserved before the v4 fields. Modern Python callers derive
those v4 fields from `MethodIR -> KsExecutionPlan`; native CPU/CUDA execution no
longer chooses RKS/UKS or the LDA/PBE/r2SCAN family from the public method ID. Old
v1/v2/v3 callers retain a narrow legacy selector fallback. The PBE-D4 correction
is retained in the snapshotted plan alongside the semilocal execution selector.

Python negotiates the highest supported descriptor version. Composition and
host-unfused scheduling still fail closed when an older library cannot represent
them. For the append-only ABI rationale, see the
[compatibility decision](../../.agents/notes/implemented/compatibility/2026-09-21-ks-v4-append-only.md).

Explicit options require complete nonzero grid counts and a positive tile size.
Optional element radii are a 119-entry positive finite array indexed by atomic
number (entry zero is unused); NULL/zero selects unit radii. Preparation copies
the descriptor and all pointees. Caller storage can be released or modified as
soon as prepare returns. Other method families reject an attached KS option.
Older libraries remain usable for default KS models; Python rejects custom
options if the version query is unavailable.
