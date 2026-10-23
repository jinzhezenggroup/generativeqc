# Prepared semilocal KS model options

`Calculator(method="lda-rks" | "pbe-rks" | "lda-uks" | "pbe-uks",
ks_options=...)` snapshots a `KsOptions` object for CPU and CUDA.
All supported KS compositions default to level-3 Treutler/Ahlrichs radial,
Lebedev angular and NWChem-pruned grids, with Treutler radius-adjusted Becke
partitions. For VV10/rVV10, an omitted nonlocal grid stays at Treutler level 3
even when the XC grid is tightened to level 4; default level-3 XC/VV10 grids
still share their storage. Explicit `GridSpec` v1--v3 XC prescriptions retain
their previous shared-grid behavior unless a distinct nonlocal grid is given. The existing compiler
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

Use `GridSpec.treutler(level=3)` to select the default family explicitly;
levels 0--9 are supported for H--Lr. `grid_accuracy="tight"` selects level 4
for XC, but leaves an omitted VV10/rVV10 grid at level 3. Supplying a nonlocal
grid explicitly always takes precedence; supplying the same level-4 grid for
both opts back into the shared-grid path. When an explicitly selected
Treutler XC level differs from 3, omitted VV10 uses level 3 too.
The grid prescriptions are identical for energy and force requests.
Independent grids currently do not support Hessians/HVPs, coupled response or
explicit resource-ledger planning. Fixed element tables
and shell pruning do not depend on geometry or density; points move with their
owner atoms and analytic responses differentiate the adjusted partition.
PySCF is not required at runtime. The pinned numerical tables are from
PySCF 2.14.0; see [third-party notices](../../THIRD_PARTY_NOTICES.md).

Point counts depend on the element. Use `spec.point_count(atomic_numbers)`
instead of `points_per_atom` for this family. No zero-weight alignment padding
is added. Signed Lebedev weights are preserved.

`GridSpec(...)` still constructs an explicit legacy tensor grid. Versions 1
and 2 use rational-Legendre radial and Legendre-trapezoid angular rules, with
an equal-radius Becke partition; version 3 adds `radial-v1` angular pruning.
Their element radii scale radial points and Jacobians. These legacy grids
remain available for reproducibility and elements outside the new table.

The native descriptor layout is unchanged. Version 4 stores level in
`radial_points` and requires both angular fields to be zero. A null radius
table selects the pinned Bragg radii; an explicit table supplies partition
radii, while radial nodes and pruning retain the pinned element rules.
`KsGrid{}` and omitted native KS options select version 4, level 3.

Changing the grid changes the discrete energy. Changing `tile_points` changes
the schedule, capacity and FP64 reduction order. With local-AO maps, larger
block unions can also admit additional AO tail work; complete numerical gates
remain necessary even though the grid and cutoff are unchanged.

(treutler-grid-values)=
## Grid descriptor helper values

`GridSpec.treutler(level=3)` returns a frozen version-4 descriptor with pinned
partition radii for H--Lr. Its level must be a Python integer from 0 through 9.
The `level` property returns that integer for version 4 and `None` for legacy
descriptors; it is not a radial-node count.

`spec.point_count(atomic_numbers)` consumes an iterable and returns the total
integer point count without constructing quadrature. Version 4 uses each
element's shell sizes; legacy versions count the entries with their uniform
per-atom prescription. `spec.atom_offsets(atoms)` consumes atom objects exposing
`atomic_number` and returns an immutable tuple of atom-major prefix offsets,
starting at zero and ending at the total point count. Empty inputs yield zero
points and the offset tuple `(0,)`.

`spec.partition_radii(atoms)` returns an immutable tuple of per-atom partition
radii in Bohr for version 4, preserving atom order and explicit radius overrides.
It returns `None` for legacy equal-radius partitions. These helpers do not use
coordinates, charge or spin, and do not construct a molecular grid.

(treutler-grid-errors)=
## Grid descriptor helper errors

The Treutler constructor rejects non-integer levels, including booleans, and
levels outside 0--9 with `ValueError`. Version-4 point counting and offsets
likewise reject atomic numbers that are not Python integers in 1--103 with
`ValueError`. Atom-based helpers require objects exposing `atomic_number`.
For an explicitly constructed version-4 descriptor with an incomplete radius
table, `partition_radii` raises `KeyError` when a requested element is absent;
it does not invent a radius. The `level` property has no additional validation
or failure branch after descriptor construction.

(treutler-grid-ownership)=
## Grid descriptor helper ownership

The constructor returns an independent frozen descriptor. Counts, levels,
offsets and per-atom radii are immutable Python values with no borrowed native
storage or device lifetime. Input iterables are consumed by the corresponding
helper but are not retained or mutated. The helpers require no native library,
GPU, or PySCF installation; numerical tables belong to the installed package.

(angular-pruning)=
## Angular pruning

The legacy `radial-v1` policy is selected explicitly:

```python
from generativeqc import GridPolicy, GridSpec

current = GridSpec.treutler(level=3)
legacy = GridPolicy(pruning="radial-v1").resolve("pbe-rks")
unpruned = GridPolicy(pruning="none").resolve("pbe-rks")
```

Version 3 requires a finite positive radial scale for every element actually
used. For an isolated angular-pruning experiment on a fixed v2 grid, use
`replace(unpruned, version=3, pruning="radial-v1")` instead; this differs from
selecting the legacy policy's radial/angular maxima. Custom radial
counts, radii and angular orders remain explicit inputs.
Maximum version-3 orders are 128 polar and 512 azimuth points. Unsupported
prescriptions fail validation rather than silently disabling pruning. Native
C/C++ callers select the same contract with `grid_version=3` / `KsGrid.version=3`
and explicit element radii; the descriptor layout is unchanged.

For increasing radial index `i`, define `u=(i+1/2)/radial_points`:

| Radial rank | Angular orders relative to the requested maximum |
| --- | --- |
| `u < 1/5` | Quarter polar and azimuth orders |
| `1/5 <= u < 1/3` | Half orders |
| `1/3 <= u <= 3/4` | Full orders |
| `3/4 < u <= 6/7` | Half orders |
| `u > 6/7` | Quarter orders |

Reduced orders round upward, with at least one polar and three azimuth points.
Each shell uses its own complete Gauss-Legendre × periodic-trapezoid rule with
the corresponding integration weights. This is an original table-free policy,
**not NWChem/Lebedev pruning**. Element radii scale the physical shell positions;
the angular schedule depends only on integer radial rank. Consequently the exact
point count can be planned without constructing quadrature, and topology stays
fixed as nuclei move. CPU and CUDA materialization, atomic-measure exports,
resident ownership and moving-grid weight response use that same shell order.
Other method/backend force-admission restrictions are unchanged.

Pruning removes constructed points; it does not change a density cutoff or imply
that an existing density-screened VV10 pair domain shrinks by the same fraction.
Changing pruning changes the discrete energy and its analytic forces. Qualify
the chosen method, basis and geometry class before using it for a production
comparison. With the independent PySCF reference dependencies installed, run:

```console
python -m benchmarks.grid_policy_convergence --include-pruning --output .artifacts/grid-pruning-convergence.json
```

This CPU oracle checks LDA/PBE energy and grid-responsive analytic gradients
against a dense grid whose stability is checked against a denser grid. Pruned
standard/tight profiles must meet the unchanged unpruned accuracy gates while
reducing point counts. It is not a VV10, large-system or GPU performance claim.
The existing matched-grid runners retain `--grid-pruning auto`: LDA/PBE select
the legacy `radial-v1` policy for both native and GPU4PySCF; other methods use
unpruned diagnostic grids. `--grid-pruning none` selects
the retained unpruned policy. Explicit `--grid NR NP NA` counts preserve the
historical one-Bohr radial scales. `README_BENCHMARK_GRID_PRUNING` forwards this
choice through `benchmarks/run_readme_dft_benchmarks.sh`.

For the legacy angular-pruning comparison, use
`python -m benchmarks.grid_pruning_gpu --atoms 96 --output .artifacts/pruning96.json`
inside a finite Slurm GPU allocation with a source-matched Release library and
GPU4PySCF installed. This runner uses GPU4PySCF's **own NWChem-pruned grids**,
not the native angular rules. It checks dense-reference stability and both
engines' complete energy/force results before selecting among timed candidates.
The energy budget scales per water unit; maximum force budgets do not scale
with system size. Qualification also reserves margin for the observed
dense-reference variation; that variation is not a rigorous error bound.
Raw samples, failed candidates, work counts and independent
reference results are retained. Publishing a README production ranking still requires
the [complete endpoint and accuracy qualification](../maintainer/performance_engineering.md#readme-production-comparisons).

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
pure-J owners, UKS, fitted-exchange, RSH, ECP, nonlocal, mixed and unclosed
incremental consumers are not admitted by the new pure-J policy.

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

### Guarded automatic cold incremental Direct-J/K and resource budgets

When both incremental selectors are **unset**, normal CUDA Standard FP64
full-range Direct global hybrids automatically select bounded incremental
J/K proposals for an eligible *cold* solve: 64–768 public AOs, 2–96 atoms,
all-electron restricted through-d basis, the canonical default grid and
qualified energy `1e-9`, density `1e-5` and screening `1e-12` controls.
The real Direct producer and complete physical/canonical final-state proof
remain authoritative. This narrow cold default is a guarded performance choice,
not a globally enabled incremental Fock setting.

Same-owner frozen-density Warm retains its one-full-Fock proof. Imported or
moved seeds, an active public resource ledger, Fast, DF, UKS, range/nonlocal
and D4 owners, strict custom settings and unqualified geometry/size/resource
domains keep the previous execution. Set
`GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK=0` (or `off`) to turn off the
automatic choice; a present legacy selector also prevents automatic
admission. Explicit `1/on` retains the broader experimental capability checks
and the existing OR semantics between both selectors. Timing an incumbent
control therefore requires an explicit `0`, not merely an unset selector.


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
legacy selector retains its PBE0 RKS restriction. The public planning guard is unchanged: automatic admission refuses an active
resource ledger rather than allocating unaccounted incremental anchor/closure
storage.

Incremental solves return to full-density J/K builds once the density-change and
physical-residual gates are satisfied, even if the energy-change gate is not yet
satisfied. This full-density energy refinement keeps the original DIIS history
and convergence tolerances; it prevents differing full/ΔD screening omissions
from blocking the energy gate indefinitely. Strict full-density physical
finalization remains mandatory. Energy convergence requires two consecutive
full-density builds. For ordinary RKS, the qualifying full-density build is
already the final physical audit; UKS and ECP retain their separate corrective
closure. A fresh solve resets this refinement phase.

All-electron RKS owners admitted to the existing Standard stopping and complete
resident final-state validation policies can use bounded screened ΔD proposals.
ΔD admission adds no functional-name restriction: J/K linearity is independent
of XC family, while the ordinary Standard model/grid and resource guards still
apply.
The requested rebuild interval bounds each proposal chain, capped at eight;
zero also selects this finite cap. These operators only propose SCF steps:
screening tolerances are not an accumulated-error certificate. Acceptance still
requires two consecutive full-density energies and the complete physical and
canonical-state proof. The density-change and RMS gates can start full-density
refinement before the AO maximum passes; all gates remain required to finish.
`GENERATIVEQC_KS_INCREMENTAL_PROPOSALS=0` restores the ordinary screened refresh
cadence while retaining the final-state and warm policies; `1` or unset enables
proposals for eligible owners. Other values are errors. Explicit strict
tolerances and owners lacking the required proof retain their ordinary
incremental policy.

Eligible proposals can also use a coarser `1e-9` J/K screening cutoff for at most
the first eight delta builds of a solve, across periodic refreshes. This applies
only to an already screened source whose prepared cutoff is tighter, with MD-J
and, when needed, generated full-range K available. Existing conservative
producer caps, including the MD-J error cap, remain in force. Unsupported providers fall
back to the prepared cutoff. The prepared source is immutable: full builds,
warm calls, final physical proof and analytic forces use the original cutoff.
The proposal cutoff is a convergence heuristic, not an accumulated-error bound.
`GENERATIVEQC_KS_INCREMENTAL_COARSE_SCREENING=0` disables this extra screening
while preserving bounded delta proposals; `1` or unset enables it, and other
values are errors. Outside the narrow automatic cold domain, the global incremental selector remains opt-in.

For all-electron RKS without range or nonlocal contributions, a validated
density reused internally at the same
geometry follows the ordinary full-density warm path, including its energy
history. Diagnostics therefore report incremental mode as requested but
inactive on that call. External seeds, changed geometry and clearing the warm
state restore the cold incremental policy. This does not expand resource-planning support or activate incremental
execution in other domains.

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
