# Application precision and explicit audits

`PrecisionCalculator(policy="standard")` runs one optimized FP64 endpoint,
while `PrecisionCalculator(policy="fast")` selects qualified FP32 arithmetic
from an application error budget, without asking users to select kernels.

`policy="fp64"` remains a backward-compatible name for Standard: both use
identical controls and execution, and the serialized `profile` preserves the
name requested by the caller for compatibility. The native CLI
`--precision fp64|auto` and low-level `Calculator(precision="fp64"|"auto")`
select arithmetic rather than these application profiles; `auto` is **not**
an alias for Fast. Neither profile changes the molecular model.

When Fast has no admitted approximation, a completed native CUDA Standard
endpoint can return as `strict_standard_fallback` without a duplicate solve or
prepared-owner rebuild. This requires strict arithmetic and Standard-or-tighter
solver controls, with the default-or-looser budget and no explicit target.
The result retains Standard's final-state checks; it does not claim measured
observable error or mixed work. Explicit targets and tighter budgets keep their
independent audits. Legacy fitted-HF work counters remain unknown.

## Module-selected Fast Fock

Every supported native CUDA XC composition uses the same shared Direct-Fock precision
policy. Admission depends on executable operator capabilities, not a functional
name, canonical exchange fraction or MethodIR identity. The current restricted,
all-electron Direct consumer admits **FP32 J and full-range K recurrence** with
at least 24 AOs, at least two atoms and shells through d. There is no fixed AO
or atom-count upper limit in this precision policy; native dimension validation
and actual resource admission still apply. It uses the requested supported
quadrature unchanged; the J/K precision decision does not require a particular grid.
Its budget floors are `1e-5` Eh energy, `1e-4` Eh/bohr maximum force and
`3e-5` Eh/bohr RMS force when requested. This domain is independent of the
384-AO mixed-gradient limit, which remains unchanged. Admission is not a
per-system observable-error certificate; explicit audits retain their contract.

Default candidate controls are `1e-6` energy, `1e-4` density/stationarity and
`1e-10` screening. Explicit stricter caller controls remain stricter. Only a
verified module-level native Fock capability permits this selection; older
libraries, unsupported providers and tighter budgets retain strict controls.
Pure semilocal models use generated strict J when it is faster than generic
mixed recurrence; their admitted XC/density contractions still follow the
budget schedule. They do not invent K work. Hybrid models share the
same full-range K consumer, including the full-range projection of a short-/long-
range decomposition. Compatible exact SR/LR K consumers also lower their shared
Cartesian recurrence, keeping radial moments in FP64. VV10/rVV10 lower bounded
pair algebra to FP32, while local scales, coordinate differences, weighted
contractions and reductions stay FP64. Pairs outside the admitted numeric domain
use the unchanged FP64 closure, without clipping inputs or dropping grid points.
These choices belong to the shared operator modules, not functional-name rules.
SR/LR K uses FP32 in both the optional whole-shell consumer and its bounded
canonical-AO fallback; both report executed mixed recurrence work.
Post-SCF corrections retain their existing arithmetic and complete-model checks;
they do not select the recurrence precision of electronic J/K. Cached primitive
preparation, whole-shell contraction/scatter, XC point algebra, eigensolution
and global accumulation retain FP64. Shared XC density and potential matrix
products use FP32 inner arithmetic in admitted physical Fast-Fock layouts,
including their existing compact and indexed schedules. Their panels and
physical outputs remain FP64; potential contributions accumulate across point
tiles in FP64. Binary column normalization preserves representable density
tails without clipping inputs or moving their dot products back to FP64.
Strict refinement and response consumers keep strict products. The bounded
canonical-AO value fallback uses its
existing FP32 scalar policy for primitive and contracted values:
this is **not an all-FP32 calculation**. An admitted phase-local integral source
uses FP32 recurrence and FP32 storage, with FP64 density products and compensated
scatter. Its immutable geometry/operator/precision identity prevents strict
consumers from borrowing it. Ordinary full-range proposals admit at most twelve
mixed iterations before bounded FP64 physical refinement; the nonlocal outer
schedule has the separate bounded proposal budget described below. At the admitted `1e-4` stationarity
ceiling, consecutive fresh FP64 physical Focks and authenticated final-state
validation replace an additional fixed-point SCF sweep. Export still constructs
the current physical-Fock projector and checks its density defect against the
selected allowance. Stricter density controls retain corrective closure.
Final physical checks run in FP64 against the selected relaxed tolerances;
electron trace, idempotency and canonical diagonalization checks stay strict.
Relaxed candidates retain FP64 force arithmetic, not the independent
strict-density mixed-gradient qualification described next.
Where the complete generated graph and molecular shape already qualify a native
prepared force consumer, budget-selected Focks reuse that consumer in FP64 rather
than repeating Python derivative preparation. Admission also checks the actual
Direct-Fock module capability and finite SCF controls no looser than
`1e-6` energy, `1e-4` density and `1e-10` screening. Final-state tokens, model
identity and resource-capacity checks still apply. Unsupported complete graphs
retain their existing bounded force consumer. Relaxed SCF controls do not
authorize mixed-gradient arithmetic or its additional resident-grid capacity.

An admitted endpoint reports `qualified_budget_fock` and consecutive final FP64
Fock checks. Its ledger requires positive FP32 `coulomb_j` and recurrence work,
or a pure semilocal graph with strict FP64 J and executed budget-selected XC or
density contractions. Positive
mixed `exchange_k` work is additionally required when the native full-range
projection contains K. Separate range K contributes executed recurrence evidence;
VV10 reports mixed pair work only after at least one FP32 pair actually executes,
and retains strict records when pair-domain fallback is used. Nonlocal exact-key
reuse includes arithmetic identity, so FP32 results cannot certify strict closure.
The operator census reports XC matrix products independently of XC point
algebra: FP32 density products have FP32 reduction, while FP32 potential
products have FP64 inter-tile reduction. Both retain FP64 storage. A mixed
matrix product does not imply mixed XC point or analytic-force arithmetic.
Module qualification is not a
measured per-call observable error or a
universal force-error bound. Explicit targets still request a separate complete
FP64 audit. Numerical failures retain the failed attempt with unknown work
counts rather than invented zeros, then use bounded strict fallback/audit solves.

Optional source residency follows the existing bounded cold/size, memory and
free-device-capacity gates. Automatic range sources can retain complete angular
buckets that fit the allowance, streaming the remaining buckets. Disabling
residency or refusing all buckets retains uncached recurrence. Full/SR/LR sources share bounded
angular-bucket selection; unselected FP32 buckets execute mixed recurrence
rather than silently selecting strict arithmetic. Before strict refinement the mixed
lease is audited and retired, then a separate FP64 source may be admitted. The
two quartic caches do not overlap. Failed mixed proposals retire their source
without certifying it and retry the strict target. Completed construction is
counted once; replay reports mixed-source J/K work but no new recurrence work.

For relaxed Fast WB97M-V restricted exact-Direct solves, the ordinary cold
nonlocal outer schedule uses a fixed FP64 J/K/range-K anchor automatically
at 192 or more AOs, without a fixed upper AO limit. This is not a whitelist of
measured AO counts or a claim that every larger shape has been benchmarked. Smaller eligible
owners keep ordinary Fast unless explicitly opted in. Provisional
initializer and frozen-VV10 iterations evaluate linear operators on the total
current-minus-anchor density and reconstruct their full matrices in FP64.
Every full target refreshes the anchor with full-density FP64 J/K; nonlinear
XC and VV10 still consume total density. Delta iterations do not advance the
anchor. Final physical checks and Fast's selected convergence tolerances are
unchanged: this does not promise FP64-mode observable accuracy.

The anchor lease is optional, allocated after mandatory preparation and included
in the native state capacity. It requires five additional FP64 AO matrices and
one scalar. Device allocation failure keeps the ordinary Fast path; host
reservation and non-allocation errors remain explicit failures. At a pressured
force handoff the idle anchor is retired before optional ERIs; density, stationary
weights, XC, the final-state token and the warm seed remain valid. Subsequent
warm and cold solves on that owner use ordinary Fast without reallocating anchors.
FP64 policy, strict caller controls, other models, DF/UKS, public resource-ledger owners,
device chunks and same-owner warm replays retain their incumbent path.
`GENERATIVEQC_KS_FAST_ANCHOR_DELTA=0` disables the optimization before preparation;
unset selects the default at 192 or more AOs; `1` opts in within the scientific
eligibility domain, including smaller owners, without forcing unsupported domains.
There is no universal iteration-count or speed guarantee.

The shared nonlocal initializer and frozen-potential proposal stages also admit
budget-selected mixed Focks. Provisional residuals never certify the target.
The initializer retains its eleven-step cap; additional mixed budget belongs to
the complete target's frozen-potential proposal, capped at sixteen inner steps.
The shared total proposal cap is at most twenty-nine mixed iterations and always
reserves two iterations for fresh strict targets when the caller's limit permits.
Finishing the frozen-potential proposal hands off to strict target evaluation,
not another mixed pair sweep that would immediately be discarded.
Strict refinement leaves the frozen/seed stage,
invalidates nonlinear history and evaluates fresh complete FP64 target Focks.
Bounded FP32 VV10 pair closures reuse the compiler's rational derivatives and
energy denominator, rather than losing those optimizations when lowering to float.
Mixed packets retain each point's immutable FP32 scale eligibility during partner
rotation; pair-distance eligibility is still checked for every physical pair.
The domain, FP64 weighted reductions and strict fallback remain unchanged. Strict
packets use a separate specialization without the mixed admission state.

Arithmetic admission is not a speedup guarantee. Compare complete energy/force
endpoints against the optimized FP64
path without disabling that reuse. `benchmarks/pbe0_fast_fock.py` retains first
calls, alternating repeats, executed J/K work and independent observable gates.

## Strict-Density Gradient Routes

All-electron CUDA
RHF and RKS with exact full-range J/K admit mixed stationary gradients at the
FP64-converged density in the following bounded domains (through d shells,
2–48 atoms):

| Complete electronic model | AO domain |
| --- | ---: |
| PBE0 | 24–384 |
| PBE, B3LYP, r2SCAN | 24–96 |
| RHF | 24–96 |
| M06-2X (2–32 atoms) | 24–96 |
| Canonical r2SCAN-3c, defining def2-mTZVPP basis | 24–96 |

Canonical r2SCAN-3c retains its existing FP64 D4 and native CPU gCP corrections;
only its electronic derivative work is narrowed. Both corrections are included
in the returned energy/forces and endpoint timing. This is not all-GPU 3c.

CUDA density-fitted PBE/PBE0/B3LYP/r2SCAN/M06-2X RKS also select mixed semilocal
geometry with 24–96 AOs, 2–12 atoms, orbital shells through d, and explicit
`GridSpec(24, 8, 16)` or `GridSpec(48, 16, 32)` (unit radii, three Becke passes).
Canonical r2SCAN-3c DF uses the same domain with its defining def2-mTZVPP basis
and unchanged FP64 D4/native CPU gCP corrections. Its electronic precision
record identifies r2SCAN DF; complete-model qualification also requires both
successful correction results at the current geometry.
This domain requires native DF J and, for hybrids, DF K with the default
`1e-10` metric threshold. The complete canonical MethodIR is checked, not merely
the XC name. Changed exchange compositions, arbitrary corrected DF, mixed
full-range J/K providers, COSX and UKS retain the audit fallback. WB97M-V has
the separate complete-model domain below.
In this fitted path **only semilocal geometry arithmetic narrows**: SCF, AO
generation, one-electron/Pulay sources, DF integrals and metric response, and
global reductions remain FP64. The direct-path AO/integral narrowing described
below does not apply to DF. This support does not promise a speedup for every
system or complete singlepoint.
DF energy-only requests have no admitted mixed operator. Eligible strict native
endpoints use the Standard fallback; explicit targets still request an audit.

CUDA DF RHF has a separate native shell-derivative candidate: 24–96 orbital
AOs, up to 512 auxiliary AOs, 2–12 atoms, orbital shells through d and auxiliary
shells through f. It requires the full-rank metric at the default `1e-10`
threshold and a verified strict final density. Only the ordinary shell schedule
narrows; the policy does not replace a faster generic schedule merely to report
FP32 work. Small generic-only and rank-reduced cases retain strict execution;
tighter-budget requests retain the independent audit. Numerical qualification
does **not** establish general HF-DF acceleration.
The cost-selected candidate narrows shell polynomial/Rys algebra and compensated
local FP32 contraction only when an orbital bra shell has angular momentum at
least two. Pairs of s/p-only orbital shells retain FP64. SCF, Gaussian/Boys/Rys
preparation, response weights, metric and
generic derivatives, one-electron/Pulay terms and global accumulation stay FP64.

Direct and density-fitted canonical WB97M-V RKS select mixed semilocal geometry with 24–96 AOs,
2–12 atoms, through-d orbital shells and explicit `GridSpec(48, 16, 32)`.
Its complete range-exchange/VV10 MethodIR and molecular VV10 density policy
must match. Direct uses exact J/K; DF uses fitted full-range J/K with the
default `1e-10` metric threshold and **exact long-range exchange**, including
the matching analytic derivative. Within this separate strict-density gradient
route, SCF, AO production, integral/metric response, VV10 pair evaluation,
nonlocal geometry and global reductions remain FP64. The application facade
selects strict SCF for this route's nonlocal candidate; this does not restrict
the module-selected Fast-Fock route above. An accepted AUTO keyword alone is
not evidence of mixed SCF execution.
Coarse grids, modified range/nonlocal compositions and energy-only
requests retain the audit fallback. No universal endpoint speedup is promised.

The separate
provisional XC warmup covers local restricted PBE-family calculations with
24–256 AOs. Outside these and the budget-Fock domain, the explicit FP64 audit
fallback remains.
This is not a global all-FP32 switch.

The AO limit is an admission boundary, not a performance crossover: larger
systems do not automatically use more FP32. Outside this domain, Fast can cost
two complete FP64 energy/force endpoints because of the audit. Use `policy="fp64"`
for one strict endpoint when an observable-budget audit is not required; inspect
`precision_validation.solves` and native precision work rather than inferring
acceleration from the preset name.

Where the separate local direct XC warmup is admitted, Fast narrows AO evaluation, PBE
exchange and correlation point algebra, supported density
and potential matrix products and local point-total sums. Matrix inner sums
use FP32, not FP32 inputs multiplied in FP64. Global arrays and inter-tile sums
stay FP64. Non-admitted point tails and small/non-tiled contractions retain
FP64. Exact SCF J/K, eigensolution, DIIS, physical residuals and final SCF refinement
remain FP64. Fast does not change the basis, grid,
screening, initial guess or final convergence criteria.

Admitted direct local gradients use FP32 weighted ERI derivative algebra and primitive sums
for orders 0–5, AO jets, density projection/features, interior PBE X/C, AO
pullbacks and point-local Becke forward/reverse sums. Global force reductions,
one-electron/Pulay sources, Boys evaluation, normalization, higher orders and
unsafe tails retain FP64. Nonfinite weighted tasks retry before scatter; a
numerical grid/geometry failure retries the complete gradient once in FP64 at
the same authenticated final density. Optional FP32 grid storage has a bounded
FP64 fallback. No density clipping or CPU reference work enters production.
For B3LYP, r2SCAN and M06-2X, the generated interior XC component graphs also use actual
float parameters, intermediates, elementary functions and outputs. Their shared
work-input conditioning and Cartesian conversion remain FP64; guarded tails
and poorly conditioned points execute the strict graph.
RHF narrows only selected native weighted schedules: generated AOT force
consumers that do not receive the arithmetic request still execute FP64.
Positive task counts are not evidence of all-FP32 integral coverage.

The gradient budget is independent of SCF AUTO admission. A method may retain
entirely FP64 SCF and still execute qualified FP32 derivatives. Ordinary
`Calculator(precision="fp64")` without an application request remains strict.
The private versioned transport can carry HF/KS and DF requests, but transport
availability alone does **not** qualify arithmetic: fitted and nonlocal Fast
endpoints require complete-model evidence. HF DF outside the shell domain and
corrected/range-separated DF outside the canonical models above still need
integration and qualification. Bounded support
does not cover every size or supported DF row in the README benchmarks.
M06-2X uses strict SCF for both energy and force requests; its native SCF owner
does not support AUTO. Budget-admitted derivatives remain independent of that
SCF restriction.

Automatic initial guesses follow ordinary FP64's native Modified Huckel policy
(MINAO on older compatible libraries). Explicit MINAO and Huckel retain their
strict FP64 preparation and resource admission; `initial_guess=None` keeps Hcore. Ordinary
`Calculator(precision="auto")` does not inherit this application-only MINAO
admission. No reference density or extra reference SCF prepares Fast's seed.

```python
from generativeqc import (
    GridSpec, KsOptions, PrecisionCalculator, PrecisionPolicy, TargetErrorBudget,
)

calculator = PrecisionCalculator(
    method="pbe0-rks", basis="def2-svp", device="cuda",
    ks_options=KsOptions(grid=GridSpec(48, 16, 32)),
    initial_guess=None,
    policy=PrecisionPolicy("fast", budget=TargetErrorBudget(
        energy_abs=1e-5, force_max_abs=1e-4, force_rms=3e-5,
    )),
)
```

`budget` reuses the adaptive numerical policy's energy/force allowances; users
do not select kernels. The native admission floors are `1e-6` Eh energy,
`1e-5` Eh/bohr maximum force component, and `3e-6` Eh/bohr RMS if requested.
Tighter allowances or unsupported domains do not silently inherit ordinary
AUTO: they select FP64 and a measured fallback audit. Fast uses the published
native FP64 stopping criteria on these strict-density routes. The separately
admitted budget-Fock route uses the relaxed physical gates described above.

An admitted mixed-force result reports `qualified_mixed_gradient` and
`gradient_admission="<model>-rks-budget-fp32-v1"` (or `rhf-budget-fp32-v1`). Warmup-only results and complete
FP64 gradient retries report `qualified_strict_final`. Each includes its requested
budget, executed precision work, and FP64 final-SCF checks. This is empirical
qualification plus strict final refinement, **not a measured per-call
observable error, mathematical bound, electronic-root or stability test**.
It never reports `observed_met` without an actual reference endpoint.
KS single-point plans are released after each call; repeated calls start fresh
SCF solves.
Approximate work and retained proposal history are bounded; failed mixed work
retries in FP64, and all final physical convergence checks use FP64 operators.

`result.precision["gradient"]` distinguishes attempted FP32 grid/geometry points,
weighted derivative dispatch, complete-gradient retries and returned arithmetic.
Version 2 also reports the execution owner's complete-model qualification.
For canonical 3c, this native record identifies the electronic r2SCAN model;
`precision_validation.gradient_admission` identifies the qualified full 3c
composition, and `result.dispersion` retains both correction results. An
electronic qualification alone does not admit arbitrary corrected methods.
Version 3 adds an explicit collection flag and per-item weighted task,
primitive/source and strict-task-retry counts. Direct RHF reports these actual
counts, without fabricating XC grid work. Its existing weighted order-0..5
consumers narrow local recurrence/contraction arithmetic; unselected schedules,
higher orders, integral geometry, density weights, Boys moments, one-electron/Pulay
terms, global accumulation and final SCF remain FP64. This is not all-FP32 HF.

DF RHF uses the distinct `rhf-df-shell-budget-fp32-v1` qualification and
`weighted_fp32_consumer="df-shell-polynomial-and-rys"`; its
`weighted_fp32_orders` is empty because it is not the direct order-0..5 owner.
Its task census includes complete pre-scatter FP64 replays. Primitive counts
describe prepared products, not pure-FP32 operations or FLOPs. A corrected
strict final density can authorize a candidate without reusing SCF factors.
The cost-selected native dispatch additionally reports
`df_shell_sp_bra_contractions` in `fp64_components`. Historical all-class
dispatch records do not receive this label from newer Python readers.

Version 4 is owned by the Python stationary CUDA consumers, not the native SCF
census. It records complete-model qualification, authenticated state/plan identity,
actual attempted/returned FP32 geometry points, numerical retries and strict
components. Geometry points count candidate enqueues; guarded XC tails may
widen and are not counted as proven float interior graphs. The composite owner
counts only its semilocal candidate; nonlocal geometry and pair work are not
relabeled as FP32. No FP32 AO/grid storage is claimed for DF or WB97M-V.
Missing optional geometry transport uses a separately
reported strict capability fallback; numerical failures after actual mixed work
permit one whole-gradient strict retry. Driver, allocation and stale-state
errors do not become numerical retries. Failed or energy-only batch items never
inherit a previous force record. Both singlepoints and the public prepared
precision fleet retain this per-item provenance.
A verified gradient does not invent a complete SCF operator inventory when
the SCF schedule does not supply one; those completeness flags remain separate.
For uninstrumented KS timing calls, per-task weighted retry counts are unavailable,
not zero; they require separate intrusive qualification. A float64 result array does not
mean that its intermediate arithmetic was FP64. For a **measured per-call**
energy/force allowance, supply `target=TargetAccuracy(...)`: the full FP64
comparison and, if necessary, verified fallback are included in its cost.

`PrecisionCalculator(policy="standard")` runs **one strict FP64 endpoint**, without
an unconditional second SCF/force solve. The CUDA KS scheduler can retain
geometry-invariant J/K integrals and reuse them across SCF iterations, within
the existing memory/lifetime guards. This exact optimization also benefits
ordinary `Calculator`: FP64 is not faster arithmetic withheld from FP64 users.
The native force executor and cooperative scheduling also benefit ordinary FP64
PBE0/PBE/B3LYP/r2SCAN RKS through 32 atoms in each qualified AO domain. Larger d-shell strict
calculations retain their faster phased executor. Fast comparisons must use this
already-optimized strict baseline, not withhold scheduling improvements from it.
It does not imply FP32 execution or promise a speedup for every workload.

Standard (or legacy `fp64`) without `target` reports `strict_fp64`, with `target=None` and no measured
errors. Native convergence/final-state checks are retained, but **no independent
observable-budget audit is claimed**. Its solver controls are at least as strict
as `1e-10` energy, `1e-8` density and `1e-12` screening; stricter caller settings
are preserved. These controls are not a bound on force error.

Supplying an explicit `TargetAccuracy`, including with Fast or FP64, requests a complete
same-backend comparison. Balanced, Strict and `audit_auto` also request these
experimental audits and cost two or three complete endpoints. A custom budget
is never silently dropped to improve a timing. `Calculator.target_accuracy`
behavior remains unchanged.

```python
from generativeqc import PrecisionCalculator

calculator = PrecisionCalculator(
    method="pbe0-rks", basis="def2-svp", device="cuda", policy="standard",
)
```

```python
from generativeqc import (
    ObservableTarget, PrecisionCalculator, PrecisionPolicy, TargetAccuracy,
)

policy = PrecisionPolicy(
    "balanced",
    target=TargetAccuracy((
        ObservableTarget("energy", "absolute", "Eh", absolute=1e-7),
        ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-5),
        ObservableTarget("forces", "rms", "Eh/bohr", absolute=3e-6),
    )),
)
calculator = PrecisionCalculator(method="pbe-rks", basis="sto-3g", policy=policy)
result = calculator.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])
print(result.precision_validation.to_dict())
```

The preset names are case-insensitive. Balanced remains the facade default;
this naming change does not alter default calculations or audit behavior.
Providing `target` replaces all preset audit requirements, so energy-only audits
use a target containing just energy. `properties` must include every explicit
target observable and energy; by default these are requested automatically.
Without a target, Fast defaults to energy and forces; requesting only energy
does not create a force-error measurement.

## Prepared fleets and frozen replays

`PrecisionCalculator.prepare_batch(...)` uses a persistent native fleet, not a
loop over scalar `singlepoint` calls. Its coordinate units, fixed topology,
input ordering and `properties` selection follow ordinary `PreparedBatch`.
Preparation selects a candidate capable of the facade's complete observable
set: if AUTO lacks forces, that prepared plan uses strict SCF, while qualified
FP32 derivative consumers remain available. This does not change subsequent
scalar energy-only AUTO selection.

```python
atoms = [("O", (0, 0, 0)), ("H", (0, 1.43, 1.10)), ("H", (0, -1.43, 1.10))]
with calculator.prepare_batch([atoms], warm_start=True) as batch:
    cold = batch.execute(strict=True)
    batch.set_warm_start_updates(False)
    warm = batch.execute(strict=True)
    print(warm.items[0].precision_validation.to_dict())
    print(warm.precision_validation)
```

Qualified Fast and untargeted Standard (FP64) retain the native plan and its own warm
densities, without seed exports or extra audits. Audited items trigger bounded
host staging of **candidate-owned** seeds and release the candidate device plan
before constructing independent FP64 auditors. The next call lazily
rebuilds the candidate plan and restores only those seeds. An auditor's density
never becomes a candidate seed, even when its energy/forces replace a rejected
candidate. Native restoration validates the source density/overlap and reruns
SCF at the requested target coordinates. Frozen seeds retain their original
source coordinates across movement and audits.
Auditors retain only their own same-geometry, same-control density proposals;
every audit still reruns SCF and recomputes the requested observables. Tighter
fallback verification starts cold. Freezing updates freezes both independent
histories. Geometry or control changes discard the affected audit seed.
Source-overlap validation uses native host work and bounded CUDA eigensolvers
for CUDA HF/KS; its cost is included in resume time. The existing physical
admission gates and near-asymmetric host fallback remain unchanged.

An admitted application-XC warmup does not perturb a converged resident seed
merely to execute FP32 work again. Same-model resident replays begin in FP64 and
must pass the unchanged physical convergence gates; their stationary gradient
can still use budget-selected FP32. Cold, external and geometry-rebuilt seeds
retain the admitted mixed warmup. Arithmetic records describe what ran: a strict
warm SCF does not claim mixed SCF work or a refinement that did not occur.

`maximum_seed_bytes` (default 256 MiB) caps combined candidate/auditor numeric
seed staging. The bound is
checked conservatively for all prepared slots **before** the native query can
materialize resident densities. Staging and validation scratch must also fit
the current `ResourcePlan`'s host headroom; auditors reserve that host storage
from the same budget, including an explicitly supplied resource plan.
Unavailable candidate seed transport or insufficient candidate headroom fails
the affected audited items. Optional audit history falls back to a complete
cold audit when its storage cannot fit. `warm_start=False` needs no seed
transport. `clear_warm_starts()` clears both suspended histories.

Each item carries a schema-2 `PrecisionValidation`. Shared candidate time and
per-item endpoint time are `None`, not an invented division of fleet latency.
`BatchResult.precision_validation` owns actual candidate, resume, suspension,
audit-attempt and complete endpoint timings, candidate precision evidence and
execution-owned Fock counts. Audit-attempt duration includes its setup and any
first-audit suspension; the duration breakdowns overlap and must not be summed.
Complete endpoint time excludes initial facade/batch construction, which must
be added for cold comparisons. Failed audits retain their attempt/error record;
unknown work counts remain `None`.

(python-precision-values)=
## Allowances, norms and evidence

| Preset | Energy allowance (Eh) | Max force-component allowance (Eh/bohr) | RMS force-component allowance (Eh/bohr) | Candidate SCF |
| --- | ---: | ---: | ---: | --- |
| Fast | `1e-5` | `1e-4` | `3e-5` | Budget-selected Direct modules or strict-density gradients; eligible Standard fallback, otherwise FP64 audit |
| Standard (`fp64` alias) | No implicit budget audit | No implicit budget audit | No implicit budget audit | Optimized FP64, one endpoint unless an explicit target is supplied |
| `audit_auto` | `1e-5` | `1e-4` | `3e-5` | Native AUTO on admitted CUDA models; FP64 otherwise |
| Balanced | `1e-7` | `1e-5` | `3e-6` | FP64 |
| Strict | `1e-9` | `1e-7` | `3e-8` | FP64 |

These are selectable **allowances**, not measured accuracy promises or
universal chemistry thresholds. Screening or coarse optimization may tolerate
larger errors; labels for AI training must use their own accuracy requirements.
Set force-error allowances well below the actual optimizer stopping thresholds.
Maximum absolute Cartesian component and RMS over all `3*Natom` components are
different from maximum atomic gradient length and RMS of atomic lengths.

When an audit is requested, every norm must pass independently. A requirement resolves to
`absolute + relative * reference_norm`; a zero reference norm supplies no
relative allowance. Values and force differences must be finite, and force
arrays must have identical `(Natom, 3)` shapes. Coordinates are Bohr, energy is
Hartree, and forces are negative energy gradients in Hartree/Bohr.

`Result.precision_validation` records the profile, selection budget, requested target, actual
scientific-model identity, measured errors and allowances, rejected-candidate
errors, fallback reasons, solve roles, requested precision, backend, solver
controls, iterations, force evaluations and complete `singlepoint` wall time.
Fock-build counts are included only when execution reports valid counters;
missing counts remain `None`. `Result.precision` retains the native arithmetic
and operator-work evidence for the returned endpoint. Requesting AUTO is not
evidence that FP32 actually ran.

An audited success has status `observed_met`. It measures agreement with the stricter
reference, whose own uncertainty remains unresolved. It does not certify an
absolute error, identify the intended electronic root, or test stability.
The identity fixes geometry, basis/representation, electron/spin state,
functional/composition, quadrature/domain, auxiliary basis and fitted metric.
Changing these cannot count as a precision audit of the same target.

## Verification and cost

Each audited call executes a complete candidate and an independent FP64 reference,
including every requested force contribution. Reference energy, density and
screening controls are capped at `1e-12`, `1e-10` and `1e-14` respectively;
tighter caller controls remain in effect. These controls are a reference
recipe, not a conversion from force error to SCF residual.

If any norm exceeds its allowance, the candidate is discarded. The FP64
reference can be returned only after agreement with one additional FP64 solve
with controls tightened by another factor of ten. There are at most three
completed endpoints. The report retains the rejected comparison as well as
the successful returned-endpoint comparison.

An explicit-target or audit-profile call costs two full
endpoints, or three after rejection. Its wall time includes audit/fallback
construction, model checks and resource release, but excludes facade
construction and Python import. Audit profiles make no acceleration claim. Reusing error
estimates across geometries or skipping intermediate audits requires separately
qualified error/conditioning evidence; this facade does neither.

(python-precision-backends)=
## Arithmetic and supported endpoints

The facade supports all-electron HF and KS single points and prepared fleets within the underlying
calculator's admitted property domain. CPU is supported; CUDA calls and FP64
audits remain on the selected CUDA device. Unsupported AUTO constructor domains
or requested properties fall back to the same model's FP64 calculator. An
energy-only request can still use AUTO when its force consumer is unavailable.
`audit_auto` uses the existing native
SCF AUTO policy with its FP64 refinement. It is distinct from Fast's
budget-selected XC warmup. FP64, Balanced and Strict keep SCF in FP64.

For CUDA KS, ordinary AUTO is **not all FP32**. Its schedule admits FP32 computation
with FP64 accumulation for Coulomb J, and for density contractions only where
the selected layout is qualified. Exact exchange K, XC point algebra and
reductions remain FP64; strict refinement restores FP64 throughout the schedule.
The mixed J implementation narrows ERI recurrence arithmetic, not all of J:
Gaussian preparation, density products and primitive sums also retain FP64.
Inspect execution-owned operator records rather than interpreting the AUTO
name or effective-bits summary as a uniform arithmetic type. The native SCF
census is not an instruction-level census of the analytic gradient.

Only the bounded complete models in the table above currently admit
application-selected FP32 gradients. `gradient_admission` and native work
records distinguish them from warmup-only and audited FP64 paths. Unqualified DF models, WB97M-V,
larger bases and other spin domains do not acquire a gradient route merely
from a looser allowance.

The facade does not expose correlated methods, ECP, Hessians,
response or an optimizer/MD driver. A successful single-point audit does not
qualify TS/IRC branches, small-curvature Hessian modes or NVE trajectories.
NVE admission additionally needs energy-drift, force/energy consistency,
continuity and sampling tests. The underlying native geometry optimizer is
unchanged by this Python facade.

## Reproducing PBE0 comparisons

`benchmarks/pbe0_precision.py` compares ordinary FP64, native AUTO and explicit
`audit_auto` at fixed PBE0/def2-SVP spherical models and moving quadrature. It requires
an independent PySCF reference and retains first calls separately from
alternating measured repeats. All runs start from the same Hcore guess; warm
timings reuse process caches, not converged densities. The endpoint includes
calculator construction, SCF, analytic forces, host export and cache release.
Audit timings include the complete audit/fallback work. The current benchmark
schema is version 2; older measurements retain their original names and source
identities and are not relabeled as results of the current revision.

With the source tree on `PYTHONPATH` and the matching native library selected,
generate the CPU reference first (PySCF is an optional benchmark dependency):

```bash
python -m benchmarks.pbe0_precision --engine reference \
  --output .artifacts/pbe0-precision/reference.json
```

On an available, permitted Slurm GPU node, run the native comparison through
the scheduler, preserving its assigned device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:30:00 \
  python -m benchmarks.pbe0_precision --engine native \
  --reference .artifacts/pbe0-precision/reference.json \
  --output .artifacts/pbe0-precision/native.json
```

Speed ratios are emitted only after independent energy/force gates pass.
The report preserves native operator evidence and explicitly flags incomplete
precision inventories instead of inferring FP32 execution from the request.

For FP32 J/K Fast against already-optimized FP64, run
`python -m benchmarks.pbe0_fast_fock --engine reference --output <oracle.json>`
in one finite Slurm job, then `--engine native --oracle <oracle.json>
--output <comparison.json>` in another. The runner retains first calls and
alternating complete endpoints, checks independent energy/force budgets, and
rejects missing FP32 exchange or hidden extra solves. A source-matched existing
independent oracle is reusable only when its complete scientific protocol matches.

For the other qualified Fast routes, run `python -m benchmarks.pbe0_fast` inside the
same finite Slurm allocation, with `--reference`, `--output` and `--repeats 5`.
This reports the current ordinary FP64 default and budget-selected Fast, including
facade construction, strict refinement, analytic forces and any fallback work.
Use `--initial-guess automatic` to measure the default native Modified Huckel seed policy,
or `--initial-guess minao` for the explicit policy; the default benchmark uses
Hcore. Both timed modes always receive the same seed policy, and the independent
oracle's Hcore protocol remains separately recorded. It checks the same
scientific target; speed ratios are not kernel ratios.
Historical no-reuse/FP64 facade and audit measurements
remain bound to their original sources and are not relabeled.

The actual README endpoint runner also accepts `--policy fast`:

```bash
python -m benchmarks.readme_dft --method pbe0 native --policy fast \
  --atoms 3 --basis-file path/to/def2-svp.json --grid 48 16 32 --repeats 5 \
  --reference path/to/independent-reference.json \
  --output .artifacts/readme-fast/pbe0.json
```

Run this inside a finite Slurm allocation. Generate the matching reference with
the same runner's `reference` engine, which remains FP64. Add
`--density-fitting cuda --auxiliary-basis-file path/to/jkfit.json` to both engines
for DF. The default native `--policy fp64` still uses ordinary optimized
`Calculator`, not an audit facade. `--fast-budget ENERGY MAX_FORCE RMS_FORCE`
sets explicit allowances in Eh and Eh/bohr; all three independent oracle gates
must pass. Fast records keep the unchanged strict-parity `fp64_gate` separately
and use a distinct schema; they must not be folded into historical FP64 figures.

Cold includes facade/plan construction; frozen post-cold, moved, and frozen
post-move calls include all requested work, including audits and seed restoration
when needed. A successful Fast request outside a qualified FP32 domain uses
the eligible Standard fallback or an audited FP64 call. Its report records which;
runner support alone is **not** qualification of all README methods/sizes, proof
of FP32 arithmetic, or a speedup claim. Private seed/force experiments cannot
be combined with Fast in this runner.

(python-precision-errors)=
## Failure behavior

Invalid policy types, requirements, properties or molecular inputs raise
`TypeError`/`ValueError`; unsupported scientific domains raise
`NotImplementedError`. Native nonconvergence, allocation/execution failures,
nonfinite observables and mismatched reference models propagate without a
successful result. They are not reclassified as an in-budget endpoint.
FP64 propagates the same native failures; it does not convert a failed solve
into a successful precision report. Invalid/nonfinite solver controls are
rejected before applying FP64's convergence ceilings.

When a converged FP64 fallback still disagrees with its tighter verification,
`PrecisionBudgetError` carries `.validation` with `observed_unmet`, measured
norms and all completed work. It does not carry a publishable partial result.
Extremely small caller solver controls that underflow during tightening are
rejected. `precision` and `target_accuracy` constructor arguments are rejected
here: use the policy target to express the application requirement.

Prepared execution with `strict=False` isolates failed candidate/audit items:
their energy is unusable and their forces are `None`; successful neighbors
remain available. Failed audited items do not retain usable precision evidence
from the candidate. `strict=True` raises after processing all peers, without
rolling back their warm proposals. Whole-fleet preparation, malformed coordinate
layout, diagnostic-query and resume failures raise in either mode. Coordinate
updates are snapshotted once so candidates and auditors see the same geometry.
Numerical failures of a budget-Fock proposal get the same bounded strict retry
as scalar execution. Its failed attempt remains in the work report; unavailable
counts remain unknown. Input and resource failures do not trigger this retry.

(python-precision-ownership)=
## Ownership and concurrency

Policies and report records are frozen. Targets own immutable requirements;
`to_dict()` returns detached data. Calculator inputs are snapshotted through the
native facade, including resolved local basis and KS settings. Calls on one
instance serialize; use independent instances for concurrent execution.

Endpoint workspaces are released before starting the next audit owner, so a
retained candidate does not double the resident resource budget. Returned
arrays retain the ordinary `Result` ownership contract and outlive those
workspaces. `clear_cache()` remains available and serializes with active calls.
Frozen `Result` fields do not make force arrays or native evidence dictionaries
recursively immutable.

Prepared plans require context-managed lifetime or explicit `close()`.
`PrecisionCalculator.clear_cache()` releases scalar caches, not independently
owned prepared plans. The precision wrapper exposes replay, warm-update control,
warm clearing and detached metadata/transport; ordinary native checkpoint and
cross-basis projection APIs are not promoted through this wrapper.
