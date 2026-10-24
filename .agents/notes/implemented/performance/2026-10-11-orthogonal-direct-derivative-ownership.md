# Decision: orthogonal optional Direct shell-derivative ownership

Status: implemented opt-in; complete endpoint qualification pending
Date: 2026-10-11

## Problem

The f80 complete water pilot retained bounded AO-task force execution for B3LYP,
M06-2X and r2SCAN-3c. Its native fallback detail reported no retained shell
derivative lease. This is not proof of a single cause at a newer source boundary.
Static inspection does expose one unnecessary preparation coupling:
`prepare_generated_coulomb` returns no owner when streaming value coverage is
incomplete and bounded fallback is disallowed; the s/p/d Direct constructor
passed `false` even when retaining first derivatives. The existing derivative
program can cover the full bounded shell domain independently of that streaming
value portfolio.

The next checkpoint is master `56bedd457`. Its only increment over `bd6f78288`
is strict-zip shape enforcement in tensor AD plus tests. The previous diff was
preserved under `.artifacts/functional-independent-xc/56bedd-integration/`.
The new full native baseline is built from an immutable source snapshot, not
the earlier f80 or bd6 point-only binaries.

## Decision

Add a constructor-only `GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK=1` request
for retained first-derivative s/p/d owners. Unset/0 preserves original behavior;
other values fail explicitly when the derivative candidate is considered.
Value-only, borrowed resident-source and through-f owners are unchanged. The
shared exchange owner remains within the original optional budget and its
existing lifetime/resource accounting. Failed optional preparation retains the
old complete bounded force route.

Use the existing bounded-shell preparation rather than a second force owner or
new scientific program. Its incomplete value coverage still fails the existing
`direct_jk_generated_full_range_value_available` gate unless a caller separately
enables experimental bounded values. The derivative request alone cannot promote
SCF value execution. Component packing and point scalar sharing remain independent
selectors. No functional, basis-name or molecule whitelist is added.

## Rejected alternatives

- Enable all bounded values together with the force lease: this changes an
  unrelated value schedule and conflates numerical/performance qualification.
- Unconditionally remove the shared owner's value gate: existing value-only
  consumers rely on it, and optional storage still needs bounded admission.
- Add a second native derivative owner: unnecessary duplicate normalized
  geometry, matrices, bounds, allocations and stream-lifetime handling.
- Treat historical f80 timings or point sanitizers as current endpoint proof:
  neither executes this new constructor policy or its whole molecular endpoint.

## Evidence and remaining gates

`tests/python/test_direct_derivative_fallback_policy.py` executes the actual
host selector and preparation join with capability/budget stubs, not reference
integrals. It checks all combinations of selection, retained order, complete
coverage, resident-source mode, through-f mode and optional failure. Uncovered
derivative owners cannot enter ordinary SCF values; tight budgets still fall
back and malformed requests fail. Alongside existing derivative layouts,
stream-drain lifetime/error probes, queue pages, KS weight staging and the new
master strict-zip contracts, **535 tests pass**. Raw output is
`.artifacts/functional-independent-xc/master56-endpoints/derivative-policy-tests.log`.

The complete source-matched CUDA baseline and its five stationary force packages
finish successfully with release sm_120 settings and existing ccache. Raw build and
endpoint drivers are under `master56-endpoints/`; remote inputs live under
`n4:/data/jzzeng/gqc-functional-independent-20261011/master56-baseline/`.
Real GPU work must run through finite Slurm allocations. Do not infer that a
candidate actually retained or executed a shell derivative lease merely because
its selector was set: inspect native availability, fallback detail and routes.

Independent larger-system and UKS references keep the same PySCF 2.14.0 /
Libxc 7.0.0, matched basis/grid, reconverged two-step finite differences and
strict convergence/uncertainty gates. Neutral OH did not converge in the initial
direct or Newton-fallback reference attempts. Those failed logs are retained;
they are not oracle acceptance. Water cation supplies a separate nondegenerate
UKS fixture without relaxing convergence or force gates. Complete native
energies/forces, actual executed work, cold/warm/moved timing and endpoint
sanitizers remain pending and do not authorize promotion or goal completion.

## Current-source ownership qualification

The baseline library SHA256 is
`3352bd185ef20b7cc86e1fb7902adc75151264c387313d11f3eacfdd0a0e0503`.
The candidate library SHA256 is
`5cebdf095178a3399b2d6010e92286cf525ee8d97666bbe02a1b2dc152fb4a35`.
Both use independently verified frozen source inputs, Release CUDA 12.9/sm_120,
the same build recipe and existing ccache. Candidate native/compiler differences
are only this constructor policy and the ingredient-layout geometry join described
in the functional-independent XC note; no integral formula or precision changes.

Slurm 766/768 executes the same-library selector-off/on complete water cohorts
with scalar sharing and bulk packing held fixed. All 56 energy/force observations
pass the independent molecular gates. B3LYP, M06-2X and r2SCAN-3c positively
report `prepared-native-stationary` and an available native derivative source with
the selector on, instead of the original bounded AO-task route. Setting the
selector is not used as a proxy for this ownership evidence. The complete
wB97M-V nonlocal endpoint also passes; its composite native integral work is
not exported, and no whole-endpoint work/performance promotion follows.

The strict larger-system/UKS reference run accepts 14 geometry/method phases;
maximum two-step FD uncertainty is `1.8000415972589205e-9` Eh/bohr. Its input
basis/grid matches the native contract, but AO evaluation, ERIs, Libxc, SCF and
reconverged finite differences are independent PySCF/Libxc work. The first UKS
endpoint attempt (Slurm 769) exposes missing B3LYP/PBE0 UKS AOT packages, not a
numerical failure. Its failed JSON is retained separately; the two existing
qualified package targets are built without rebuilding or replacing the native
library. Package inventories/checksums are verified before subsequent endpoints.

The broad host run reports **4,124 passed, 642 conditional skips** and three
unmet configuration prerequisites: two default-CLI CMake checks need unavailable
CLI development dependencies, and the CPU AOT export check needs a CPU-force-AOT
enabled library. The two cache checks pass in the same non-CLI configuration as
the native qualification builds; the CPU AOT export check explicitly remains
skipped without an applicable library. No tests or numerical gates are weakened
to conceal these prerequisites. CUDA ownership inventory reports 354 maintained
files, no reclassification, zero scientific LOC growth and one fewer runtime LOC.

### Complete molecular and sanitizer receipt

Slurm 773 completes both selectors for B3LYP, M06-2X and the PBE0 control on
water cation (UKS), water dimer (RKS, 48 AOs), and the PBE0 water control. Together
with the baseline and same-library water cohorts, **182 complete energy/force
observations** pass: maximum energy error is `3.694822225952521e-13` Eh and maximum
force error is `5.75830405580291e-9` Eh/bohr. Native derivative availability and
the `prepared-native-stationary` route are positive for every applicable
selector-on cohort, including UKS and the larger system. No method whitelist,
weaker convergence criterion or omitted force/model correction is used.

Complete-water endpoint memcheck (Slurm 770/771) and initcheck (772/774) execute
both selectors across B3LYP, M06-2X, wB97M-V and full r2SCAN-3c. All four
sanitizer logs report zero errors; their additional 16 cold energy/force
observations also pass the independent molecular gates. This is complete
endpoint qualification, not reuse of the earlier point-only sanitizer receipts.

The paired full-API warm medians below are diagnostic observations, **not
performance/default promotion**: native/composite executed integral work still
is not exported. Logical quartets, shell capacities and absent counters must
not be substituted for executed work. The selectors therefore remain opt-in.

| Fixture | Method | Selector off (s) | Selector on (s) |
| --- | --- | ---: | ---: |
| water | B3LYP | 4.001094 | 0.084724 |
| water | M06-2X | 4.027988 | 0.114168 |
| water | wB97M-V | 0.287874 | 0.210904 |
| water, prescribed corrections | r2SCAN-3c | 9.273647 | 0.214368 |
| water cation, UKS | B3LYP | 4.032995 | 0.124570 |
| water cation, UKS | M06-2X | 4.075517 | 0.167370 |
| water cation, UKS | PBE0 | 4.037337 | 0.124278 |
| water dimer, 48 AOs | B3LYP | 51.752022 | 0.250630 |
| water dimer, 48 AOs | M06-2X | 51.838536 | 0.286261 |
| water dimer, 48 AOs | PBE0 | 51.881748 | 0.249582 |
| water control | PBE0 | 4.000595 | 0.085164 |

The water off/on requests use Slurm 766/768; the extended pairs share allocation
773. Each median contains three full energy-plus-force executions. Cold
including preparation, moved-geometry executions, SCF iteration diagnostics,
resource snapshots, native availability, source/library/script hashes and raw
work records are retained under
`.artifacts/functional-independent-xc/master56-generic-final/` and the immutable
remote checkpoint of the same name. `verify_endpoints.py` checks the independent
oracle gates, pair correspondence, matching candidate library/runtime hashes,
all sanitizer summaries and the explicit false promotion decision;
`complete-endpoint-verification.json` is its machine-readable receipt.

## Revisit when

Promote only after positive source-bound native ownership, independent forces,
complete endpoint/resource/work evidence and explicit fallback controls pass.
If native work is unavailable, retain an unqualified measurement rather than
substituting shell capacity, logical quartet bounds or zero counts.
