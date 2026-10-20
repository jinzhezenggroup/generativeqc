# Decision: revalidate Fast's stationary warm seeds directly in FP64

Status: implemented
Date: 2026-10-20

## Problem

Default-grid README measurements at source `02337e0cf` found Direct PBE/PBE0
warm Fast running two mixed and two strict Focks, while Standard took one
strict Fock. The approximate restart perturbed an already converged same-owner
seed. The prior strict warm selection covered experimental mixed XC but omitted
application budget-Fock. Merely bypassing native mixed work would also fail the
Python controller's positive-FP32-work admission and trigger another endpoint.

## Decision and invariants

Extend the existing same-owner warm FP64 selection to budget-Fock. Keep energy,
density and physical residual gates, finite iteration limits, final-state
validation and cold/external/moved seed handling. This changes arithmetic
selection, not stopping tolerances or requested scientific outputs.

The Python policy separately recognizes strict prepared warm results only with
complete operator inventory, no executed mixed work, positive strict Fock work
and a final physical audit bound to the returned state identity. Required
operators follow the actual model, including exchange and nonlocal correlation.
No synthetic mixed work or per-call observable error is reported. Missing or
inconsistent evidence retains the full audit fallback. An explicit accuracy
target always retains its independent endpoint audit.

## Evidence and remaining qualification

The policy/batch host suites pass 178 tests, including missing inventory,
approximate arithmetic, missing exchange, stale state, cold/fallback and explicit
target rejection cases. The new default-grid real-device test checks independent
CPU energy/forces for frozen and moved 12-atom PBE/PBE0/r2SCAN/M06-2X endpoints.
Real-device execution and complete endpoint performance are pending at this
checkpoint; no universal Fast speedup or README production ranking follows.

Revisit source reuse and force scheduling if eliminating the mixed restart still
leaves warm Fast slower. Keep those measured causes separate from the stopping
policy and from independent audit requirements.

## Follow-up diagnosis: strict final consumers

Source `6bce35757` eliminates the mixed warm restart on n1 jobs 11201/11202:
96-atom PBE/PBE0 warm each take one SCF iteration and the state-bound strict
warm admission, with no independent endpoint audit. Default warm medians are
10.227/10.940 s. In the same jobs and binary, explicitly enabling the existing
resident final-state proof and relative-density force schedule gives 6.776/7.864 s.
These are diagnostic selector comparisons, not final default performance claims.

Both default selectors incorrectly used historical SCF precision instead of the
arithmetic of their own consumers. The final-state proof now admits strict FP64
refinement following mixed proposals, retaining all model, token, scratch and
physical validation gates. The force selector consumes the actual per-invocation
precision request. Strict Python fallback and FP64 retry paths can use the
existing spatial source schedule; any requested mixed force component retains
the incumbent route. No convergence thresholds or operator approximations change.

The policy/host suites pass 198 tests, including executable precision combinations
for the force consumer. The first GPU test exposed an invalid pure-J range-field
lookup, fixed in `6bce35757`; endpoint probes also required the missing SPD AOT
packages. A later apparent moved-force error occurred in Standard as well:
`execute(None)` selects the preparation geometry, not the preceding explicit
geometry. Keep the moved coordinates on every replay. The diagnostic confirms
the actual moved PBE endpoint passes the original independent force gates.
At source `798f9ddd7`, Slurm job 11382 passes all four frozen/moved GPU tests
against independent CPU level-4 energy and full moving-grid forces. Jobs
11383/11388 and same-build Standard controls 11389/11390 retain complete
96-atom cold plus three warm populations. Every sample passes energy/atom 1e-5,
force maximum 1e-4 and force RMS 3e-5 against the independently stable dense
reference; the CPU/GPU spot and grid-stability checks use one quarter of those
gates. Producer identities are retained separately, not relabeled.

Same-build Standard/Fast cold/warm seconds are PBE 41.169/6.776 versus
34.479/6.709 and PBE0 42.546/8.340 versus 38.393/7.842. PBE warm differs by
only about one percent and does not establish a robust universal speedup.
HF/DF fallback audits and auxiliary-g response performance remain outside this
change. Raw receipts, failed attempts, reference arrays and validation are in
`benchmarks/results/fast-warm-20261020/`.

## Follow-up diagnosis: strict range-value leases

On the same frozen `798f9ddd7` build, WB97M-V Direct48 Standard warm takes
7.024–7.054 s and Fast 9.712–9.780 s. Both execute one strict target Fock and
no implicit endpoint audit. Python profiling locates the difference in native
SCF (1.534 versus 4.536 s), not force evaluation (5.566 versus 5.267 s).
The profiler populations are diagnostic and are not publication timings.

An explicit 4-GiB range-residency diagnostic (job 11417) reduces Fast cold
to 69.550 s but leaves warm at 9.744 s. There are two distinct causes:
automatic refinement discards the mixed lease without building a strict one,
and warm replay rejects even an explicit strict lease because the cold
precision schedule contained mixed operators.

Admit automatic strict range construction only for the existing application
budget-Fock/stationarity refinement domain. It retains the substantial-grid,
restricted SPD, nonlocal outer-solver, complete-resource and 4-GiB bounds.
Retire the mixed lease first; never retain mixed values as strict. The strict
lease can serve the remaining cold target iterations and same-owner strict
warm consumers. Ordinary AUTO retains its streamed refinement tail.
Warm never constructs quartic storage. Failed/refused admission, explicit
seeds, changed policy or geometry, force-memory eviction and public resource
ledgers retain the ordinary-provider fallback and separately rebased energy.

The replay decision follows the next consumer's arithmetic, including the
existing application-budget strict warm rule, and explicitly verifies that
the retained source is FP64. Physical publication and force gates are unchanged.
GPU qualification and endpoint measurements for this extension are pending;
it does not yet establish an additional performance claim.
