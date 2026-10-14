# Decision: admit larger resident DF guesses within the caller's phase budget

Status: implemented
Date: 2026-10-13

## Problem

The [resident-only policy](2026-10-09-df-rhf-work-admission.md) correctly rejected
streamed preliminary factors, but its fixed 512-MiB allowance also rejected a
profitable resident proposal for the 434-AO QM7 C7 reference. A 64-GiB caller
budget did not change that independent guess cap. The resulting Hcore start was
not evidence that DF preconvergence was unavailable or slower.

## Decision and invariants

Keep compatibility, exact-source cache exclusion and the existing work proxy.
Start the allocation-free preparation/tile-plan preview at 512 MiB, retaining
that allowance for already-admitted cases. On refusal, double up to the smaller
of 4 GiB and the caller budget remaining after correlation metadata and retained
response caches. Require full retained factors, not streaming, before beginning
GPU work. Neither the caller's budget nor the 32-cycle/single-attempt work cap
is increased. The planner remains authoritative for actual live allocations.

Transfer only a validated density. No fitted Fock, orbitals, DIIS history or
reference/response approximation is accepted. Existing preliminary failure and
seeded-target cold retries remain bounded. Exact target refinement, physical
reference publication and downstream analytic equations are unchanged.

Replace the policy audit's internal source-expression assertions with the
existing compiled behavioral tests, extended to cover planner refusal/exception,
budget growth, retained-cache subtraction and the 4-GiB ceiling. The inventory
still discovers the public rollback selector. Refactoring a boolean expression
must not break the admission audit when executable semantics are preserved.

## Evidence

Source is `51ca8eb1d` plus this `df_hf_guess.cpp` change (file SHA-256
`e8efa405d72537556b233a0c58e2d221f8e018957bbb5c9e4dd663b926d095f7`). The frozen
library SHA-256 is
`6c5f76b76e0ab0587d2f2b39497dd127c3c1018578ee020df836fe2e59f30226`.
The subsequent `085aea326` test-only repair does not change this native source.
Build: CUDA 12.9.1, Release sm120/auto, eight AOT shards, fast compile off,
direct device link, verified sccache 0.16.0. These measurements precede the
separate cooperative materialized-Fock experiment; do not pool its timings.

Node4 RTX 5090, UUID `GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`, driver
580.178.04, 575 W, finite Slurm job 818. The pinned QM7 index 5604 input has
23 atoms, 58 electrons, 434 spherical/485 Cartesian cc-pVTZ AOs, 1047 correlation
auxiliaries and input SHA-256
`696b783d661883745a71f15cb529e83802773d7ace6cbaf38f3e69f2e6cc16d0`.
One fresh-process pair, same binary/device and target controls:

| Initial guess | Complete RHF (s) | DF stage (s/cycles) | Screened/strict exact Focks |
| --- | ---: | ---: | ---: |
| Hcore control | 346.850328884 | 0 / 0 | 23 / 7 |
| Automatic resident DF | 223.673319219 | 34.583853771 / 21 | 8 / 6 |

This is 1.5507x faster, or 35.513% less **complete reference** time, including
DF preparation and final exact refinement. The bundled aug-cc-pVTZ-JKFIT proposal
has 1464 functions; the admitted allowance is 2,147,483,648 bytes and previewed
resident value-plan peak is 2,143,648,567 bytes. There are no preparation/seed
fallbacks, exact retries or post-SCF Focks. DF physical-Fock counts remain unknown,
not inferred from its 21 SCF cycles. The complete DF timer must not be replaced
by the smaller nested SCF timer.

Independent PySCF 2.14 conventional-RHF maximum errors for the candidate:
energy `2.8421709430404007e-13 Eh`, density `1.8944493543338048e-11`, orbital
energies `7.021627723702295e-11 Eh`. Commutator/canonical drift are
`8.36886115962443e-12 / 4.144476428713517e-12`. Gates are respectively
`2e-9 / 3e-8 / 3e-8 / 1e-8 / 1e-8`. Every admitted final frame passes.

An explicit 1033-function cc-pVTZ-JKFIT preparation in Slurm 817 takes
270.692038801 s including 34.398141511 s preparation, with 21 DF and 17 exact
cycles. Tightening its preliminary tolerance from `1e-4` to `1e-6` leaves the
same cycles/state and takes 270.800226552 s. That different auxiliary is an
experiment, not the automatic policy or an independent repeat of it.

Host qualification: 188 passed, 12 optional skips. Slurm node5 job 1513:
28 source-matched screened/bounded RHF and DF-MP2/DF-CBS E+F tests pass, followed
by 36 preconvergence tests including native methane E+F and failed-one-cycle
fallback. Independent reconverged energy differences qualify small-system
forces. Run these suites in separate processes: the combined process run hit
`std::bad_alloc` in a child probe, consistent with retained parent GPU state;
isolated execution and standalone cold probes pass without a production workaround.

Raw receipts remain ignored under `.artifacts/rhf-bounded/` and node4
`/data/jzzeng/qc-cbs-rhf-20261013/receipts/rhf-fix/`, notably
`c7-guess-{auto-direct,direct}.json` and `df-guess-c7.log`.

## Limits and rejected alternatives

This is not a full C7 CBS E+F timing or a claim of size-independent speedup.
The separate C7 DF-MP2 E+F experiment (Slurm 819) was stopped without a result
after exposing a roughly 251-s exact-response Arnoldi action. Its unfinished
receipt cannot certify a total time or force gate. Small complete-force and
larger complete-reference qualification must not be relabeled as a completed
large-force benchmark. Response work remains a separate bottleneck.

Native MINAO exists, but its qualified CUDA provider currently belongs to
restricted KS, not the correlated conventional-RHF reference owner. Removing
only a Python guard would not connect charged GPU eigensolver workspace, strict
seed admission and bounded retry/publication. No PySCF/CPU target SCF seed or
silent MINAO fallback is added here. MINAO may be useful, but is not assumed to
beat a density-fitted molecular guess without complete timing/root checks.

Do not lift the resident requirement, use the entire caller budget by default,
or fit a geometry/AO cutoff to this one pair. Revisit the 4-GiB cap and broader
element/backend admission with fresh independent endpoint evidence.
