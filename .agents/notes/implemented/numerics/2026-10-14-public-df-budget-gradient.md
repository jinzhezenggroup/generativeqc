# Decision: public budget admission for fitted semilocal geometry

Status: implemented; bounded DF force domain, not all-method Fast completion
Date: 2026-10-14

The corrected/RSH exclusions recorded here are superseded for bounded canonical
r2SCAN-3c and WB97M-V by `2026-10-14-fitted-composite-budget.md`. That later
qualification does not change this note's original measurements or admit HF DF.

## Problem

The private DF diagnostic already executed typed geometry and passed independent
fitted-Hamiltonian derivative gates. Public Fast still performed strict work and
a second audit because it neither selected that consumer nor retained its work
in per-item precision provenance. Relabeling the diagnostic timing or native SCF
census would not fix either missing production boundary.

## Decision

The ordinary batch force consumer selects the existing shared typed geometry
path from the application's method-level budget. Admission requires a native
current state, strict SCF controls, exact canonical full MethodIR, no external
corrections, restricted all-electron CUDA semantics, and live DF provider proof.
The bounded domain is 24–96 AOs, 2–12 atoms, orbital shells through d, unit-radius
GridSpec(24,8,16)/(48,16,32), and metric threshold 1e-10. Both J and any K must
actually be fitted. Names, point codes, requested DF options and electronic-only
identity after stripping a correction are insufficient. Missing optional proof
denies admission; stale proof is an error, not optional unavailability.

SCF, AO production, fitted integral/metric response, one-electron/Pulay sources
and global reductions stay strict. No extra FP32 producer panels are allocated.
The existing consumer resource bounds remain unchanged; its diagnostics still
exclude independently owned DF-provider response scratch where specified.

Only completed consumer results produce schema-4 gradient evidence. It is
explicitly Python-consumer-owned, distinct from native SCF/gradient ABI versions.
It binds state and plan, counts actual attempted/returned candidate geometry
points and batches, and lists strict components. Points count enqueues, not every
interior XC graph taking the float branch. Per-item merging occurs only for a
successful force publication; energy-only or failed peers cannot inherit stale
gradient evidence. The facade accepts this qualified record without fabricating
native operator-inventory completeness or calling intentional strict AO storage
an allocation failure. Explicit targets and tight budgets still cost real audits.

Missing optional mixed transport has a narrowly typed, separately reported
strict replay. It cannot catch driver/allocation/stale-state failures. The
existing positive-work numerical retry remains separate and bounded; strict
replay failures are never retried recursively. Drain/discard obeys the prepared
lease's failed-before-invalidate transition.

## Validation and retained failures

Host tests cover complete model changes, correction/RSH/UKS/provider/budget
rejection, stale-provider errors, actual work publication, schema isolation and
bounded typed recovery. Initial focused run:158 passed. The expanded run passed
269 before three native CPU batch tests stopped on the absent local native
library; those tests must execute with the source-matched library on n1. This
environment failure is not numerical or GPU evidence.

Slurm8362 passed five public small-grid methods, then stopped in the Cartesian
test fixture because the pinned auxiliary file explicitly describes spherical
functions. The corrected fixture deliberately selects Cartesian auxiliary
functions in both engines from the same unchanged exponents/contractions.
It does not weaken the production representation check or oracle gates.
The failed log/XML are retained, not overwritten by the corrected run.

Public tests call PrecisionCalculator without overriding arithmetic selection.
They cover five methods, small-grid spherical water, dense-grid Cartesian water,
NH3 and the 96-AO water tetramer; strict and mixed forces use an independent
PySCF fitted-model oracle with metric/auxiliary response and moving grids. Other
cases cover explicit target/tight-budget audits, real pending-work numerical
retry and native per-item evidence expiry/isolation. Private moved/frozen
diagnostics remain separate from the not-yet-implemented public batching API.

Slurm 8363 passed all 23 public cases after correcting the representation fixture.
The final host suite passed 270, including optional capability absence before a
source lease exists. Later hardening also requires optional precision counters
before any mixed submission and exposes capability fallback even when zero
mixed work leads to an audit. Its additional real-device test and final-source
check run separately from 8363.

## Public complete-endpoint timing

Exclusive Slurm 8364 measured the actual public facade against unchanged ordinary
Calculator FP64, not against the former two-solve Fast fallback. All 80 attempts
are retained (first call plus three interleaved measured repeats per mode and
population). Every call starts from Hcore; construction, complete SCF/forces,
host export, cache release and synchronization belong to the endpoint. The
independent PySCF DF oracle uses the same pinned cc-pVDZ-JKFIT inputs and dense
GridSpec(48,16,32), but no native density, weights, metric or derivatives.
All 80 attempts passed; their maximum oracle differences were 1.819e-12 Eh,
2.006e-7 Eh/bohr maximum force component and 1.060e-7 Eh/bohr force RMS. These
are observations for the two timing populations, not bounds for arbitrary inputs.

| Method | 24-AO strict/Fast | 96-AO strict/Fast |
| --- | ---: | ---: |
| PBE | 1.0102 | 1.0234 |
| PBE0 | 1.0037 | 1.0121 |
| B3LYP | 1.0154 | 1.0497 |
| r2SCAN | 1.0167 | 1.0594 |
| M06-2X | 1.0238 | 1.1090 |

These are small complete-singlepoint gains, not 2–7x, not prepared README timing
and not a large-system crossover. Earlier private small-grid losing populations
remain in the predecessor note; changing grid/protocol does not erase them.
Raw source-matched receipt: `df-public-timing-8364.json`. The native library is
unchanged `5b7fc0ef88c97ee56246c91c55428b3cb8df972ca5d68fe576a1ce5792bd9599`.
The receipt records the Python source hashes: it precedes only optional-counter
availability/reporting hardening, not a successful-path arithmetic/schedule
change. Its regression run and final public-source checks are separately logged.

The same allocation subsequently passed all 65 regression cases, including the
native CPU batch tests unavailable locally, CUDA batch lifecycle, private moved/
frozen DF and composite candidates, both pending-work fault injections and all
public direct-RKS checks. Receipt: `df-public-regression-8364.xml` and
`measure-df-public.log`. No deselections or numerical failures were hidden.

Final-source Slurm 8366 passed all 24 public DF cases, including the added real
missing-capability fallback and public energy-only replay. Its receipt binds the
final Python hashes after hardening; numerical, strict-audit, retry and evidence
lifetime gates all pass. No GPU/build job remains owned by this checkpoint.
Final receipts: `df-public-8366.xml`, `qualify-df-public-final.log`, the 270-test
host log and `df-public-precommit-all.log`. The complete README Fast goal remains
active: these are bounded fitted-force endpoints, not all-method completion.

## Consequences

This is real but partial FP32, not an all-FP32 DF claim. Complete public endpoint
timing must be measured independently of the prior private-flag results and
must retain losing populations. WB97M-V, HF DF, corrected/RSH DF, broader domains
and the public precision batch facade remain unfinished. Do not infer a large
system crossover or universal speedup from an admission boundary.

## References

- Supersedes the public-integration boundary in
  `../../proposed/2026-10-14-df-fp32-geometry.md`; its diagnostic evidence remains
  historical and is not relabeled as public timing.
- Current contract: `docs/user/application_precision.md`.
- Local/remote receipts: `.artifacts/readme-budget-fast-20261013/` and
  `n1:/data/jzzeng/readme-budget-fast-20261013/`.
