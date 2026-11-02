# Experiment integrity and public delivery audit

Date: 2026-11-02. Overall verdict: **WARN; no blocking defect in the declared supporting-evidence delivery scope.** Delivery and deterministic replay checks pass. Historical provenance limits and dormant packaged helpers remain explicit limitations.

This was a fresh read-only experiment-audit review. Requested route: native Codex, `gpt-6.1-sol`, `ultra`; actual returned model and effort are **unverified**. Auxiliary arithmetic reviewer was requested as `gpt-6.1-sol`, `high`, with returned identity likewise unverified. Review independence is **same-family** and acceptance is **provisional**. Deterministic checks are separately reproducible. See `review-trace.json`.

Aliases below: `P = h100-production-20261102/cohort-cd2b/public-delivery-v2/`, `C = h100-production-20261102/cohort-cd2b/`, and `R = generativeqc-h100-current-evidence/`. Primary code, archives, manifests and raw operands were read directly; earlier audit conclusions were not substituted for primary evidence. Writes, extracted copies, modified controls, Python caches and temporary source metadata were confined to this audit directory. No GPU, native-library, SCF or remote execution was performed; the audited receiver itself creates only temporary `.git` metadata pointing at local objects.

## Deterministic delivery results

- Both staged-wrapper full runs return the exact original 1,292-byte `P/public-replay-third.stdout`. The latest private wrapper SHA256 is `3bad9245731663b7d611c4c2ca8d6c008cfb467e27bb67bed8ec488a8f7f3513`. All eight recomputed build/DF/screening/support proof JSON files are **byte-identical** to original receiver files in `C/`; they are not copied in as admissions. Fourteen successful reader invocations include all five support phase readers. Evidence: `latest-positive-results.json`, `reader-invocations.json`, and `P/h100_production_cohort.py:260`, `:306`, `:318`, `:325`, `:389`, `:400`.
- The Nsight parts independently concatenate to the exact original **1,440,220-byte** archive, SHA256 `6e957e8d9a3787a12a8e307ef835194634ac45460cc0c212edc6fcf233c2c1c8`. All three full SQLite databases are reduced again by the public receiver, and separate independent event-sweep/count/correlation arithmetic matches the saved timelines. These are actual databases, not JSON-only reconstructions. Evidence: `deterministic-controls.json`, `raw-science/independent-arithmetic.json`, `P/h100_production_cohort.py:326`, `P/packet/readers/replay-nsys-v2.py:71`, `C/nsys-timeline-v1.py:62`.
- The 23 fixed file bindings and fixed ordered five-phase/status inventory are checked. Missing/changed parts, changed exact reader bytes, missing/duplicate bindings, unsafe paths, missing phase, changed expected status, reordered parts, linked reader files, terminal archive missing/duplicate/mutated/traversal/absolute/symlink/unbound-root members, and mismatched original admissions all refuse. Nsight-specific duplicate/traversal/symlink/member-byte/full-archive-identity controls refuse even after rebinding the copied outer hashes. Unoptimized Python3.12+ is enforced, including an actual `-O` refusal. See `deterministic-controls.json`, `nsys-deep-controls.json`, `optimized-python-control.json`, `copy-admission-controls.json`.
- Eighteen science controls reject malformed/nonfinite operands or fail unchanged gates after deliberately excessive energy/force perturbations. They exercise AST-selected pure functions only; they do not turn dormant helpers into public runtime gates. See `raw-science/science-negatives.json`.
- Source snapshots are made with local `git archive` from native `cd2b4514…` and reference `15ab66a6…`. The native snapshot's object alternate points only to the existing local object store. Reader subprocesses are separate module contexts, with no dependence on the caller's working directory or the private measured-worktree paths. `GIT_NO_LAZY_FETCH=1` and `GIT_ALLOW_PROTOCOL=''` are carried to child readers. Evidence: `P/h100_production_cohort.py:17`, `:166`, `:209`, `:274`, and `latest-positive-results.json`. The archives contain a separate `000517f8…` source-helper snapshot for the six fresh references.

## A. Ground-truth provenance — WARN

The numerical references are independently implemented PySCF/stock GPU4PySCF calculations, not dataset-provided ground truth and not targets manufactured from native outputs. CPU attribution uses a separate scientific solver and complete E/F operands. Fresh-reference endpoints accept geometry/model inputs rather than native E/F or densities, construct fresh independent solvers, block native entrypoints, and record empty native mappings. The ten archived scientific helper hashes match the protocol. Independent AST comparison finds the new scientific solver equal to the archived `solve` apart from its declared name/signature and optional nonlocal-grid override. Evidence: `C/uks-pbe-df-cpu-diagnostic-v1.py:6`, `:31`, `C/oracle_replication6_solver_v2.py:10`, `:20`, `C/reference-six-endpoint-v1.py:18`, `:28`, and `raw-science/reference-provenance.json`.

The original fourteen historical oracle recipe producer bytes remain unrecovered. The six fresh references corroborate six old targets without establishing historical producer identity or revising old raw values. This is accurately disclosed at `P/packet/README.md:38` and `P/packet/readers/replay-reference-six-v6.py:77`. The grid3 raw receipts for the older guard reference stability are absent from this supporting packet; this audit's supplemental fourteen dual-grid arithmetic checks used separately sealed historical originals and are not public packet members. This is a provenance limitation, not a detected target-from-native-output substitution.

## B. Score normalization and scalar/full-vector arithmetic — PASS

Metrics use raw absolute energy differences divided by atom count, full `N x 3` force maximum error and RMS. The denominator is a fixed physical count, not an output maximum/mean. Exploratory speedups are reference/native endpoint-time ratios with a fixed thirty-case denominator; they remain exploratory. Evidence: `P/packet/readers/replay-standard-guards-v2.py:11`, `:14`, `P/packet/readers/replay-reference-six-v6.py:14`, `R/benchmarks/results/h100-sm90-cd2b-20261102/readers/replay_screening.py:11`, `:145`, `:157`.

Independent arithmetic reproduced:

| Retained evidence | Independent result |
|---|---|
| Base DF regression | 299 public full E/F; 295 force and 259 scalar comparisons pass original tolerances |
| Base screening | 264 full E/F; 528 full-vector native/reference cross-comparisons pass; 30 primary inputs; 21 exploratory ratios >1.05; geometric mean exactly 1.265755834772816 |
| Original Standard guards | 84 E/F; 75 passing outputs and nine failed energy outputs; 7/8 populations qualify |
| Default UKS PBE DF failures | All nine energy errors remain 1.1404164624e-5–1.1434822980e-5 Ha/atom, above the unchanged 1e-5 gate; force gates pass |
| CPU attribution | Twelve CPU E/F; default native-vs-matched-DF CPU maxima E/atom 6.1580370433e-14 and Fmax 1.1994760740e-10; the default auxiliary approximation reproduces the failed dense-energy gate |
| Separate explicit refined fit | Nine E/F pass unchanged dense and stricter matched-DF gates; original84 remains 7/8 |
| Fresh six references | Twelve full E/F pass same-level reproduction and new dual-grid stability at unchanged quarter-gates |
| Nsight | Twelve intrusive full E/F pass independent dense gates; all three SQLite reductions pass independent arithmetic |

Complete operands, maxima and source lines are in `independent-base-arithmetic.json`, `raw-science/independent-arithmetic.json` and `raw-science/findings.md`. No score normalization by prediction statistics was found.

## C. Result existence and preservation — PASS

All advertised five phase readers run and return the expected statuses from authenticated archives. All eight derived proof byte hashes equal the original receivers. All 252 independently extracted support files equal existing raw originals; all eight public reader source files equal original reader bytes. `verify_consumed_proofs` rejects recomputed-vs-archived admission disagreement rather than trusting a precomputed PASS. The original Standard failure status is both required by the wrapper and checked by the separate fit reader. Evidence: `P/h100_production_cohort.py:96`, `:260`, `:409`, `P/packet/readers/replay-uks-fit-v1.py:28`, `:30`, and `latest-positive-results.json`.

Native execution evidence retains four records/two unfiltered all-process zero-error memchecks. Production and fixture core hashes stay distinct; other 32 recorded artifacts agree. These are source-bound remote execution records, not local reinspection of the actual binaries. `P/packet/readers/replay-native-v2.py:20`, `:33`, `:58` preserve that boundary and explicitly record the unavailable full RSH displaced-energy transcript.

## D. Dead-code and phantom-gate detection — WARN, nonblocking

The actual five scientific phase readers and their advertised metrics execute. However, `P/packet/readers/cohort-replay-v1.py` and `budget-capacity-replay-v1.py` are bound and copied by the wrapper without being imported by those five readers. The latter only imports the former when itself imported. Their `errors`, `authenticate`, and `replay_plan` functions are dormant in this public entry; testing selected pure helpers in this audit does not make them public runtime gates. Evidence: `P/h100_production_cohort.py:47`, `:382`, `P/packet/readers/cohort-replay-v1.py:24`, `:32`, `P/packet/readers/budget-capacity-replay-v1.py:5`, `:8`.

The CPU12 diagnostic is authenticated, counted and retained by `replay-uks-fit-v1.py:25–32`; runtime arithmetic at `:47` uses the refined-fit matched samples. The public entry does not separately recompute every CPU12 default-vs-dense/native diagnostic comparison. This fresh audit did independently recompute all twelve observations and verified the attribution. The README's twelve-observation count is supported; it must not be upgraded to a claim that a dedicated public CPU12 reduction ran.

Neither dormant helper is used to claim public ResourceBudget/facade acceptance: `P/packet/README.md:46` explicitly excludes that scope. Retain this distinction, remove unused baggage in a later cleanup if desired, or add a dedicated reduction before advertising broader runtime gates.

## E. Scope and source distinctions — PASS within declared scope

README/native/core/reference boundaries agree with the actual packet. Native `cd2b4514…`, stock-reference `15ab66a6…`, and fresh-reference helper source `000517f8…` stay distinct. The later restricted Direct-force compiler change is explicitly unqualified (`P/packet/README.md:5`; `C/upstream-overlap-051e2411-v1.json:2`). Nsight is intrusive attribution with zero clean-timing votes. Fresh six references provide accuracy corroboration, not native dispatch/timing. Original84 failed population is not retroactively replaced by the separate refined9.

The fixed final thirty-case denominator and >=20-win rule remain unchanged; output always sets final-three-pair acceptance false (`P/h100_production_cohort.py:425`, `P/packet/README.md:46`). No A100, RTX5090, universal MD, full public ResourceBudget/facade, custom-RSH UKS or later-source qualification follows. Remote installed-wheel/binary attestations and remote large QDSTRM/nsys-rep hashes remain records, not locally inspected bytes.

## F. Evaluation classification

**simulation_only, subtype independent scientific implementation comparison**, with explicitly labeled native/control observations. This is numerical computational-chemistry evaluation against separately implemented scientific references. It is not dataset `real_gt`, human evaluation, or an unlabeled same-output synthetic proxy.

## Final repository delta

The installed `R/benchmarks/results/h100-sm90-cd2b-guards-20261102/` has all 25 packet files byte-identical to the private audited package. The final installed wrapper/test delta consists only of import ordering/type annotations, an explicit default `subprocess.run(check=False)`, and equivalent short-circuit conjunction formatting. A normalization accounting for those exact semantics matches the original executable AST; no numerical or gate logic changed.

`R/benchmarks/evidence-policy.json` keeps the original 80 MiB result cap, 1 MiB per-file limit and 2 MiB change-review budget. Six new exceptions bind exact archive/part hashes; only the two Nsight parts carry `review_change=true`. The four new large-review rows exactly match file sizes/hashes. Every public packet file is below 1 MiB. Evidence: `repository-delta-review.json`; final 29-file delta is sealed in `repository-delta-hashes-before.json` and `repository-delta-hashes-after.json`. The installed wrapper returns the exact original stdout and empty stderr. An independent installed nine-test run passes (9 passed in 8.23s), using Python3.12.14, NumPy2.5.4 and existing cached pytest8.4.2 with cache/conftest/plugin autoload disabled. The two earlier setup attempts failed before collection because pytest was absent and an ephemeral interpreter had expired; they are retained as setup records. All 49 deterministic controls exhibit their expected accepted/refused or numerical gate-failing behavior.

## Findings and safe local recommendations

No blocking finding was detected for publishing this bounded supporting packet.

1. Keep historical14 provenance, native transcript, remote binaries/raw traces, source-transfer and final-performance exclusions attached to any downstream claim.
2. Treat dormant helper functions and CPU12 authentication as described above; do not count them as executed public ResourceBudget or full diagnostic reductions.
3. Optional path hardening: `checked_publication` checks the leaf with `path.is_symlink()` at `P/h100_production_cohort.py:137`, but a symlinked `readers/` parent directory with exact authenticated reader bytes is accepted. A copied control demonstrates this. Hash integrity remains intact, so this is not a delivery blocker. If strict physical containment is desired, reject symlink path components or require resolved paths to remain under the resolved packet directory before opening them.

## Input preservation and assurance

The first seal covered 528 input files. During review, the executor explicitly notified an early Python-version/optimization guard addition to the private wrapper; that is the only change from the first seal. The prior hash is preserved in `initial-preservation.json`; the latest wrapper/test were snapshotted and sealed before the latest full replay. The scientific packet/readers remained unchanged. The final primary seal covers 529 files plus the separately sealed 29-file installed delta. Final preservation results record exact unchanged counts; supplemental historical grid3/grid4 files have a post-read hash inventory with that boundary labeled. No claim of whole-workspace immutability is made because the executor separately installed the reviewed repository delta during this audit.
