# Native Modified Hückel on exact CUDA RHF

This reviewed copy retains compact scalar gates, exact samples, scientific work
counts and the measured production patch. `qualification.json.gz` holds the
compact records. `recovery.json` binds original full records/source audits to
existing Git history; exact bytes remain recoverable without exceeding the
checkout evidence budget. Dense density exports and transient build/job receipts
remain in ignored local run storage; regenerate those exports
for a full density-replay audit. Historical policy statements below describe the
decision at measurement time; the later default/DF decision supersedes them.

This campaign qualifies the CUDA HF execution-owner integration, separately from
the earlier PBE0 initial-guess campaign. It measures native atomic preparation,
not an oracle-backed seed or an isolated Hückel microbenchmark.

## Source and numerical qualification

- Frozen base: `ab5282f74c0acf98c18ec05b343a5494de85cf49`, plus native Hückel,
  eigenpair ordering and the CUDA HF integration. The recoverable `source-audit.json` inventories
  1,998 production/qualification files and all 24 changed production files.
  `measured-production.patch` and the recoverable `measured-source.patch` reconstruct the actual
  qualified source, not the complete dirty checkout or an earlier deployment.
- Library SHA-256:
  `d598040711420953ece33ecabeff61751c5c2651911290560d54a59959981886`.
- Slurm 7861 passes the native preliminary contract suite, 111 native/CPU Python
  tests and all 17 real-GPU CUDA HF tests. The separate host control regression
  passes 134 tests, including seven evidence-reader regressions, skipping the 17
  GPU cases without an allocated-device opt-in.
- The independent CPU PySCF tests gate energies, analytic forces, final densities
  and physical F(D) commutators for exact STO-3G/def2-SVP inputs in spherical and
  Cartesian representations. Cartesian component normalization is converted
  explicitly; rounded packaged tables are not compared to different raw inputs.
  No numerical gate is relaxed to make those tests pass.
- Mixed imported-warm/cold batches, zero hydrogen atomic J/K, preparation caps,
  atomic failure, one seeded/warm core retry and exact device budgets are tested.
  Checkpoint import's separate host staging has explicit headroom. Used seeds do
  not pass through warm normalization or inherit an unrelated core orbital frame.
- The first 574-step CUDA build uses verified ccache 4.5.1, checkout-root path
  normalization and reused caches. Actual CXX/CUDA launcher commands, configure,
  build, before/after cache statistics and subsequent qualification receipts are
  retained. Failed candidates are separate receipts, never benchmark observations.

## Complete-endpoint screen

Slurm 7869 measures 48/96 atoms; Slurm 7873 adds the 3-atom scope control. Both
run on node1/n1 with a finite allocation, one assigned RTX 5090 and eight CPU
cores. The assigned physical GPU's UUID/PCI identity is checked before and after
every endpoint, rejecting any other CUDA process on that device. The long-lived
job on another GPU is not cancelled. Node-exclusive job 7865 never executes;
its pending receipt is retained separately. Device visibility is never changed.

The model is exact FP64 RHF with the frozen spherical offline def2-SVP H/O basis
and nested README water geometries. Native energy/density/screening tolerances
are `1e-12`/`1e-10`/`1e-12`, with 100 target iterations. The independent GPU4PySCF
1.8.1 / PySCF 2.14.0 reference uses full Focks, `1e-12` energy, `1e-10` gradient
and `1e-14` direct screening. No reference density is imported into a native solve.

Each arm has three fresh-process/owner samples, rotating `core,huckel` and
`huckel,core`. Complete time includes preparation, native atomic construction,
strict target admission, synchronization, energy, analytic forces and any retry.
Hückel's retained ordinary force canonicalization is included, not optimized away
by assuming that density admission proves a matching orbital frame. Library
hashing, final-density export and independent F(D) verification are outside the
timer. Persistent compiler/runtime caches remain enabled for both arms.

All 18 native endpoints pass the independent `1e-8 Eh` energy, `1e-7 Eh/Bohr`
force, `1e-10` physical residual and `1e-8` paired final-density gates. Maximum
errors are `2.05e-11 Eh`, `3.15e-11 Eh/Bohr`, `1.09e-11` residual RMS and
`6.05e-11` paired AO-density error. All nine native Hückel seeds are used with
complete preparation counts and exactly one target attempt, without retry.

| Atoms | Hcore median (s) | Hückel median (s) | Complete wall reduction | >5% gate |
| ---: | ---: | ---: | ---: | --- |
| 3 | 1.427763 | 1.376158 | 3.61% | fails |
| 48 | 11.586258 | 10.781192 | 6.95% | passes |
| 96 | 40.352622 | 36.235778 | 10.20% | passes |

Paired reductions are 2.31%, 3.81%, 0.006% at 3 atoms; 6.92%, 6.95%, 6.82% at
48 atoms; and 13.16%, 10.21%, 10.20% at 96 atoms. The first 96-atom core sample
has 26 Focks rather than 25; it is retained, not removed to manufacture a gain.
The reported gate requires both a strictly >5% median reduction and every paired
repeat exceeding 5%. A positive small-system median is not sufficient.

| Atoms | Core target Focks | Hückel target Focks | Hückel preparation median (s) |
| ---: | --- | --- | ---: |
| 3 | 19 / 19 / 19 | 18 / 18 / 18 | 0.002043 |
| 48 | 24 / 24 / 24 | 22 / 22 / 22 | 0.176473 |
| 96 | 26 / 25 / 25 | 22 / 22 / 22 | 1.260207 |

Every Hückel sample performs seven atomic iterations/Focks; hydrogen performs
no J/K. Hückel target counts include one post-SCF physical Fock for ordinary
force canonicalization. The legacy public CUDA HF scalar Fock/residual fields
can be absent; target work comes from the certified precision ledger, and the
physical residual is independently rebuilt at the exported target density.

## Decision and evidence replay

Explicit `InitialGuessSpec("huckel")` is admitted for exact FP64 closed-shell
all-electron H–Ar CUDA RHF energy/force consumers, including resident/bucketed and
independent Fock execution. UHF/DF/ECP remain unqualified, and CUDA HF MINAO stays
unsupported. CUDA HF automatic selection remains Hcore. The 3-atom result fails
the existing >5% gate; two favorable larger water cases do not justify a global
default or a hard-coded atom-count whitelist. Other methods retain their existing
MINAO/Hcore defaults; this HF campaign is not a CPU/KS profitability claim.

There are 21 original lossless raw JSON gzip records, recoverable through the
revision and SHA-256 inventory in `recovery.json`. For any listed payload, use
`git show <git_revision>:<path> > <scratch-path>` and verify its hash before replay.
The original run also retains 18
lossless NumPy density gzip exports in local run storage. Recover/regenerate
these exports and decompress the records into a temporary directory without changing bytes;
`runner.py --summary <directory>` rechecks the large cases, and
`runner.py --summary <directory> --atoms 3` rechecks the small case. `summary.json`
and the recoverable `work-ledger.json` combine accepted observations without modifying them.

Job 7869 completes every measurement but exits during postprocessing: its old
reader accidentally parses the initially empty redirected `summary.json`.
The fixed reader selects measurement filenames only and validates the retained
records on the host. Source/library/harness before/after hashes still match.
The exact measured large-case runner is recoverable losslessly as
`receipts/measured-large-runner.py.gz`; its digest matches every large record.
The recoverable `receipts/checker-update-audit.json` proves the timed and independent verification
functions are unchanged in the current runner. Only reader handling and the
explicit 3-atom CLI/summary selection change. Small job 7873 then completes with
the corrected reader. Expired final Slurm queries are retained as query errors,
not substituted with invented completion states.

See the [execution-owner decision](../../../.agents/notes/implemented/numerics/2026-10-12-cuda-hf-huckel-admission.md)
and the [current user contract](../../../docs/user/initial_guesses.md).
