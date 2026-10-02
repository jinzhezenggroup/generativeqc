# Preliminary SCF: actual Python API evidence

This publication records `Calculator(initial_guess=...)` measurements at the
frozen source snapshot below. Later source mappings admit NEW reproductions;
they do not inherit these historical timings or qualification.
It is separate from [the earlier native prototype](../preliminary-scf-prototype/README.md):
none of that prototype's timings, oracle claims or sample counts are reused here.
The performance decision remains **inconclusive/non-promotable**.

## Experiment

Two exact earlier inputs are used: water/6-31G (13 Cartesian AOs, 9 shells,
22 primitives) and the 12-atom water tetramer (52 AOs, 36 shells, 88 primitives).
The primitive coefficients and Bohr coordinates are explicit in `inputs.json`;
no network basis lookup or name-dependent coefficient substitution is involved.

Each of three repeats launches a fresh process for each of `initial_guess=None`,
`InitialGuessSpec("hf")`, and `InitialGuessSpec("lda")`: 18 requested endpoints.
Route order rotates as direct/HF/LDA, HF/LDA/direct, LDA/direct/HF. Each route
occupies each ordinal position once per system; case ordering also alternates.
The separate HF smoke is excluded from all campaign statistics and samples.

Every final target is PBE0/RKS, CPU FP64, exact two-electron providers,
`GridSpec(24,12,24)`, energy/density tolerances 1e-10/1e-8, maximum 150
iterations, DIIS history 8 and screening 1e-12. Preliminary policies use the
actual API defaults: maximum 32 iterations, DIIS history 8, tolerances
1e-6/1e-4, numeric cap 256 MiB; LDA uses `GridSpec(8,6,12)`. The earlier
prototype's maximum 150 preliminary iterations is not silently substituted.

## Observed results

All 18 endpoints completed and passed every row check; all 12 candidate/direct
pairs passed the native numerical gates. No campaign failures or samples were
discarded. Three-repeat medians below are descriptive, with seconds rounded only
for display; the full unrounded samples are retained.

| Input / route | Endpoint seconds | Whole-process seconds | Target iterations | Process RSS high-water KiB |
| --- | ---: | ---: | ---: | ---: |
| Water / direct | 0.481423 | 0.891274 | 12 | 51676 |
| Water / hf | 0.432690 | 0.942733 | 8 | 52948 |
| Water / lda | 0.408909 | 0.885479 | 8 | 52964 |
| Tetramer / direct | 43.623370 | 44.178972 | 18 | 327704 |
| Tetramer / hf | 30.640891 | 31.102033 | 11 | 388016 |
| Tetramer / lda | 31.494415 | 31.969313 | 11 | 388016 |

Largest candidate/direct energy difference: 2.046363078989089e-12 Eh.
Largest total AO-density difference: 7.122084033639453e-10.
Largest native reported physical residual: 8.949678861161504e-11.

All requested preparations were used; no target retry or discarded target work
was reported. HF and LDA both reduce the tetramer target from 18 to 11 iterations
in this sample. That endpoint reduction is accompanied by a higher observed
child-process RSS high-water: median 327,704 KiB direct versus 388,016 KiB for
both preliminary routes. This is a real process-level resource observation,
not an attribution of all extra memory to the preliminary owner.

Negative/uncertain results remain visible: water HF has a lower endpoint median
but a higher whole-process median than direct, with imports and process overhead
outside the endpoint timer. Large first-repeat variation, only three repeats
and non-exclusive affinity prevent a general profitability claim. The original
prototype also contains its distinct same-grid-PBE and triple-stage negative
results; they are not remeasured or pooled into this API matrix.

## Timing, diagnostics and checks

The complete endpoint starts before input materialization and includes
Calculator construction, native library loading within construction,
`prepare_batch`, and `execute` through returned diagnostics. Python API imports,
read-only final-density export, closing the batch, interpreter/process launch
and process termination are excluded from that endpoint and reported separately.
`process_wall_s` retains the full subprocess wall measurement. In particular,
`process_nonendpoint_s` includes imports/export/cleanup/IPC as well as launch;
it must not be mislabeled pure operating-system launch overhead.

Every batch is newly constructed and executes exactly once with no imported
state. Warm retention is enabled equally for all routes solely to expose the
final density. The fresh-state checks require no warm density was used.
`save_checkpoint` cannot represent PBE0 in its HF/MP2-only model schema, so the
harness uses the existing read-only native DFT warm-state buffer export. This
is a total AO-density observation, not an HF checkpoint or an extra SCF solve.

Each row retains constructor/preparation/execution totals, policy diagnostics,
preliminary/target/discarded work counters, full final-target KS iteration
history, final energy, full density array/hash, physical residual and all
acceptance checks. Basis shell/primitive/AO counts, grid, backend, fresh state,
requested provider use, export energy/coordinates and convergence all affect
acceptance. An exported-state failure preserves the completed solve but blocks
numerical acceptance. No failed attempt may be removed from the sample set.
Native direct/candidate gates are 1e-8 Eh energy, 1e-6 maximum AO-density error,
and 1e-8 reported physical residual. Gates use full raw arrays, not medians.

Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss` is observed after cleanup,
before final record serialization. It is a child-process high-water mark that
includes imports, calculation and density export, not incremental owner
allocation or an exclusive endpoint numeric-memory measurement.

## Provenance and limits

The library was built from local revision
`6bf930be60b688fe5917d8fb03464cb44c37be58`, tree
`5783753bfad4cbef30d318df1e14a22bef8b8a43`. The measured checkout was
`154e3207d08207b950a93db7e4064a95098270fa`, tree
`2a2b0789c2b0d3e6d520097ca7061cb9b0568352`. Their production paths
`src/include/python/cmake/CMakeLists.txt` were verified equal; later test/note
changes are not claimed as a library rebuild. The measured binary SHA-256 is
`82856b490be279ecde772d2feb6684953271246abac94c5f3748fe945f922c24`.

`native-build-provenance.json` records CPU RelWithDebInfo with `-O2 -g0 -DNDEBUG`,
GCC 14.2.0, ccache launchers and corrected OpenBLAS/LAPACKE detection. A partial
build was interrupted before probe correction, and shared cache counts are
not isolated build attribution. The corrected build interval is not a complete
cold-compilation cost. No objects were copied from the old prototype build.

Selected linked regression receipts are summarized separately: 7 native suites,
62 new-API Python tests and 67 existing/default-path Python tests, no skips in
those Python suites. These are regression counts, not an independent oracle
for every benchmark endpoint. The supplied manifest separately reports 39
host/source tests; that count is labeled manifest-only here.

The observed host is Intel Xeon Platinum 8573C with nine visible/allowed CPUs;
thread limits were one and affinity was observed, not pinned or exclusive.
Only three repeats are available. Compilation cost, owner-allocation peak and
an independent full-matrix oracle are missing. Native paired matches do not
establish the global ground state or unrestricted stability. No GPU, forces,
large-molecule win or production/default promotion follows from this campaign.

## Reproduction from explicit reviewed source snapshots

`run.py` and `inputs.json` remain byte-exact historical files. The new launcher
verifies every publication-bound checksum before reading the source allowlist
or loading code. `reproduction-sources.json` admits only full exact aggregate
production-source hashes; undeclared source changes fail closed.

The mappings distinguish:

- `frozen-measured-2026-10-02`: exact historical production source, available
  publicly at [f9551220](https://github.com/jinzhezenggroup/generativeqc/commit/f955122039e0f90d5adb11c921396a90a0b3f0e7).
  This public tree equals the original native build tree
- `cuda-integrated-2026-10-02`: the 16 listed CUDA-related integration changes,
  available at [04cef391](https://github.com/jinzhezenggroup/generativeqc/commit/04cef391cbe2d2ca13be60f9b336f9704b891301).
  It is an untimed follow-up, not the measured source
- `planner-abi-follow-up-2026-10-02`: subsequent CPU Python LDA g-shell
  decline/resource-planning and portable C option-tag-width repairs. Both
  source changes and exact before/after hashes are listed. These semantic/ABI
  changes are explicitly untimed; this is not pure-CUDA or CPU-byte equivalence

All executions, including exact-source replays, are labeled **NEW reproduction**.
The existing 18 raw rows, solver history, numerical gates and source/library
receipts are unchanged. The original `provenance.json` retains its historical
launcher-policy description; the new checksum-bound mapping is the current
reproduction admission policy.

Native binary provenance is separate from the checkout's Python/header source.
Known `82856b49...` retains its original build receipt. Known `062fde24...`
retains its actual `c49e93cf...` integration-build receipt even when used with
the reviewed planner/ABI repair. That binary was not rebuilt after those
repairs. Explicit compatibility covers the passed focused/API and normal/
`-fshort-enums` client regressions; it does not create new timing evidence.
Unknown rebuilt library hashes are allowed and recorded as unknown-build or
caller-declared provenance, never equated with either known binary. An optional
`--build-source-revision` is a caller declaration, not a compiler attestation.
No old local commit must exist merely to read a known byte-bound receipt.

For the exact frozen experiment, use a checkout of the public `f9551220...`
revision and a compatible CPU library. For a reviewed current-source experiment,
use the repaired checkout and its explicit compatible or rebuilt library:

```bash
python benchmarks/results/preliminary-scf-api/reproduce.py \
  --source-root . --library build/cpu-review/libgenerativeqc.so \
  --output .artifacts/preliminary-scf-api-new-run
# This is a read-only mapped dry-run. It does not load the native library.
# Add --run to produce NEW measurements in a new scratch directory.
# Optionally require --source-map frozen-measured-2026-10-02 for exact-source replay.
```

`new_campaign.py` owns only new-run orchestration and truthful receipts. It
reuses the frozen per-process worker, row acceptance function and standard
summarizer; no scientific solver or timing worker is copied. Every new run
records its actual source revision/tree/hash, actual library SHA-256, known or
unknown native build provenance, selected mapping, and `reproduction-context.json`.
The historical publication is never overwritten or relabeled. The three-repeat
performance decision remains non-promotable; no new benchmark was run to make
this reproduction repair.
