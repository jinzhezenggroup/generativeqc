# Preliminary SCF: frozen prototype evidence

This is a compact, reproducible **exploratory CPU experiment**, not performance
qualification of the preliminary-SCF API introduced by the accompanying PR.
The performance decision is **inconclusive/non-promotable**. No production
selector or default is qualified by this publication.

## Scope and results

Five campaigns completed **224 energy endpoints, with zero failures in their
campaign records**. All smoke files, including the abandoned reversed-direction
`hf/` experiment, are excluded. Campaigns remain separate:

| Campaign | Routes | Endpoints | Repeats and thermal scope |
| --- | --- | ---: | --- |
| root | direct, same-grid LDA, same-grid PBE, xTB charges | 96 | 3; first and second fresh contexts per process |
| coarse | direct, coarse LDA, coarse PBE | 72 | 3; first and second fresh contexts per process |
| cluster | direct, coarse LDA | 6 | 3; first endpoint in fresh process |
| hf-preliminary | direct, HF, coarse LDA | 18 | 3; first endpoint in fresh process |
| hf-lda-preliminary | direct, HF, coarse LDA, HF→coarse LDA | 32 | 4 Latin-square rotations; first endpoint in fresh process |

The target is unchanged PBE0/RKS: 25% HF exchange, 75% PBE exchange and 100%
PBE correlation, FP64, the same basis and native 24×12×24 grid, with SCF
energy/density tolerances 1e-10/1e-8 and maximum 150 iterations. HF/LDA/PBE
preparations use 1e-6/1e-4 and the same basis; coarse LDA/PBE uses an 8×6×12
grid. The xTB route hands off atomic charges through the existing Löwdin mapper,
not a cross-basis density copy. See each captured protocol for full controls.
Exact geometry and basis numbers are in the retained source captures.

Final four-route, fresh-process medians (seconds, rounded only for this table):

| System / route | Full native endpoint | Preparation | Target iterations |
| --- | ---: | ---: | ---: |
| water/6-31G direct | 0.441981 | 0 | 12 |
| water/6-31G HF | 0.345205 | 0.007962 | 8 |
| water/6-31G coarse LDA | 0.350957 | 0.021740 | 8 |
| water/6-31G HF→coarse LDA | 0.372336 | 0.024886 | 8 |
| 12-atom/52-AO tetramer direct | 44.928923 | 0 | 18 |
| tetramer HF | 30.352931 | 1.602693 | 11 |
| tetramer coarse LDA | 31.957829 | 2.558308 | 11 |
| tetramer HF→coarse LDA | 32.676477 | 3.878219 | 11 |

These descriptive medians do not establish a general winner. Preserve the
negative findings: same-grid PBE reduces target iterations but its preparation
makes the total endpoint slower in all eight root case/thermal comparisons
(1.462–1.858× direct). Adding coarse LDA after HF reduces the LDA-stage work,
but does not reduce final PBE0 iterations versus either single preparation; its
extra preparation makes the triple route slower in these four-repeat medians.
Other routes/cases and every original timing are retained, not selected away.

All native candidate/direct pairs satisfy energy ≤1e-8 Eh, density max error
≤1e-6 and native reported residual ≤1e-8. The actual largest pair differences
are 2.0463630789890885e-12 Eh and 7.122084033639453e-10 in total AO density.
These checks establish the same restricted solution branch as direct SCF, not
unrestricted stability or the global ground state. One independent
PySCF 2.14.0/Libxc 7.0.0 water/6-31G spot-check uses exactly the native grid;
its energy difference is 1.7053025658242404e-13 Eh. This is not an independent
oracle for the whole matrix. `oracle-result.json` retains its full scope.

## Boundaries and missing evidence

- Endpoint `total_s` includes system/basis normalization, plan/grid construction,
  seed preparation and target SCF. It excludes process launch, JSON output and
  destruction after the timer stops. Full `process_wall_s` is also retained.
  A process containing two endpoints has its process wall time on both rows;
  do not sum those duplicate process observations as independent timings.
- Process-warm means the second call creates a fresh context, grids and plans.
  It is not reuse of a converged density or a prepared plan.
- O/H restricted singlets only, 2–12 atoms and 7–52 Cartesian AOs. No GPU,
  force, open-shell, large-molecule, transition-metal or ECP claim.
- CPU model, peak/allocated memory and compilation duration were not measured.
  There was no CPU-affinity isolation. Do not infer historical hardware from
  the machine running this publisher or a later reproduction.
- Three/four repeats are below the standard five-pair comparison gate, and
  these multi-route schedules are not relabeled as a standard A/B experiment.
  No complete per-iteration solver trajectory or whole-matrix independent
  residual/oracle exists. Native residuals and all semantic work counts remain.

## Source provenance

The measured clean local revision is
`8ba615aacd9f02708b39782cc71b68f27bbe1af6`, tree
`42cc3f8f871523a4d48304f520a5e967e61cea6e`. Public revision
[`3b04ecbda28dcecab6fb9db3ca2b4cdf10ba99bc`](https://github.com/jinzhezenggroup/generativeqc/commit/3b04ecbda28dcecab6fb9db3ca2b4cdf10ba99bc)
has exactly that tree ([PR 1683](https://github.com/jinzhezenggroup/generativeqc/pull/1683)).
The public mapping does not rewrite the measured revision or claim the
prototype measured the new API. The CPU RelWithDebInfo library SHA-256 is
`670c81365bde477f32e5a7fc6af1cff5e489a42e118651aca77090fd00626258`.
`integration-manifest.json` records audited incremental source integration;
this was not a clean/full build. Its absolute paths are historical provenance,
not reproduction dependencies. Compiler/BLAS versions and exact bridge compile
flags were not recorded. The portable runner records its actual new compiler,
command and binary identity instead of stamping the historical ones.

`source-comparison.json` compares six selected implementation files against an
already-fetched master snapshot. It is not a latest-live-master claim.

## Retained data and reproduction

`evidence.json` uses the existing `generativeqc.validation` envelope, with
checksum-checked `record_parts` for all 224 samples in their original order.
Each sample retains every original scalar, timing, residual and work count.
Only repeated density arrays are replaced by canonical-JSON SHA-256, shape,
paired energy/density error blocks and a direct-baseline row index. Errors
and gates were computed from full raw arrays, never rounded summaries.
`summaries/` contains verified descriptive projections; the historical
`qualified_matched_state` label is clarified as `native_matched_state_only`.
No timing/failure samples were removed. Full original raw records remain
unchanged in the measurement workspace; their hashes are in `provenance.json`.

`inputs/*.json` captures the exact UTF-8 contents of `staged.cpp`, `basis.inc`
and original orchestration scripts, including missing final newlines, plus
parsed historical protocols. Source text is encoded to prevent formatting
hooks from silently altering historical sources. `reproduce.py` verifies
source hashes and restores original filenames. Historical `run.py` is for
audit only: it embeds measured-run provenance and must not be used for a new
run. The oracle grid is regenerated by the captured `oracle_grid.cpp`; its
historical exact-file hash is retained, rather than a 1.7 MB repeated grid dump.

Read-only validation (requires the repository's existing NumPy environment):

```bash
python benchmarks/results/preliminary-scf-prototype/reproduce.py --verify-only
```

For a new experiment, use a separate clean checkout at the public-equivalent
revision and an already built CPU RelWithDebInfo library. Build that library
using the repository instructions and verified CXX/CUDA `ccache` launchers;
do not reuse headers from the new API branch with an old library. A new library
may have a different binary hash; the runner records that fact and does not
claim bitwise identity with the historical binary. No installation, source
fetch, library build or scientific solve is performed by `--verify-only` or
`--dry-run`.

```bash
python benchmarks/results/preliminary-scf-prototype/reproduce.py \
  --source-root /path/to/clean-public-source \
  --library /path/to/clean-public-source/build/cpu/libgenerativeqc.so \
  --output .artifacts/preliminary-scf-reproduction --dry-run
# Remove --dry-run to compile the small bridges and execute all five campaigns.
# Add --campaign hf-lda-preliminary for only the final four-route experiment.
# Add --oracle to rerun the root water spot-check in an existing PySCF environment.
```

The runner requires `ccache` (or `--ccache /path/to/ccache`) and a C++20 compiler.
It preserves measured process ordering and thermal behavior, exports one-thread
limits, and writes raw endpoint densities and failures to a new scratch output.
It cannot overwrite this publication or write raw data into `benchmarks/results`.
The portable bridge command uses `-O2`; that is a reproduction choice, not an
invented record of historical bridge flags. Reproduction compile/execution
was not rerun during packaging while another build was active.

The repository's `tools/evidence.py publish` validates the standard envelope
and creates `publication.json`. It writes local files only. Scientific
promotion remains blocked by the explicitly unavailable evidence above.
