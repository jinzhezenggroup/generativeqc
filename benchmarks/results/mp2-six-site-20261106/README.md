# Six MP2 materialization dispositions

This is measured qualification for #1626, not a source-fingerprint CI gate or a
claim that every tensor can be fused. All six remaining MP2/oracle sites have
final retention decisions in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-11-06-mp2-six-site-dispositions.md).
The production change remains the occupied-support second-stage compaction;
none of the disposable alternatives silently replaces production with an oracle.

## Inputs and controls

- AMD EPYC 7K62, affinity CPU 0, one OMP/OpenBLAS thread, Release CPU build with
  packaged stationary-force AOT enabled. Compiler commands use verified sccache
  0.16.0; before/after cache statistics are retained. Cache state is not a timing
  performance claim.
- H2/LiH/water complete cold/warm MP2 E+F with native STO-3G primitives. Independent
  PySCF/libcint reconstructs exactly those primitives, solves RHF/MP2 and computes
  analytic gradients; differently rounded built-in basis tables are not used.
- Six component variants: production retention, lazy first, lazy second, lazy
  third, reusable local view, and matched experimental-helper retention.
  Nine separately relinked endpoint libraries also include dense oracle,
  dense-oracle helper retention, and dense-oracle lazy first.
- Two ABBA repeats per comparison: 64 fresh-process histories / 384 timed
  complete E+F calls. Preparation plus first E+F and warm E+F are both recorded.
  Each alternative is compared with its matching helper, not credited with the
  helper's independent loop-reorganization difference.
- Separately instrumented fresh-process Memray 1.20.0 histories add 54 complete
  E+F calls, not timing samples. Their tracked native-heap peaks include imports,
  constructors, preparation and all three molecular cold/warm histories.

All **438** complete calls pass unchanged independent gates: maximum energy
error **4.94e-10 < 1e-9 Eh**, maximum full force error
**2.33e-8 < 2e-6 Eh/Bohr**, and force translation sum <2e-8. The direct tensor
component oracle checks 384 invocations (8 cases × 4 representations × 2 sparse
populations × 6 variants); independent physical libcint derivatives check 288
comparisons (8 × 3 × 2 × 6). Callback cleanup and the production staging bound
remain enforced in the experimental harness; retention-specific allocation and
operation equalities are replaced by actual measurement, not false requirements
that a fusion have exactly retention's work counts.

## Measured decisions

Example shell payload peak, N=5/O=2, fully populated input:

| Alternative | Dense / factorized peak bytes | Relevant work / allocations |
| --- | ---: | --- |
| Retain; matched helper-retain | 6,528 / 3,360 | identical work and 125 allocations |
| Lazy first | 3,528 / 2,448 | populated first contributions ×5 |
| Lazy second | 4,728 / 2,640 | dense second accumulations 3,125→15,625 |
| Lazy third | 5,448 / 2,280 | dense/factorized third accumulations ×5 |
| Reused local view | 6,528 / 3,360 | unchanged work; allocations 125→47 |

Matched water complete cold E+F seconds (control→alternative):
first **0.550464→0.550833**, second **0.551643→0.553961**, third
**0.550623→0.553906**, local view **0.548524→0.547091**.
Small warm differences do not establish broadly profitable recomputation or
justify its work amplification. Dense-first versus its dense-helper control is
**0.045583→0.045398 s** warm, not a general dense-first fusion win.

The bounded dense oracle is faster on these tiny examples: water warm
**0.157246→0.045584 s**, cold **0.554954→0.443321 s**. This does not authorize
rank-four oracle storage or CPU/reference execution in a bounded production
contract. Its pre-allocation/overflow gate remains mandatory. All five
production stages and the budget-gated oracle therefore retain their current
ownership/storage contracts, with explicit revisit conditions in the note.

Every matched variant's tracked process-heap peak is **55,057,817 bytes**,
dominated by imports. No endpoint-memory reduction is claimed. The exact
simultaneous shell allocation peaks above measure a different, narrower scope;
neither measurement is RSS or GPU storage.

An earlier local-view experiment exceeded the existing staging bound because
vector growth briefly retained old and new capacities. Its source/error are
retained. The qualified alternative reserves the checked maximum once. The
earlier unmatched experiment is retained separately and is not used to attribute
the helper's common timing difference to each storage decision.

## Artifacts and reproduction

`qualification.json.xz` contains every matched timing median, exact independent
oracle inputs/results, full work/storage example rows and heap summaries.
`component.json.xz` retains every direct tensor/work/allocation record;
`samples.json.xz` retains every timed complete-endpoint value and compiler-rule
descriptor; `independent-oracle.json.xz` retains exact independent inputs/results
and all timed/memory endpoint comparison identities. Full generated experimental
C++, physical derivative inputs, compiler/link commands, cache statistics,
rejected/unmatched runs and Memray aggregated high-water allocation records stay
in ignored local `.artifacts/issue-1626-1629/materialization-matched/` and
`.artifacts/issue-1626-1629/evidence-before-compaction/`, with recovery identities
in the compact qualification. This preserves reproducible scientific samples
without treating full run bundles as accepted benchmark publications. Full
native-stack pilot streams remain in ignored local
`.artifacts/issue-1626-1629/materialization-memory-native-stacks/`
and are not mixed into the final matched memory comparison. No executable,
object file or shared library is distributed. The shared native source/build
provenance is in the sibling
[scheduled evidence](../residency-qualified-n4-20261106/README.md).
`checksums.sha256` binds the retained artifact bytes, not current production
source spelling. The unchanged 82 MiB aggregate, 1 MiB per-file and 2 MiB
incoming-change guards remain enforced. Live scheduled replay inputs have an
explicit lossless migration to permanent per-test reference fixtures, documented
in `manifests/evidence_migrations/mp2-residency-20261106.json`.

From a verified cached Release CPU build with stationary force AOT enabled:

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/experiments/mp2_materialization/qualify.py \
  --build build-cpu --output .artifacts/mp2-six-site-new
PYTHONPATH=python:. GENERATIVEQC_LIBRARY="$PWD/build-cpu/libgenerativeqc.so" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/experiments/mp2_materialization/oracle.py \
  --observations .artifacts/mp2-six-site-new --output .artifacts/mp2-six-site-oracle.json
```

For each generated library, separately run `memory.py OUTPUT.json` with
`GENERATIVEQC_LIBRARY` selecting that library in an isolated Memray environment.
Then rerun the oracle over both timed and memory endpoint JSONs. Required CI
continues testing actual production numerical/resource contracts, not generated
source hashes or exact implementation call counts.
