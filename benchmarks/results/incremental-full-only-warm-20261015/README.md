# Full-only warm HF: compact diagnostic record

Base: `4e28477e96d257528a60bcb68c00529f22585bc1` plus the measured patch;
exact source/library/patch/harness hashes are in `diagnostic.json.xz`.
Release/sm120 CUDA 12.9.1, generic/non-AOT, not production/default qualification.

Complete RHF/def2-SVP spherical FP64 energy-plus-force median seconds, six
balanced interleaved repeats per arm on one Slurm-assigned RTX 5090:

| AOs | Phase | OFF | fixed8 | Change |
| ---: | --- | ---: | ---: | ---: |
| 96 | cold | 1.447577 | 1.274201 | -11.98% |
| 96 | warm | 0.187664 | 0.186917 | -0.40% |
| 96 | moved | 1.151983 | 0.963352 | -16.37% |
| 192 | cold | 13.556668 | 11.716461 | -13.57% |
| 192 | warm | 1.954380 | 1.959842 | +0.28% |
| 192 | moved | 11.474454 | 8.561851 | -25.38% |

Both warm arms have one full SCF and zero final builds; fixed8 has zero deltas.
Warm is baseline parity, not a significant speedup over OFF. Cold includes
construction, preparation, initialization, SCF, finalization and host E/F,
excluding common global bootstrap; teardown is separate. All 72 formal and
12 priming endpoints pass independent CPU PySCF gates. Maximum E/F errors:
8.413e-12 Eh / 9.325e-11 Eh/Bohr. Bounded-streaming 192-AO quartet work is unknown.

The 384-AO n4 moved priming fails convergence at 100 returned iterations plus
100 discarded target iterations, with no forces. The remaining cohort is
cancelled: no 384-AO speedup claim. All partial/failed points remain in the
record. No default promotion or complete HF synccheck acceptance is claimed.

The record retains every endpoint's timing, work, gates/errors, provenance,
reference hashes and full-bundle inventory. The repository's aggregate
evidence budget prevents retaining the full bundle here: exact samples with
force arrays, references, reconstruction patch, frozen harness, runner and
validation outputs remain in ignored local
`.artifacts/delta-anchor/warm-full-reviewed-bundle`. That is not a permanent
remote archive or a standalone reproducibility guarantee. No old evidence or
budget policy is changed. Full methodology, gates, failures and fallback
rationale: [Agent Note](../../../.agents/notes/implemented/performance/2026-10-15-incremental-full-only-warm-finalization.md).
