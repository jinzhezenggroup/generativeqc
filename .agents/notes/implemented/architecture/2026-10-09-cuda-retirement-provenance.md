# Decision record: generated one-electron/DF CUDA retirement provenance

Status: implemented (retrospective evidence index)
Date: 2026-10-09
Historical work: [#231](https://github.com/jinzhezenggroup/generativeqc/issues/231), closed

## Problem and choice

Compiler-owned scientific equations initially coexisted with handwritten
CUDA implementations and method-specific contraction glue. Counting only
`.cu` lines could also confuse a native-to-C++ move or role reclassification
with actual removal of duplicate science.

The project retained generic CUDA runtime facilities, independent oracles
and measured exceptions, while promoting code-generated arithmetic only
after independent numerical, resource and complete-endpoint qualification.
The active rules now live in
[scientific CUDA ownership](../../../../docs/maintainer/cuda_ownership.md).

## Implemented migration history

Phase 1 (including PR #252) promoted a generated S/T/V one-electron value
path after a clean matched comparison and retired the redundant handwritten
value kernel/selector. Normal derivative/response recurrences still used
by other operators were not deleted merely to simplify the census.

Phase 2 promoted compiler-owned DF value, center-derivative and weighted
response paths with shared normalized AO/primitive traversal. Native
cuBLAS/cuSOLVER, metric factors, J/K equations, plan/resource ownership and
explicit nonreplaced scientific exceptions were retained. Generated
weighted response became the CUDA HF finalizer for the qualified domain;
the previous coordinate-wise raw/metric/source response implementation
and selector were removed.

## Source-matched evidence

The [one-electron benchmark bundle](../../../../benchmarks/results/cuda-ownership/README.md)
records a baseline and candidate checkout, full original per-case values,
20 cases/four phases, a five-repeat ABBA timing protocol and a per-workload
2% median non-regression gate. Its observed maximal median ratio was
1.019427; this was evidence of **non-regression**, not a universal speedup.

The [DF retirement bundle](../../../../benchmarks/results/cuda-ownership/df/README.md)
retains the second phase's baseline `4f36c6e`, immutable snapshots,
numerical gates, checked source hashes, semantic role changes and physical
line-count reconciliation. Against **that pinned baseline**, the historical
physical changes were scientific +23/-512 and runtime +243/-444; another
37 unchanged lines were reclassified. The reported net reduction of
690 maintained CUDA lines does not mean 690 scientific equations vanished.
Against the separate pre-phase-1 baseline, the total handwritten-source
and classification changes must be read from the corresponding report,
not estimated by adding those two different comparisons.

The stable semantic ledger is sharded, and the derived current report is
generated in CI rather than tracked; the contemporaneous decisions are in
[generated report](2026-09-19-generated-cuda-ownership-report.md) and
[sharded ownership](2026-09-20-cuda-ownership-shards.md).

## Invariants and rejected approaches

- A file named `reference` is not automatically an independent oracle.
- Native generic traversal and runtime operations are not scientific debt
  merely because they are CUDA.
- A failed, slower or unqualified generated candidate cannot replace the
  current production default by a source-count target alone.
- Role reclassification, non-CUDA C++ movement and physical deletion must
  be reported independently.
- A retired selector cannot be reconstructed on the current checkout;
  historical reproduction must use its pinned original version.
- Retained performance exceptions need explicit owner, domain, evidence
  and retirement conditions in the semantic ledger.

## Consequences and revisit criteria

Do not append milestone-by-milestone timings to the current maintenance
guide. The authoritative state is the checked live ledger and reproducible
report; the immutable historical campaign is indexed here. Revisit a
retained exception when an independently qualified generated replacement
meets the same complete science and endpoint conditions.
