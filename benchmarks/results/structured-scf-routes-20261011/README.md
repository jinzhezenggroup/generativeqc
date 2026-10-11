# Borrowed structured SCF primitive qualification

This compact [receipt](qualification.json) qualifies primitive arithmetic,
resource facts, masks and replay for scientific source commit
`12c96bfde81f79b4fd6aa0185826a9cfb5a37ba6`. It is **not** complete scientific
endpoint or default-route qualification, and closes none of #1877/#1878/#1883.
Subsequent receipt-only changes do not change the qualified scientific source.

The independent diagnostic driver checks generated and cuBLAS routes against
long-double coordinate loops over 1,680 scenarios, twice per scenario. Captured
graphs replay with changed data/occupations/masks at the same device addresses.
Cases cover positive and signed Gram, inverse-root reconstruction, rectangular
basis stages in both shape directions, spin broadcast, sparse shell spans,
hcore publication, poisoned virtual orbitals, invalid eigenvalues, inactive
byte preservation and library scratch alias/capacity rejection. Memcheck and
initcheck each report zero errors. The receipt records the numerical acceptance
threshold; no endpoint timings or speedups are inferred.

Reproduce from the qualified commit on an allowed GPU node using a fresh output:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:10:00 \
  bash tools/qualify_structured_scf_routes.sh \
  "$PWD/.artifacts/structured-reproduction" /group/software/cuda-12.9.1 \
  120 /home/jzzeng/miniconda3/bin/python
```

The script requires verified ccache and clean committed source, records build
commands/dependency files, generated-header/object/binary hashes, compiler/cache
versions and statistics, loaded libraries, source provenance and Slurm device
visibility. `build-only` performs compilation without device execution;
`run-only` checks the recorded source identity before reusing the binary.
Run both `compute-sanitizer --tool memcheck --error-exitcode 99` and
`compute-sanitizer --tool initcheck --error-exitcode 99` against the resulting
driver inside a finite Slurm allocation. Do not alter `CUDA_VISIBLE_DEVICES`.

Raw objects, binary, full dependency/command/device logs and sanitizer receipts
remain in the ignored local directory named by the compact receipt. No archive,
Release, release asset or external backup is published for this qualification.
The [current contract](../../../docs/developer/lowering_providers.md#borrowed-structured-scf-regions)
and [decision note](../../../.agents/notes/implemented/architecture/2026-10-11-borrowed-structured-scf-regions.md)
describe ownership and the remaining default-promotion gates.
