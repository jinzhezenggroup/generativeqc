# Guarded shared FP32 Boys qualification

The production change lowers only normal mixed-precision Boys tables. It
reuses the shared range-moment recurrence, with a representability guard for
subnormal highest moments. FP64, derivatives, Fock accumulation and SCF
publication policies remain unchanged. There is no functional-name dispatch.

`summary.json.xz` contains compact independently audited receipts, not raw logs,
energies/forces, binaries or profiles. Historical failures stay in local
evidence. No Release, external backup or publishing workflow is involved.

## Isolated v16 paired performance

Each of three pairs per method runs baseline Fast, guarded Fast and guarded
strict on the same physical RTX5090, with fresh processes/calculators/hcore,
96 atoms/768 AOs, spherical def2-SVP and GridSpec(48,16,32). Timing includes
complete E+F, preparation, export and teardown, excluding build attestation.
Independent reference timings never enter a denominator.

| Method | Baseline Fast / guarded Fast, three pairs | Median |
| --- | --- | --- |
| PBE | 1.024647, 1.024135, 1.024786 | 1.024647 |
| PBE0 | 1.024906, 1.024542, 1.026851 | 1.024906 |
| WB97M-V | 1.050706, 1.051289, 1.055462 | 1.051289 |

All 27 returns pass the unchanged independent accuracy/precision gates.
Fock counts match within each pair. WB recurrence work differs by -5431
out of approximately 20.39 billion, so identical operator work is not
asserted. PBE/PBE0 precision work matches. Their conservative force-semantic
comparison remains false: an exhaustive audit finds provenance/artifact
hashes/keys/paths, stream handles and timers, but no arithmetic-counter
differences. Its exclusions were not widened to manufacture equality.

## Exact integrated v19 qualification

Tested head `17fc57d69` integrates master `8141cc7db`, including axis PR2569.
Its production source is `3f82094939da7d22ad91e99139d8716767686f36efb7f9addae4e0c7aa6e76ee`.
The compiled identity getter, core hash and six official force manifests
authenticate. The mathematical Boys/range headers match isolated v16.
This integration is not relabeled as a new incremental-speed experiment.

- Job3283: 157 native tests, zero skips, CUDA memcheck zero errors.
- CPU/CUDA-FMA/CUDA-no-FMA each pass all 420 scaled Coulomb seed cases.
- Job3284: six moved CUDA matrix tests and 32 family E/F returns pass.
- Job1083: all six clean RTX5090 PBE/PBE0/WB97M-V96 Fast/strict returns pass.
- The additional node2 96-atom attempt times out in WB's independent
  reference before any WB native sample; its incomplete receipt is retained.
  This is not a native Fast timeout. No repeat replaces it.

Fast E/maxF/RMS limits are `1e-5 /1e-4 /3e-5`; strict limits are
`1e-7 /1e-6 /3e-6`. Fast requires genuine mixed work, solve1/retry0; strict
recovery does not qualify as Fast. Node2's known NVML mismatch restricts its
evidence to numerical qualification, never performance.

## Reproduction boundary

Run real GPU tests through finite `srun` while preserving assigned device
visibility. `tests/python/test_mixed_boys_table.py` exercises the production
owner for CPU and opt-in CUDA (including no-FMA). The endpoint protocol is
the repository's `benchmarks/pbe0_precision.py` complete native E/F consumer,
using the independent oracle and grid/basis specified above. Source/core,
sample and independent-auditor SHA256 receipts in `summary.json.xz` identify the
retained local populations; they are not substitutes for rerunning a changed
source. See the implemented Agent Note for the rejected unguarded alternative.

The JSON summary is losslessly compressed to preserve the aggregate evidence
budget, including when this branch is merged with newer master evidence.
Read with Python's `lzma.open` or `xz -dc summary.json.xz`. The earlier axis
campaign's summary receives the same storage-only transformation; decompressed
bytes match Git blob `51352dd114b1aae7540c4f2b8201d117349161ad` exactly.
