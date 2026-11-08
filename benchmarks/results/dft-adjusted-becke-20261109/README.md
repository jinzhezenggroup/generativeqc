# Rejected radius-adjusted scaled-Becke experiment

This capsule is a **rejection**, not a performance/default qualification. It does
not change the qualified production runtime from PR #3051. Complete original
scientific records, including all forces, failed allocations, profile rows,
native-owner fixture source, build/restore receipts and actual measured test
overrides are retained byte-for-byte by the existing Exact FP64 decoder.
Full AOT compiler stdout alone stays in the original ignored archive, indexed
by byte size and SHA256. The repository's 85 MiB evidence cap is unchanged.

To stay under that cap, two existing scientific input NPZs use stdlib ZIP-LZMA,
as the existing water-octamer fixture already does. This saves 266242 bytes;
every original NPY member name/byte, numerical metadata and scientific identity
stays unchanged. `input-storage-receipt.json` and replay authenticate all members
and the new container hashes. Other exploratory recompression trials are not
applied. This is storage maintenance, not a new reference or GPU measurement.

Run from the repository root, without a GPU:

```bash
PYTHONPATH=python:. python benchmarks/results/dft-adjusted-becke-20261109/replay.py
PYTHONPATH=python:. python -O benchmarks/results/dft-adjusted-becke-20261109/replay.py
```

Both arms use experimental source `9d11d755` / ELF `f44e143a`. The control
explicitly disables scaled products; the candidate leaves the selector unset.
Neither arm is an execution of the restored source `75d94b32` / ELF `549ed9a2`.

Subsequent synchronization with master `8b9983820` changes the checkout source
identity to `8b1b284f`. `master8b-plan-audit.json` records unchanged explicit
schema-v2 build selections, not a newly linked image or GPU measurement. The
restoration identity in this historical capsule remains `75d94b32`; no old
timing is relabeled as a measurement of the later checkout.

| GPU | Direct96 complete cold | Control | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| A100 | PBE0 | 11.751200 s | 11.686492 s | -0.551% |
| A100 | B3LYP | 11.848846 s | 11.796362 s | -0.443% |
| RTX 5090 | PBE0 | 16.814291 s | 16.525521 s | -1.717% |
| RTX 5090 | B3LYP | 16.877596 s | 16.539189 s | -2.005% |

A100 does not clear its predeclared cold/noise improvement gate, even without
the 2% floor. All primary paired warms pass, but 5090 Direct48's single-pair
cold ratio is 1.03618. Its selection is unchanged, so this does not establish
causation or qualify nonregression. All present independent scientific and
strict native E/F/work gates pass.

**A100 job230464 exits 1:** its native tests and all 18 complete noninstrumented
populations finish, then missing `libcupti.so.12` prevents all four profiles.
These diagnostics remain absent, not zero-filled. 5090 job16868 completes
native tests, endpoints, four profiles and three sanitizers; failed test-only
name-shadowing job16867 is retained. The actual measured test overrides contain
36 cases per card, distinct from the restored production-policy tests.

Fresh same-allocation GPU4PySCF PBE0/B3LYP cold means are 11.064521/11.615196 s.
The experimental incumbent remains 0.686679/0.233650 s slower. Historical PBE0
reference anomalies have no established cause and are not rewritten. Neither
this capsule nor PR #3051 establishes all-README leadership. See issue #2976
and the rejected Agent Note for the decision and revisit conditions.
