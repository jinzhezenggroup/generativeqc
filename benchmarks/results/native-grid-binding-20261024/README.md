# Native grid identity at consumer admission

Measured source `bdd14167ec9edd4c8813029fabfda3e03fbe0aaa` binds the current native
library once in its consuming grid, after resource admission. It retains complete
SHA256 hashing, immutable artifact identity, external expected-digest checks and
prepared geometry-refresh validation. It does not cache digests by stat/mtime.

Five alternating fresh-process pairs per method/mode use the same authenticated
source and native binary, with stock production defaults. The baseline runner
binds the native request in the caller, then passes the artifact through the
unchanged consumer expected-SHA check. This reproduces the prior two complete
reads inside the timed endpoint. It is a controlled before/after on one binary,
not a benchmark of a separately built older master. The candidate binds in its
consumer. Construction, preparation, SCF and analytic forces are timed together.

| Endpoint | Baseline cold (s) | Candidate cold (s) | Gain | Gate |
| --- | ---: | ---: | ---: | --- |
| PBE DF | 20.696223 | 20.082784 | 2.964% | pass |
| PBE Direct | 20.827612 | 20.289087 | 2.586% | pass |
| PBE0 DF | 32.380982 | 31.863541 | 1.598% | below 2% |
| PBE0 Direct | 24.111333 | 23.567553 | 2.255% | pass |

Warm changes are at most 0.20%. Each process contributes one median of its three
warm replays. The acceptance floor is `max(2%, 2*(relative baseline MAD + relative
candidate MAD))`. All 160 complete calls pass the existing independent energy and
full-force gates. Matched pairs have identical settings, iterations and Fock
counts. These results do not establish GPU4PySCF parity or a broad method/device
performance envelope. Separate combined-stack experiments are not pooled here.

The head also passes 442 host tests, independent PBE0/B3LYP prepared E/F with
same-geometry replay and displacement, unprepared PBE0 (energy error 4.12e-13 Eh,
maximum gradient error 5.62e-12 Eh/bohr), and memcheck with zero errors. The small
hybrid fixtures explicitly select the compatibility consumer used naturally by
water96. Actual binding observations and all original logs remain retained.

`receipts.json.xz` stores exact original paired JSON text, full vectors and work,
source manifest delta, precompile source verification, library/package hashes,
original scripts, sccache 0.16.0 statistics, device test receipts and host logs.
Real GPU work used finite `srun` allocations on node1/RTX 5090 with scheduler
visibility preserved. Run:

```bash
PYTHONPATH=python:. python benchmarks/results/native-grid-binding-20261024/reproduce.py
```

The reproducer rejects `-O`, authenticates the complete frozen source manifest,
checks the runner/binary identity, requalifies all 160 calls and recomputes the
paired statistics. Existing metric-GEMV receipts are gzip-compacted byte-for-byte to
fit the unchanged evidence caps; no historical scientific values are modified.

The separate `integration.json.xz` cohort authenticates source
`79f906eb4ce2f099e73294ed56a3e6db4d24c7ca` after integrating master `be28a867c`.
Its 24 complete calls cover PBE/PBE0/r2SCAN-3c, Standard/Fast and DF/Direct, with
one cold and one warm sample each. All independent full-vector gates pass;
prepared/unprepared hybrid E/F, geometry reuse/displacement and memcheck also
pass. The reproducer rechecks this cohort separately. These correctness smoke
samples do not replace or extend the earlier paired performance population.

The integration archive is 113,628 bytes with SHA256
`65bc8f0fc52eae06b47de86a6098a0fe0df84512934ca899aa337f268ed9c0d7`.
