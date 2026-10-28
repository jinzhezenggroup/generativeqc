# Shared correlated convergence policy

Scoped CPU R4/R6 evidence for #2227. Python CBS planning and CC admission query
the existing `core::CorrelationConvergence` policy through an additive C API.
CBS coefficient/error allocation, explicit tighter settings, its additional
canonical-frame/residual caps, and all solver equations remain unchanged.
This does not complete native CBS composition, general policy ownership, R3,
resource authority, or CUDA qualification.

Baseline: `78e0c8afab504bbdf7ecd73ede2e4179195eaf49`. The candidate is reconstructed
from that revision plus `source.patch.xz`; `provenance.json` records verified
reconstruction of all six production paths and their actual hashes. The baseline
and candidate have separately built Release libraries. Both use GCC 11.4.0,
CUDA OFF, CPU stationary force AOT ON and verified sccache 0.16.0. A generated
compiler command demonstrates the launcher; before/after shared-server statistics
are observations, not isolated cache-speed measurements. The scientific policy
header is byte-identical to the baseline.

## Complete endpoint comparison

`benchmark.py` compares separate baseline/candidate processes in both ABBA and
BAAB order, with three warmups and 21 samples per block: **672 timed endpoints**.
Each endpoint includes facade construction, both native component groups,
energy/full analytic forces, and ordinary cleanup. Imports and process startup
are excluded. One OpenMP/OpenBLAS thread is requested on a shared CPU host.
Every endpoint prepares fresh state and alternates H2 bond lengths 1.40/1.44 Bohr.

The distinct low/high basis records are the pinned `water-algebra-*.json`
fixtures, restricted to their H records (two AOs per H2). This tests complete
CBS composition/solver/policy ownership; it is not a physical CBS limit or a
performance claim for water, def2-SVP, large bases or CUDA. Standard, Tight,
custom budget without RMS, and explicitly tighter settings are all included.

Full energies/forces, selected ceilings and the available native iteration,
CC/response/residual/resource/work receipts match exactly across corresponding
samples. Unavailable counters retain their native validity flags. Candidate
endpoints forbid both Python policy compatibility functions.

| Case | Order | Candidate/baseline median |
| --- | --- | ---: |
| standard | ABBA | 0.974444 |
| standard | BAAB | 0.977266 |
| tight | ABBA | 0.975146 |
| tight | BAAB | 0.973064 |
| custom | ABBA | 0.972542 |
| custom | BAAB | 0.983252 |
| explicit | ABBA | 0.983759 |
| explicit | BAAB | 0.973929 |

All eight ratios pass the unchanged 1.05 gate. Raw timings, signatures and
relative median absolute deviations are retained in `samples.json.xz`. The
result is scoped non-regression evidence, not an exclusive-host speedup claim.
An exploratory def2-SVP timing smoke was stopped without a completed comparison;
it is explicitly incomplete in provenance and supplies no acceptance evidence.

## Correctness and reproduction

- 127/127 post-build CTests.
- 206 Python passes plus 25 subtests; 3 CPU-build CUDA skips and 23 explicit
  CUDA deselections. Existing pinned PySCF water CBS/component E/F checks pass
  `1e-5 Eh`, `1e-4 Eh/Bohr` max and `3e-5 Eh/Bohr` RMS aggregate gates, with
  the stricter coefficient-scaled allowances applied independently to components.
- Public SDK boundary/atomicity/overlap checks run against the installed library
  under `env -i PATH=/nonexistent`, without Python or compiler runtime search.
- 600 random budgets and bracket neighbors match legacy decisions; concurrent
  queries, native-only CBS/Calculator consumers, selected-library isolation,
  missing-protocol compatibility, malformed receipts and native failures pass.
- ty 0.0.82, Sphinx warnings-as-errors and applicable hooks pass.

An early CTest run overlapped CLI probe relinking and failed two probe cases;
the final complete 127-test run started after the build finished and passed.

Build baseline and reconstructed candidate independently with the settings in
provenance, then run:

```sh
python benchmarks/results/correlated-policy-20261028/benchmark.py \
  --baseline /absolute/baseline-checkout \
  --candidate /absolute/candidate-checkout \
  --output /absolute/ignored-directory/samples.json
```

Each checkout supplies its own `build/libgenerativeqc.so`. The runner fails on
any scientific/work difference or median ratio above 1.05. Read compressed JSON
with `json.loads(lzma.decompress(path.read_bytes()))`. These historical source
hashes do not impose permanent source-spelling tests on future refactors.
