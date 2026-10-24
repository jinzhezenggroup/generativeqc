# Observation: qualified range-separated DF after stable response

Status: implemented (bounded endpoint evidence)
Date: 2026-10-15

This supersedes the practical-auxiliary numerical limitation in
`2026-10-15-range-df-endpoint-observations.md`, without rewriting that negative
evidence. The inverse-root reverse decision is documented in
`../numerics/2026-10-15-stable-whitened-df-response.md`.

The final campaign uses the same cc-pVDZ-JKFIT basis, metric cutoff, SCF
tolerances and controlled small grid. At CPU water/TZVP, complete energy plus
forces changes from 15.729 to 3.258 seconds for warm replay (4.83x), from 40.407
to 7.168 seconds including preparation (5.64x), and from 40.044 to 6.197 seconds
for changed geometry (6.46x). Fixed-geometry energy replay is slightly slower;
the old Direct LR owner already caches its values there. SVP warm energy plus
forces improves 1.43x. These are complete endpoints, not kernel timings.

All 56 CPU measured samples pass independent reconverged libcint energy and
component/directional force gates. Warm force changes fall below 8e-10
Eh/bohr, replacing the old 1e-6 instability. A displaced coarse-grid component
also requires a resolved finite-difference step in **both** Hamiltonians:
1e-4 and 5e-5 agree near 1e-9, while 2e-4 does not. Retain that original failed
coarse-step reference and its convergence sweep rather than loosening the
unchanged 3e-7 force gate. This is distinct from the original LR response error.

CUDA passes 14 selected real-device endpoint/response tests and a zero-error
memcheck through finite Slurm jobs. Absolute fitted-candidate timings are
retained for the normal-optimized DF kernels in the numerical library. They
are not an optimized Direct comparison: two legacy Direct units in that
library are fast-compiled. Do not advertise historical preliminary ratios as
normally optimized GPU speedups. Production-grid, larger-system, constrained
memory, batch and whole-process memory qualification remain separate work.

The exact settings, raw receipts, source/library hashes, test receipts and
reproduction harnesses are preserved losslessly under
`benchmarks/results/range-separated-df-stable-20261015/`. Its verifier restores
the historical and final evidence into ignored local storage under the fixed
repository-wide evidence budget; no external archive or Release is published.
