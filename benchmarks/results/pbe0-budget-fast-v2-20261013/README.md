# Public budget-selected FP32 gradients: fair PBE0 comparison

Fast now uses the qualified mixed stationary gradient through the public
`PrecisionCalculator` interface. This is not a private-kernel speedup or an
all-FP32 claim. Ordinary FP64 receives the same exact native-executor improvement
through 32 atoms. Above that boundary its faster existing phased force executor
is retained; forcing a slower native FP64 comparator would inflate the ratio.

## Complete public endpoints

Slurm **8197**, node1 RTX 5090, assigned visibility `1`, CUDA 12.9.1,
Release/sm_120, fast-compile disabled. Exact non-DF PBE0/RKS, spherical def2-SVP,
moving grid `(48,16,32)`, energy/density/screening `1e-10/1e-8/1e-12`, 180 maximum
iterations. Each row has one retained first-call pair and three alternating
measured pairs. Timings include fresh construction, unchanged seed policy,
SCF/refinement, complete analytic forces, host export, release and synchronization.
Only process caches are warm; no converged density is reused between endpoints.

| Seed | System | AOs | Optimized FP64 (s) | Fast (s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Hcore | Water | 24 | 0.599156 | 0.487177 | 1.2299× |
| Hcore | Ammonia | 29 | 0.803390 | 0.581250 | 1.3822× |
| Hcore | Water tetramer | 96 | 2.130445 | 1.880907 | 1.1327× |
| Hcore | Water octamer S4 | 192 | 7.188223 | 6.143476 | 1.1701× |
| Hcore | Water hexadecamer | 384 | 24.318866 | 22.033182 | 1.1037× |
| Automatic MINAO | Water | 24 | 0.515988 | 0.406743 | 1.2686× |
| Automatic MINAO | Water octamer S4 | 192 | 6.586288 | 5.767102 | 1.1420× |

These are descriptive medians, not confidence bounds or a universal speed
guarantee. The larger system does **not** show a larger endpoint gain. All 56
complete endpoints, including first calls, pass the independent oracle gates.

## Numerical and arithmetic contract

Independent CPU PySCF SCF and full moving-grid gradients use exactly the same
Hamiltonian, basis, representation and quadrature; they never supply native
production densities, integrals or forces. Energy acceptance is `1e-7 Eh`;
maximum force-component acceptance is `1e-6 Eh/bohr` for strict and `1e-5` for
mixed, with mixed RMS `3e-6`. Largest public errors are approximately
`1.069e-11 Eh`, `1.241e-6 Eh/bohr` maximum and `3.853e-7 Eh/bohr` RMS.

Every timed Fast call performs one solve and one force evaluation, reports
`qualified_mixed_gradient`, positive FP32 grid/geometry work, and zero full
gradient retries. This is independent qualification, **not a per-call observable
error certificate**. Explicit `TargetAccuracy` still requests a complete FP64
audit and verified fallback, including its extra endpoint cost.

FP32 includes weighted ERI derivative orders 0–5 and primitive sums, gradient AO
jets, density projection/features, interior PBE exchange **and** correlation,
AO pullbacks and point-local Becke sums. Global sums, normalization, Boys,
one-electron/Pulay, higher derivative orders and unsafe tails retain FP64.
Exact SCF J/K, eigensolution, DIIS and final convergence remain FP64. SCF XC
warmup remains limited to 24–256 AOs; the 384-AO gain is gradient-only, not an
approximate SCF trajectory. No TF32/FP16 route is claimed.

Hcore mixed/strict SCF-stage Fock counters are respectively `9+5`, `10+4`,
`10+6`, `10+9`, and `0+22`; ordinary FP64 reports `0+13`, `0+14`, `0+14`,
`0+17`, and `0+22`. Thus extra refinement is included rather than selecting
samples with favorable iteration counts. Full precision timelines, final
audits, solve records and reported operator counts remain in the raw files.

## Isolated gradient diagnostics

Separate same-controller, unchanged-FP64-SCF probes in job 8197 show gradient
speed ratios **1.686×** at 192 AOs and **2.157×** at 384 AOs. Each has just one
measured pair; these are not the public speedups above. Instrumented replay is
excluded from timing. It records 5,961,644 / 23,141,636 mixed weighted tasks,
91,654,664 / 352,656,912 primitive contractions, 589,824 / 1,179,648 mixed grid
and geometry points, and zero weighted-task or whole-gradient retries.
Semantic pair visits, pair-state evaluations, AO values and projection FMA work
are retained in `summary.json.xz`. Clean public calls deliberately do not collect
intrusive per-task retry counters: unknown is not zero.

## Rejected preliminary comparison and fixes

The earlier public comparison against the old Python strict force executor gave
3–4× small-system ratios, mostly exact executor overhead. It is retained under
`before-common-executor/`, **not** presented as FP32 speedup. The common strict
native executor removes that misleading advantage. For 48 atoms, native inline
strict FP64 remained slower than the incumbent phased strict path, so that
promotion was rejected rather than making FP64 slower.

The first public integration also reserved only strict cached capacity, leaving
zero headroom for optional float grid panels/provider. Tests correctly rejected
its zero mixed-grid count. A shared checked storage formula now charges both
panels and the second contraction provider in the cached tier; the unchanged
strict minimum still runs without that optional storage. Failed logs remain.

## Provenance and validation

- Base: `da7185262dd77c67d16dd72614b96a30e315bebf` plus the retained source archive.
- Native source identity: `ef562e63496eb346d3587cd495a9813beb629549612cbf8fa2dcd07de80e5ebe`.
- Library SHA-256: `5330d80cc4284ca982831c2aac0b19203693a1b9d8af3ca7c9a80c344e8abb6a`.
- Public/native policy tests: **33 passed**, job 8197; strict-sized resource
  fallback: **1 passed**, job 8206.
- Memcheck: **5 passed, zero errors**, job 8198, on the identical library.
- Final focused host contracts: **657 passed**; compiler dependency audit:
  **537 modules, zero errors**; Sphinx warnings-as-errors passes.

`summary.json.xz` retains all measured times, first-call times, work counters,
protocol identities, maximum errors, device/build metadata and raw-file hashes.
Full protocols, geometries, references, forces and diagnostics are in the ignored
`.artifacts/pbe0-budget-fast-v2-20261013/` campaign and
`n1:/data/jzzeng/pbe0-budget-fast-v2-20261013/`; nothing is published as a Release.
Changed production inputs were checked byte-for-byte against the built tree.
ccache 4.5.1 was verified and reused; before/after stats and failed builds/tests
are retained, not cleared or silently overwritten.

Reproduce public timings with `benchmarks.pbe0_fast --reference <reference.json>
--output <new.json> --repeats 3 --cases <case...>` inside a finite Slurm allocation,
preserving assigned device visibility. The raw reference and source identities
must match; private `benchmarks.pbe0_mixed_gradient` ratios are not substitutes
for this public endpoint.
