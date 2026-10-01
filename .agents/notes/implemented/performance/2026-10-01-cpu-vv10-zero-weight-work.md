# Decision: omit exact-zero CPU VV10 work only for weighted molecular potentials

Status: implemented; measured endpoint qualification below
Date: 2026-10-01

## Problem

The molecular VV10 domain already replaces density-screened tails with benign
features and exactly zero effective quadrature weights. CPU execution nevertheless
called the pair kernel for every row/partner, including the zero contributions.
On a water/STO-3G 3,456-point energy endpoint, a timing interposer attributed about
80% of elapsed time to `Vv10Plan::execute`. Across its eleven calls, 131,383,296
pairs were evaluated; only 76,107,148 had two nonzero effective weights.

## Decision

The existing molecular E/V integration explicitly opts into a weighted-potential
contract. Under a conservative finite-arithmetic envelope, skip exactly-zero
weight rows and partners. Keep every nonzero partner in ascending original order,
including negative and subnormal weights. Use the original effective weight as
the predicate, never `weight * density`, which can underflow for an active point.
No density threshold, quadrature, functional algebra, precision, SCF setting,
AO/integral operation, or pair formula changes.

The outer loop still traverses N rows; each active row still traverses N partner
indices with cheap weight checks. The expensive pair kernel executes A² times
for A nonzero effective weights, rather than N². This is not a compacted index
schedule and does not claim A² total loop iterations. No index array, cache,
allocation or geometry-dependent resident state is introduced.

The feature arrays on zero-weight rows may be zero only because this consumer
uses `weight * vrho/vsigma` to build the AO potential. Raw fixed-grid calls retain
all feature derivatives, including nonzero derivatives at zero quadrature weight.
Geometry/weight-derivative requests, energy-only calls, rVV10, CUDA execution,
and out-of-envelope inputs retain their original route. The public C API and
resource admission are unchanged. One uint64 metadata counter records successful
CPU pair calls; it is zero before execution, after failure and on CUDA execution.
`resources().pair_evaluations` remains the N² capacity bound, not measured work.

## Failure-preserving envelope

Validate every input and calculate/validate every local scale before masking.
Enable the optimization only with all of:

- |coordinate| <= 1e6 and density <= 1e6
- omega and kappa in [1e-12, 1e12]
- |local feature derivative| <= 1e24
- |weighted density| <= 1e12
- coefficient <= 1e6 and beta <= 1e24

The molecular AO consumer also certifies |all AO values/first jets| <= 1e40
and |effective density gradient| <= 1e40 during its existing first pass.

The existing strict-positive/finite checks also apply. These are sufficient
optimization conditions, not new scientific-domain restrictions: otherwise run
the unchanged dense algorithm and its failure checks.

The admitted point count is at most floor(UINT32_MAX/3). The bounds imply
r² <= 1.2e13, gi in [1e-12, approximately 1.2e25], a pair denominator between
2e-36 and approximately 3.456e75, |phi| <= 7.5e35,
|dphi/dgi| <= 1.125e48, and |dphi/domega| <= 1.35e61. Complete feature outputs
stay below approximately 2e118 and energy below 8e83. Consequently omitted
pair arithmetic and zero-row reductions would be finite, and cannot conceal a
numerical failure that the dense runtime would have rejected. An outlying zero
row with a finite coordinate whose squared distance overflows still follows the
dense route and fails. All retained pair arithmetic and reduction ordering are
unchanged; there is no symmetry reassociation or approximation.

The caller-side bound is also necessary: finite individual AO jets alone do not
prevent raw finite feature fields from overflowing before multiplication by zero
weight. The additional factors bound each three-axis AO weak sum below 1.2e241,
so omitted zero-weight AO contributions would be finite. A regression uses an
accepted near-cancelling tight/diffuse contraction with finite huge AO values and
a tiny-radius all-zero-weight grid to ensure the original nonfinite AO-potential
failure remains observable. Normalization alone is not a sufficient AO bound.

## Rejected alternatives

- Masking generic zero-weight rows would incorrectly discard raw feature and
  weight derivatives
- Masking underflowed weighted densities would incorrectly discard active
  nonzero weights
- Unconditionally skipping pairs could conceal overflow at inactive points
- Index compaction adds resource accounting and storage; weight checks obtain
  the measured benefit without it
- Semilocal branch-aware expression reuse is a separate, wider compiler change
  involving lazy branch domains and is not included

## Numerical and regression gates

The native fixture compares raw and optimized energies/weighted potentials
exactly, covers signed/zero/subnormal weights, all-zero inputs, tile boundaries,
raw zero-row derivatives, geometry/rVV10 fallback, invalid inactive inputs,
overflow fallback and recovery, extreme finite AO pullback overflow, and unchanged
budget admission. It asserts actual
A²/N² pair counts. The existing independently compacted molecular active-set
oracle checks energy and AO potential to 1e-13 and checks a zero-density vacuum.

Independent fixed-grid Python references check every VV10/rVV10 output at the
existing 2–3e-14 relative gates. Complete molecular SCF is independently checked
against PySCF 2.14.0 / Libxc 7.0.0 on identical basis/grid definitions. No oracle
is used by production execution.

Final local validation before publication:

- Native CTest: 57/57 passed, including the adversarial AO overflow regression
- Focused Python VV10/domain/composition gates: 60 passed, 6 CUDA skips
- Complete WB97M-V gradient and reconverged finite-difference tests: 7 passed;
  this is correctness coverage, not a force-endpoint performance claim
- Five native matched-grid PySCF reconvergences (RKS/UKS): maximum energy-at-density
  error 3.1e-14 Eh, complete Fock error 4.6e-14, density error 8.6e-12
- Water/STO-3G and water/def2-SVP on the 16/6/12 grid independently reconverge
  within 1.6e-13 and 2.7e-13 Eh, using tighter 1e-12/1e-10 energy/density criteria
- OH/STO-3G on 16/6/12 agrees with independently evaluated PySCF energy and full
  Fock at the native density within 1.6e-13 Eh and 7.8e-14, respectively.
  Independent OH reconvergence from core/minao guesses did not converge (including
  a Newton retry); it is not counted as a successful independent SCF oracle.
  Baseline/candidate OH parity and the separate small-system UKS reconvergence
  gates remain required; failed reference attempts are retained with the evidence
- Compiler/SCF/electronic-structure boundaries, native complexity audit, generated
  method manifest, CUDA ownership, default-promotion inventory, formatting and
  lint checks passed. Full Python-suite and real GPU execution are not claimed

## Endpoint qualification

Baseline: `8154ab3df56a10900dd267039b857021a1054721`, which already contains
PR #1643's AO-axis optimization. These are incremental gains, not a repeat of
that earlier change. Both builds use GCC 14.2, RelWithDebInfo (-O2), identical
CPU-only CMake/AOT/OpenBLAS settings, and one OpenBLAS/OMP/MKL thread on a shared
AMD EPYC 9V74 host (nine exposed vCPUs, about 10 GB RAM).

`benchmarks/cpu_wb97mv_endpoint.py` records complete energy endpoints, native
binary hashes, scientific settings, energies, SCF iterations, Fock builds, and
grid/AO diagnostics. Each process performs a first-call cold endpoint, two
one-shot warm calls and a 1% changed geometry. Warm does not reuse a prepared
plan or converged density. Cold is not an OS-page-cache or library-load benchmark.
Imports and builds are excluded; preparation and
complete SCF are included. Three baseline/candidate process pairs alternate
order. All paired samples are checked numerically; none are filtered by timing.
No compilation, tests or reference calculations run during endpoint timing.

The explicit small/medium grids are controlled benchmark inputs and do not
establish quadrature convergence. Production grid defaults are unchanged.
No CUDA, force-endpoint, default-grid, large-molecule or multithread-scaling
speedup is claimed.

### Final measured complete energy endpoints

| Case | Grid points | Warm baseline (s) | Warm candidate (s) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Water/STO-3G, 12/4/8 | 1152 | 0.39079 | 0.33590 | 14.05% |
| Water/STO-3G, 16/6/12 | 3456 | 1.95049 | 1.32389 | 32.13% |
| Water/STO-3G, 24/8/16 | 9216 | 11.74594 | 7.44102 | 36.65% |
| Water/def2-SVP, 16/6/12 | 3456 | 5.46055 | 4.81071 | 11.90% |
| OH UKS/STO-3G, 16/6/12 | 2304 | 1.37452 | 0.96632 | 29.70% |

Each warm median contains six samples per build. All 60 corresponding
cold/warm/changed-geometry energies and complete KS diagnostics matched exactly,
including iterations, Fock builds and grid work. The separate VV10 pair work is
reduced as described above; it is not part of the unchanged public KS work fields.

| Case | Cold baseline -> candidate (s) | Changed geometry baseline -> candidate (s) |
| --- | ---: | ---: |
| Water/STO-3G, 12/4/8 | 0.38178 -> 0.33969 | 0.38304 -> 0.32839 |
| Water/STO-3G, 16/6/12 | 2.14969 -> 1.29467 | 1.92128 -> 1.29856 |
| Water/STO-3G, 24/8/16 | 11.62123 -> 7.70770 | 11.91009 -> 7.69519 |
| Water/def2-SVP, 16/6/12 | 5.71162 -> 4.76379 | 5.37559 -> 4.49848 |
| OH UKS/STO-3G, 16/6/12 | 1.50988 -> 0.99058 | 1.35271 -> 0.96263 |

Cold and changed-geometry medians each contain three samples per build. These
single-thread shared-host measurements are evidence for the tested systems and
explicit grids, not universal speedups. No sample was removed for a timing outlier.

Native library SHA-256:

- Baseline: `16f47c1f474c1cc3c2cf7fab56aff7882a4412bd7050ef645106910ca926ac32`
- Candidate: `dff79c6e144d0cb0e4add0aba92c882e500135e6b61fbf8a18e5282ba182fd92`

A separate final 3,456-point singlepoint profile records 11 VV10 calls and exactly
131,383,296 -> 76,107,148 pair-kernel evaluations (42.1% less pair work). Inclusive
VV10 time was 1.582 -> 1.023 s. AO work stayed at 462 calls / 798,336 AO-point
pairs, with 0.149 -> 0.147 s inclusive AO time. These instrumented component
timings are not substituted for the complete endpoint medians.

Separate nominal-geometry profiles confirm the work reduction in every case:

| Case | VV10 calls | Baseline pair kernels | Candidate pair kernels | AO calls / AO-point pairs (both) |
| --- | ---: | ---: | ---: | ---: |
| Water/STO-3G, 12/4/8 | 11 | 14,598,144 | 8,504,480 | 165 / 266,112 |
| Water/STO-3G, 16/6/12 | 11 | 131,383,296 | 76,107,148 | 462 / 798,336 |
| Water/STO-3G, 24/8/16 | 11 | 934,281,216 | 545,150,324 | 1,188 / 2,128,896 |
| Water/def2-SVP, 16/6/12 | 15 | 179,159,040 | 106,452,024 | 630 / 3,888,000 |
| OH UKS/STO-3G, 16/6/12 | 17 | 90,243,072 | 51,863,225 | 459 / 705,024 |

Raw JSON, test/build logs, independent-oracle scripts, timing drivers and profiler
source are retained in the local qualification bundle. The initial unguarded-AO
campaign is retained separately and excluded from these final claims.

## Reproduction

Build the baseline and candidate with identical CPU settings, then alternate
fresh processes with each library:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
GENERATIVEQC_LIBRARY=/absolute/baseline/libgenerativeqc.so \
  python benchmarks/cpu_wb97mv_endpoint.py --case water --basis sto-3g --grid 16 6 12
GENERATIVEQC_LIBRARY=/absolute/candidate/libgenerativeqc.so \
  python benchmarks/cpu_wb97mv_endpoint.py --case water --basis sto-3g --grid 16 6 12
```

Repeat three pairs, reversing execution order on alternate pairs. Also run water
on grids 12/4/8 and 24/8/16, water/def2-SVP on 16/6/12, and OH-doublet UKS on
16/6/12. Compare every energy, iteration and recorded work field before timing.
Use `generativeqc_wb97mv_scf_tests` with a JSONL output path and
`tools/verify_wb97mv_scf.py` for the five independent native SCF fixtures.

## Revisit when

If cheap full-index scans become measurable at higher inactive fractions,
qualify bounded ordered compaction with explicit storage admission. Additional
variants, unweighted consumers, derivative paths or an expanded arithmetic
envelope require their own scientific contract and endpoint evidence.
