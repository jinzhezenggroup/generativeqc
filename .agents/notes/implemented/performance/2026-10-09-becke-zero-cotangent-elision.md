# Decision: exact zero-cotangent elision for admitted Becke first derivatives

Status: implemented, experimental; no production promotion
Date: 2026-10-09

## Problem

The [exact AO map and normalized-adjoint experiment](2026-10-09-exact-ao-bitmasks-becke-adjoints.md)
does not remove steady-state point/pair work. Its warm calls already avoid
map discovery, and normalized atom adjoints retain seven phases per tile and
four-word reverse panels. Reducing retained bytes or scalar arithmetic alone
did not establish an endpoint improvement. Follow-up profiling and negative
probes must not become evidence for repeating those approaches.

## Decision

For the authenticated, admitted equal-radius normalized-adjoint first
derivative (primitive mode 2), elide a point's partition VJP if and only if its
actual FP64 seed compares equal to zero. Both signed zeros qualify; no amplitude
threshold, product-zero heuristic or partition-weight cutoff is involved.
Nonfinite seeds do not qualify. Keep point/center distance validation, owner
validation, and point-motion/publication. Explicitly write four gather outputs
to zero before the latter consumes them, including the first use of an
uninitialized pair panel and an all-zero replay after a nonzero replay.

Pair primal, atom logs, normalization arithmetic, pair reverse and pair-panel
gather reads have no remaining consumer for that point. Only this partition
VJP is removed. AO/grid motion and external/nonlocal contributions remain
independent and are not skipped. This is not a Hessian or response shortcut.

Admission additionally requires geometry tolerance at least `1e-12`. Other
primitive selections, unadmitted/fallback owners and smaller tolerances retain
the original computation and error behavior. The production primitive default
remains `off`; the existing coefficient experiment is not promoted either.
`GENERATIVEQC_STATIONARY_BECKE_ZERO_SEED=off|on` permits causal qualification
with the same binary. Native configuration is once-before-topology, never a
mutation of a live geometry owner. Unspecified controls do not require a new
ABI from old artifacts; explicit controls on those artifacts fail closed.

## Numerical admission rationale

Zero multiplied by a nonfinite intermediate is not a valid reason to remove
work. The finite bound here depends on the recognized equal-radius canonical
graph, at most 128 atoms, validated geometry, and positive separation floor:

- The nearest atom's switch factors are at least approximately one half, so
  one product remains finite and positive (at least approximately `2^-127`).
  The maximum log product is finite and the normalized scaled products are
  bounded by one. The normalized denominator cannot vanish.
- A one-zero excluded product scaled by the maximum is bounded by approximately
  `2^127`, still finite. Multiple-zero products have the canonical zero
  pullback. The switch's nonzero FP64 factors have a finite reciprocal bound
  from the rounding of `0.5 * (1 - polynomial)` and its complement.
- Prepared center reciprocals and live ratio partials remain finite under the
  `1e-12` floor. Saturated switches already have zero slope and the incumbent
  reverse does not consume their rematerialized ratio partials.

Thus this admitted first-order VJP with an exactly zero seed has finite zero
results; distance errors must nevertheless remain observable. Do not extend
this argument to radii-adjusted graphs, smaller separation floors, arbitrary
atom counts, high-order derivatives or a graph that fails recognition.

## Ownership, resources and accounting

No new retained allocation, global scratch panel or phase launch is required.
The aligned uint64 cumulative count resides eight bytes after the error pointer
inside the already charged 256-byte control panel, outside scratch aliases.
It is initialized at owner construction, not cleared by geometry resets;
Python reports per-call before/after differences. Cooperative normalization
uses eight extra shared unsigned flags and at most one integer atomic per
block. Existing CUDA function-attribute admission checks the actual static
shared memory footprint, not a stale hand-computed estimate. Serial
normalization records the same semantics with one atomic per elided point.

The metric ABI downloads eight bytes for a mode-2 owner, with bytes, D2H calls
and synchronization counted. Ordinary mode-0 metrics incur no new device
readback. Retain the original launched-domain counters, and separately derive
evaluated primal/reverse pair counts and gather incident counts from the
actual elided-point count. Elided pair-panel byte counts describe logical
values, not executed instructions, DRAM traffic or freed capacity. The
retained pair reservation and seven-launch-per-tile schedule are unchanged.

## Rejected alternatives

These are frozen `fb9586569` diagnostics, not measurements of the later master:

1. Changing pair blocks among 64/128/256/512/1024 threads did not produce a
   useful phase improvement; 1024 threads was slower.
2. A nearest-two-atom scout proving two exact-zero factors before pair primal
   was 6.6–14.5% slower. Its preliminary reduction/control overhead outweighed
   avoided work. This is distinct from the already rejected late product-zero
   truncation; neither is retained.
3. Fusing ordered atom gather with motion using 2/4/8 point lanes was 4–99%
   slower. Reducing launch count alone did not improve this schedule.

Each isolated probe used interleaved ABBAABBA, 64 sampled tiles, signed
synthetic cotangents, all phases plus publication, and preserved exact output
equality against its incumbent. These are microprobes, not complete E/F
promotion evidence. Prototype sources, libraries, hashes and raw JSON remain
in ignored evidence storage; none is added to production.

An intrusive 96-atom profile on the same old population locates pair
primal/switch-log as the largest Becke phase (approximately 1330 ms), followed
by reverse (371/394 ms), gather (300 ms), motion (197 ms), atom logs (175 ms)
and normalization (67/94 ms, ordinary/normalized). Per-tile profiling fences
make these diagnostic numbers, not clean endpoint components or proof of a
hardware bandwidth bottleneck.

The existing opt-in cooperative/materialized integral-derivative route was
also checked on that old source. Its grouped 96-atom warm medians around
18.76/18.78 seconds pass numerical gates, but different-GPU, nonpaired results
do not authorize promotion. The earlier
[cooperative weighted-shell evidence](2026-10-06-cooperative-weighted-shell-force.md)
already records both winning and losing angular classes; defaults are unchanged.

## Qualification protocol and evidence

The local ignored evidence root is `.artifacts/issue1893-1894`; remote frozen
populations reside at `/data/jzzeng/qc-exact-becke-20261009-fb9586569` on n1.
All GPU work uses finite Slurm `srun` on `main`, `gpu:5090:1`, preserves assigned
device visibility, and reuses ccache. No evidence is published as a release.

`paired-zero.py` compares elision off/on with one binary, normalized adjoints
and exact AO bitmasks in both arms. There is one native SCF batch and one
publicly frozen post-solve warm density, not a reference-seeded solve. Close
and reconstruct the whole gradient executor between arms, then prime it
outside timing; the configure-once guard is not bypassed. Retain all priming
calls and their discovery work. Each geometry has five interleaved measured
complete E+F calls per arm. The two setups plus 20 primes plus 20 measurements
give 42 complete calls per atom count; warm calls retain one iteration/one
Fock build. Intrusive profiling is off. Setup cold/moved histories are retained
but this shared-state experiment does not compare cold/moved timing.

The independent verifier recomputes `E <= 1e-8 Eh` and
`F <= 1e-7 Eh/bohr` against fresh GPU4PySCF cold/moved oracles, convergence,
profile state, both endpoint resource bounds, selected mode/zero override,
actual pair-count formulas, unchanged AO/local work, zero measured discoveries
and one priming discovery. It recomputes medians from raw samples. Performance
acceptance requires an improvement greater than both 2% and the robust MAD
noise floor in every compared workload; a `not-run` assessment is not a pass.

### Old population: perf4

Base `fb9586569`, original experiment source `30efb9f43`; these results must not
be relabeled as the current master. Every numerical/work/resource gate passes.

| Atoms | Phase | Elision off, s | Elision on, s | Improvement |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 6.417269 | 6.393678 | 0.368% |
| 48 | moved-warm | 6.390483 | 6.373934 | 0.259% |
| 96 | warm | 19.515841 | 19.247869 | 1.373% |
| 96 | moved-warm | 19.732823 | 19.599972 | 0.673% |

All four comparisons fail the performance significance gate. Raw/summary
hashes and oracle errors are preserved in `perf4-verified-summary.json`; max
energy/force errors are `8.6402e-12 Eh` / `3.0234e-11 Eh/bohr`.

### Current master population: perf5

Base `5cddaa4641797280d24da987d839e2205c49333e`, which includes the merged
exact-map/normalized-adjoint work and corrected benchmark wrapper. It also
contains the newer exact-K default; perf4 endpoint timings cannot be substituted
for this qualification. Both native libraries are freshly built from this
base plus the elision patch, and both GPU4PySCF references are fresh. The
source archive SHA-256 is
`414898076a2778ce75727ca5c75b321f3acfaee3ca58ab289c9bba0519af197d`;
the runtime and stationary library SHA-256 values are respectively
`0346d2dde90a1e0f1b3b9568e028387229124ee6073f9ad3b6d9334bbc4d7c23`
and `c46bd5ad89ee472effc08693af7791f3356872d5c4747532fafd896ab162624d`.
The paired-driver SHA-256 is
`4e7b7bc01204859b50635e749718957a16d2165364753eb73896a7f6d72d6d2f`.

| Atoms | Phase | Elision off, s | Elision on, s | Improvement |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 6.239048 | 6.211919 | 0.435% |
| 48 | moved-warm | 6.249490 | 6.231947 | 0.281% |
| 96 | warm | 19.520877 | 19.405895 | 0.589% |
| 96 | moved-warm | 19.570450 | 19.422776 | 0.755% |

All 84 complete calls pass independent numerical, work and resource gates.
Every measured/priming warm call retains one iteration and one Fock build,
without fallback. All four performance assessments remain `not-run`: their
median improvements are below the 2% noise floor. This is not a stable
full-endpoint speedup, nor a comparison of the default mode-0 owner against the
joint experiment. It isolates zero-cotangent elision within mode 2.

Actual elision is identical in both measured geometries:

| Atoms | Points | Elided points | Primal/reverse pairs, off | Evaluated pairs, on |
| --- | ---: | ---: | ---: | ---: |
| 48 | 1,179,648 | 114,484 | 1,330,642,944 | 1,201,504,992 |
| 96 | 2,359,296 | 211,908 | 10,758,389,760 | 9,792,089,280 |

These are 9.705% / 8.982% reductions in evaluated Becke domains. Distance and
motion domains, dense launch counts (16,128 / 32,256), exact-map AO/local work,
and retained numeric reservations remain unchanged. Device numeric peak bounds
are 257,657,680 / 527,399,824 bytes under the current 536,870,912-byte device
budget; host numeric bounds are 94,603,824 / 256,066,992 bytes under 268,435,456
bytes. These are contract bounds, not measured allocator peaks.

Maximum independent oracle errors are `1.04592e-11 Eh` and
`3.78082e-11 Eh/bohr`. The verifier and summary SHA-256 values are respectively
`e329b62c0c4e4102aac1d89e434fb2fac5877712ea30bba05fb2e4e8829ccbf9`
and `3367fc14508bfacd336697e3ec970dbad01f4da36af9d5f719d7dedc398d5467`.
Raw endpoint/reference hashes are retained in `perf5-verified-summary.json`.

Host qualification passes 1776 cases (511 device/optional cases skipped). The
final test-only serial-control extension passes an additional focused host
cohort of 197 cases (360 device cases skipped). Native shared-owner validation
passes the frozen 288-case matrix plus 72 separately frozen serial-normalization
cases on the same native libraries: 48/96/128 atoms, full/subset/empty indexed
AO selections, explicit/implicit owners, external/no-external contributions,
changed geometry, small-tolerance fallback, mixed/all-zero cotangents,
nonfinite/coincident zero-seed points, tail tiles and failed-then-valid reuse.
Primary memcheck, racecheck and initcheck each pass 16 targeted mode-2 cases;
all report zero errors/hazards/warnings. The serial extension does not rewrite
the original frozen endpoint source/test archive. Final Ruff/clang-format,
public-API docstring, compiler structure (504 modules), CUDA ownership
(338 files), and repository pre-commit gates pass.

## Consequences and revisit conditions

This mechanism genuinely removes an evaluated domain, unlike the prior
arithmetic-only change. It does not compact launched grids, free pair capacity,
remove phase launches or accelerate other endpoint work. A roughly 9% Becke
domain reduction therefore does not imply a 9% endpoint speedup. Neither the
historical/current medians nor the negative probes justify changing production
defaults.

Revisit with complete paired evidence and independent gates if a new route
reduces the nonzero pair domain, its representation, or the larger integral/AO
endpoint consumers without weakening correctness or bounded fallbacks. Any
broader derivative domain needs its own finite-intermediate proof and oracle.
