# Decision: admit fitted LR after complete endpoint crossover

Status: implemented in this change; not a claim of an upstream merge
Date: 2026-11-04

## Decision

Unset `GENERATIVEQC_DF_RANGE_EXCHANGE` now means `auto` for the admitted
fitted-primary range-exchange/VV10 composition. The native metric owner admits
an independently fitted LR correction before the first energy. An unresolved
cutoff or optional resource/capability refusal retires the whole candidate
before one bounded exact-LR attempt. Each batch item freezes the actual
approximation across properties, replay, geometry replacement and retries.
Explicit `df` requires LR fitting; explicit `direct` retains the mixed model.
Invalid controls, eigensolver and non-allocation CUDA failures still propagate.

The performance changes reserve optional immutable Hermite geometry inside the
existing value allowance, preserving resident B/U and response capacity, and
automatically select the qualified symmetric shell response under work and
metadata gates. Generic source/response fallbacks and explicit caps remain.
There is no new cutoff, automatic LR screen, mixed LR arithmetic or CPU oracle
in the CUDA production path.

The default change qualifies energies and analytic first derivatives. Hessian
and HVP consumers still require the fitted-primary/exact-LR approximation;
fitted LR is explicitly rejected, never silently differentiated with exact-LR
response equations. See the [current DF contract](../../../../docs/user/dft_density_fitting.md).

## Final source-matched qualification

Slurm 15807 compares explicit direct and genuinely unset default in fresh
processes using the same release binary and GPU allocation, RTX 5090/sm120,
water96/def2-SVP/def2-SVP-JKFIT, stock grids, SCF tile 256 and force tile 1024.
No response/pair override, LR screening or tracing is enabled. No other GPU
work from this campaign overlaps timing; the reference and broad qualification
jobs finish first, and the test-hook recovery job is dependent on its completion.

| Complete E/F observation | Exact LR (s) | Default fitted LR (s) | Improvement |
| --- | ---: | ---: | ---: |
| Cold construction/preparation/SCF/forces/export | 225.346998 | 216.859267 | 3.7665% |
| Three frozen-seed warm calls, median | 29.946756 | 23.522215 | 21.4532% |
| Cold plus three warm calls | 315.229689 | 287.421781 | 8.8215% |
| Moved-geometry rebuild | 222.177260 | 201.072594 | 9.4990% |
| Moved warm replay | 30.481874 | 23.504100 | 22.8915% |
| Entire six-call lifecycle | 567.888823 | 511.998476 | 9.8418% |

Both cold SCFs take 28 iterations, with 29 exact / 28 fitted Focks; moved takes
19 iterations, with 20 exact / 19 fitted Focks. Every stationary or moved warm
call takes one iteration and one Fock. These actual semantic work differences
remain in the receipts rather than being hidden by equal-iteration labels.
Cold and warm improvements reproduce the earlier explicit-schedule and
automatic populations in both execution orders; this is not a microbenchmark
or a first-warm setup transfer.

All twelve observations pass the independent stationary/moved GPU4PySCF full
RI-J/K energy and complete-force gates. Default's worst errors are
8.351e-10 Eh/atom, 2.448e-7 maximum and 9.007e-8 RMS Eh/bohr, versus limits
1e-5, 1e-4 and 3e-5 respectively. These stock-grid checks do not establish a
universal basis/size/backend performance ranking or grid-converged accuracy.

V5 Slurm 15806 passes 10 default/auto admission and 22 automatic RKS/UKS
selection/auxiliary-g endpoint cases. Its release-image legacy population
passes 61 source/anchor cases; the two hook-dependent failure/recovery cases
pass separately under Slurm 15808 with the same-source test-hook build.
Slurm 15807 passes all five CUDA admission cases under memcheck, zero errors.
Water12 default cold/warm/moved/moved-warm correctly retain exact LR and agree
with explicit direct to roundoff. Earlier V4 broad qualification additionally
passes 43 independent endpoint cases, with one intentionally CPU-only diagnostic
skip. Host controls include 430 policy/budget/envelope cases, 253 final adjacent
cases and 130 frontend/precision cases; scientific and lifecycle coverage is
not traded away for the performance result.

The final source identity is
`649dd4802509b936af5f82c41b5af5a97291c709033cc97514b02e4a7fc51cfd`,
release library SHA256
`09ddd5616948a0f7a1802bac950b0f62ca77e4af025f4710a878508c637ba502`.
The restored cached build reproduces the release library byte-for-byte and
reports no work. Logs, complete/failed journals, source additions/patches,
cache statistics and analysis are retained under ignored
`.artifacts/lr-df-default-20261104/`, mirrored in the n1 qualification campaign.
The [endpoint runner](../../../../benchmarks/wb97mv_lr_crossover.py) records
actual snapshot identity, not an inference from the selector.

## Consequences and revisit conditions

Initial fallback is bounded but not free: water12's optional attempt adds about
0.19 seconds to the 6.19-second explicit-direct cold endpoint in this campaign;
warm and moved owners do not retry LR admission. Explicit `direct` is available
when that setup cost or cross-preparation approximation identity matters.
Cross-size resource availability can select a different initial approximation;
use explicit `df` or `direct` when every independent preparation must match.

Requalify complete endpoints and independent forces when changing source
staging, shell/pair policy, metric response, resource envelopes or backend
admission. A fitted-LR Hessian needs its own matching response implementation
and independent numerical gates; this decision is not permission to promote it.

## Historical investigation

The sections that follow retain the original candidate stages, rejected
approaches and pre-promotion limitations. Their interim default/status statements
are historical, not the current policy above. The initial source was master
`9cfb77727`; the final fetched `origin/master` is `195561fae` and has no
intervening changes to the compared WB97M-V production paths. Measurements are
not relabelled as a build of that newer revision.

## Objective and baseline

The requested endpoint is faster complete WB97M-V density-fitted calculations
with fitted long-range exchange than with the current mixed fitted-primary /
exact-LR default, followed by a qualified default change. Beating GPU4PySCF or
an older fitted implementation alone is insufficient. A cold improvement must
not hide work transferred into the first warm call.

The initial baseline is master `9cfb77727`, built in Release with tests and fast
compilation disabled, sm120, and the WB97M-V restricted/unrestricted stationary
profiles. Its scientific source identity is
`d28ab04b4866199ebc45681d3acf1974553715da12e4af7910e35c78c158e426`.
The build configuration and cache statistics are retained in ignored
`.artifacts/lr-df-default-20261104/`. Compiler caching remains enabled.

## First optimization candidate

The preceding Rys investigation measured substantial derivative work in bra
classes with A<B, which still used polynomial lowering. The existing Gaussian
moment program supports either bra orientation: raised B is recovered from
raised A and the center difference. Generate those missing Rys classes through
the same cooperative lowering rather than introducing another recurrence or
symmetrizing away metric-adjoint terms.

Mathematical availability is separate from execution-policy promotion. The
first host controls compare the expanded family against independent libcint
contracted Cartesian and spherical derivatives, including positive erf omega.
Runtime selection and complete device performance are not yet qualified.

The subsequent source audit identified a stronger work reduction to measure
first: the explicit LR shell route defaults to ordered full bra pairs. Its
existing symmetric consumer folds both actual adjoints, including nonsymmetric
weights, before traversing one shell orientation. This can remove the reverse
classes entirely rather than merely accelerating their recurrence. The mirrored
Rys candidate is not a reason to retain avoidable semantic work. Its development
edits were withdrawn after symmetric-pair measurements: reverse recurrence
coverage adds compilation footprint without serving the selected folded path.

## Matched original-image crossover

On the RTX 5090, Slurm 15788 runs all three arms with the same original
production image, water96/def2-SVP/def2-SVP-JKFIT, stock grids, force tile 1024
and SCF tile 256. Each cold call includes constructor, preparation, SCF, analytic
forces and host return. Three warm calls start from each owner's frozen cold
density:

| LR route | Cold (s) | Warm median (s) |
| --- | ---: | ---: |
| Incumbent exact LR | 225.683626 | 29.942668 |
| Fitted LR, full shell pairs | 269.684804 | 55.037973 |
| Fitted LR, symmetric shell pairs | 259.572983 | 45.151355 |

Cold SCF takes 28 iterations; the incumbent counts 29 Fock actions and both
fitted arms 28. Every warm call takes one iteration and one Fock. All twelve
observations pass the existing independent GPU4PySCF DF energy/full-force
gates. Full/symmetric fitted forces differ by at most 1.6615154e-10 Eh/bohr.
All eight original-image auxiliary-g GPU endpoint cases pass, but none of
these observations establishes a crossover. Do not promote.

A later fetch finds origin/master at `b735793be`; the two intervening commits
do not change this WB97M-V LR path. Do not relabel the measured `9cfb77727`
image as a newly compiled master image.

## Remaining source-work bottleneck

A separate same-image diagnostic, Slurm 15790, reproduces the crossover with
tracing enabled. The symmetric fitted LR response takes about 33.22 seconds
per force. Disjoint CUDA-region accounting attributes about 24.48 seconds to
raw three-center regeneration, 3.36 seconds to unclassified response algebra,
and most of the remainder to derivative contraction. The response generates
57 packed panels covering 3616 auxiliaries, rather than one source slice per
auxiliary. This is not the older single-slice amplification problem.

Primary resident tensor materialization uses prepared Hermite E and costs
4.67 seconds. LR materialization does not retain this optional source and
costs 28.25 seconds for the same 8,542,322,688 raw output bytes. The existing
Hermite owner already accepts immutable positive omega and has independent
range-separated raw-value tests. Diagnose its admission refusal before adding
another recurrence, tensor cache or pseudoinverse shortcut. Optional source
storage must stay within live headroom, explicit caps and the unchanged
response allowance; refusal must retain the bounded generic source.

The warm exact-LR Fock action itself takes about 14.76 seconds, versus about
1.00 second for fitted LR. Fitted LR's advantage there is lost to force source
work. Optimizing only derivative shell kernels misses the dominant cold and
warm costs. Traced observations are diagnostic, not promotion timings.

## Bounded raw-source admission candidate

The progress diagnostic identifies the refusal precisely: the primary source
retains 105,901,184 device bytes and needs a 115,640,616-byte host peak. LR has
only 264,254 bytes left after its value planner consumes the allowance with
interchangeable staging; it retains B but no optional occupied U. Hermite E is
refused before even its component metadata fits. The existing LR raw math is
not absent or scientifically unsupported.

The candidate reserves the existing 256-MiB optional-source ceiling inside
the already resolved LR value budget and asks the ordinary packed planner to
prove B and the incumbent occupied capacity still fit. It then prepares the
same omega-aware source transactionally and replans against the actual retained
source bytes. No extra total allowance or response borrowing is introduced.
Explicit storage/mapping selectors retain their existing behavior. A failed
capacity proof or optional allocation keeps the original value plan and bounded
raw source. This may reduce a staging panel, but does not discard admitted B/U.
The diagnostic confirms the LR source now retains E without increasing its
12,354,066,476-byte value allowance or changing occupied rank capacity zero.
LR tensor materialization falls to 6.91 seconds, response raw regeneration to
2.82 seconds and the complete LR response to 11.81 seconds. This run disables
all shell screening, including the primary automatic screen, and is not the
production comparison.

An unprofiled same-binary/same-allocation pair, Slurm 15797, keeps production
primary screening and selects explicit LR shell/symmetric execution. Exact LR
takes 226.022796 seconds cold and 29.934455/29.957942/29.962611 seconds warm;
fitted LR takes 216.788616 cold and 23.518712/23.533470/23.528289 warm. Cold,
median warm and cold-plus-three-warm improve 4.0855%, 21.4623% and 9.0252%.
All eight independent energy/full-force gates pass. Every force tile is 1024;
cold Fock counts remain 29 versus 28 and each warm call has one. The candidate
scientific source identity is
`1ac1f81aa7144f89892bfa3a6535baaf8dabce3c3e380ae0e7faffe3ea84e170`.
This initial endpoint crossover is not default-promotion qualification. The
repeat and lifecycle follow-up below retain their own source-matched receipts.

The reverse-order Slurm 15798 population reproduces the improvement: exact LR
is 225.986908 seconds cold / 29.968907 median warm, fitted LR is 216.846796 /
23.489798. Moved-geometry rebuilding is 222.732054 versus 201.025908 seconds;
subsequent moved warm calls are 30.489917 versus 23.504844. The complete six-call
lifecycle improves 10.0602%. Stationary reference gates all pass. Fitted/exact
moved observables differ by at most 1.91e-9 Eh/atom and 7.32e-9 Eh/bohr per
component; this is approximation parity, not an independent moved reference.
Slurm 15799 passes all 23 selected auxiliary-g, radial-mapping fallback and
explicit-screen tests. The final cached build has no work and retains the exact
qualified source identity. Automatic selection, cutoff admission and final
default-policy qualification remain separate unfinished work.

## Explicit LR shell screening experiment

The existing shell screen bounds positive Gaussian majorants in the Coulomb
norm after summing absolute Cartesian/public-basis weights. For positive omega,
`0 <= erf(omega*r)/r <= 1/r` pointwise. Applying this inequality to those positive
majorants proves the same envelope bounds the LR derivative contraction;
raised/lowered derivative norms already cover shared-atom center sums.
Metric derivatives and retained/discarded spectral response stay unchanged.

The candidate honors an explicit shell error budget for LR instead of forcibly
resetting it to zero. Unset LR behavior remains strict/unscreened, and neither
LR defaults nor automatic shell policy change. Optional norm-storage refusal
still retains the exact unscreened route. Host tests compare the emitted
production envelope with independent libcint at omega 0, 0.001, 0.3 and 10 for
orbital s/p/d/f, auxiliary s/p/d/f/g, signed contractions, random geometries and
coincident-center gradient subsets: 332 checks pass. Scheduled endpoint tests
must also demonstrate real skipped work, bounded full-force error, independent
finite differences, replay and auxiliary-g behavior. Screening is secondary,
not a substitute for eliminating raw source cost.

## Required endpoint evidence

- Compare the current exact-LR route, fitted-LR generic response, and fitted-LR
  shell response from matched production builds and explicit scientific inputs.
- Measure fresh-process constructor/preparation/SCF/host-returned energy and
  analytic forces; retain cold, each of three warm calls from the owner's frozen
  cold density, and changed-geometry/replay observations separately.
- Authenticate actual primary and LR approximations from native snapshots;
  preserve Fock work, solver controls, basis/auxiliary/grid identity, response
  work, binary/source identity, scheduling and device visibility.
- Independently validate energy and full forces against the matching fitted
  reference. Compare with the incumbent observables as a separate approximation
  check. Retain the declared README common gates and independent derivative,
  finite-difference, auxiliary-center and metric-response tests.
- Keep failure, capacity, lifetime, unsupported-class and bounded fallback
  behavior explicit. Fitted-LR resource failure cannot silently select exact LR.
- Require repeated complete populations exceeding the existing variability
  floor, not a component timing, before default promotion. Inspect warm1 and
  the full lifecycle rather than pooling away transferred setup work.

## Reproduction and failure retention

`benchmarks/wb97mv_lr_crossover.py` provides the native comparison runner. It
reserves an exclusive output journal, records partial failures and restores
preparation-time controls. Raw development output stays outside the reviewed
benchmark result tree. Every real-device command uses a finite Slurm allocation
on an admitted node; n3 is not used for GPU execution.

The first independent reference attempt returned no complete sample because
its post-execution auxiliary reference metadata file was absent. Its failed
journal and log remain separate. The retry regenerates the complete PySCF
2.14.0 input/metadata pair before executing a new population; it does not relabel
the original attempt as a timing sample.

## Small-system rank-boundary admission

The original production image completes water12 exact-LR energy/forces and
explicit fitted-LR energy, but rejects fitted-LR forces with
`DF metric rank crossing: retained/discarded subspaces are unresolved`.
Generic, shell and symmetric shell all reject the same input. A native
snapshot derivative query exposes that typed detail, independently of the
batch API's generic numerical-failure text.

An independent PySCF 2.14.0 / SciPy eigensystem confirms that this is a real
cutoff-admission issue, not a pair-folding regression: for water12/def2-SVP-JKFIT
and omega=0.3, the LR metric has 452 auxiliaries, rank 282, largest eigenvalue
536.8368642032228 and relative-1e-10 cutoff 5.3683686420322285e-8. The closest
eigenvalue is about 5.36783398e-8, only 0.3504 times the existing
`128*epsilon*largest` resolution away. Water3 and water96 have respective
nearest-cutoff distances about 164.5 and 34.88 times that resolution.

Do not lower this guard or tune the cutoff to obtain passing timings. A default
promotion must account for this existing numerical admission boundary while
preserving preparation-time Hamiltonian identity and explicit-request errors.
The 96-atom crossover campaign is separate from that smaller admission failure;
neither observation can stand in for the other's complete-endpoint evidence.

The batch already has a suitable lifetime boundary for a future automatic
admission decision: `KsPreparedBatch::Item` outlives replaceable geometry plans.
Initial construction eagerly creates each item's `KsPreparedState`; changed
geometry materializes its last-good seed, destroys the old geometry owner and
calls `make_plan` again. A future automatic LR choice must therefore be frozen
per item, not merely in the replaceable state's local execution-plan copy.
Otherwise an initially exact fallback could silently become fitted after a
displacement, or a fitted model could switch to exact on the first force error.
Do not implement promotion by changing only the unset environment selector.

A proposed admission contract is to resolve optional automatic fallback only
before the initial energy, report the actual radial approximation through the
existing snapshot, and reuse that choice for property changes, warm replay,
geometry rebuilding and failed-build retries. An initially fitted item's later
unresolved cutoff remains an explicit failure; an explicitly requested `df`
never becomes permission for exact fallback. The native metric owner already
knows response admissibility after factorization, so its read-only preparation
result should be the authority, not a CPU/reference eigensystem or a trial force.
Capacity reservation must cover every admitted initial alternative without
weakening explicit caps. This design is not implemented by the current raw
source optimization and requires its own lifecycle/error-injection tests.

## Automatic-admission candidate (not default promotion)

The follow-up candidate adds explicit `GENERATIVEQC_DF_RANGE_EXCHANGE=auto`;
unset still selects exact LR. Both CPU and CUDA expose response validity from
the native metric's existing preparation eigensystem. This diagnostic neither
changes eigenvalues/cutoffs nor evaluates a trial force. `KsPreparedState`
admits an optional fit only before its first energy. Unresolved rank, native
unsupported capability or allocation refusal destroys the whole optional state
before the one bounded exact attempt; invalid controls, solver failures and
non-allocation CUDA errors propagate. Explicit `df` retains its existing errors.

`KsPreparedBatch::Item` freezes the resolved approximation independently of its
replaceable geometry plan, including a failed moved-geometry force and retry.
The existing fleet NumericCap reservation explicitly excludes range-separated
models, so the optional initial alternatives do not borrow an energy-only
promise for an unreserved exact provider. Explicit DF budgets still split the
initial fitted primary/LR allowances. A primary split-budget refusal must also
admit the original exact alternative: the batch retires the complete failed
candidate before reconstructing exact under the original total cap. Keeping
only a smaller primary and retrying LR would miss this initial refusal boundary.

Automatic LR response selects the measured shell/symmetric/warp FP64 route
under the existing work and metadata-capacity gates. It does not inherit the
full-range compact schedule, screen budget or occupied/full-rank requirement.
Explicit execution/pair controls and bounded generic response are preserved.

The candidate's native source identity is
`ff28514e6925205c25256d3ee23f90718f95ef128eed6e5e5677b4eda91cb7c9`.
Slurm 15800 passes six CPU/CUDA admission cases and twenty complete endpoint
cases (automatic auxiliary-g shell/symmetric dimer response and frozen
RKS/UKS selection across properties/replay/movement). Slurm 15801 passes all
three CUDA admission/error/retry cases under Compute Sanitizer memcheck with
zero errors. Host controls include 430 selected policy/budget/failure/envelope
checks, 107 adjacent owner/reverse-memory checks and native CPU endpoint gates.
No final automatic performance population has been collected for this source.
These correctness checks are not default-promotion performance evidence.

The cutoff stress fixture must use exact shared primitive inputs. The native
and PySCF same-name STO-3G tables differ in their final digits: their H2 LR
metric eigenvalue ratios are respectively 0.019095277266131912 and
0.01909527709268948, whereas the relative admission resolution is only
2.842170943040401e-14. Treating those tables as identical fails to exercise
the intended unresolved-cutoff branch. The executable test therefore supplies
one explicit 0.7-exponent normalized H Gaussian to both backends and libcint.
It does not tune a production cutoff or relax a numerical guard.

At that stage, unset still selected exact LR. Final automatic endpoint timing
and lifecycle receipts were still required before default promotion.

## Automatic performance and independent moved reference

Slurm 15802 completes the automatic-source candidate with no execution, pair or
LR screening override. Automatic fitted LR versus explicit exact LR takes
215.950908 versus 225.863238 seconds cold, 23.435243 versus 29.969299 median
warm, 201.030922 versus 222.668586 moved and 23.531302 versus 30.463682 moved
warm. Improvements are 4.3886% cold, 21.8025% warm and 10.2081% for the six-call
lifecycle. Cold work is 28 iterations with 28 fitted / 29 mixed Focks; each warm
observation has one iteration/Fock. The automatic arm overlaps a separate
qualification job on another GPU of the same node: retain that caveat instead
of using this population alone as clean default-promotion timing.

Slurm 15805 independently reconverges the moved96 geometry with stock
GPU4PySCF full RI-J/K, full grid response, identical auxiliary inputs and no
native seed/grid reuse. Both moved observations of both native arms pass the
unchanged README gates: energy per atom 1e-5 Eh, maximum force error 1e-4 and
RMS force error 3e-5 Eh/bohr. Fitted LR's moved errors are 8.35e-10 Eh/atom,
2.323e-7 maximum and 8.130e-8 RMS Eh/bohr. This closes the earlier independent
moved-reference gap. It is a stock-grid endpoint check, not a grid-converged
accuracy ranking or a GPU4PySCF performance claim.

## First-order default candidate and second-order boundary

The final candidate makes unset equivalent to `auto`, retains explicit `df`
and `direct`, and migrates exact-specific legacy tests to explicit `direct`.
Its initial admission and frozen item identity are the policies described
above; the small water12 cutoff is not weakened to make a force call succeed.
Real water12 default cold/warm/moved/moved-warm endpoints choose exact LR before
the first energy and retain it, agreeing with explicit direct to roundoff.

Default promotion must not accidentally extend second-order qualification.
The existing Hessian/CPKS consumers reconstruct fitted full-range primary J/K
plus exact LR; they do not implement the independently fitted LR response.
Native CUDA already authenticates mandatory exact LR. The shared Python
second-order entry now explicitly rejects a fitted LR source before response
construction, even when its primal is numerically close enough to exact LR to
pass an incidental Fock tolerance. Hessian/HVP users explicitly request
`GENERATIVEQC_DF_RANGE_EXCHANGE=direct`; no force-dependent fallback or silently
different second-order Hamiltonian is introduced. Twelve real CPU RKS/UKS
default/auto/df HVP/matrix refusal cases and the existing independent exact-LR
energy-difference tests cover this boundary.

The final source identity is
`649dd4802509b936af5f82c41b5af5a97291c709033cc97514b02e4a7fc51cfd`.
The release library SHA256 is
`09ddd5616948a0f7a1802bac950b0f62ca77e4af025f4710a878508c637ba502`.
The release image intentionally omits test injection hooks. Its legacy
qualification passes 61 source/anchor cases; two fault-injection cases require
a separately frozen same-source test-hook image. Missing symbols in the first
attempt remain recorded as harness/configuration failures, not scientific
passes or a reason to remove the recovery coverage. The test-hook image is
never used for promotion timing.

## References

- [Earlier LR Rys evidence](2026-11-02-long-range-rys-derivatives.md)
- [Warm-anchor work transfer](2026-11-01-default-range-warm-anchor.md)
