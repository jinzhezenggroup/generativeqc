# Decision: native compiler-AD WB97M-V fixed-grid density response

Status: implemented component; not a complete molecular endpoint
Date: 2026-10-16

## Problem

The native CPKS controller has an authenticated orbital frame and a coupled-spin
CUDA callback, but its original numerical qualification used a linear model.
Production WB97M-V induced Fock also needs semilocal meta-GGA and bilateral VV10
density curvature. The ordinary native XC response selector does not admit
WB97M-V, and promoting the existing hand-propagated point jets would introduce
a second derivative authority instead of meeting the compiler-AD requirement.

## Decision

Add `dft::CudaWb97mvGridResponse` to the opt-in native Hessian library. It borrows
validated native packed basis records, device points/weights and the reference
density under an actual owner authenticator, plus an exactly sized caller arena.
Preparation evaluates first AO jets, physical spin fields, a deterministic
reference-fixed VV10 active list and the native value-side grid potential.
Actions evaluate signed density directions without any PSD factorization.

`method/wb97mv_grid_response.py` composes the existing scalar WB97M-V gradient,
its AD JVP and the existing coefficient pullback/JVP graphs. It rehomes graph
nodes verbatim rather than introducing another scalar algebra. Its only extra
compiler ownership exception is the neutral scalar graph/interner module, whose
historical location is IntegralIR; integral recurrences and schedules remain
forbidden dependencies. Method scalar composition stays above XC/TensorIR.

The Libxc work projection is outside functional differentiation, as in the
incumbent stationary-grid consumer. A floor suppresses the feature-Hessian seed,
not the already projected potential or the coefficient's raw-gradient tangent.
Differentiating an energy with clamps inserted beforehand would change that
contract. RKS has total D and its existing doubled floors; UKS retains physical
alpha/beta fields, the cross-sigma bound and cross-spin derivatives. Lazy scalar
branch coalescing avoids duplicating the large polarized graph while preserving
vacuum and spin-tail evaluation domains.

AO feature extraction reuses `feature_programs`. Potential assembly is its
TensorIR density VJP, demand-pruned to AO panels and feature cotangents. Thus
tau's one-half and both AO legs are derived, not handwritten native equations.
The initial lowering materializes full-range affine nodes and does not claim
GEMM optimization. No point-by-AO-pair tensor is retained.

VV10 uses the existing pair energy/first-leg gradient/JVP graph, with directions
on **both** legs and alpha+beta total fields. Every ordered active pair is
visited, including the diagonal. First-leg derivatives account for both energy
legs; no extra one-half is applied to potentials. Deterministic column-tile
reductions lift the same four rho/gradient cotangents to each physical spin.
The generated beta constant contributes to the reference potential only; its
density-response derivative on a fixed grid is zero.

The physical grid owner belongs to native `dft/`, not generic `response/`.
The generic CPKS controller still has no dependency on DFT mathematics. CUDA
ownership records explicitly count the retained pair integration/spin-lifting
glue as scientific native code; compiler AD does not mean all runtime code has
become generic or that a full endpoint exists.

## Resource and failure invariants

- The arena includes dense first jets, base/directional fields, coefficients,
  both retained/reference and candidate potentials, tensor scratch, status,
  active indices and one bounded pair tile. All size arithmetic is checked.
- Pair tile storage and total ordered-pair work have independent limits. The
  actual device-discovered active count squared must fit before any pair work.
  Discovery transfers only count/status metadata, never density arrays.
- Base masks are fixed on both legs for every response action. They are not
  rediscovered from a signed direction. Vacuum performs no pair work.
- All numerical operations execute on the supplied device/ordinary stream.
  There is no allocation, interpreter, JIT, device switch or CPU numerical
  fallback. Build-time generation cannot import runtime/CuPy/PySCF/Torch.
- Admission checks capture, pointers, alignment, disjoint storage and current
  device. Input/output matrix symmetry and all numerical results are checked,
  never repaired. Failures drain queued work.
- Publication follows numerical acceptance and fresh authentication. Failure
  before publication preserves the caller output; CUDA failure during its final
  copy invalidates that output. A stale authenticator revokes the instance.
- Work receipts distinguish reference/response ordered pairs and attempted
  response actions. They do not hide failed or repeated work.

## Independent acceptance

`tests/python/test_wb97mv_native_grid.py` uses Gaussian polynomial jets computed
independently of native AO emission, including real-spherical d/f records.
Interior value potentials use PySCF's pinned Libxc plus the separate real-space
VV10 feature-derivative oracle. Response gates use displaced value potentials at
steps `1e-3` and `3e-4`, not compiler response equations. Interior acceptance is
`atol=3e-9, rtol=2e-6`; the empty-minority case has the explicit separate gate
`atol=5e-9, rtol=3e-6`. Value gates are `atol=3e-10, rtol=3e-9`.

Exact empty minority spin requires the existing **independent upstream Maple C
libquadmath oracle**, emitted by `tools/qualify_wb97mv_tail.py::oracle_source`,
not binary64 Libxc. The source archive is SHA256
`8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`;
the oracle contains no GenerativeQC graph or AD evaluation. It is compiled with
a verified compiler cache. Set `GENERATIVEQC_WB97MV_WIDE_ORACLE` to that compiled
executable for the opt-in acceptance test, in addition to
`GENERATIVEQC_CUDA_WB97MV_GRID_PROBE`. Test-only processes provide references;
the CUDA executable has no Python/oracle dependency.

The initial binary64 comparison failed at the empty beta potential by up to
`0.8052`, whereas upstream quad values pass the unchanged value gate. This is
the previously diagnosed Stoll cancellation, not grounds to alter production
algebra or widen acceptance. See the September 24 stability note below.
Quad displaced potentials also expose a genuine step-size truncation: maximum
response differences at steps `3e-3`, `1e-3`, `3e-4`, `1e-4`, `3e-5` are about
`1.048e-3`, `1.164e-4`, `1.048e-5`, `1.169e-6`, `1.689e-7` respectively.
The empty-spin test therefore requires greater-than-fivefold error reduction
over `1e-3 -> 3e-4 -> 1e-4`, and applies the **unchanged** response tolerance to
the latter two converged steps. Both spin orientations are tested. Do not
silently replace this convergence study with a looser threshold or a double
oracle.

Coverage includes RKS/UKS, unequal and empty minority spin, signed/indefinite
directions, vacuum, cross-spin response, both VV10 legs, self-adjointness,
different/tail pair tiles, and a 133-point multiblock runtime shape. Failure
tests cover stale entry/action/publication, capture, device/stream mismatch,
aliases, insufficient arena, pair work overflow, bad reference and
NaN/asymmetric/overflowing directions. The executable also runs with PATH,
PYTHONHOME and PYTHONPATH unavailable. Independent tests are optional oracle
consumers, not production dependencies.

Source-matched Slurm job receipts, binary SHA256, compiler commands, compiler
cache statistics and sanitizer logs are retained under ignored
`.artifacts/wb97mv-hessian-aot-20261015/grid-*`. Failed staging/collection runs
are retained separately and are not numerical acceptance evidence.

### Source-matched qualification

Normal Release/sm_120, `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`, `--fmad=false`,
verified ccache 4.5.1 and explicit CXX/CUDA launchers. The unavailable sccache
executable was not assumed usable. Actual cached NVCC commands and shared cache
counters are retained; those counters are not exclusive per-task measurements.

On node2/main/PRO6000, finite Slurm job **3086** passes all **29** grid tests in
13.71 s and all four sanitizers (memcheck/initcheck/racecheck/synccheck) with zero
errors/hazards. Job **3088** repeats all four sanitizers on the multiblock
133-point UKS case, also with zero errors/hazards. Its reference and response
each perform exactly **17,424** ordered pairs; the smaller case performs 64.
Both jobs preserve Slurm-assigned `CUDA_VISIBLE_DEVICES=1`.

The native grid probe SHA256 is
`4cfb405362994911898e1645f3b1d403ea32aef846da42186d38af51edf8d229`;
the independent upstream quad oracle is
`d2a45d6437881c9430b517c3b35304dcc1a14db7bc8027b45d5932f17ee28964`.
Readelf confirms that the grid executable depends on cudart and standard C/C++
libraries, not libpython. The unavailable-Python-path numerical test passes.

After fast-forwarding the concurrent atomic-GMRES status repair `d7d78952e`
(without reverting or duplicating it), all four native probes are rebuilt.
Finite Slurm job **3087**, same node and preserved visibility, passes the full
**159** leaf/GMRES/CPKS/grid test suite in 59.38 s. These are component/regression
times, **not Hessian endpoint timings**. CPU/compiler regression passes **76**
tests with **137** explicitly opt-in skips; ty 0.0.82 and the compiler structure
check (554 modules, zero dependency errors) pass.
The pinned all-files pre-commit suite passes, retaining only pre-existing
non-blocking structural CUDA warnings outside this change.

## Boundary and consequences

This is a dense fixed-grid component, not full nuclear response or a public
method capability. A production prepared-owner binding still must establish
the basis/quadrature/density identity and compose it with exact or admitted
fitted J/K and the declared long-range exchange correction. Do not substitute
full exchange for that correction or reuse a DF PSD factor for a response D.
Moving grids, complete integral shell closure, raw DF metric response, ECP,
stationary HVP/Hessian assembly and production CLI replacement remain separate
required parts of the full migration. PR #2527 must remain draft meanwhile.

The synchronous status gates and correctness-first contraction schedule are
deliberate. No performance promotion or GPU4PySCF speed ratio follows from
component qualification. Revisit panel scheduling, retained point Hessians and
replay only after a complete endpoint has matched independent molecular gates
and reports its complete work and timing.

## References

- `docs/developer/dft_hessian.md`
- `.agents/notes/implemented/numerics/2026-10-15-native-coupled-ks-response.md`
- `.agents/notes/implemented/numerics/2026-09-24-wb97mv-empty-spin-stoll-stability.md`
- `tests/python/test_wb97mv_native_grid.py`
- `tests/native/cuda_wb97mv_grid_probe.cpp`
