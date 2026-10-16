# Decision: compiler-AD native WB97M-V J/K nuclear source

Status: implemented component; complete Python-free Hessian remains incomplete
Date: 2026-10-16

## Problem

Primitive integral AD leaves and a native one-electron source do not provide
molecular Coulomb/range-separated exchange nuclear response. Four-center public
projection cannot reuse the one-electron public-pair-by-component scan: repeating
raw quartet evaluation or scanning every raw component for every public quartet
amplifies angular work toward an eighth-power transform.

## Decision

Keep integral derivatives in the incumbent compiler AD. Build scalar quartet
energy programs from the stationary gradient plan, retaining its RKS/UKS density
and SR/LR conventions. Rebind the four distinct density incidences before AD;
shared scientific integral input names must accumulate both Coulomb/exchange
cotangents. Transpose supplies integral weight and four density VJPs. Linearize
the weight with all density incidences active, but linearize the Fock incidences
with only the integral active. Frozen-density Fock nuclear motion and stationary
right-density weight motion are different quantities and must not be conflated.

Bounded inline lowering now admits rational add, row-major reshape, unit-step
slice and explicit-axis broadcast. Output-demand specialization, fail-closed
unsupported operations and the existing 64-term einsum gate remain intact.
The generated maps are compiled at build time; no interpreter is linked into
the numerical consumer.

The native source admits complete ordered family/angular/radial/omega closure
before CUDA work. It owns a mandatory shell-quartet cache of 25 raw fields:
value, twelve first derivatives, twelve gradient JVPs. Through f this is at most
25*10,000 doubles (2 MB), independent of molecular size and public left tiles.
Each primitive component is evaluated once. After source completion, each public
quartet enumerates the actual normalized AoBasis polynomial term products
(at most 3^4), gathers cached fields, and invokes compiler-owned stationary maps
and the shared weighted integral contraction. Radial coefficients and public
component/spherical factors are each applied once.

Four sequential incidence reductions own one Fock cell per spin/thread; a
physical-atom-axis owner sums all coincident mathematical centers. Energy and
vectors have fixed-order reductions, without floating atomics. Independent
primitive, projection-term and reduction-term limits accompany arena admission.
Projection work does not grow with primitive streaming chunks.

The live KS adapter borrows the actual accepted D and normalized grid basis.
Integer shell topology comes from the actual prepared Fock System. Shared
physical-model admission validates the same complete WB97M-V model for induced
and nuclear response. A DF owner can request only the exact LR correction;
replacing fitted primary J/K by exact integrals is explicitly rejected.

## Invariants and boundaries

- Actions allocate nothing, never switch devices, execute Python, JIT, contract
  numerical data on CPU, or mutate borrowed provider scratch.
- Signed/indefinite symmetric D/dD are allowed. Inputs and all private outputs
  have finite/symmetry gates; arenas/inputs/outputs cannot overlap writable data.
- Original stream/device and live generation bindings are authenticated before
  execution and again before joint five-output publication. Prepublication
  failures leave public energy/Fock/Fock-direction/gradient/HVP unchanged.
- Coincident physical atoms do not eliminate mathematical integral centers.
- Cache capacity is mandatory bounded scratch, not an optional global ERI fast
  path. Insufficient resources fail explicitly rather than repeat source work.
- This is not fitted primary response, residual ECP response, stationary
  Hessian assembly or a migrated production CLI. No endpoint speed follows.

## Rejected alternatives

Python orchestration leaves a runtime interpreter dependency. Handwritten native
second derivatives or HF spin factors duplicate compiler authority. A global
ERI/derivative tensor violates molecular memory bounds. Re-evaluating sources
per public quartet/left axis has unacceptable semantic work amplification.
Treating D response as Fock nuclear motion double counts later CPKS relaxation.
Using exact J/K on a DF owner silently changes its physical approximation.

## Evidence

Independent signed-density equations test every emitted quartet map as compiled
C++, including distinct/repeated/permuted incidences and invalid selectors or
nonfinite inputs. CUDA-free injected registry tests cover the largest through-f
cache, complete nameless closure, overlap/missing/malformed programs, work/arena
limits and shape overflow.

Final source-matched qualification is Slurm **3124**, node2/main/PRO6000,
preserving scheduler-assigned visibility **0**, with terminal success:

Independent oracle versions are PySCF 2.14.0/libcint, NumPy 2.4.6 and Python
3.13.9. They run only in the test driver, not in the native numerical process.

- **43 tests pass**: all six radial/method parts with RKS/UKS and both public
  representations on s/p fixtures; displaced libcint value/direction/gradient/
  mixed-response oracles; two finite-difference step sizes; tile 1/3/64 and
  rigid-translation invariance; malformed basis; actual Direct/DF live owners;
  independent induced-Fock/CPKS regression and build-only generation admission.
- **24 sanitizer runs pass**: memcheck/initcheck/racecheck/synccheck across two
  standalone spin/representation fixtures and four actual KS Direct/DF modes.
  All terminal summaries are checked; racecheck has zero errors and warnings.
- Standalone native actions also pass with PATH, PYTHONHOME and PYTHONPATH
  unavailable. The qualified probes/library have no libpython link.
- Synthetic energy/Fock gates are 3e-12 absolute/relative; frozen-D first Fock
  direction uses 5e-11 absolute/3e-9 relative; weighted gradient uses 3e-10
  absolute/2e-9 relative; mixed response uses 3e-8 absolute/2e-7 relative against
  fourth-order displaced values. Actual owner Fock/vector comparisons have
  separate 2e-11 and 2e-7 gates. No finite difference executes in production.
- **101 local tests pass**, four opt-in GPU cases skip, in the focused compiler/
  plan/AOT suite. A separate local source suite passes 40 tests and explicitly
  skips 36 Slurm-only cases. The linked nameless through-f host closure/work/
  overflow/arena gates pass without initializing CUDA.
- Pinned ty 0.0.82, 559 compiler-module and 249 shared-SCF-module dependency
  checks, 383-file CUDA ownership admission, and all-files pre-commit pass.
  Pre-existing structural warnings in unrelated SCF files are unchanged.
- Release strict-FP64 sm_120 builds use CUDA 12.9.1 and verified ccache 4.5.1,
  with actual launcher commands and before/after shared-cache stats retained.

Each standalone tile-64 exact action has 32 ordered groups, gathers 1,602 radial
records across component tiles, evaluates 13,122 primitive components, and admits
41,472 public polynomial-term products. RKS/Cartesian uses 67,584 arena bytes
and 29,746 reduction terms; UKS/spherical uses 67,840 bytes and 33,842 reduction
terms. Projection and source evaluation counts are invariant to radial tile
size; only radial accumulation reduction work changes. These are component
work/storage receipts, **not endpoint timings**. Since s/p public spherical
polynomials coincide with Cartesian ones, this does not qualify nontrivial d/f
spherical ERI transforms or numerical through-f closure.

Qualified binary SHA256 values:

```text
aa685c170118fd5d7e18d5e0d5b13184ff45bf08e95bd546d0257b5f432e6759  generativeqc_cuda_wb97mv_nuclear_jk_probe
197a0ee18a39abcd5556bd737ba60fc4c5813f511a12f5199d50b77ed3b81344  generativeqc_ks_wb97mv_grid_probe
f58f48b9a87c4684c301bb12cc98465ed512a3a67f4f73d87cff785c374cfd7c  libgenerativeqc.so.0.1.0
```

Ignored receipts, source patch, compiler/cache/linkage logs, fixtures and the
reproducer script are under `.artifacts/wb97mv-hessian-aot-20261015/jk-*`.
The retained compute stage is
`n2:/data/jzzeng/wb97mv-hessian-native-response-20261015/nuclear-jk-20261016`.
Initial Slurm 3123 passed the 36 new native cases and two existing map tests,
then stopped because a build-only generator was missing from the staged tools;
3124 reruns the complete corrected harness and all sanitizer phases. No code or
numerical acceptance gate was weakened for that staging repair.

## Remaining endpoint work

Full DF metric/factorization second response, residual ECP/refinement, practical
full ERI/auxiliary spherical-g numerical closure, stationary Hessian/HVP assembly
and production CLI replacement remain required. The CLI still invokes its
Python/CuPy worker, PR #2527 remains draft, and no native/GPU4PySCF endpoint speed
claim is justified by this source-component qualification.

## Revisit when

The compiler can emit a reusable source schedule that removes duplicate radial
record construction between component tiles, or symmetry-related ordered leaves
can share exact derivative permutations without weakening independent closure
qualification. Full through-f ERI packaging still needs practical qualification;
host admission alone is not numerical angular coverage.
