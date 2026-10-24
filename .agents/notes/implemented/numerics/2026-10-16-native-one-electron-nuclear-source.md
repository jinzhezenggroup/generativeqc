# Decision: native normalized one-electron nuclear source

Status: implemented component; complete Python-free Hessian remains incomplete
Date: 2026-10-16

## Problem

Linked primitive value/gradient/gradient-JVP leaves did not provide a native
molecular source. A Python-side shell schedule or numerical primitive-record
builder would preserve an interpreter requirement. A moving nuclear source also
needs public spherical projection, physical-atom scatter, externally supplied
stationary weights and their response, and live-owner publication gates.

## Decision

Keep the existing compiler integral AD as the sole derivative authority. Extend
the immutable AOT descriptor with scientific family/angular/radial selectors,
independent of program-name spelling. Build-time closure tooling replaces only
selected integral families and preserves scalar/AO and unselected entries.
Native admission requires exact, non-overlapping ordered Cartesian component
coverage before any CUDA execution; it never requests runtime compilation.

The caller-arena `CudaOneElectronResponse` owns bounded integer scheduling only.
CUDA gathers radial records directly from actual normalized AoBasis storage,
executes the linked AD leaves, projects into public matrices, contracts the
compiler-AD weighted envelope and deterministically scatters physical atoms.
The source returns S and T+effective-point-charge attraction values/directions,
plus weighted nuclear gradient/HVP vectors. Weights and their right response are
generic signed public matrices; the method supplies D/W and spin/HF policy.

The scalar weighted envelope is derived with the existing Graph algebra:
left weight motion is zero, right weight motion remains live, and projection is
geometry-independent. Native code does not introduce a second derivative rule.
Mathematical centers are retained until scatter, even when two or three map to
the same physical atom. Axis-owned reductions use fixed pair/center ordering
instead of FP64 atomics.

`CudaKsOneElectronResponse` borrows the actual canonical native KS grid/Fock
owners. Integer shell topology and ionic charges come from the prepared Fock
owner's normalized System; numerical basis records come from the grid lease.
The adapter owns separate arena/charge storage and reborrows both bindings
before execution/publication. Failed preparation clears the previous source.

## Invariants and boundaries

- AoBasis primitive coefficients contain radial normalization; AO polynomial
  terms already contain Cartesian component normalization and spherical
  expansion. Apply polynomial coefficients exactly once, not in both primitive
  scale and public projection. Multi-component tiles cannot use a single common
  Cartesian normalization scale.
- Effective nuclear charges use `Atom::ionic_charge()`, including ECP core
  removal. **Local/nonlocal residual ECP projectors are not supplied.** Comparing
  this partial core with full ECP Hcore would be incorrect.
- Actions allocate nothing, change no device, mutate no provider scratch and
  execute no Python, runtime compiler, CPU numerical contraction or production
  finite differences. Input packing/preparation is not claimed to be device-only.
- Radial tiles are bounded to 1..64. Separate admission limits cover jobs,
  primitive-component work, conservative public projection terms and arena bytes.
  Primitive records count repeated record gathering across component tiles;
  component work counts every primitive-component evaluation in successful
  actions; failed actions retain conservative successfully dispatched work.
  Projection work bounds nine polynomial products per public
  pair/component/radial tile.
- All four matrix panels and two atom vectors remain private until finite,
  symmetry, stream/device and live-owner gates pass. Every queued failure path
  drains the original stream. Prepublication failure leaves all six outputs
  unchanged; a CUDA publication failure invalidates all six.
- The development inventory remains explicitly partial. Complete S/T/V closure
  through f has 34 tiles per family, 400 raw ordered components per family and
  131 programs when combined with retained development entries. This is not a
  complete ERI/DF/ECP package or stationary Hessian capability claim.

## Rejected alternatives

Python primitive construction or source orchestration violates the final runtime
contract. Native handwritten second derivatives would duplicate compiler AD.
Normalizing raw Cartesian leaves before a second spherical projection applies
component factors twice. Inferring shell/center symmetry from names risks losing
ordered derivatives and coincident-center contributions. A molecular derivative
tensor would violate the bounded, early-contraction design. Floating atomic
scatter was unnecessary for this small-center deterministic source schedule.

## Qualification

Independent libcint/PySCF displaced *values* qualify all ordered s/p/d/f public
matrices and their directions, signed weighted gradient/HVP and ionic core
removal in Cartesian and real-spherical representations. The probe constructs
the actual C++ normalized System/AoBasis, not Python-reconstructed AO records.
Fourth-order value differences are test oracles only. Tiles 1/3/64, rigid
translation, malformed packing, every output-pair alias, input/arena aliases,
nonfinite/asymmetric weights, capture, stale/late owner revocation and failed
repreparation have behavioral gates. A separate CUDA-free linked-registry test
injects missing, overlapping and malformed nameless closures.

Initial standalone qualification: Slurm 3121, node2/main/PRO6000, assigned
visibility 0, 63 tests passed. Final integrated qualification uses Slurm **3122**
on the same node/partition/GPU, preserves assigned visibility **0**, and exits
successfully after every phase:

- **67 tests pass**, including all four actual RKS/UKS Direct/DF KS owner modes.
- **24 sanitizer runs pass**: memcheck/initcheck/racecheck/synccheck across both
  synthetic public representations and all four actual molecular owner modes.
  All 24 terminal receipts are checked; racecheck has zero errors and warnings.
- Both standalone representations run with PATH, PYTHONHOME and PYTHONPATH set
  to nonexistent locations. Qualified probes/library have no libpython link.
- Independent oracle: PySCF **2.14.0**/libcint, NumPy **2.4.6**, Python **3.13.9**.
  Matrix value gates are 2e-11 absolute/relative; first directions have 3e-10
  absolute and 2e-9 relative gates. Weighted HVP gates are 3e-7 absolute/2e-7
  relative against fourth-order displaced values, with two step sizes, tighter
  weighted-gradient gates and independent translation/tile invariance checks.
- **36 local tests pass**, 17 opt-in tests skip. The separately selected
  CUDA-free host admission test passes; linked nameless/missing/overlapping/
  malformed/omega closure gates also pass without CUDA initialization.
- Pinned ty **0.0.82**, **558** compiler-module and **249** SCF-module dependency
  checks, **379**-file CUDA ownership admission and all-files pre-commit pass.
  Existing structural-lint warnings in unrelated SCF files remain unchanged.
- CUDA 12.9.1 strict-FP64 sm_120 builds use verified **ccache 4.5.1**; launcher
  commands and before/after global cache statistics are retained.

Each synthetic tile-64 action dispatches 170 jobs, gathers 765 radial records
across component tiles and evaluates 8,405 primitive components. Cartesian uses
125,440 arena bytes and a 959,220 projection-term bound; spherical uses 120,832
bytes and a 544,500 bound. These are source work/storage receipts, **not timings**.

Final qualified binary SHA256 values:

```text
0995be2bc9ab7186392ba968e056866b6b328b51f23823299d243e5ce0786997  generativeqc_cuda_one_electron_probe
04613a4b10c13eb142343c6af6dad7e0f7032673212d773e563f8802ad7067ed  generativeqc_ks_wb97mv_grid_probe
75e3a66d85a614374276e0561c11971fa72191e821510885db8b90a7dd613b15  generativeqc_hessian_aot_probe
a058e78984a428c401df1ba6c5bf421533dcd8f58b2e0fe775c8589f0b9f1ae8  libgenerativeqc.so.0.1.0
```

Ignored source-matched local receipts, compiler/cache logs, source patch and
reproducer script are under `.artifacts/wb97mv-hessian-aot-20261015/one-electron-*`.
The retained remote stage is
`n2:/data/jzzeng/wb97mv-hessian-native-response-20261015/one-electron-20261016`.
No release/tag/assets publishing is used for evidence retention.

## Remaining endpoint work

Full/LR/SR J/K nuclear sources, full DF metric/factorization second response,
residual ECP/refinement and auxiliary spherical-g closure, stationary Hessian/HVP
assembly, and replacement of the production CLI Python worker remain required.
No complete-endpoint native/GPU4PySCF timing or speed claim follows from these
source-component tests. PR #2527 remains draft until that contract is delivered.

## Revisit when

Compiler-generated multi-source schedules can reuse geometry/radial work without
changing exact closure, normalization, semantic work admission or transactional
publication. Any faster resource-dependent schedule must retain a bounded
fallback and independent complete-consumer qualification.
