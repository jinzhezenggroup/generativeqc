# Decision: generation-bound occupied projections for CUDA KS XC

Status: implemented, opt-in; not a promoted performance default
Date: 2026-10-17

## Problem

Source-matched Direct r²SCAN-3c/96 cold profiles put roughly 101 seconds of
device time in SCF Direct J and 95 seconds in XC/AO/grid. Warm solves largely
skip SCF iteration work, so their dominant stationary order-6/7 derivatives
are a separate optimization problem. Repeating page-width or candidate-only
scheduling experiments cannot address the cold density-feature arithmetic.

An earlier MD source-fusion population improved complete cold/warm endpoints
by only 1.049%/0.477%; that separate experiment is not part of this change.
The retained XC experiment instead exploits an exact density factor already
available to the native KS owner, without changing the Direct J/K contract.

## Decision

After accepting a structured density proposal, borrow the exact AO canonical
frame still in `tmp1` for the next density generation. Consume this same-stream
lease in XC before residual/DIIS scratch overwrites the frame. Bind the view to
the density pointer, generation, AO/spin dimensions and integer occupations.
Beginning a solve clears it. Imported, held and initial warm densities cannot
manufacture a witness; an accepted later proposal can publish a fresh lease.

The XC owner additionally requires strict FP64 density contractions, FP64 AO
jets, four work jets and five feature terms. Invalid or unsupported views use
the existing matrix path. An otherwise valid frame must be device-resident and
must not alias density or XC scratch. The identity witness is an internal owner
contract, not numerical authentication of arbitrary caller-supplied orbitals.

For a tile with actual selected AO count `a` greater than occupied rank `r`,
project all four orbital jets `C_occ^T A`. Reuse the existing generated tiled
product panels and scalar rho/gradient/tau feature algebra. Logical projection
products decrease from `4*p*a*a` to `4*p*a*r`, where `p` is the tile point count.
This is an algorithmic expression, not a measured GPU instruction/FLOP count.
Other tiles keep the dense path. RKS scales quadratic orbital features by 2;
UKS uses separate unit-occupation frames, including unequal and empty spins.
Virtual columns are never needed and may contain NaNs in the independent tests.

The original four AO jets remain available to the unchanged Vxc contraction.
Reuse dead density-work panels for the smaller orbital panels: no extra device
allocation, global atomics or synchronization is introduced. No functional,
molecule, AO-count or GPU-model selector chooses the scientific algorithm.

Enable only with `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1`. The default remains off.
Nonlocal, mixed-precision, response, device-chunk and unsupported feature paths
retain their existing bounded fallbacks. Existing screening, moving grids,
convergence gates, final-state validation, forces and exact Direct J/K remain.

## Rejected alternatives and resource boundary

- Do not substitute the final rediagonalized F[D] frame for the exact factor
  that formed retained D. Those are not interchangeable density identities.
- Do not cache a witness across imported/warm calls or through residual scratch
  reuse. A coefficient pointer alone says nothing about density provenance.
- C1 used a runtime occupied branch in the common batch kernel. Its inactive
  FP32 matrix arm grew from 3200 to 7552 bytes shared memory and from zero to
  56 bytes stack; the structural CTA ceiling fell from six to five. Complete
  cold gains do not justify this resource coupling. C1 is retained as negative
  evidence, not a production candidate.
- C2 emits separate `bool Occupied` batch specializations. Linked FP64/FP32
  matrix kernels use 36/38 registers, zero stack and 5376/3200 bytes shared,
  restoring the six-CTA structural bound. The occupied FP64 arm uses 46
  registers, 56 bytes stack, 5376 bytes shared and a five-CTA bound. These are
  static cuobjdump facts, not measured occupancy. Zero LOCAL is not a claim
  that stack storage has no local-memory consequences.
- Resource admission covers both batch sources and the standalone occupied
  allocation/map fallback, including noncompact plans. Missing specialized
  evidence fails closed; inactive FP32 facts cannot inflate a strict region.
  The standalone product uses its actual scheduled block width in the bound.

## Qualification and source honesty

The matched clean C2 campaign freezes master
`b8102246862f68603adae0394dd41f2205d87425`, without the unmerged #2558 force
overlay. Node4 job1099 runs all eight method/size/arm populations; node5
job1695 reverses the 96-atom arm order and then profiles each frozen library.
Every clean population keeps cold, five warm, moved and five moved-warm calls,
the original 1800-second cap, canonical basis, strict FP64, moving
48×16×32-per-atom Becke grid, screening and convergence. PBE0/HF retain their
existing def2-SVP controls. No iteration normalization or timing subtraction.

Primary 96 cold improves 296.461741→275.539082 seconds (-7.0575%) with
identical 24/24 SCF iteration/Fock counts. Moved improves
205.431849→185.186882 seconds but changes 14/14→13/13; do not attribute the
whole gain to a kernel. Warm and moved-warm change +0.0500%/-0.0810%.
The reverse repeat has cold 306.917294→276.879606 seconds (-9.7869%), with
25/25→24/24 work, and moved 198.229687→186.560340 seconds, both 13/13.
Its warm changes +0.0606%/+0.1465% are not a warm optimization.

At 48 atoms cold/moved improve 73.381158→65.852037 and
55.105953→50.175575 seconds, with unchanged 19/19 and 13/13 work.
PBE0 cold changes +0.2550%, not the earlier 3.06% regression; HF cold changes
+0.0163%. Full untrimmed observations, independent gates, actual work and
source/library/Slurm identities live in the linked evidence packet.

Master subsequently advances through #2574 (guarded FP32 Boys) and #2573
(native KS host resource ownership) to
`a67846c22d415c889e63d9530e15b6da9e4cbfe3`. The separate integrated build
passes 157 host tests against its actual library and, in finite node1 job9379,
all eight native cases, 24 sanitizer runs and 24 complete 3/12-atom E+F calls.
All small-case calls pass the frozen independent homepage-reference gates;
maximum energy/force errors are 2.78533e-12Eh/2.50964e-11Eh/Bohr. These are
compatibility/accuracy checks, not new matched performance comparisons or a
reason to replace the already-qualified homepage observations from #2572.
Own pending node4 job1101 was canceled without executing a population after
its resource estimate moved three hours later; no foreign/running job was
canceled. The seven delivered compiler/native implementation files exactly
match this integrated source. Its four native files equal clang-format23.1.1
of the frozen C2 sources; two differ only in whitespace. The one-off comparison
is provenance, not a required-CI production-source fingerprint. Never relabel
the b810224 timing populations as measurements of a67846.

## Tests, evidence and unsuccessful qualification attempts

Executable tests compare generated features to independently formed dense
density matrices and cover RKS/UKS, rank tails, maps, partial CTAs, empty spins
and NaN virtual columns. Native owner tests compare XC energy and potential
against the dense path and inject stale generation, density/shape/occupation
mismatches and alias failures. Every invalid identity must fall back or reject
without poisoning the next valid generation. Node4 C2 qualification passes all
eight native cases and 24 full-leak memcheck/synccheck/racecheck invocations.
The final combined host/evidence/homepage/native-resource suite passes all
188 cases, and every configured pre-commit hook passes.

Retain failures rather than quietly changing admission or scientific gates:
an early compact fixture requested an unsupported 17-AO/13-point-tail shape;
the corrected fixture uses two full 32-point tiles and 65-AO/rank-33 domains
to exercise both factor and dense fallback in one group. An omitted SONAME
alias, ccache libhiredis path, CUDA void-pointer cast, and a host-test invocation
without `GENERATIVEQC_LIBRARY` each failed before scientific execution. Their
fixed reruns preserve source/library provenance. Node2's driver mismatch is
not an optimization result; do not retry it unchanged. Three historical
diagonal-panel API tests also fail on the frozen baseline; unrelated coverage
is not removed or weakened.

Selected C2 96/48 populations and all retained profiles preserve returned FP64
forces losslessly. Other cohorts are explicitly collection-time independent
force-gate receipts, with original JSON/force-stream hashes and complete scalar
samples; do not claim offline force replay for them. The evidence reader checks
independent energy/force gates, convergence/work, source/protocol/library
identity, partial populations and mutually exclusive timing partitions.
Kernel device durations remain separate from exclusive wall-time accounting.
Unavailable recurrence counters remain null, and `ERR_NVGPUCTRPERM` precludes
hardware instruction/FLOP claims. No evidence is discarded to fit a speedup
story, and the aggregate 64-MiB storage cap is unchanged.

## Consequences and revisit conditions

This is a cold SCF/XC optimization, not the original ≥10% stationary warm-force
target. Warm solves have no accepted-proposal lease before their first XC
submission and retain the dense fallback. Mainline warm remains roughly
82–83 seconds versus the historical, differently sourced GPU4PySCF 26.59
seconds. The remaining large masses are SCF Direct J and stationary order-6/7
J′, not order-8 task enumeration. Do not mix the separate #2558 composition's
roughly 43-second warm endpoint with this mainline comparison.

Revisit default admission only with enough incumbent-route/resource evidence
and explicit review. Revisit new force work through recurrence-DAG/primitive
reuse and shared contractions, measuring actual executed work and complete
endpoints instead of page widths or candidate counts.

## References

- `benchmarks/results/occupied-xc-20261017/README.md` and its frozen packet.
- #1117, #2486, #2538, #2558 and #2572.
- Ignored raw build/cache/native/profile evidence under
  `.artifacts/md-source-reuse/`; original frozen source/runtime arms under
  `/data/jzzeng/gqc-md-source-reuse-20261017/`.
