# Decision: share one-electron second-integral programs across orientations and charges

Status: implemented
Date: 2026-11-11

## Problem

The negative Ni HVP experiments in
`../../proposed/2026-11-11-ni-uks-hvp-first-integral-work.md` continued compiling
component/coordinate-specific one-electron second-order sources after the coupled
response finished. The shared four-center ERI factor jets do not eliminate S/T/V
compilation. Retuning GMRES alone does not address this source of amplification.

## Decision

For CPU fixed-weight second-integral HVPs, visit unordered shell pairs in
angular-momentum order. S/T/V and their physical nuclear derivatives are invariant
under swapping the two complete Gaussian slots. Fold distinct ordered cotangents
as `W_ab + W_ba.T`; do not assume symmetric weights inside this schedule and do
not double or average diagonal shell blocks. The public pair-weight API keeps its
existing symmetric-matrix admission contract.

For an admitted pair, nuclear attraction uses the existing compiler-owned
unit-charge IR. Multiply the external cotangent by the fixed physical charge;
do not specialize the generated expression separately for each element. This is
linear constant-weight contraction, not differentiation of nuclear charges or a
new recurrence. Mathematical-center mapping, normalized Cartesian weights and
the existing coordinate/component tiling remain unchanged.

Account conservatively for three shell-weight buffers plus the output accumulator
before taking this optional schedule. If that workspace cannot fit, retain both
ordered orientations and charge-specialized programs. Keep the existing bounded
primitive/public-stream resource plans. Release each charge-scaled array before
moving to another nucleus and each pair view before advancing the generator;
no cross-pair tensor cache is retained. CUDA emission/execution remains unchanged.

The source diagnostics count ordered weight blocks, canonical shell pairs and
unit-charge nuclear shell pairs. Counts include every invoked S/T/V family and
must not be confused with SCF iterations or complete endpoint work.

## Validation and boundaries

`test_one_electron_hvp_reuse.py` checks arbitrary ordered cotangents against
independent symmetric tensors, unequal s/p/d spaces, diagonal/repeated shells,
mixed-capacity fallback, the unchanged pure-host CUDA schedule and finite-input
overflow rejection. Its compiled CPU cases compare spherical mixed-charge S/T/V
HVPs with multistep differences of independent Libcint gradients, preserve
translation and direction linearity, and replay successfully after rejected
nonsymmetric public weights. The initial group passes all six tests.

These are source/schedule checks, not a returned complete Ni molecular HVP,
installed raw Hessian, mixed scalar-ECP qualification or molecular TS receipt.
The Ni source-boundary audit and full coupled/public regressions retain their
own source hashes, native binary identity, logs and finite time limits under
`/tmp/gqc-ni-one-electron-hvp-reuse*` and
`/tmp/gqc-one-electron-reuse-*`. No paired endpoint speedup is asserted.

The isolated canonical NiH3 source audit completes on sealed numerical sources
with the master-merged native binary
`cca47c31b6031dd9c5d3673847a70094003f5fd1407e6e980df3b7a71adbb49b`.
Using the previously state-matched fixed cotangents, all three independent
gradient-difference steps are retained. The one-electron HVP error is
`1.0808690054098236e-10`, its step spread `9.453179350416008e-10`, and its raw
translation error `8.743006318923108e-16`. The overlap/Pulay errors are
`1.44251104117199e-12`, `1.4442513157630898e-11` and
`8.456776945386935e-18`, respectively. No displaced SCF or reference work enters
the generated source provider.

The source processes take 636.985238 seconds for T/V and 13.541685 seconds for
S, including their actual artifact work; the complete source-audit process takes
651.55 seconds and peak RSS 1,221,184 KiB. These are not complete SCF/response/
molecular-HVP timings. T/V covers 800 ordered blocks with 420 canonical shell
pairs and 210 unit-charge pairs, using 94 distinct program identities (14 kinetic,
80 attraction), 2,427 executions and 10,431 primitive records. S covers 400
ordered blocks with 210 canonical pairs, 14 programs, 165 executions and 591
records. The returned provider-phase host peaks are 11,736 and 6,672 bytes;
these exclude compiler processes and are not process-RSS claims.

The master-merged complete regression group passes **85 tests**, with only the
two existing JUnit `record_property` warnings, in **871.13 seconds**. It includes
both Cartesian/spherical independent coupled UKS Hessians, public PBE0/B3LYP
Hessians, failure recovery/lifetimes, point-panel and spin-gap tests, backend
admission and one-center bounded-source checks. Production numerical source and
native binary hashes are unchanged across the run. The new test subsequently
adds only an explicit non-null spherical transform assertion and is replayed
separately; do not restamp the original test-source hash. Configured compiler
type checking, compiler ownership checks and warnings-as-errors Sphinx pass.

## Rejected alternatives

- Increasing component tiles without a liveness/compiler-cost qualification can
  replace many small programs with one pathological compilation.
- A handwritten one-electron Hessian recurrence duplicates scientific algebra;
  the current change only reuses canonical compiler programs.
- Assuming a symmetric internal cotangent, averaging diagonal shell blocks or
  adding a factor of two for every pair can change signed contractions.
- Changing density floors, discretized grids, scientific acceptance gates or
  substituting displaced SCF/reference work is not a performance fix.
