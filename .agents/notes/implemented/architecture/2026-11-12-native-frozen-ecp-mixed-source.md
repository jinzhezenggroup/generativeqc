# Decision: contract analytic residual-ECP mixed sources before coordinate scatter

Status: implemented (development component; not a stationary endpoint)
Date: 2026-11-12

## Problem

First nuclear Fock directions do not supply frozen nuclear energy curvature.
The existing analytic ECP weighted-second API retains a global coordinate
Hessian. Wrapping it to obtain one bilinear contraction repeats that storage
cost and obscures the bounded source schedule needed by a native Hessian owner.

## Decision

Add a separate mandatory-cap analytic mixed source over the same normalized
order-two pair jets and primitive/frontier authority. Consume each shell-pair
group once and immediately apply a compiler Graph projection to arbitrary
ordered AO weights and two nuclear directions. Preserve independent A/C and
B/C mathematical slots until this projection, even when physical owners repeat.
Accumulate four scalars per residual part using the existing compensated sum.
Keep local and nonlocal parts separate; do not allow cancellation between them
to conceal nonfinite arithmetic. Query complete order-two kernel-block work
before execution and retain only eight scalar outputs. Authenticate the actual
immutable owner during traversal and again before returning.

## Rejected alternatives

- Forming the existing coordinate Hessian and multiplying afterward keeps an
  avoidable coordinate-squared output and is not a bounded mixed consumer.
- Native handwritten bilinear projection duplicates scientific algebra owned
  by the compiler. The new native layer only binds owners and accumulates the
  compiler-produced scalar outputs.
- Displaced production operators, a reference engine, or an unchecked general
  quadrature would change derivative semantics or dependencies. Unsupported
  analytic operators remain explicit refusals here.
- A zero final weight or direction must not bypass the underlying source's
  numerical checks. Complete source work remains scheduled for these inputs.

## Invariants and boundary

The linked analytic specialization remains power two with local/s/p/d residuals
and normalized Cartesian/spherical s/p/d/f AOs. Ordered weights may be signed
or nonsymmetric; there is no implicit occupancy factor. Direction arrays use
physical atom/xyz order. Neither ionic point attraction nor moving D/W belongs
to this frozen residual source. General checked-grid second sources,
stationary assembly, current-source installed qualification and a native TS
endpoint remain separate work. Synthetic primitive controls and basis-free
25/40-atom storage controls are not representative Ni/I/Cs or complete
electronic endpoint performance claims.

## Validation

`tests/python/test_native_ecp_mixed.py` compares eight Cartesian/spherical
s/p/d/f controls against independently normalized libcint primal residuals at
three four-corner displacement steps. It also exercises ordered weights,
bilinearity, exact translation, caps, late/final owner failures, malformed
frames, transactional publication and recovery. The existing Python-free
native ECP smoke consumer also executes nonzero analytic mixed curvature.
Source-matched results belong in
`build/qualification/host-ecp-mixed-20261112/`. The eight independent primal
controls pass with maximum final left/right/mixed errors approximately
`1.04e-11`, `1.89e-11`, `8.09e-11`. Broader qualification is recorded only by
actual command exits and a source-matched receipt, not inferred from those
component errors.

The first Cartesian f control passed its numerical parity gate but failed the
smallest-step convergence-floor check at `2e-4`. A separate seven-step sweep
with the native result held fixed resolved the cause: independent libcint
four-corner mixed differences lose accuracy on refinement. Its maximum error
is approximately `8.1e-11` at `2e-3`, `6.7e-9` at `2e-4`, and `9.8e-8` at
`5e-5`, while first partials remain resolved. Retain that failed run and all
seven primal-corner observations. The permanent three-step control uses
`2e-2`, `6e-3`, `2e-3`, with unchanged `atol=3e-8`, `rtol=2e-6` and convergence
floor. Do not claim quadratic mixed convergence below the reference's
subtraction-noise floor or weaken the native source's admission policy.

## Revisit when

The subsequent [checked curvature-column decision](2026-11-12-native-checked-ecp-curvature-columns.md)
adds structural general-grid dispatch with raw physical second-order gates and
whole-column reuse. It does not change this specialized scalar source's domain
or retroactively qualify a stationary endpoint.

A shared, independently validated general-grid order-two source can join this
consumer without changing its operator, quadrature-admission or publication
contract. Complete electronic work/time measurements, not scalar storage
bounds alone, are required before promoting a larger molecular endpoint.
