# Decision: supplied right-weight response in native residual-ECP columns

Status: implemented
Date: 2026-11-12

## Problem

The checked residual-ECP column previously froze its ordered AO weights.
Stationary assembly also needs total D_right contracted with the left ECP
gradient. Replaying one first-derivative source for every left coordinate, or
performing a second complete column sweep solely for that response, discards
the existing source-driven reuse. The raw jets already contain every required
first derivative.

This extends [the checked-column design](2026-11-12-native-checked-ecp-curvature-columns.md)
and composes with the separately implemented integral D/W and XC D response
columns. It does not itself qualify a stationary molecular Hessian or TS.

## Decision

`checked_ecp_curvature` takes ground ordered weights, an explicit empty frozen
span or supplied finite ordered `weights_right`, and one nuclear right vector.
The left partial holds ground weights fixed. The canonical scalar Graph forms
the weighted value and left gradients, then differentiates those roots using
external integral-jet rules and the weight tangent. Both analytic Gaussian and
checked-grid providers consume that same generated closure and the same raw
integral jets. No new native product-rule implementation, occupancy factor,
primitive pass or molecular-left source replay is introduced.

In a KS consumer, the caller supplies total D and total D_right; UKS spin
blocks must be summed without another occupation factor. Arbitrary supplied
right weights are not certified SCF/CPKS responses. The right scalar includes
their contribution and need not equal the frozen-weight gradient dotted with
the nuclear direction. Arbitrary supplied responses do not imply stationary
Hessian symmetry. Ordered nonsymmetric weights remain legal, with the existing
off-diagonal pair folding preserved for both primal and moving weights.

## Ownership and bounded resources

The checked consumer snapshots ground weights, right weights (zeros for an
empty span), and the nuclear direction only after resource admission. It uses
those owned copies throughout the source traversal and authenticates all
external contents after the final immutable-system validation. An exception
returns no partial columns. The mandatory numeric query explicitly adds
`sizeof(double)*(2*nao^2+3*natoms)` to the existing provider bound; source work
ceilings are unchanged. This is a numeric-capacity bound, not RSS or time.
The raw Gaussian provider borrows these stable panels; its provider-only plan
does not double-count the checked consumer's snapshots.

The general grid still checks raw unweighted physical primal, gradient and
Hessian components before either weight or direction projection. Weight-zero,
response-zero and rigid directions cannot hide a failing second-order grid.
Both grids, scalar measures, coincidence incidences, FP64 compensated scatter,
analytic structural dispatch and empty-operator behavior are unchanged.

## Validation contract

Independent libcint displaced-primal controls use Cartesian/spherical f AOs
and both structural backends, with weights displaced only on the right leg.
Independent left-only primal gradients avoid the four-corner gradient's
right-frame bias. Keep the incumbent ECP gates: primal atol2e-11/rtol2e-10;
derivatives atol3e-8/rtol2e-6, eightfold convergence/5e-9 floor, and steps
1e-2/3e-3/1e-3. Retain all primal corners and source counts.

Frozen-plus-weight-only action, identical source work, explicit empty response,
malformed/null/nonfinite responses, all external input/output aliases, actual
final mutation of all three inputs and recovery are separate executable
controls. The test probe's validator authenticates only the system for column
calls, so external-input mutation refusal must come from the real native
snapshots rather than a duplicate test-side guard. Native CTests also execute
nonzero response columns with no Python involvement.

## Remaining work

Actual stationary S/F/D/W composition with every Hamiltonian, nuclear and D3
source, installed Python-free CPU TS admission, representative Ni/I/Cs electronic
controls, complete 25--40-atom resource/time gates and the requested saddle and
unscaled frequencies remain separate requirements. Synthetic residuals and
spectator-coordinate controls do not qualify those endpoints.
