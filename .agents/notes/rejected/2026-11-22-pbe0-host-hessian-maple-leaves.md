# Decision: do not admit CPU PBE0 from generic Hessian scalar leaves

Status: rejected
Date: 2026-11-22

## Problem

The linked CPU B3LYP Hessian/HVP owner cannot admit a different electronic
model just because it can prepare that model for SCF. An experiment based on
`aff6ad925` added PBE0 RKS scalar leaves and keyed the integral contractions
and their work estimates by exact MethodIR. That package linked but did not
produce a complete molecular HVP.

## Evidence

With CPU FP64, spherical STO-3G, a 12/4/8 product grid, and H atoms at
`(0, 0, 0)` and `(.15, .13, 1.5)` Bohr, the direction
`[[.02, -.03, .1], [-.02, .03, -.1]]` failed in `HostGgaNuclearXc` with
`linked host nuclear GGA field response failed`. The SCF state and physical
spin-field checks had passed. The generated PBE expressions form inverse
powers in the tails; using the production compiler flag does not replace
those expressions with the scaled physical PBE SCF evaluator. The exact
failing lane and first invalid intermediate have not been isolated.

A separate one-lane control used a total density of `1e-48`, Cartesian
gradient `rho * [-12, 3, 2]`, and a direction of
`rho * [0.125, -1, 0.5, -0.25]` in `(rho, gx, gy, gz)`, with unit weight
and zero weight direction. The generated leaf rejected its fifth output
(the density coefficient direction). The shared scaled PBE0 point and
direction evaluators both accepted this input and returned finite
coefficients. At total density `1e-120`, the generated leaf also rejected the
primal density coefficient. These are diagnostic controls, not molecular
qualification or a proposed lower density bound.

## Revisit when

Use the shared scaled PBE point and direction evaluators (`src/dft/xc_point.hpp`
and `src/dft/xc_point_response.hpp`) without a new density floor or copied
equations. Align total-density versus per-spin feature conventions in all
three consumers: moving-grid CPKS RHS, frozen curvature/pullback, and fixed-grid
induced response. Admit only after independent same-grid full-molecule and
tail checks, bounded work/admission, and bare versus corrected parity. The
established Python PBE0 route is unchanged; the linked CPU aggregate still
admits canonical B3LYP only.

Refs #2227 (R3/R6).

The [implemented opt-in route](../implemented/numerics/2026-11-22-scaled-pbe0-cpu-hessian.md)
uses the scaled physical PBE evaluators instead of these leaves.
