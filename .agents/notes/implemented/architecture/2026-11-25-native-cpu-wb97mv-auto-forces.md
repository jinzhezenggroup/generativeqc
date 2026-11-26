# Decision: Use the prepared CPU force owner for automatic wB97M-V fitting

Status: implemented
Date: 2026-11-25

## Problem

All-electron CPU wB97M-V with `density_fitting="auto"` prepared the same
fitted Fock strategy as `density_fitting="cpu"`, but the force admission
check used the requested enum. The installed SDK and CLI could not request
the native derivative for that item, and Python routed it through the
stationary compatibility controller.

## Decision

Admit automatic fitting through the existing prepared CPU wB97M-V composite
gate when the selected strategy is CPU and the orbital system is all-electron.
The existing J/K strategy checks still require fitted full-range providers;
the range/nonlocal and final-state checks still authenticate the other
derivative sources. Neither selecting the long-range provider nor replaying
its frozen choice changes. The patch does not qualify other automatic-fitted
methods, scalar ECP, new grids, or a complete composite host budget.

Normalizing the requested enum globally would also change callers that use it
for context-specific inventory negotiation. The force gate is the narrower
boundary: admission changes without altering the model or resource request.

## Evidence

The installed force parity test exercises RKS/UKS, Cartesian/spherical, and
explicit exact/fitted long-range selection against independent PySCF energy
and analytic forces. SDK and CLI run without a Python executable on `PATH`;
the Python endpoint forbids its compatibility force controller. The separate
mixed-DF compatibility test opts into that controller explicitly so its
source bounds and independent finite-difference checks remain covered.

Refs #2227
