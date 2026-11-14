# Decision: retain signed atomic quadrature measures in mixed geometry

Status: implemented
Date: 2026-11-12

## Problem and evidence

The first complete default-grid frozen-XC curvature control refused in the mixed
geometry reader with `invalid mixed grid atomic measure`. Small geometry tiles
and grid versions 1--3 had passed. Some Lebedev angular rules used by the retained
default grid contain legitimate negative atomic weights; those are not a negative
partition or a malformed quadrature. The failed full-grid control is retained in
`build/qualification/host-xc-curvature-20261112/initial.log`.

## Decision and invariants

Atomic measures remain signed, finite, geometry-independent inputs. Zero atomic
measure still requires zero retained weight. The normalized dimensionless primal
partition, compared with retained weight divided by its actual nonzero measure,
remains the identity gate. Multiplication by the signed measure occurs only after
partition AD. No clipping, absolute value, quadrature replacement, functional
change or derivative-tolerance relaxation is introduced.

Full-grid frozen-XC controls for both spins and a direct signed-measure mixed
geometry control protect this boundary. A finite small tile does not qualify
every angular shell of an element-dependent production prescription.

## Consequences and revisit conditions

The earlier mixed-geometry component receipt is a historical source cohort whose
sampled tiles did not expose this admission bug, not complete default-grid source
qualification. Later current-source evidence must retain the signed-grid control.
Revisit only when the authoritative quadrature prescription itself changes;
generic geometric response must not impose positivity on signed quadrature rules.
