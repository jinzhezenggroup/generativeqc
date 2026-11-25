# Decision: Register overlap centres without copying the trial system

Status: implemented
Date: 2026-11-24

Mass-centroid registration changes only the coordinates in the occupied-overlap
diagnostic. Reconstructing a normalized `System` at the shifted centres needlessly
copied basis and ECP records and left another system owner to charge during that
phase. Borrow the existing trial and accepted systems and subtract the target
translation while constructing the overlap provider's already-budgeted centre array.

The raw positions, basis/representation, primitive coefficients, grid and bound
electronic state are unchanged. This preserves the previous subtraction order, does
not rotate the angular grid, and still rejects nonfinite shifted coordinates before
writing the metric. An explicit shifted-system comparator covers mixed spherical g /
Cartesian d overlaps and checks that the original target is unchanged.

The public no-offset overlap call remains on the same kernel. Storage for the
original trial and accepted systems, the optimizer and the other workspaces still has
to be composed before publishing a unified saddle host limit.
