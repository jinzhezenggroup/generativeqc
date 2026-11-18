# Decision: admit WB97M-V Direct full-range forces to the density-first s/p/d schedule

Status: implemented
Date: 2026-11-17

## Problem

After the cold anchor, Direct tiles and the VV10 weight domain, an Nsight Systems
capture of one H100 water-96 WB97M-V cold + warm endpoint attributed 6.49 s to the
generic bounded full-range J/K force queue and 2.99 s to the long-range K force
queue: about 4.7 s per force evaluation. GPU4PySCF's whole J/K gradient took
about 0.3 s per evaluation. Global hybrids already use the qualified
density-first s/p/d force schedule (DensityReuse) by default, but
`qualified_spatial_force_default` excluded every range-separated or nonlocal owner,
and the RSH derivative consumer did not forward the qualification to the
full-range traversal it already splits off.

## Decision

The method owner admits a restricted strict Direct owner whose Fock strategy is
exact full-range J plus exact global-hybrid full-range K and which also carries
nonlocal correlation with a separately owned range correction (WB97M-V). Other
range-separated hybrids, DF, ECP, D4 and mixed-precision force requests remain
excluded, as does the existing 384-AO floor. The RSH consumer forwards the
qualification only to its full-range traversal; the LR traversal is unchanged.
The existing `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS` selector still overrides it.

## Evidence

`benchmarks/results/h100-wb97mv-spd-forces-20261117/`. Water-96 Direct: cold
36.6 → 33.9 s, warm 7.48 → 4.98 s; paired against GPU4PySCF 1.165/1.165/1.169.
Water-48: 13.98 → 13.09 s cold, 3.41 → 2.54 s warm. All 36 processes pass the
original independent gates. `relative-reuse` measured the same as the default
DensityReuse; `spatial-reuse` was 1.4 s slower.

## Revisit when

Other range-separated hybrids (wB97X, CAM-B3LYP) need the same admission: measure
them first; their LR fraction and force composition differ.
