# Decision: screen near-zero quadrature weights from the molecular VV10 domain

Status: implemented
Date: 2026-11-17

## Problem

Policy `vv10-molecular-rho-ge-1e-10-v2` kept every point with rho >= 1e-10,
including points whose quadrature weight is essentially zero. GPU4PySCF also drops
|w| <= 1e-14 in its SCF and force pair domains. On H100 water-96 that is 8.9% of
the density-active points and about 17% of the quadratic pair work; it was the
largest remaining difference in the WB97M-V cold endpoint after
[the cold anchor and tile changes](../performance/2026-11-17-h100-wb97mv-cold-anchor-and-tiles.md).

## Decision

Policy `vv10-molecular-rho-ge-1e-10-absw-gt-1e-14-v3`: a point is active only if
rho >= 1e-10 and |w| > 1e-14. The compiler policy owns both thresholds; the
generated `kMolecularVv10WeightThreshold` feeds the CUDA domain kernel (SCF, the
stationary force controller and the C API owner), the CPU energy/potential and
stationary-force paths, the Python fixed-density oracle and the Hessian grid
second-order owner. Screened points leave both pair legs, the beta term and the
weight derivative, exactly like density-screened points. Exactly zero weights are
now inactive as well (previously active with zero contribution). The KS payload
reports the weight threshold alongside the density threshold.

This is a quadrature-domain change, not a convergence, precision or grid change,
and it has no runtime selector: one policy version names one domain.

## Evidence

`benchmarks/results/h100-wb97mv-vv10-weight-domain-20261117/`. Paired same-image
v2/v3 processes give identical total energies and force differences of at most
9.4e-13 Eh/Bohr. Water-96 Direct cold falls from 39.3–39.5 s to 36.5 s; every
paired GPU4PySCF comparison is now faster (ratios 1.100, 1.032, 1.044). DF water-96
falls from 23.9 to 21.6 s. All 25 processes pass the original independent gates.

## Rejected alternatives

- **A runtime selector.** A domain is part of the scientific model identity;
  selector-dependent domains would need separate policy versions anyway.
- **Screening only the CUDA SCF kernel.** CPU, oracle, force and Hessian
  consumers would then integrate a different domain from the SCF.
- **Other tile sizes or looser frozen inner solves** (measured on the v3 image).
  Explicit 512/2048-point SCF or force tiles are 1.0–4.5 s slower cold; stopping
  frozen inner solves at a fraction of the last target residual saves two
  provisional Focks but needs four more full targets and pair sweeps (+8.7 s).

## Revisit when

A tighter accuracy target needs points with |w| <= 1e-14, or a grid family whose
weights are legitimately tiny but numerous; the threshold is a policy constant.
