# Decision: Borrow moving AO panels and replay spatial jets under the grid cap

Status: implemented
Date: 2026-11-20

## Problem

The stationary Hessian copied its geometry candidate into a second full moving
AO panel. Separately, it required a full order-three spatial cache even if the
mixed output would fit without that cache. Default-grid 34-AO r²SCAN-3c/DF water
failed the default 128 MiB per-grid-owner limit before response could start.

## Decision

Borrow gated geometry results on the owning stream until the next geometry
attempt, keeping transactional standalone copies for callers that need them.
Prefer cached spatial jets when they fit. Otherwise reuse the same compiler-
generated CUDA evaluator over contiguous point tiles, with a separate checked
endpoint work bound. Tile-local spatial planes scatter into the unchanged
full mixed layout; there is no change in quadrature, per-lane arithmetic, or
density/response reduction order. The fallback does not promise that the full
mixed panel or the independent XC response arena fits on arbitrary grids.

## Qualification

On Slurm `node1` (RTX 5090), the fitted, complete named water HVP at the default
quadrature used 349,560,488 response bytes under the 128 MiB grid limit versus
601,981,864 with a 512 MiB limit and resident jets after removal of the copy.
The HVPs were identical. The complete CLI wall times were 5.94 s and 6.35 s,
respectively, single runs, not a speedup claim. Both had one DF value pass, one
DF contraction pass, 13 response iterations, and one correction direction.
Replay charged a conservative 491,990,800 AO primitive/polynomial terms over
the ten mixed actions, versus 49,199,080 once at resident preparation. The
tighter budget also turned off optional staged affine graph replay. Local JSON
receipts are under `build/hessian-gpu-sm120/.r2scan3c-water-receipts/`.

The reproducible gates include an independent Gaussian/Becke oracle and
borrow-vs-copy comparison for a non-divisible grid. The default-grid raw matrix
on the same GPU had `max|H-H^T| = 4.23e-11 Eh/bohr²`, translational residual
`2.85e-11 Eh/bohr²`, and `max|Hv-HVP_resident| = 2.50e-12 Eh/bohr²`. Its native
endpoint took 35.03 s with 4,427,917,200 replay terms and nine each of DF value,
DF contraction, and classical correction passes. A separate default-grid test
also passed the `8e-7 Eh/bohr²` gate against Richardson differences of
reconverged complete native gradients. This tests the second derivative
without invoking the Hessian implementation in the reference, but shares the
native SCF and force implementation; third-party energy oracles remain on the
smaller grids.
