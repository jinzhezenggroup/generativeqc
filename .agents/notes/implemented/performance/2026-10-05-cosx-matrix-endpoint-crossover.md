# Decision: retain generated COSX matrix selection after endpoint crossover

Status: implemented
Date: 2026-10-05

## Problem

COSX projection (`AO*D`) and donated exchange accumulation (`seed+AO^T*potential`)
now consume shared prepared contraction sites. Their regular algebra makes them
plausible GEMM candidates, but isolated matrix timing cannot predict an endpoint
that also generates every ESP integral and publishes host matrices and energy.

## Decision

Keep the production generated incumbent. A dedicated test-only endpoint harness
independently qualifies projection, accumulation and their combination, retaining
the original and changed geometries, exact full/tail work, resource reservations,
all candidates and every timed numerical comparison.

The [retained receipt](../../../../benchmarks/results/cosx-contractions-1884/README.md)
uses actual 384/768-AO spherical def2-SVP water systems, a positive diagnostic
density, tiles 64/256 and explicit 3×3×6 per-atom quadrature. That quadrature is
a coarse diagnostic domain, not a qualification of production COSX quadrature,
converged PBE0 SCF or forces. CPU subset oracle work at the actual AO dimension
runs outside timing. First prepared replay and warm replay are distinguished;
process-cold CUDA/library initialization is not measured.

## Invariants

- Method owners do not select a library or create provider handles per tile.
- Projection and accumulation retain the same semantic identity and scalar
  summands across generated and library routes, including the final partial tile.
- A library route shares one charged provider allowance across four sites;
  it does not reserve four independent handles or change quadrature/screening.
- Every sample passes raw/symmetric K and energy gates. The small independent
  scalar/CPU/failure/resource tests remain required alongside paired timing.
- A small difference within the spread of these repeated endpoint measurements
  does not establish a profitable production domain.

## Consequences and remaining work

The benchmark supplies negative endpoint evidence without adding a vendor switch
to COSX science. The composed Fock owner still reserves only baseline staging
bytes, so this diagnostic admission must not be described as production GEMM
selection. Enclosing resource composition and full PBE0/COSX qualification remain
separate work.

ESP application is still a weighted point-batched contraction. Its next provider
slice must represent the weighting and complete finite/publication behavior,
retain an executable fused generated alternative, and charge any split epilogue
or materialization. ESP integral recurrence itself remains with the integral
owner. Derivative consumers still require migration and independent force gates.

## Revisit when

A changed matrix recipe, device, layout, quadrature or ESP schedule gives a
measurable complete-endpoint win with unchanged scientific work. Qualify actual
production cold/warm/changed-geometry SCF and force endpoints before promotion;
do not infer them from either isolated GEMM throughput or this coarse grid.

## References

#1884, #1886, #1966 and #1967. The receipt records exact source/binary/input/data
hashes, per-case Slurm assignments and ccache compiler evidence.
