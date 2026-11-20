# Decision: Bound native CPU unrestricted GGA contraction tiles

Status: implemented
Date: 2026-11-20

## Problem

The cold public B3LYP-D3(BJ) HVP at the frozen 25-atom Ni hexaammine
geometry previously exhausted two hours in its UKS primal before making a
response call. On `7ad2f06db`, whole physical SCF evaluations again take about
a minute with 205 AOs and 280,596 grid points. The semilocal grid stage, not
the roughly two-second exact J/K contraction, dominates the first measured
projection-only prototype. The existing dense ERI build and SCF convergence
are separate costs; reducing grid contraction time does not certify either.

## Decision

Use existing FP64 task-local BLAS for fixed-grid GGA fields and potential
assembly, with at most two point-by-AO panels in the current tile. Keep the
point functional, all grid points, their signed weights, and the density and
spin identities. Reuse each `Phi D` row for the potential panel only after
evaluating the coupled spin point, so field and potential do not need separate
buffers. Project one potential leg and add the transpose once. The complete
scalar engine covers small tiles, non-GGA ingredient layouts and missing or
unavailable optional storage/provider. For ill-resolved positive density and
matrices that only pass tolerance-based symmetry, retain the existing scalar
field arithmetic instead of clamping or assuming the two gradient terms match.

Reaching the derivative planner is not sufficient to admit the action. For the
frozen 25-atom all-electron case, a single column plans 1,925,254,939,358 XC map
terms, above the default `2**40`. Expose the existing native cap explicitly at
the public boundaries rather than increasing the default for everyone. A retry
uses the same physical and resource gates; a finite HVP override does not prove
that a full matrix fits or that either request is practical.

## Evidence and limits

- The native fixed-density gate compares both spin potentials to the scalar
  point-wise quadrature on a ragged tile and independently differentiates the
  discrete energy in each spin channel. It also covers an admitted
  near-symmetric matrix.
- Relative timings of SCF stages are diagnostic, not complete-endpoint speedups.
  Capture the cold preparation, complete public request, semantic work counts
  and peak RSS before making an endpoint claim. Keep different native binary
  identities and independent reference receipts separate.
- The change does not qualify practical Ni/ECP second order or a Ni molecular
  saddle. It does not promote any force or curvature capability.

## Revisit when

Re-evaluate the shape and capacity gates when complete cold endpoints cover
larger AO dimensions, other grid densities or a scalar-only provider. Consider
optimization of the force/response phases separately once the cold primal
reliably reaches them with unchanged physical acceptance.
