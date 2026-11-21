# Decision: compiler-derived gCP pair Hessian action

Status: implemented (correction component only)
Date: 2026-11-20

## Decision

Differentiate the existing Slater-overlap and damped pair slopes, including
both derivatives of the supplied overlap, rather than fitting a curvature or
differencing the native gradient. Preserve the current series/exponential
selection, which depends on atom parameters only, and the finite series
actually evaluated by the energy. The Cartesian action also differentiates
the pair unit vector; a radial-only projection misses transverse motion.

As in the existing GPU force owner, each lane owns an atom and traverses its
neighbors in fixed order. The pair action uses the same canonical orientation
for both legs and applies opposite signs; no atomic reduction, dense Hessian,
or per-pair storage is needed. Reserve one additional direction buffer only
for an explicitly second-order `GcpCudaPlan`, retaining the original force
capacity contract. Invalid geometry and numerical failures publish no result.

## Evidence and limits

The RTX 5090 test uses the 16-atom MB16-43/06 fixture. Central mixed differences
of host energies at two steps bound all 48 action entries; the largest error
was `2.70e-10 Eh/bohr²` for the chosen direction. The independently transcribed
reference under `tools/generativeqc_gcp` gives the directional curvature
`1.41315809626e-4` by differencing its analytic directional slope at `2e-4`
and `1e-4` and extrapolating. The probe also tests translation, coincident
atoms, exact-capacity refusal, and replay after a failed action. Run
`generativeqc_gcp_cuda_tests` via finite Slurm `srun`.

This work does not admit an r²SCAN-3c Hessian. The D4 charge, coordination,
interpolation, pair and ATM response and the complete additive endpoint still
need independent qualification.
