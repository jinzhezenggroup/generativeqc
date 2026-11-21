# Decision: Bind molecular r²SCAN+rVV10 to its accepted CUDA SCF state

Status: implemented
Date: 2026-11-21

## Decision

The compiler owns the Ning et al. r²SCAN+rVV10 MethodIR (Phys. Rev. B 106,
075422, Sec. II A): unscaled r²SCAN, rVV10 with `b=11.95` and `C=0.0093`, and
no exact exchange. Generate the native value identity from the same spec. This
fit is not the `b=12.3` r²SCAN+VV10 model. Select the variational r²SCAN point
program before the first SCF iteration. The accepted nonlocal owner uses
MolecularV1 on its own quadrature, without inheriting WB97M-V's condition
that a range-separated exchange provider also be present.

Bind the induced and moving-grid response to the SCF token's complete model,
including nonlocal parameters, domain and quadrature. With a separate mesh,
the primary integrator computes semilocal response only and the secondary
integrator computes rVV10 only. A pair-cap failure never replaces a grid or
publishes a derivative. Neither a post-SCF rVV10 correction nor the incumbent
nonvariational r²SCAN point energy differentiates this model. The standalone
CUDA force registry is not extended by the Hessian binding.

## Evidence

- Slurm 2086 (`node5`, RTX 5090), linked `hessian-gpu-sm120` build: raw
  molecular presentations for direct RKS and fitted UKS with each grid layout.
  Native probes reject an insufficient pair budget without writing the caller
  buffer and exercise both dense and action paths.
- Independent PySCF/Libcint integrals, Libxc 7 point work-domain pullback,
  long-double Becke weights and NumPy rVV10 values/potentials reconverge the
  displaced determinant. Compare center energy, Fock and density, then every
  Hessian entry against Richardson-extrapolated energy stencils at 0.004 and
  0.002 Bohr. Check raw symmetry, translation and dense/HVP parity at `3e-9`.
  Maximum absolute matrix errors (Ha/Bohr²):

  | SCF provider | Shared grid | Separate grids |
  | --- | ---: | ---: |
  | Direct RKS | `1.07e-9` | `1.09e-9` |
  | Fitted UKS | `3.62e-7` | `6.94e-9` |

  The worst shared-grid UKS diagonal crosses the MolecularV1 cutoff on the
  coarse stencils. At `0.0005` Bohr stencil step neither displacement changes
  the active mask, and the unextrapolated entry differs by `2.89e-8`. The native
  derivative is local to the accepted active set, not a derivative across a
  cutoff crossing.
- Check dimensionless partition fractions (before multiplying large atomic
  radial weights). Recompute the physical density cutoff during displaced SCF;
  do not constrain the oracle to the native center's active set. Bounded NumPy
  row batching made this reference tractable without production kernels or
  changing the analytical sums (30 scalar/blocked VV10 and rVV10 cases were
  bitwise equal).
- Slurm 2087: 11 Python/CLI tests passed, including both rVV10 spin endpoints
  and previously admitted r²SCAN and split-grid WB97M-V regressions. Host
  catalog/frontend tests: 71 passed; MethodIR and nonlocal reference tests:
  166 passed. Keep the raw probe JSON, stencil receipts and compiler cache logs
  under `build/hessian-gpu-sm120/rvv10-molecular-*`.

## Revisit when

An additional composition, first-order rVV10 force owner, non-FP64 model, or
ECP inventory can demonstrate matching physical energy and derivative
qualification. None should be inferred from the semilocal family alone.
