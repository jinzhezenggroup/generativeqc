# Decision: Admit scalar ECP Hessians for linked CUDA split hybrids

Status: implemented
Date: 2026-11-21

## Decision

Canonical M06-2X and MN15 use the same opt-in variational point in SCF
and Hessian/HVP for all-electron and scalar semilocal ECP orbitals. The
ECP contribution comes from the existing stationary core/projector owner,
not from a different split-hybrid point. A separate first-order gate keeps
standalone ECP forces unavailable. Selecting that point only for a Hessian
request would borrow a density converged on the wrong energy surface.

ECP DF preparation requires an explicit ECP-free fitting basis, full-range
fitted J and K with the method's exact exchange, strict FP64, fused XC,
three Becke iterations, and the linked method-specific domain. Orbitals may
go through f and the fit space through g in spherical representation; the
Cartesian ECP shape is limited to s/p. Energy-only budgeting accounts for
the simultaneous SCF, ECP, and response reservations without adding a force
consumer or CPU numerical fallback.

## Evidence

- RTX 5090 Slurm 2277 passed 16 low-angular Python/CLI dense/HVP cases:
  both functionals and spins, all-electron/ECP, Direct/DF. Slurm 2278 and
  2281 passed four f/g DF and four f-orbital Direct ECP cases, respectively.
  Checks include point identity, zero numerical host fallbacks, ledger peak
  and release, matrix symmetry, translation, and action replay.
- RTX PRO 6000 Blackwell Slurm 5490 passed four native UKS f-orbital cases
  on the same SM120 binary. Both f/g DF matrices matched the RTX 5090
  element-for-element; the two Direct f cases separately passed independent
  energy reconvergence (maximum Hessian error `2.35e-8 Ha/Bohr²`).
- Slurm 2279 captured eight low-angular native ECP comparisons; all passed
  independent Libcint/Libxc energy reconvergence, with maximum Hessian error
  `3.38e-7 Ha/Bohr²` and replayed HVP error `3.42e-8`. Slurm 2280 captured
  additional UKS f/g DF cases for both functionals; both pass with maximum
  Hessian and HVP errors of `2.67e-8 Ha/Bohr²` and `3.12e-9`, respectively.
  The CPU oracle uses
  independent integrals, upstream Libxc and an explicit chain rule at the
  smooth work coordinates, and authenticates the center Fock before taking
  energy differences. It does not use the native response or AD.
- For high-angular M06-2X, hard-screened Libxc alone misses the Fock gate.
  The smooth work coordinates close that gate. At the tail join, a coarse
  mixed-coordinate stencil is not in the local quadratic regime: the
  extrapolated error of the affected element drops from `1.26e-6` at
  `0.002/0.001` Bohr to `1.94e-8` at `0.0005/0.00025` Bohr, without changing
  the matrix acceptance tolerance. Keep the explicit work-coordinate test
  and the resolvable stencil rather than widening the tolerance.
- CUDA 12.9 Compute Sanitizer on Slurm 2283 reported zero errors for the
  complete M06-2X RKS Direct endpoint. The shared 12-minute allocation ended
  during the MN15 UKS DF pass; isolated Slurm 2284 finished the same DF
  endpoint with zero errors.
- Host checks: 241 passed, 11 skipped, including AOT and package contracts.
  Build/cache, binary hashes, Slurm and
  oracle receipts live in `build/hessian-gpu-sm120/r2scan-orbital-f/`
  (`split-hybrid-ecp-*`). These qualify the measured SM120 inventory, not
  every target architecture or partial AOT package.
