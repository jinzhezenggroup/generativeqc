# Decision: Admit Direct f orbitals to the variational CUDA r²SCAN domain

Status: implemented
Date: 2026-11-21

Direct r²SCAN previously kept an orbital-d ceiling shared with the generic
CUDA force profiles. In an f-shell calculation this selected the incumbent
point energy before SCF and therefore prevented the stationary Hessian. The
same orbital and grid fixture with CUDA DF entered the variational domain and
completed the matrix (RTX 5090, Slurm 2113). This was not evidence that the
Direct f-shell integral leaves or Cartesian normalization were missing.

Use the shared variational selection through f for both providers. Admit the
same strict canonical bare r²SCAN context to native forces (including the
capacity and property query), not just to the response owner. Energy-only, force
and second derivatives therefore authenticate the same point domain, independent
of requested properties or temporary resource limits. The rVV10 composition
inherits the f-shell point domain, but still has no standalone native force
endpoint. Keep the linked shell inventory checks, per-provider work limits,
strict FP64, fused execution and three Becke iterations. Other Direct force
profiles and the incumbent AUTO/budget/host-unfused domain are unchanged.

This extends the [original deployment](2026-11-20-cuda-r2scan-variational-deployment.md),
not its point mathematics. Do not just remove the Hessian admission check or
start a second, derivative-only SCF for f-shell inputs: either would differentiate
an energy other than the one the user requested.

## Qualification

RTX 5090 SDK build with the default full Hessian AOT inventory and `sccache`
0.16.0. The raw matrices and independently reconverged Libcint/Libxc energy
stencils are retained in `build/hessian-gpu-sm120/r2scan-orbital-f/`. Tests check
native forces (bare r²SCAN), symmetry, translation, and matrix/HVP parity as
separate gates. Slurm 1703 passes the public Python/CLI f-shell path plus the
unchanged strict/AUTO/alternate-grid routing regression.

Slurm 2114 and 2115 produced eight complete matrices: bare real-spherical
RKS/UKS with both providers, bare Cartesian RKS/UKS with Direct, and Direct
real-spherical r²SCAN+rVV10 (RKS shared grid, UKS separate grids). Across
the 9–13 AO fixtures, the worst raw entrywise error was `3.34e-8 Ha/Bohr²`;
the largest independent bare gradient error was `1.42e-10 Ha/Bohr`. Raw
symmetry, translations and separate HVPs all passed without symmetrization.
The paired state-domain and f-shell public frontend tests pass. Host-only
point and budget checks report 13 passed, with 12 device-only point cases
skipped in that host invocation.

## Remaining size qualification

The 44-AO real-spherical HF/cc-pVTZ RKS Direct unit HVP, with an explicit
16 × 6 × 12 grid, produced no result within six minutes (Slurm 1704).
A bounded trace (Slurm 1707) completed all 16 Fock builds with a converged
full-density audit and passed resident final-state validation before its
55-second diagnostic cutoff. The scalability gap is after SCF, not evidence
of an incorrect or unavailable SCF point domain. Separate the mandatory
provider preparation, frozen nuclear integral source and coupled-response
work before changing budgets or making a performance claim. Independent
reconverged energy and unit-column references for this case are retained
under `build/hessian-gpu-sm120/r2scan-orbital-f/hf-ccpvtz/`; no native
comparison is claimed yet.

The later [canonical streaming decision](../performance/2026-11-21-cuda-hessian-canonical-streaming-resident-cap.md)
records the diagnosis, fix and completed 44-AO endpoint qualification.
