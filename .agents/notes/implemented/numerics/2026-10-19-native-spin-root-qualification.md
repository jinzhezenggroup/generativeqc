# Decision: Native spin-root qualification around shared response and SCF owners

Status: implemented
Date: 2026-10-19

## Problem

A converged unrestricted SCF result can be an internal saddle or a different
localized root. Fixed-geometry tool UHF recovery did not qualify public UKS
energy/force publication or geometry continuation.

## Decision

Use one bounded complete real-spectrum owner and one restart controller for
UHF, polarized LDA/PBE and the actual UHF references of native UMP2.
The physical orbital Hessian is twice the existing spin response Jacobian.
Rebuild full nonlinear energy/Fock at every certificate; UKS point response
cannot reconstruct the nonlinear XC potential. Keep native SCF, exact J/K,
SCF-domain XC, spin response and metric projection as their existing owners.
Each restart gets its own batch to preserve earlier snapshot leases. Import
seeds through the existing atomic warm-state ABI and require actual consumption.

Projector identity uses cross-geometry AO overlap, not raw AO coefficients or
density norms. Equal-spin global inversion and occupied gauge are equivalent;
unproven spatial symmetries remain separate. The explicit tracking threshold
can reject a large geometry step. Full work/action and numeric storage gates
are independent; native SCF/correlation use their existing resource contract.

## Rejected alternatives

A partial lowest-eigenvalue Krylov estimate cannot prove a stable full domain.
An energy-only downhill filter can accept an unconverged or unstable state.
Applying linear XC response to the density as if it were nonlinear Fock changes
the model. Reusing a mutable batch for all attempts invalidates retained snapshots.
SCF screening zero in the public Calculator is not an unscreened contract: it is
rejected there and the native common adapter interprets zero as its default.
Therefore keep the existing small SCF screening and independently rebuild an
unscreened qualification model, with explicit energy/stationarity gates.

## Invariants and evidence

Every spectrum column/eigenpair is freshly applied. Curvature uncertainty,
self-adjointness, stationarity, actual nonlinear energy and retained lifetimes
must pass before publishing. Signed finite rotations preserve spin counts and
projector structure. No oracle or tool-package import is a production fallback.
The analytic Hubbard dimer and pinned PySCF 2.14.0 molecular tests independently
qualify energies, spin/populations, response spectra, finite rotations, forces,
geometry tracking and failure publication. Build compiler commands use verified
sccache 0.16.0; cache use is separate from cache hits.

## Consequences and revisit conditions

The opt-in CPU endpoint requires SciPy and a bounded full real OV domain. It
may execute several full correlated endpoints; it makes no performance or global
minimum claim. Revisit partial spectra, complex/spin-flip stability, additional
XC models, spatial equivalence and CUDA only with separate mathematical,
resource/lifetime and independent scientific gates.

## References

- Issue #1826; earlier fixed-geometry UHF PR #2338.
- `docs/user/spin_roots.md` and `docs/developer/uhf_stability.md`.
