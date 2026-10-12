# Decision: native multistage analytic-force geometry optimization

Status: implemented
Date: 2026-10-12

## Problem

Analytic conventional and DF CCSD(T) energy-plus-force endpoints already exist,
but the native CLI cannot find stationary structures. A high-level method should
be able to refine a cheaper method's geometry without copying method-specific
scientific kernels or depending on Python at runtime.

## Decision

Expose a standalone C++ Cartesian L-BFGS + Armijo optimizer over an analytic
energy/force callback, independent of the electronic method and CPU/CUDA
provider. The CLI alone plans a deterministic ladder (GFN2-xTB, spin-aware
PBE for costly targets, requested final method) or accepts explicit stage
selectors. Each stage has its own freshly converged energy surface and
curvature history. A stage never imports a result, intermediate or claim of
stationarity from a different Hamiltonian as a finished target value.

Only automatically selected *preparatory* steps can be skipped, and only
for an explicit native NOT_IMPLEMENTED status. Electronic convergence,
resource, analytic force, and line-search failures all abort. The final
target stage is mandatory and must independently reach its stricter fmax
limit. Unconverged trajectories cannot publish an optimized XYZ.

## Rejected alternatives

- Energy-only finite differences (unnecessary cost and noise for CC).
- Reuse BFGS curvature across different electronic Hamiltonians.
- Treat a preceding DFT gradient or xTB structure as evidence of CCSD(T)
  stationarity.
- Fall back from the user's requested method/basis/precision silently.

## Invariants

Coordinates are Bohr; energies Hartree; native forces are -dE/dR
Hartree/Bohr. Actual native response/derivative owners are the only
scientific force producers. Every accepted move satisfies the same-stage
Armijo decrease and per-atom Cartesian step cap. Bad curvature is discarded.

## Evidence and gaps

Independent C++ test: quadratic and quartic analytic-force oracles, target
surface switch, invalid input/nonfinite output, max-step exhaustion,
line-search failure, native electronic failure propagation. CLI integration
test: plan-only method ladders, invalid targets/tolerances, and full native
CPU RHF H2 force optimization.

Full multi-geometry CUDA CCSD(T)/DF-CCSD(T) qualification and end-to-end
performance measurements still require real GPU runs. Cross-geometry
density/amplitude warm starts, atomic restart checkpoints and molecular
trust-region/RFO are not included in this first native endpoint.

## Revisit when

Explicitly versioned and independently accepted SCF/CC warm-start contracts,
bounded snapshots/checkpoints, or fully qualified Hessian support become
available. Keep new policies behind the same method-neutral objective.
