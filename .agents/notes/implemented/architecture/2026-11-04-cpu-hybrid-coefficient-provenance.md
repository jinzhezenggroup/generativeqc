# Decision: authenticate CPU hybrid exchange from the physical execution plan

Status: implemented (native composition prerequisite; public Hessian admission unchanged)
Date: 2026-11-04

## Problem

The initial [full-range composite qualification](2026-11-04-full-range-composite-second-order.md)
exposed a native failure before Hessian execution: changing full-range exchange
from 1/5 to 37/100 retained the fixed B3 semilocal graph but the SCF wrapper
reconstructed a canonical 1/5 Fock strategy. Merely removing its comparison
would lose the proof that preparation and execution use the same Hamiltonian.
Final-state validation independently assumed the canonical coefficient too.

## Decision

Freeze the compiler-validated full-range coefficient in `NativeKsExecutionPlan` and
require it explicitly in both existing B3-family SCF wrappers. Each wrapper
constructs its expected strategy with that coefficient and retains exact
prepared-strategy equality, including spin and approximation. No new SCF or
Hessian solver is introduced. Migrate all native callers rather than retaining
a default coefficient or compatibility overload.

For CPU fixed-semilocal global-hybrid states, the immutable determinant Fock
identity owns the full-range coefficient. Physical and eigenframe identities
must still match it; finite/full-range operator, semilocal-scale, generation,
occupation and token checks remain active. CUDA retains its independently
qualified canonical/matched-composition restriction. This does not widen
force/ECP/DF/public-Hessian admission.

## Rejected alternatives

- Dropping prepared-strategy equality would permit execution with a different
  Hamiltonian from the one prepared or claimed in a snapshot.
- Keeping a canonical default would conceal unmigrated internal callers.
- A coefficient-specific functional alias or Hessian solver would split the
  shared scientific owners rather than fixing coefficient provenance.

## Evidence and limits

Native RKS/closed-shell UKS tests converge with 37/100 exchange and agree in
energy, while differing from 1/5. A mismatched prepared coefficient still
rejects. Detached CPU-state tests accept coherent 1/5 and 37/100 identities for
both spin layouts and reject changed physical/eigenframe coefficients or
semilocal scales. Existing CUDA host-identity tests remain part of that suite;
these are not real-device qualification.

The independent molecular test freezes maximum raw Hessian error at
1e-5 Hartree/Bohr² and directional-curvature error at 1e-6 Hartree/Bohr² before
acceptance. It compares independently converged PySCF/Libxc energies and
densities for H2 and asymmetric HeH+, each with 1/5 and 37/100 exchange. It
supplies exact native quadrature primals as inputs but never native densities
or derivatives to the reference SCF. Complete reference matrices use three
energy-difference steps (1e-3, 5e-4, 3e-4 Bohr), with each step required to pass.
The asymmetric HeH+ maximum errors were 4.78e-6/1.19e-6/4.26e-7 for 1/5 and
4.85e-6/1.21e-6/4.31e-7 for 37/100, respectively. Production remains analytic;
displaced SCFs and the external engine are test-only oracles.

These tiny all-s fixtures do not establish real-spherical angular chemistry,
PBE0 support, installed-package/resource behavior or Ni-scale qualification.
#2224 and #2843 remain incomplete until their independent gates pass.

## Revisit when

The later public domain and its separate acceptance protocol are recorded in
[qualified public full-range execution](2026-11-05-full-range-public-second-order.md).
This does not turn the earlier tiny-fixture evidence into Ni-scale evidence.

Additional functional families or backends prove a matching native numerical
domain and immutable preparation/final-state contract. Do not widen admission
based only on accepting a MethodIR or passing a synthetic identity test.

## References

- Issues #2224 and #2843.
- `docs/developer/dft_hessian.md`.
- `tests/native/test_uks.cpp` and `tests/native/test_ks_final_state.cpp`.
- `tests/python/test_global_hybrid_molecular_hessian.py`.
