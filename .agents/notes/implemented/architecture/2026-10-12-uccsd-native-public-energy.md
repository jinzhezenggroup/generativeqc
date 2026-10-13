# Decision: productize unrestricted energies under native method ownership

Status: implemented
Date: 2026-10-12

## Problem

The internal unrestricted energy/response owners orchestrate generated native
TensorIR through Python. Registering those owners as Calculator methods would
violate the migration to a Python-free C/C++ computational runtime. Existing
prototype force/CUDA qualification is not transferable to a new execution owner.

## Decision

Add distinct UCCSD/UCCSD(T) IDs and an energy-only native provider. Generate
physical unrestricted residuals and four standard triples channels at build
time from the existing scientific compiler. Share emission, arena planning,
bounded CC iteration and host DIIS machinery, not restricted mathematics.
The C++ owner runs native UHF, ordered spin MO transformations, independent
packed excitations, fresh CCSD replay and singleton triples tiles. Python
Calculator only forwards controls and translates existing diagnostics.

Public admission is deliberately CPU-only and bounded to twelve AOs with
all-electron real-FP64 unscreened s/p/d Hamiltonians. No force, CUDA, DF, ECP,
batch or reference-library fallback is advertised. Numerical/work admission
includes the complete configured residual iteration and triples domains.

## Rejected alternatives

- Calling PreparedUCCSD/PreparedUCCSDT from the native or Python public owner:
  leaves scientific state/solver orchestration outside native ownership.
- Reusing RCCSD for unrestricted singlets or unequal spin populations:
  changes the equation/state contract and loses five independent spin blocks.
- Exposing prototype force/CUDA capabilities before native migration:
  changes ownership, resource lifetime and backend behavior without qualification.
- Treating bounded triples memory as bounded work: singleton tiles still
  traverse the complete admitted ordered external coordinate domains.

## Invariants

- Full physical Fock feeds; level shift is preconditioning only.
- Same-spin packed representatives use i<j,a<b; mixed spin is not symmetrized.
- Fresh all-five-block physical replay binds the exact triples parent state.
- Nuclear repulsion belongs solely to the UHF reference.
- Unsupported domains fail explicitly; late failures publish no partial state.
- Preserve existing ABI layouts/IDs; append two distinct method IDs.

## Evidence

`tests/python/test_uccsd_public.py` checks independent shell-matched PySCF
UHF/UCCSD/(T) energies, unequal-spin four-channel triples, s/p/d input,
restricted-limit parity, model identities, replay thresholds, memory admission
and unsupported requests. Runtime imports of prototype/reference scientific
owners are forbidden during the Python user endpoint. Native CLI qualification
runs with no interpreters on PATH. Total-energy tolerance is 3e-9 Hartree;
triples tolerance is 5e-10 Hartree and physical residual gates are 1e-11.
Source-matched local execution evidence belongs to ignored `.artifacts/`, not
to claimed inherited GPU/force qualification.

## Consequences and revisit conditions

This is staged native energy integration, not completion of unrestricted
response/analytic-force/CUDA delivery. Extend public properties/backends only
after native Lambda, full-Fock triples response, UHF Z-vector and contracted
nuclear derivative ownership passes independent native endpoint gates. A
larger AO domain requires complete work/data-movement qualification, not only
a larger allocator budget.
