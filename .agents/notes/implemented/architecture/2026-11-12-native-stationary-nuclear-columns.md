# Decision: assemble stationary CPU columns around one verified nuclear response

Status: implemented
Date: 2026-11-12

## Problem

Frozen or supplied-weight integral/XC/ECP columns are not a stationary molecular
Hessian. They must consume the actual coupled density and full energy-weighted
density response from the same verified state, Fock and moving grid, then add
ionic nuclear repulsion and the requested geometry-only dispersion once.
Individually correct sources can still produce a wrong total through mixed
owners, occupation factors, frozen W response or misinterpreted Lagrangian values.

## Decision

`HostKsNuclearHessian` creates the existing nuclear response and curvature
consumers from a shared authenticated state/system/prepared Fock/column geometry.
The nuclear response owns its induced-Fock reader and solves coupled metric/CPKS
for D′ and full W′. Integral and XC columns consume the correct spin panels; ECP
receives the total D/D′ with no added occupancy factor. The nuclear seven-root
AOT leaf reuses the existing Coulomb scalar Graph. Optional `HostD3Column` uses
real atomic numbers, even on effective-ionic ECP centers. No production finite
differences, reference engine, Python/JIT, optimizer or lower-level method runs.

`HostKsNuclearResponse::validate_current` exposes full reference validation
without another source or solve. The column geometry exposes its actual mutable
base reader so the new owner can use the same reader without a const cast.
These are direct current interfaces, not compatibility facades.

## Energy semantics

The integral column's value includes `-Tr(W S)`. Adding source values directly
would report a Lagrangian value rather than the physical molecular energy.
Assembly restores `Tr(W S)` and checks the reconstructed physical B3LYP energy
against verified component energy before adding D3. The supplied-weight right
scalar is also not a physical stationary energy derivative; the public internal
column layout explicitly reports gradient dot right instead. Full W′ remains in
the Hessian action and is never replaced by orbital-energy-scaled D′ alone.

## Resource and lifetime boundaries

The nuclear response currently offers instance resources rather than a complete
allocation-free preparation plan. Instantiating every child with a global cap
and checking the sum afterward can violate that cap before refusal. The new
owner instead admits the sum of explicit enforced child reservations and its
shape-only assembly panel bound before any child numeric allocation. The
response reservation includes GMRES. Actual composed fixed bytes and requested
GMRES bytes are reported separately from the conservative reserved total.

This deliberately does not copy private response workspace formulas into a new
owner. Borrowed SCF/Fock/basis/grid/geometry owners, provider/BLAS storage, static
tables, metadata/allocator overhead and RSS/time remain separate. All complete
endpoint resource/work/time gates must account for those owners explicitly.

The assembler owns D/W/F/C/orbital-energy and overlap/core snapshots. This is
stronger than checking borrowed pointer/shape identity, which cannot detect
in-place orbital or Fock mutation. Physical component energies and the accepted
stationarity tolerance are also authenticated. The required caller validator
continues to cover immutable geometry/provider contents and epochs. External
right contents are snapshotted and checked across the entire composition, not
only inside each child call. All fields publish together, never on response or
source failure. Reentry/borrowed outputs refuse and right/output aliases stage.

## Validation and remaining scope

`test_host_ks_nuclear_hessian.py` contains native stationary-frame controls,
independent multistep reconverged primal-energy gradient/HVP comparisons on the
same primal grid, raw symmetry/translation/bilinearity/H@v controls, exactly-once
D3 composition, capacity/nonconvergence/mutation/reentry/alias recovery, and an
actual no-Python native executable. Scientific gates are declared before running:
energy2e-9; gradient atol3e-7/rtol2e-6; HVP atol3e-6/rtol2e-5; eightfold step
convergence with5e-8floor; raw stationary invariants2e-8. Compact synthetic
empty/analytic/checked-grid ECP controls do not qualify real heavy basis sets.
Source-matched raw outputs belong under
`build/qualification/host-stationary-column-20261112`.

Installed public endpoints, real Ni/I/Cs electronic states, practical25--40atom
complete electronic work/time/memory, the requested Ni saddle and final unscaled
frequencies remain separate acceptance requirements. Ordered quartet source work
is still quartic; bounded capacity is not a practical scaling claim.

## Rejected alternatives and revisit conditions

Frozen D/W sources, independently bound readers, doubled UKS ECP density, raw
Lagrangian energy reporting, handwritten nuclear/D3 Hessians and post-hoc raw
symmetry/translation repairs change the required semantics. Runtime Python/JIT
does not meet the native installed endpoint requirement. Revisit reservations
when all children expose authoritative preparation plans; revisit scheduling only
with complete endpoint evidence, not microkernel or allocated-byte improvements.
