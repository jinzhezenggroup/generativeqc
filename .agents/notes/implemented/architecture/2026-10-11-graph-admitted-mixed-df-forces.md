# Decision: admit complete mixed DF forces by provider graph

Status: implemented
Date: 2026-10-11

## Problem

Native KS already composes fitted full-range J/K with an exact Direct LR
correction and self-consistent nonlocal correlation. The private stationary
bridge differentiates that same provider composition, but Python force
capabilities rejected every fitted range/nonlocal graph. Separately, the
stationary compiler allowed combined exchange/nonlocal planning only for the
semantic payload of one named functional. That coupled mathematical source
planning to a runtime qualification decision and prevented reuse for other
functionals with the same primitive inventory.

## Decision

The stationary compiler plans additive primitive derivatives independently of
functional names. Its existing primitive-count, ingredient, spin, Hamiltonian
and derivative-capability checks remain. Planning is not backend qualification.

The compiler owns one complete external-source inventory for the current
resident range/nonlocal CUDA consumer. Both the public mixed-DF force predicate
and the execution owner use that inventory. The predicate additionally requires
the generated native family's semilocal, range-exchange, nonlocal and molecular
density-domain capabilities. Native family resolution still enforces the
registered electronic composition; unsupported graphs are not promoted simply
because they have a mathematical gradient plan.

Public admission retains all-electron, FP64, device-fused CUDA and through-f
orbital-basis boundaries. CPU mixed forces, other external-source inventories,
ECP and automatic mixed precision remain unqualified.

The existing native bridge remains the scientific owner. For physical Fock
coefficients `s` and `delta = l - s`, its fitted full-range exchange response and
exact LR correction are recomposed as
`s * (K_DF(full)' - K_LR')` and `l * K_LR'`. Their sum is precisely the derivative
of the SCF energy's mixed exchange. Spin factors and signs belong to the resolved
Fock terms, not this public gate. A zero correction cannot recover separate
SR/LR rows from the weighted LR owner and remains explicitly rejected.

The DF descriptor retains `AUTO` after native provider resolution. The bridge
must validate the actual CUDA backend and resolved DF approximation, rather
than demand the descriptor spelling `CUDA`. Complete endpoint execution caught
that previously dormant mismatch: energy with `density_fitting="auto"` worked,
but its derivative bridge rejected the same live CUDA DF provider. Both explicit
and automatic descriptors now use the same checked scientific owner; CPU and
non-fitted owners still fail before allocation.

The former composite schedule assumed all Direct derivative storage was already
charged to SCF. That holds for the ordinary Direct composition, but not the
mixed owner's derivative-only LR provider. Complete water/SVP execution exposed
the missing reserve even though the small H2/H3 endpoints passed. A private live-
snapshot capacity query now returns the same conservative native bound used for
lazy LR construction, without constructing that owner. The compiler planner
accepts this caller-supplied capacity and charges it alongside AO, geometry and
full-grid nonlocal storage before selecting a tile. It also conservatively
charges the reserve to the host envelope. Cold and warm admission use the same
bound, rather than shrinking to the already allocated owner's observed bytes.
No integral capacity formula is duplicated in Python, no user cap is enlarged,
and DF response storage retains its separate envelope.

## Rejected alternatives

- A new functional-name switch in `Calculator`: duplicates the generated native
  family registry and fails to extend when another provider graph is qualified.
- Treating any stationary plan as production admission: mathematical contraction
  rules do not certify native AO, integral, nonlocal or final-state consumers.
- Calling this range-separated RI-K: neither metric nor three-center DF
  integrals depend on omega; the LR owner is deliberately exact Direct.
- Replacing the force with CPU/PySCF or energy finite differences: references
  belong to qualification only, never to the CUDA production endpoint.

## Invariants

- Differentiate the same fitted auxiliary basis, metric and exact LR operator
  used by the final physical SCF state; include Pulay and moving-grid response.
- Preserve the native final-state token checks and geometry/auxiliary rebinding.
- Charge the lazy LR derivative owner on cold creation and warm reuse. The
  bridge's additional-resource report excludes the separate DF response owner;
  it is not a whole-force peak-memory guarantee.
- Keep the complete production endpoint on its requested backend, without
  reference-oracle or CPU-integral retry.

## Evidence

`tests/python/test_mixed_rsh_df_forces.py` executes the actual public predicate
without a GPU, tests renamed semantics and missing provider capabilities, and
constructs a different range-separated PBE/VV10 graph. Simulated future registry
qualification proves that extending the provider metadata needs no new
functional dispatch, while the current unqualified graph remains rejected.

The real-device gates reconstruct an independent PySCF mixed-DF SCF for every
energy displacement, including both auxiliary centers and both moving grids.
They require energy agreement within `2e-8` Hartree, force agreement within
`3e-6` Hartree/Bohr, translational residual below `2e-9` Hartree/Bohr, cold/warm
force agreement within `2e-8`, and moved-batch agreement with a fresh owner.
PySCF's ordinary fully range-fitted hybrid is not a matching oracle.
The device matrix includes RKS/UKS H2/H3 in both representations and spherical
water/SVP and water/TZVP, covering d/f orbital derivatives. Host-only execution
of the actual native bridge checks explicit/AUTO descriptors and allocation-free
cold/warm capacity queries; planner tests charge the reserve to both live bounds
and reject infeasible totals before choosing a smaller tile.

Qualification on master `637466c62` plus this change: 131 host checks passed;
the final combined mixed-DF and existing DF public matrix passed 55 tests with
one CPU-only budget test skipped. All six complete mixed endpoints passed in a
finite Slurm `main` allocation on `n1` with one RTX 5090. The native build used
CUDA 12.9, `sm_120`, and verified ccache 4.5.1, with the PBE RKS/UKS, R2SCAN RKS
and PBE0 RKS stationary AOT profiles for the existing regression matrix. These
are correctness gates, not a performance or whole-endpoint memory claim.

## Revisit When

Add another production inventory only after its complete native and grid
consumers pass independent energy/force gates. Implement omega-dependent DF
integrals and their metric response before claiming range-separated RI-K.
