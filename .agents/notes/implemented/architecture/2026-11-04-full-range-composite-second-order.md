# Decision: qualify full-range exchange through the shared composite owner

Status: implemented (low-level qualification path; public admission unchanged)
Date: 2026-11-04

## Problem

The shared stationary planner already owns the full-range exchange weights,
and the composite molecular owner already handles spherical AO pullbacks and
one coupled response solve. Its runtime binding nevertheless assumed an SR/LR
pair and a VV10 primitive. Reusing it for B3LYP also exposed a distinct identity
hazard: `state.identity.method` can name the internal PBE execution carrier,
whereas `state._source.method_ir` retains the actual B3LYP graph. Resolving the
former reconstructs the wrong semilocal Hamiltonian.

## Decision

Bind the token-verified physical MethodIR into the existing composite grid,
Fock and second-integral owners. Resolve exchange operators and coefficients
through the existing KS execution plan; full-range exchange binds the existing
`exact_exchange` source rather than inventing another Hessian equation.
Keep the public Calculator gate unchanged while scientific qualification is
incomplete. Low-level full-range qualification is CPU/direct/all-electron/RKS;
it does not imply UKS, ECP, fitted or CUDA admission.

Allow the compiler grid model to omit VV10 completely. The native
B88/VWN-RPA/LYP family uses the existing production scalar graph and derivative
builder, including its density-tail policy. An absent pair primitive has no
pair graph, no pair work and no pair-work rejection. Retain the independent
grid-memory bound and the qualified WB97M-V worker floors.

## Rejected alternatives

- A B3LYP-specific Hessian solver would duplicate the stationary and spherical
  owners and risk different overlap, density and weighted-density conventions.
- Resolving the carrier name can change the requested chemistry to PBE.
- Borrowing WB97M-V's floors or an interior-only XC graph differentiates a
  different numerical energy in the tails.
- Combining separately converged HF and semilocal Hessians omits the shared
  method-correct response.
- Declaring public support from a planner, a grid test or an H2 calculation
  would bypass the independent molecular and Ni/installed-resource gates.

## Invariants and evidence

The executor retains one response solve per direction and one contribution per
declared source, raw unsymmetrized output, explicit provider budgets and native
owner/token checks. No oracle or displaced SCF is added to production.

`test_global_hybrid_grid_second_order.py` freezes independent Libxc primal,
Cartesian-gradient and multistep directional-gradient gates for both spin
layouts and noncanonical component fractions. It checks bilinear symmetry,
zero nonlocal work, native vacuum behavior and memory rejection.
`test_global_hybrid_molecular_hessian.py` exercises the low-level physical
full-range source inventory, reconverged native-gradient differences, raw
Hessian/HVP agreement, translation, lifetime and resource rejection. These
small fixtures are not independent complete electronic/Ni qualification.
An additional independent PySCF/Libxc sentinel matches the actual raw basis,
VWN-RPA graph, overlap, converged density and total energy, then compares the
HVP's directional curvature to reconverged reference energy differences at
three steps with a fixed 1e-6 Hartree/Bohr² gate. It supplies the exact native
quadrature primal as an input at each geometry, not a native density seed or
derivative; the reference electronic solves are independent. This scalar
curvature check is not a substitute for an independent full-matrix/Ni oracle.
Exporting that primal also avoids treating far-tail Legendre/Becke roundoff
between NumPy and native grid reconstructions as a different quadrature model;
the scientific energy, density and curvature gates are not relaxed to hide it.

The initial native trial with an unchanged B3LYP semilocal graph and exchange
fraction 37/100 failed during SCF (`invalid argument`), before response or
Hessian execution. The native canonical B3LYP SCF wrapper still authenticates
its prepared Fock strategy against the generated 1/5 fraction. Do not weaken
that identity check or claim custom-coefficient molecular qualification from
the independent compiler grid tests. Extending the shared native composition
contract and its final-state/force proof remains part of #2224's prerequisite.

### Follow-up to the initial native coefficient failure

The initial wrapper limitation recorded above is superseded by
[CPU hybrid coefficient provenance](2026-11-04-cpu-hybrid-coefficient-provenance.md).
The follow-up passes the authenticated physical coefficient into the shared
SCF wrappers, preserves prepared-strategy and final-state identity checks, and
adds independent full-matrix H2/HeH+ evidence for 1/5 and 37/100 exchange.
It does not supersede the remaining #2224, Ni or public-admission gates.

## Revisit when

The independent matched-state and matched-grid molecular checks required by
#2224 and #2843 pass, including real-spherical Ni and installed endpoints. Only
then widen the public admission gate. UKS, D3 and ECP retain their own source
and complete-endpoint requirements under #2844–#2847.

## References

The public-admission deferral is superseded by
[qualified public full-range execution](2026-11-05-full-range-public-second-order.md).
The earlier failures and low-level-only evidence above remain historical facts.

- Issues #2224 and #2843–#2849.
- `docs/developer/dft_hessian.md`.
