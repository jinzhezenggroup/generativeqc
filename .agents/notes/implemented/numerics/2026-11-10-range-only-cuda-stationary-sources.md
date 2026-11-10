# Decision: consume range-only mixed KS derivatives without a VV10 lease

Status: implemented
Date: 2026-11-10

## Problem

Issue #2840 retained three different failures. Its mixed DF-primary J/K plus
exact Direct LR-K energy fixture is not a WB97M-V force fixture: its PBE point
program has no nonlocal-correlation primitive. The final-state provider fix
admitted the intended energy Hamiltonian, but public force routing coupled
every SR/LR source inventory to the separate resident range/VV10 controller.
An energy/state-only pass therefore did not establish the original force gate.

## Decision

Reuse the ordinary generated CUDA geometry consumer for the range-only GGA
inventory. Admission requires its qualified SCF point domain, no intrinsic
semilocal range parameter, both SR/LR primitives and a nonzero LR correction.
CPU, ECP, nonlocal and other unqualified domains retain their existing rules.
No new functional alias, handwritten derivative kernel or independent exchange
algebra is introduced.

The generated artifact keeps the complete plan's source order, including SR/LR
slots. The existing native mixed bridge authenticates the independently owned
fitted primary and exact LR correction and supplies five contracted rows:
one-electron, overlap/Pulay, fitted Coulomb, SR and LR. Both range rows must be
included exactly once in the final sum, not just in diagnostic components.
The compiled geometry owner supplies the moving-grid XC response and nuclei.
Unavailable native integral rows fail closed; an exact full-range or CPU oracle
substitution would change the production contract and is not a fallback.

The existing range/VV10 controller remains separate and still requires its
nonlocal feature/grid leases. Prepared artifact identity and cache recipes bind
the complete source layout. Resource accounting retains the native bridge's
declared scope; this is not a resident-memory or performance promotion.

## Rejected alternatives

- A WB97M-V force run cannot replace the original range-only PBE Hamiltonian.
- Leaving the original public force request unsupported would avoid a required
  endpoint even though its native derivative sources already exist.
- Relabeling a semilocal-only generated artifact or summing only its device
  reduction omits the separately published SR/LR derivatives. An exploratory
  run exposed a 0.1292169721472736 Eh/Bohr force error despite correct E/D/Fock;
  that failed observation is retained, not relabeled as a passing endpoint.
- Weakening canonical, commutator, energy or force gates is not a repair.

## Validation contract

`tests/python/test_ks_original_rsh_admission_cuda.py` preserves the native H2/H3
primitive decimals, geometry and moved-coordinate rule, original grid and SCF
controls, identical orbital/auxiliary bases and exact LR approximation. Both
spin and representation variants exercise cold/warm/moved consumers. Independent
PySCF reconverges the same fitted-full-range/exact-LR Hamiltonian for E/D/Fock;
five-point derivatives of its independently rebuilt moving-grid energy check
the complete native analytic force, including auxiliary-center response.
The original large-spd UKS controls retain their separate analytic-force oracle.
Measurements must retain native-library, compiler-artifact, source-patch and
device identities; no receipt qualifies another binary or GPU architecture.

## References

- Issue #2840 and PR #3060.
- `.agents/notes/implemented/numerics/2026-11-09-ks-physical-force-and-canonical-proof.md`.
- `docs/developer/xc_scf_domain.md`.
