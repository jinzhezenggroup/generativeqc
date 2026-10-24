# Decision: distinct corrected owners, one physical/nuclear implementation

Status: implemented
Date: 2026-10-12

## Problem

Corrected Lambda and complete same-space Fock moments do not individually
provide physical UCCSD(T) forces. Their raw Hamiltonian and symmetric metric
chains must reach the shared UHF solve and every nuclear derivative. Letting
corrected records inherit standard CCSD qualification would erase the method
boundary; copying the standard force owner would introduce divergent orbital
sign, spin ownership and nuclear-repulsion algebra.

## Decision

Factor private physical/nuclear implementation bases and private immutable
record bases. Public standard and corrected types are siblings, with distinct
input/type gates. The standard public signatures, scientific qualification,
complete all-rotation gate and nuclear derivative schedules remain unchanged.

Borrow corrected Lambda and full-Fock owners with the exact same live triples
owner, parent/state/backend. Freshly qualify Lambda and replay full Fock, then
compare all fresh source bars/channel energies at the existing bounded drift
tolerance. Admit both full replay endpoints and simultaneous arrays. Do not
assume detached records are fresh capabilities or numerical hashes are stable.

Seed the existing raw-Hamiltonian generic VJP with complete correction bars
and zero reference-electronic-energy bar. Share its general-frame transport
with the composed CC pullback: stationarity `G-G.T`, overlap
`-(G+G.T)/4`, and ov stationarity for shared UHF `C exp(-K)`. Add correction
weights before both the initial RHS and final CC+reference+Z stationarity.
The denominator full-Fock moments replace, rather than accompany, eps bars.

The corrected nuclear owner shares spin frame transforms, native one-electron
binding, one-electron metric derivatives, all ordered shell derivatives and
one nuclear-repulsion contribution. Primal CC checks remain CCSD checks;
only corrected records publish fresh CCSD+(T) correlation and total energies.

## Rejected alternatives

- Standard-result subclasses or disabling standard Lambda qualification hide
  the corrected stationarity contract.
- Full-Fock moments without symmetric overlap transport omit Pulay response.
- An ov-only final gate permits nonstationary same-occupancy rotations.
- One raw-Hamiltonian replay per parameter duplicates complete scientific work.
- Trusting an earlier detached triples response avoids necessary freshness
  checks without an explicit, qualified lifetime capability.
- The independent PySCF corrected gradient cannot reuse `_native_oracle`'s
  cached CCSD Lambda. Explicit `uccsd_t_lambda.kernel` results do not fix the
  measured discrepancy here: the installed PySCF 2.14.0 unrestricted analytic endpoint
  differs from its own complete-energy FD by 1.2526e-3 (restricted H4 z) and
  1.2766e-3 (NH2 z). This is an invalid reference on these fixtures, not evidence
  for relaxing production forces. Independent restricted corrected analytic
  gradients agree with the same UCCSD(T) energy derivative. Retain that valid
  analytic gate and strengthen every open/broken force to all-coordinate FD at
  2e-7; do not omit failed coordinates or infer them through symmetry.

The opt-in local self-FD diagnosis separately reconverges the reference's own
UHF/UCCSD/(T) energies, with no production equation call, and compares the
restricted analytic endpoint to the same total energy. Preserve this diagnosis
when revisiting independent-oracle suitability; do not attribute the whole
discrepancy to initially cached standard Lambda without reproducing the
explicitly corrected result.

## Evidence and boundaries

`tests/python/test_uccsd_t_gradient.py` qualifies broken H4, asymmetric H6,
NH2, native nuclear binding, independent reconverged energies/analytic forces,
restricted force parity, freshness, distinct types, admission and same-space
stationarity rejection. Real CUDA qualification forbids CPU equation and
nuclear-derivative fallback and requires finite Slurm allocations with assigned
visibility. Raw measurements remain ignored local artifacts; current contract
and reproduction guidance are in `docs/developer/uccsd_physical_response.md`.

No Calculator registration, frozen core/ECP support, performance claim or
process-RSS cap follows from this internal composition. Revisit repeated
triples replay only when a fresh, exact-state borrowed capability can preserve
every binding and admission gate without trusting detached convergence data.

Qualification snapshot: all 16 host cases are covered by a successful 14-case
host run and the final nine-case failure/admission run (seven overlap). Four
complete real-CUDA force cases pass under Slurm 7828 on node1, with finite
40-minute admission and assigned visibility preserved; CPU equations and
nuclear derivatives are forbidden. These cover collapsed H4, broken H4, NH2
and asymmetric H6, including all four triples sectors. Standard shared-path
regression passes 32 cases (three CUDA skips). A one-off opt-in comparison
preserves 12 original raw/CC/metric graph identities; this is evidence for the
refactor, not a permanent required-CI source fingerprint. CI ownership passes
99 cases without reducing worker counts, budgets or scientific coverage.
