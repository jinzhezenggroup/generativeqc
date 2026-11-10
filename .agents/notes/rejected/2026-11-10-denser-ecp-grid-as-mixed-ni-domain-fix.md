# Decision: reject simple ECP grid densification as the mixed-Ni fix

Status: rejected
Date: 2026-11-10

## Problem and preserved experiment

Issues #2846–#2847 require canonical real-spherical def2-SVP with I/ECP28,
Cs/ECP46 and mixed all-electron Ni. The retained mixed fixture is Ni
`(-2.7,0.2,-0.1)`, I or Cs `(0.03,-0.02,0.01)`, H `(0.17,0.11,3.1)` Bohr,
neutral singlet. Do not move its atoms to relabel the failed domain as passing.

Independent PySCF `ECPscalar` matrices, normalized to the native AO convention,
were compared to the generated native raw provider without SCF. All canonical
records come from `tests/data/global_hybrid_ecp/manifest.json`. The production
160x32 / 224x44 gates remain `2e-9` for matrices and `2e-8` for derivatives.

| Element | Raw grid | Maximum independent matrix error (Eh) |
| --- | --- | --- |
| I | 224x44 | 1.8152884689e-7 |
| I | 384x80 | 5.6287521328e-9 |
| I | 512x96 | 6.7909078790e-9 |
| Cs | 224x44 | 5.4326528479e-5 |
| Cs | 384x80 | 1.7204432333e-6 |
| Cs | 512x96 | 2.0786914520e-6 |
| Cs | 768x128 | 2.3515723186e-7 |
| Cs | 1024x160 | 1.4545283329e-7 |

The 384x80 to 512x96 nonlocal derivative differences are `1.4154403836e-7`
for I and `4.2874256224e-5` for Cs. The **768x128 to 1024x160 Cs difference
is 7.2967910650e-5**, not a converged force result. Those latter raw endpoints
take 317.83 and 672.31 seconds respectively; they are not complete molecular
timings or a performance claim. The largest reported independent matrix error
is a total local-plus-nonlocal comparison; channel-wise two-grid differences
are separately recorded and must not be substituted for independent accuracy.

## Decision and invariants

Do not replace production matrices with a dense raw export, weaken the gates,
or make primal/force/response select different quadrature Hamiltonians. The
existing Hessian validation-only 320x60 check cannot rescue an inaccurate
224x44 native Hamiltonian. This negative evidence rules out that simple fix;
it does not establish a unique numerical cause.

Extend only explicit CPU **raw diagnostic** bounds to 1536 radial / 192 polar
points so the failed refinement can be reproduced. CUDA stays at 512 / 96.
The generated root iteration, ordering, fixed checked grids and production
capability selectors are unchanged. An independent closed-form normalized
s-Gaussian test exercises each newly available raw axis and translation
derivatives; it does not qualify mixed Ni/ECP chemistry.

An ECP-centered angular/radial rule may have difficulty resolving tight,
off-center Ni functions, but that remains a hypothesis. A replacement needs
consistent primal, first- and second-order operators, authenticated actual
quadrature/provenance, independent matrix/derivative gates and bounded work
and storage. An analytic angular/local treatment or a properly differentiated
bounded refinement is future algorithm work, not a delivered fallback here.

## Additional unrestricted-state blockers

The all-electron, native-default-grid NiH3 neutral doublet reaches native SCF
convergence at iteration 548 with energy `-1509.7972253482067` Eh and physical
residual `1.5003539351e-13`, under energy/density controls `1e-13` / `1e-11`
and a 960-iteration cap. Its public force call separately rejects the f-shell
stationary-force resource domain. A successful energy-only repeat is not a
successful force endpoint. Its independent cold UKS reference fails its
200-cycle convergence gate; no state match or curvature is inferred.

A NiH neutral-doublet probe converges natively at iteration 198 to
`-1508.6876441475483` Eh with residual `8.6972588701e-14`. The independent
cold 960-cycle reference fails, at `-1508.6614558340173` Eh and orbital
gradient norm `6.4417691496e-7`. Do not accept its density or construct
finite differences from it. Independent root search is validation work;
native and oracle densities must never be interchanged to conceal a mismatch.

Four independently initialized Newton retries (`minao`, `atom`, `huckel`,
`1e`; 200 macro cycles at unchanged energy/gradient controls) also report
false convergence. The atomic guess comes closest in energy (`2.05e-12` Eh
from native), but its orbital gradient is `9.43277e-8` and density difference
`7.36062e-7`, outside the declared state gates. Repeating with exact in-core
integrals and the prior Ni protocol's `ah_lindep=1e-24` does not resolve it.
This is an explicit example of why energy agreement alone is insufficient.

These failures keep the requested unrestricted, ECP and molecular TS gates
open. D3's already qualified restricted composition cannot erase them.

## Evidence location

The raw local probes and unabridged failed logs are retained under
`/tmp/gqc-complete-*` in this checkout environment. This note preserves the
compact, source-recorded negative findings without adding large transient
archives to the repository or changing its evidence budget.
