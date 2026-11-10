# Decision: complete the original KS scientific tasks

Status: implemented
Date: 2026-11-10

## Problem and decision

An altered-grid ownership test did not settle #2915's original discrepancy;
a range/VV10 endpoint did not replace #2840's range-only PBE Hamiltonian.
Repair the complete consumer contract without changing original inputs,
scientific assertions, memory limits or iteration budgets.

Force intent is bound before SCF even when retained storage needs no additional
workspace. Its physical stationarity ceiling is independent of energy-only
stopping. Eligible pure UKS publishes only after physical F[D] and independent
canonical reconstruction agree, avoiding a destructive unextrapolated update.
Range-only forces consume the existing mixed native integral bridge and
compiled CUDA moving-grid owner, summing both SR/LR contributions exactly once.

## Controlled original-grid water96 evidence

Preserve the exact 96-atom geometry/displacement, spherical def2-SVP orbital and
auxiliary bases, FP64, 512 MiB DF cap, Treutler level 3, force tile 1024, and
cold/rejected import/atomic restore/frozen replay/moved/fresh sequence.
Default E/D controls stay 1e-9/1e-5; only the separate tight arm uses 1e-12/1e-10.
Acceptance remains 1e-8 Eh / 1e-7 Eh/Bohr.

Same-device parent/repair pairs on RTX 5090 reproduce the reported failure:

| Controls | Parent force difference | Repair force difference |
| --- | ---: | ---: |
| Original default | 4.794283938736044e-7; 45/288 fail | 3.398517023356362e-10 |
| Tighter SCF only | 1.4027001782324078e-11 | 1.4004797321831575e-11 |

Parent default moved/fresh residual maxima are 2.1630536855967753e-8 and
1.1205372236498749e-8; repaired defaults reach 1.1013669043422849e-11 and
5.663237955655786e-12. Tightening only SCF independently demonstrates stopping
sensitivity on the original grid. A matching parent failure alone would not
establish that cause. This identifies a present-day base defect, not whether
historical #2910 introduced it. Old timeouts remain incomplete attempts.

## Integrated independent gates

Source `d3f8a62df6aa371ad75449406053d7ab3333c97a`, native input
`02dd0d238d7572be343496b4f3267a48b48ee2b32b9f8917035996a7ae2365cb`, library
`41efe7a154cfd8e428edbc9ff4c25c9a99f42c380839866d893dbc9002c8c34e`:

- Original water default/tight force differences pass at
  3.398525905140559e-10 / 1.4007461857090675e-11 (jobs 17048/17049).
- Eight original UKS Cartesian/spherical E/E+F x masks 0/127 controls pass:
  40 states, 20 analytic-force observations, canonical max 9.267731582163208e-9
  below 1e-8, independent force max 1.0525362970081886e-8 Eh/Bohr.
- Original range-only RKS/UKS Cartesian/spherical cold/warm/moved consumers
  pass: 16 E/F/physical-state observations, independent force max
  6.686293893228869e-12 Eh/Bohr. Reconverged five-point moving-grid energies
  include auxiliary-center response; CPU/PySCF is an oracle, not production.

Release/sm120/fast-compile builds use verified sccache 0.16.0. These are
diagnostic numerical receipts, not performance/production promotions, actual
Graph-launch proofs, or qualification transfers to A100/PRO6000. Overlapping
Slurm steps are disclosed.

## Retention and consequences

The source-bound publication retains full force/reference vectors, final
diagnostics, parent and exploratory RSH failures, artifact witnesses, exact
receipt identities and history hashes. Full iteration streams remain ignored.
Git recovery preserves earlier data without rewriting measured statuses,
raising evidence limits, deleting unrelated evidence or creating external
backups. The 2026-11-09 report stays historically inconclusive.

Prepared host fixtures must describe the complete source inventory. Empty
range sources on semilocal mocks preserve, rather than bypass, their artifact,
lifecycle and resource tests.

The follow-up resource review found that the ordinary owner decoded the mixed
bridge's lazy LR peak but read only the full-range one-electron field during
reserve validation and publication. The corrected route queries the native
lazy capacity before geometry planning and charges the mixed peak consistently.
DF response scratch remains explicitly excluded. Range-only public force
admission is through d, matching the ordinary compiled geometry layout; the
separate nonlocal owner retains its through-f domain. Native integral support
alone does not qualify a geometry layout that rejects the requested basis.

Two latest-source UKS attempts encountered independent PySCF DIIS nonconvergence
at the original 1e-13 energy / 1e-9 orbital-gradient gates and 200-cycle cap,
in different Cartesian force controls. Those failed reference observations
remain failures, not native passes or permission to relax a gate. The targeted
force oracle uses PySCF's independent second-order orbital solver with at most
one inner orbital update per macro cycle and the same 200-cycle cap. Geometry,
quadrature, Hamiltonian, native controls, basin seed, and all E/D/Fock/force and
physical-state comparisons remain unchanged. Solver identity and declared
bounds accompany every new independent-force observation. PySCF adapts its
effective inner-cycle count near convergence even when `max_cycle_inner=1`;
therefore `ah_max_cycle=1` also bounds each macro cycle to one augmented-Hessian
step. The initial Newton attempt without that second bound is retained as an
intermediate protocol, not evidence of the stricter one-step bound.

## References

- `benchmarks/results/ks-original-state-gates-20261110/` is the current compact
  publication. Its authenticated Git recovery record retains the complete
  vectors at `de57dc9c3118d47bc3eb2692a7b51320fad311db` without an external archive.
- `2026-11-09-ks-physical-force-and-canonical-proof.md`.
- `2026-11-10-range-only-cuda-stationary-sources.md`.
- Issues #2915 and #2840; PR #3060.
