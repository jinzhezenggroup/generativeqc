# Decision: bounded cold incremental proposals with ordinary warm proof

Status: implemented in the local candidate; global incremental default remains off
Date: 2026-10-23
Base: `afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e`

## Problem

The [V3 experiment](../../proposed/2026-10-22-incremental-warm-policy.md)
removed unnecessary convergence work but still alternated a screened delta with
a full rebuild. PBE0/96 performed six full Focks and only three deltas. A matched
GPU4PySCF comparison instead performed one full plus nine deltas and gained about
5% at its default screening. Its looser screening case needed another iteration
and lost the gain: complete endpoint work, rather than delta kernel speed alone,
must decide whether the change is useful.

## Decision

For all-electron full-range RKS owners, distinguish prepared incremental storage
from effective per-solve execution. Internally
retained same-geometry warm states use the ordinary full-density proof and their
validated energy history. Changed geometries, external seeds, and cleared warm
state restore incremental execution. This selection is recomputed on every begin.

Admit incremental owners to the existing Standard convergence policy when its
original model/grid, spin, precision and complete resident physical/canonical
validation conditions hold. Delta J/K adds no functional whitelist: its linear
operator algebra is independent of the XC family. Explicit
strict requests retain the conservative policy. Start the full-density tail when
density change and physical RMS predict stationarity; AO maximum and all final
canonical gates still decide acceptance. A just-executed full Fock may supply
the first closing energy, whereas a delta energy never may. Successful closure
can use the ordinary resident final-validation products.

Within that capable owner, screened incremental operators are bounded proposals.
Their refresh interval is the smaller of the positive requested interval and
eight; zero also selects eight. They never supply the closing full-energy pair or
final physical proof. `GENERATIVEQC_KS_INCREMENTAL_PROPOSALS=0` restores the V3
screened cadence for diagnosis. The shared exact SCF/HF incremental controller
still enforces its existing one-delta screened refresh rule.

The provider does not expose an actual omitted-error bound. Summing nominal
screening thresholds would not certify a chain, so no such claim or artificial
error ledger is introduced. The finite proposal cap and ordinary total iteration
limit bound work; fresh full-density closure decides publication.

## Boundaries and invariants

- Global incremental mode remains an explicit experimental opt-in. The public
  KS resource inventory still rejects its unqualified capacity contract.
- Incremental admission adds no functional-name restriction. Existing Standard
  model/grid qualification remains in force; UKS, ECP, range/nonlocal, lower
  precision, fitted providers and incomplete physical-proof owners retain their
  previous incremental policy. Ordinary warm proof is selected for retained
  all-electron full-range RKS states, independently of XC family.
- Full-density XC, unchanged force operators, two consecutive physical full-Fock
  energies, original energy/density/residual gates, and final canonicality remain
  mandatory. Proposals cannot create accepted energy history or reusable states.
- Warm diagnostics truthfully report requested=true and active=false. Cold work
  counters classify actual full/delta builds; unavailable quartet counters stay
  unavailable rather than being inferred from timings.

## Evidence

The V4 precursor (before removing its extra PBE/PBE0 guard) was measured in
three rotated-order rounds on n1/RTX5090, native initial guesses, PBE0/def2-SVP,
spherical AOs, stock grid and strict FP64 Direct. Cold includes construction,
preparation and full E/F; each process then executes two frozen warm E/F calls.

| Atoms | Master cold | V3 cold | Candidate cold | Candidate/master |
| --- | ---: | ---: | ---: | ---: |
| 12 | 3.16902 | 3.16159 | 3.08973 | -2.50% |
| 48 | 13.12015 | 12.81732 | 12.69048 | -3.27% |
| 96 | 30.40414 | 29.83235 | 29.03728 | -4.50% |

Seconds are medians; every paired cold round improved. Total cold Focks stay
10/9/9 while deltas increase from three to six. Every warm call remains one full
Fock. Independent-process 96-atom warm medians are 8.26583/8.36141 seconds
(master/candidate, +1.16%); retain this variance. Same-binary alternating warm
controls also retain all long tails and show no systematic on-mode regression.
Neither comparison establishes exactly zero latency impact for arbitrary inputs.

All 160 recorded E/F endpoints pass independent tighter full-density GPU4PySCF
references: PBE/PBE0 water clusters, ammonia and methane, displaced geometries,
frozen warm, reset, strict E/D, refresh intervals 0/1/100, and proposals disabled.
RKS references suppress both density and potential history; direct_scf=False
alone is insufficient. References use E=1e-12, gradient=1e-9, screening=1e-14,
and moving-grid response. No oracle density enters native timing. This qualifies
solver error at the sampled default grids, not independent quadrature convergence.

Gates are E/atom <2.5e-6 Eh, force max <2.5e-5 and RMS <7.5e-6 Eh/Bohr.
Observed maxima are 6.02e-12, 2.04e-6 and 6.35e-7 respectively. Eighty-one host
tests and two real-CUDA independent-reference regression tests pass. GPU work
uses finite Slurm allocations; compilation reuses verified ccache 4.5.1.

The final head removes the additional XC-family guard and adds 40 independently
validated endpoints: B3LYP/r2SCAN/M06-2X water-12 original/moved cold/warm/reset,
B3LYP/r2SCAN strict controls and water-48 resident validation, and a PBE0/96
master/V4/head bridge. All 200 recorded endpoints pass; the new cohort maximum
force error is 1.17e-6 Eh/Bohr. Five real-CUDA reference tests cover PBE, PBE0,
B3LYP, r2SCAN and M06-2X. The 81 controller/resource and 130 inventory tests pass.
The bridge's cold times are 30.18497/28.69142/28.83080 seconds, with nine Focks
in every arm, six deltas for V4/head, and one full Fock for every warm call.
This single bridge verifies the final revision without relabelling V4's
three-round measurements as final-head data.

## Rejected alternatives and consequences

Warm-only repair leaves cold much slower than master. Counting reconstructed
energies or omitting physical closure would change the acceptance contract.
Unbounded screened chains have no supported accumulated-error certificate.
The previously tested late-density threshold did not consistently improve full
endpoints. An initial V4 candidate added a PBE/PBE0 whitelist; this was removed
before PR submission because J/K linearity is an operator capability rather
than an XC-family property. Cross-functional endpoint qualification supplements
the shared final-proof contract; it does not create another method-name gate.

The retained full tail limits potential gains relative to continuous GPU4PySCF
deltas but preserves the required physical proof. Revisit the cap, broader
provider/proof admission, and default promotion only with independent displaced
E/F evidence, complete cold/warm timings, and public resource accounting.

## References

- [Retained source-matched receipts](../../../../benchmarks/results/incremental-cold-proposals-20261023/README.md)
- [Current incremental contract](../../../../docs/user/ks_options.md#experimental-incremental-direct-jk-and-resource-budgets)
- External full workspace evidence: `../evidence-delta-d-opt-20261023/`
- Measured V4 CUDA KS source SHA256: `ab6e1e94a3423b22af0d010630e03f34f800f77da89f2510d449c69339e3c086`
- Measured V4 library SHA256: `4789e2e8722c4151f82946bb40e7d29ae53add737363d9c3d5968836c1120b95`

Final-head CUDA KS SHA256: `7218190a885f66e287a4e8ee65052e3fbca05426a42c51229412cb30cab8b9ad`.
Final-head library SHA256: `16174b3618d8124e2b18e872c9b0a4908e7c7f9f3453508da2f04d78602e395f`.
