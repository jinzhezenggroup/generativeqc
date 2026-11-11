# Accepted production Ni UKS raw full Hessian, not public endpoint admission

Date: 2026-11-11

## Scope and immutable source cohort

The separate serial full-matrix protocol now finishes with actual process exit
zero and receipt stage `accepted`. This advances the earlier
[directional-only acceptance](2026-11-11-ni-production-directional-hvp-acceptance.md);
it is not a relabelled directional action. The run starts clean at
`591c2ff40235f48f3201366005e42a6e602ad26b` and all eleven frozen numerical,
input, driver and binary hashes remain unchanged through final acceptance.
Subsequent concurrent documentation and unrelated saddle additions do not
restamp this run as a newer installed artifact.

The physical model remains direct CPU FP64 B3LYP/VWN-RPA, external spherical
def2-SVP, native default Treutler grid, neutral asymmetric NiH3 doublet and no
ECP or density fitting. There are 46 public/51 Cartesian AOs, 20 shells and
50,064 grid points, with alpha/beta occupations `(16,15)`. The geometry and
pre-frozen gates are unchanged from the linked action note. The library SHA is
`a451331d1b2fb2d613d0bf4c80dd0dc4a36cd08bd57183c0a0e1532fbcb668d4`.
No production callable is replaced, no production source is rewritten, and no
displaced production SCF, reference-derived integral or post-hoc matrix
symmetrization supplies the result.

## Complete scientific acceptance

Central native SCF again converges in 123 iterations to
`-1509.5720618893429` Hartree. Independent density error is
`7.872758001070679e-12`, energy error `2.7284841053187847e-12` Hartree and
orbital-gradient norm `4.003441143467981e-11`. Separate alpha/beta induced-Fock
errors are `3.3134606169937797e-13` and `2.987263214571101e-13`.

All twelve raw nuclear columns are computed by the real production
`composite_rks_hessian` endpoint, not assembled from independent references or
replayed from a saved matrix. Every original matrix gate passes:

| Quantity | Observed maximum | Frozen gate |
| --- | ---: | ---: |
| Coupled response true residual | 1.5455936231119348e-10 | 2e-10 |
| Raw matrix asymmetry | 4.2634451524747874e-11 | 2e-8 |
| Raw common-translation error | 4.672240372372016e-11 | 2e-8 |
| H@v versus separately accepted native action | 1.7274298311220626e-12 | 2e-8 |
| H@w versus fresh second native action | 2.7811529121346545e-12 | 2e-8 |
| Raw bilinear reciprocity | 1.4157078287446723e-14 | 2e-8 |
| Independent full matrix, 0.001 Bohr | 1.9040764520639186e-6 | 1e-5 |
| Independent full matrix, 0.0005 Bohr | 4.925145891121474e-7 | 1e-5 |
| Independent full matrix, 0.0003 Bohr | 2.05990007495771e-7 | 1e-5 |

The independent matrices retain all 72 converged displaced-gradient points,
their original failed/cold-retry history, charge/spin occupations and occupied
root-continuity gates. No native density seeds the independent reference. Every
accepted point's raw arrays and receipt hashes are rechecked before comparison.
No column, negative/soft mode, failed state or numerical gate is removed.

## Work, resources and boundaries

The full matrix consumes twelve nuclear response solves, 274 coupled-response
iterations and 572 grid actions. Its first-source diagnostics record 23,508 raw
first calls, 314,388 directional calls, 5,934,384 records, 113,113,476 component
evaluations and 5,827,932 factorized-geometry evaluations. The fresh second
native action adds one response solve, 22 iterations and 46 grid actions.
These are observed semantic work counts, not permanent spelling/count tests.

The measured serial process, including native SCF, state verification, full
matrix and the additional native action, takes 19,628.04 seconds in its receipt
(process wall time `5:27:08`), peak RSS 3,118,300 KiB, under its original finite
eight-hour limit. Independent displaced references and the prior native action
were prepared separately; their generation time is not hidden inside this
process measurement. The receipt does not record a native SCF Fock-build census.
This expensive four-atom endpoint is not a scalability or TS timing claim.

Receipt flags remain `low_level_native_full_hessian_accepted == True` and
`public_curvature_qualified == False`. Installed/public UKS/Ni admission,
broader geometry/failure/root lifecycle checks, D3/ECP totals, mixed Ni/I/Cs,
25–40-atom work qualification and fixed-method native TS/final frequencies
remain separate requirements. In particular, this does not authorize silently
removing a public admission guard or closing #2844–#2848.

## Verified recovery

The ignored local archive is
`build/qualification/ni-production-full-matrix-20261111/full-matrix-recovery.tar.gz`,
1,953 member/hash-verified entries, 165,823,016 bytes, SHA
`3eea90011bf3ff4fddf7cfdca0bef4f1f4f220dfe69315313b239e7b7ffdac15`.
It preserves the final raw matrix, both native directions, central states,
independent complete matrices and all point receipts/arrays, cold retries,
original protocols, frozen numerical source/library and actual JIT artifacts.
The linked previous directional archive remains unchanged. No Release, external
asset or new backup host is published.
