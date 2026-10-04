# Decision: retain the complete 24-atom TZVPD pair without relabeling earlier runs

Status: implemented
Date: 2026-10-04

## Evidence and interpretation

The full spherical def2-TZVPD water-24 case has 464 AOs and 589,824 matched
semilocal/VV10 grid points. Slurm 5725 on n1 device 2 completed all 12 native
and 12 GPU4PySCF original/displaced calls with five warm repeats per geometry.
All calls pass the unchanged 1e-8 Eh / 1e-7 Eh/Bohr independent gates; the
reference reports CUDA LibXC for every recorded XC component.

Native/reference prepared cold is 5752.509670 / 471.074847 s; original warm
medians are 584.661243 / 57.365026 s. Native uses 22 cold iterations versus
reference 39. Preparation costs 1.644111 s and the recorded force stage costs
350.091709 s, so their difference from complete cold is 5400.773851 s. This is
an arithmetic remainder, not a measured exchange or SCF component. Missing
Fock-build and screened/executed molecular quartet counts remain unavailable.
The force geometry/pair drain is combined and does not time VV10 separately.

These observations support the ongoing high-order value/derivative work; they
do not show parity, a performance improvement, or default-policy qualification.

## Identity and publication contract

The newer capacity source is `302039f8e26a30c2efb6578d409d6c83f8930493`,
source identity `bd67b5aa0b87fc329d84a002c6fae8c518f41177246a5b9d06415b2897810178`,
and library SHA-256
`4da2d760e4b85e6c8ab5aee199cd541f3240656967f706c0765bdebf62ba488e`.
Its force host/device budgets are each 4 GiB. The earlier 3/6/12 records retain
their original source, library, automatic budgets and three reference/none/LDA
variants. No LDA source was run at 24 after the 12-atom regression.

The shared publication therefore fixes its per-point variant inventory:
reference/none/LDA at 3/6/12, reference/none at 24. It does not infer absent
variants from files. Each point must retain complete original/displaced warm
repeats, matching reference bytes, actual source and library identity, paired
Slurm allocation/device, correct basis, explicit resource controls, and source
lifecycle accounting when present. Missing seeds are not zero-cost seeds.

The enlarged evidence record is deterministically compressed through the shared
publication tool. Raw scalar/force data and all negative seed observations are
preserved in the existing samples bundle. The graph connects recorded points
with explicit build differences; it is not a same-binary scaling experiment.
48/96 remain pending and receive no inferred timing points or timeout bounds.

## Validation and revisit condition

The read-only verifier accepts 132 retained calls. Forty-three tests include
forged forces, missing legacy LDA or capacity records, changed capacity/library,
cross-device pairing, invalid lifecycle durations and untrusted bundle code.
The numerical gate still executes under `python -O`.

Only add a later 48/96 point after the entire native/reference pair is complete
and independently accepted. A numerical oracle from a different allocation may
not silently substitute for the paired timing reference. Future optimization
claims require same-source/GPU controls and their own complete endpoints.

Publication: `benchmarks/results/wb97mv-tzvpd-cold-20261004/`.
