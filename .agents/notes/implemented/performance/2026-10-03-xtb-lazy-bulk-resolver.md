# Decision: initialize bulk-XC resolution only for its consumer

Status: implemented
Date: 2026-10-03

## Problem and decision

The first GFN2 calculator imported `resolve_bulk_ks` with the common method
types. That lazy compiler export initialized bulk-XC resolution and its compiled
evidence, although native selectors never called the resolver. A constructor
profile attributed approximately 7 ms to those imports.

Import the resolver inside the existing automatic Libxc selector branch. Keep
selector admission, compiler ownership, scientific controls and native loading
unchanged. Automatic KS requests still resolve through the same compiler API;
no result, device or native-handle cache is introduced. Moving imports before
the measured constructor would only hide work and is not the chosen approach.

## Evidence

Existing bulk-KS resolution/method discovery tests pass (16), as do Libxc public
API tests (11). Each of n1 RTX 5090 and n2 RTX PRO 6000 passes 41 public CUDA
tests. The final metadata-refreshed native build passes 43 tests on n2, including
the preflight harness. Builds use ccache; GFN2 objects were unchanged by the
metadata refresh.

On each GPU, parent, candidate and xTBloom each execute 110 complete energy/force
samples over ten cases, followed by six fresh-process H2 sequences per variant.
Variant order alternates for the startup sequences. Every energy/force and SCC
count gate passes against both parent and xTBloom. Maximum candidate versus
xTBloom errors are 5.69e-14 hartree and 5.17e-15 hartree/bohr; versus parent,
energies agree exactly and forces differ by at most 5.56e-17 hartree/bohr.

Constructor plus first-call medians and ranges, milliseconds:

| GPU | Parent median (range) | Candidate median (range) | xTBloom median (range) |
| --- | ---: | ---: | ---: |
| n1 | 752.565 (521.447–890.962) | 740.893 (529.676–846.895) | 720.576 (685.544–838.406) |
| n2 | 403.613 (381.246–409.277) | 394.116 (375.124–395.632) | 388.336 (361.253–390.505) |

n2 supports an approximately 9.5 ms median reduction against the parent, while
n1's spread is much larger than the change. Neither establishes startup
superiority over xTBloom. A separate n2 cProfile shows first-constructor calls
falling from 7908 to 2351, with bulk-XC imports absent. That profile's total
constructor time is not endpoint performance evidence.

This cohort has nine winning cold-total medians on each GPU. Warm comparisons
win 9/10 on n1 and 10/10 on n2; changed-geometry comparisons win 10/10 on both.
n1 H2 warm is 2.858847 ms versus xTBloom's 2.853639 ms, a retained regression
of about 5 microseconds. Candidate/parent warm speedups range from 0.9971 to
1.0012 on n1 and 0.9992 to 1.0011 on n2: this import change does not establish a
steady-state gain. Other single-sample cold regressions remain in the receipts,
including n1 CO (15.148 to 17.336 ms) and water-32 (228.125 to 229.485 ms).

The timing contract is `calculator-cold-v3-lazy-library-load`: package imports
precede measurement, actual lazy native loading is charged, and cleanup is
separate. These are not OS-launch or cold-filesystem measurements. All samples
retain FP64, fresh SCC, 300 K, Broyden 8/0.4, maximum 300 iterations and
energy/charge tolerances 1e-10/1e-8. GPU work uses finite Slurm allocations.

## Reproduction and identities

Ignored receipts: `.artifacts/{n1,n2}/lazy-bulk/`, with `before.json`,
`after.json`, `xtbloom.json`, comparisons, and six `startup-*` sets;
`.artifacts/run-lazy-bulk.sh` records the complete commands. Profiles and final
qualification are `.artifacts/n2-lazy-bulk-cprofile.log` and
`.artifacts/n2-lazy-bulk-final.log`.

- Parent source: `da9efd7f1`.
- Candidate `calculator.py` SHA256:
  `a98a0be42627fd1c7a3587e2131a4f0ad9e9c60e29e71fa4ba64121f7253d455`.
- n1 measured native SHA256:
  `fc23ea615764d458bb3281ceed72749ac0466bd27e32b62bbd0fdbb7fc467145`.
- n2 measured native SHA256:
  `2bb5afe520e8c079785e759f6a621a32b0f2fc987c16cd90c1f7a3c37b12a408`.
- Final metadata-refresh native SHA256:
  `0ae0d819ca11c3218167c46f4f4c8e81dcaabcb0eb0459024091954e20d985c6`.
- xTBloom revision: `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`; native SHA256:
  `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

Parent and candidate use the identical native binary within each GPU cohort.
Revisit the import boundary if a native selector acquires a real bulk-XC
consumer; preserve automatic KS resolution and count initialization in endpoint
timing.
