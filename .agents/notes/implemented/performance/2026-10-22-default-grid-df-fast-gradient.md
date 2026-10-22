# Decision: admit qualified DF Fast gradients on the default Treutler grid

Status: implemented
Date: 2026-10-22

## Problem and decision

The fitted stationary precision policy admitted only two historical tensor
grids and at most 96 AOs / 12 atoms. After adopting default Treutler/Lebedev/
NWChem quadrature, Fast therefore returned strict Standard even when the
existing semilocal geometry consumer could execute mixed arithmetic.

Retain the original tensor-grid domain and add a separately identified
`<method>-df-rks-treutler-budget-fp32-v1` domain: canonical PBE, PBE0, B3LYP,
r2SCAN and M06-2X RKS, exact default level-3 grid descriptor, 96–1200 orbital
AOs, 2–96 atoms, and orbital shells through d. Canonical r2SCAN-3c additionally
requires both successful typed D4/gCP correction results and its defining
basis. Keep native fitted J/K, the default metric threshold, strict SCF controls,
full grid response and all pre-existing complete-model/provider checks.
Only semilocal geometry narrows; this does not introduce mixed DF SCF or
change quadrature, convergence, corrections, integral derivatives or metric
response. Explicit accuracy targets retain their independent audits.

The 96-AO floor is a profitability decision, separate from numerical evidence.
Measured small default-grid endpoints did not consistently amortize conversion
and dispatch: PBE and B3LYP monomer warm timings regressed. They retain the
existing one-solve Standard fallback. This is not a guarantee that every
single cold timing or every future molecule improves.

## Evidence and limits

Ignored `df-md-fast-domain-v1/` and `df-md-fast-domain-v2/` receipts bind frozen
source manifests, Python overrides and the independently qualified native
`df-md-force-plan-v1` core. Across six canonical models, 3/12/48/96 atoms and
Standard/Fast, all 48 complete cold/three-warm populations finished. 46 pass
the unchanged independently stable dense-grid/CPU-spot common gates. Both
M06-2X/3 arms retain the known default-grid RMS-force failure; they are not
ranked or relabelled as passing. Mixed-versus-Standard force differences in
the second campaign are at most 3.85e-7 Eh/Bohr, far below the common force
gates. The complete returned precision records show one strict SCF solve and
actual mixed geometry work, without hidden observable-audit endpoints.

All measured 12–96 atom warm populations improve. Examples (Standard / Fast
median seconds): PBE96 7.067 / 6.549, r2SCAN96 7.891 / 6.894, r2SCAN-3c96
18.489 / 17.026, PBE0/96 11.311 / 10.844, B3LYP96 11.343 / 10.866 and M06-2X96
11.812 / 11.407. Cold benefits are smaller and PBE/B3LYP examples do not show
a consistent improvement. The final floor is separately exercised by
`df-md-fast-domain-v3/`: all six three-atom Fast endpoints keep one strict
Standard solve, with the same known M06-2X grid failure.

Host policy tests check default versus legacy grid domains, size/resource
qualification boundaries, untrusted sources, stale provider errors, actual
precision-work records and complete correction ownership. This static
qualification does not establish arbitrary-molecule observable-error bounds,
universal superiority over GPU4PySCF, or a finished README campaign.

## Revisit when

Extend the scientific domain only with independent complete energy/force
qualification. Revisit the small-input profitability floor when conversion
or launch overhead changes. Mixed DF SCF needs its own provider/arithmetic
admission and strict final-state evidence; this gradient proof cannot confer it.
