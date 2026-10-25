# Decision: Share the authenticated DF-HF source and metric with correlation

Status: implemented
Date: 2026-10-25

## Decision

Full-DF MP2/CCSD(T) already retains its DFReference for the orbital and nuclear
response. Its correlation source now borrows that same immutable integral source
and forward metric eigensystem by default. It checks the published physical
reference, exact orbital and auxiliary basis/geometry, device, dimensions,
relative metric cutoff, radial operator and frozen generated scheduling/math.
A mismatch or `GENERATIVEQC_DF_SHARED_SOURCE=0` retains fresh construction.
Exact-RHF plus DF correlation has no eligible reference owner and is unchanged.

The ordinary SCF B/E factors are not interchangeable with the correlation
contract. Fresh raw high/low expansions and compensated AO-to-MO contractions
remain mandatory. Sharing removes duplicate source construction, metric
transfer and factorization, not the remaining raw MO-factor generation. The
owner's symmetric inverse root remains separate from SCF's spectral B gauge.

## Ownership and capacity

The source builder's caller already reserves the DFReference lease outside its
remaining budget. The builder charges only fresh projections/output during
forward execution and reports the overlap explicitly. A retained response state
holds a strong reference to DFReference before its stream-dependent coefficient
buffer. Its standalone response census includes the complete reference lease;
the method subtracts that overlap exactly once from other live owners. Tight
budgets still refuse before publishing a partial result. These owners retain
the existing sequential reference/correlation execution contract.

Identity is not inferred from dimensions, approximate coefficients, or a hash.
A copied physical reference cannot authenticate itself by reproducing values.
A changed auxiliary exponent of the same shape must build a fresh source.

## Validation and limits

The compiled lifecycle probe compares all factor/block outputs against fresh
construction, rejects same-shape auxiliary changes, threshold changes and a
detached frame and changed source policy, exercises budget/backend refusal, and performs a physical source
pullback after the outer HF handle is released. It checks final owner release.
Complete PySCF energy and reconverged directional-force gates cover full-DF
MP2 in spherical/Cartesian bases and CCSD(T), including auxiliary g functions.
Complete endpoint timings and source-bound receipts accompany qualification.

This change does not share arbitrary SCF factors with the separate legacy RI-MP2
builder, cache changing MO projections, or add new Hessian/Lambda intermediates.
Those require their own precision/frame contracts and measured endpoint gains.

Initial [source-bound endpoint evidence](../../../../benchmarks/results/df-shared-source-20261025/README.md)
shows a small setup saving: full-DF water MP2/cc-pVTZ cold decreases from 1.2860
to 1.2798 seconds; CCSD(T)/def2-SVP from 0.8099 to 0.8033 seconds. Both arms
retain identical compensated source/transform work. Sharing is useful for
eliminating duplicate ownership and metric setup; it is not evidence that
ordinary SCF factors can replace compensated correlation factors.
