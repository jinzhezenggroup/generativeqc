# Decision: admit auxiliary g through the complete CUDA HF/KS DF owner

Status: implemented
Date: 2026-10-20

## Problem

GPU4PySCF's default def2-SVP-JKFIT auxiliary contains oxygen g shells. The
default even-tempered fit for canonical r2SCAN-3c does too. Python preparation
rejected both at `cuda/df_metric derivative 0 supports only l<=3`, before
native execution. This was an auxiliary-basis limitation, not an orbital-g
requirement or a grid limitation.

Generated auxiliary-g values and derivatives already existed. The complete
HF gradient adapter had gained six-term spherical metadata for the full-DF
correlation interface, but ordinary HF/KS admission still used the old generic
through-f role. Deleting only the Python check would encounter the native
system-construction check and would not establish a valid handle contract.

## Decision

Add a separate private `scf_auxiliary` system role for complete CUDA HF/KS
fitting values and first nuclear derivatives. Python selects it only when the
linked library exports its versioned constructor. Older libraries retain their
existing through-f behavior. Native preparation rejects this handle as an
orbital, in CPU DF, and in correlation-only/full-DF correlated methods, which
retain their own role. The values-only KS/Hessian role remains separate.

Reuse the existing generated value and six-term bounded derivative owners.
There is no recurrence, normalization, ordering, metric cutoff, precision,
resource policy or scientific default change. In particular, this does not
replace the reference's auxiliary basis with a smaller through-f fit.

## Rejected alternatives

- Globally raise CUDA angular limits: orbital-g and generic standalone exports
  have independent implementation and ABI boundaries.
- Reuse a correlation-only handle: that would erase method-role ownership and
  misleadingly broaden a previously narrower API.
- Remove the g shells from JKFIT: that changes the fitting model and cannot
  repair a comparison of the default auxiliary spaces.

## Validation

`test_scf_df_auxiliary_g_cuda.py` checks complete public cold and frozen-density
warm energy/force endpoints for HF, PBE, PBE0, B3LYP, r2SCAN, r2SCAN-3c,
M06-2X and WB97M-V in both Standard and public Fast, plus UHF and PBE-UKS.
The independent reference constructs its default auxiliary, which is asserted
to contain g. XC/VV10 grids remain at their defaults. Full moving-grid and
auxiliary responses are included. WB97M-V's oracle explicitly matches native
exact long-range exchange; the fully fitted stock comparator is not relabeled
as an identical-operator oracle.

Acceptance is `2e-7 Eh`, force maximum `2e-6 Eh/Bohr`, and force RMS
`7e-7 Eh/Bohr` for Standard; public Fast uses its declared `1e-5 / 1e-4 / 3e-5`
budget. Translation must cancel within `2e-8 Eh/Bohr`. The test prohibits the
existing CPU scientific derivative fallbacks. Existing independent native
auxiliary-g block/weighted-derivative tests cover Cartesian, spherical, mixed
representations, signed contractions, tail tiles, budgets and finite
differences. Metadata tests retain rejection of orbital g, auxiliary h,
higher derivatives and incompatible native handle roles.

Raw local qualification and build/cache receipts are retained under
`/data/jzzeng/df-g-auxiliary-20261020/receipts` on n1. All real-device runs use
finite Slurm allocations; these correctness checks are not README timings.
The previous default benchmark's failed DF receipts remain observations of
their original source and must not be relabeled as successful new-source runs.

Final qualification: 24 independent complete endpoints passed (20 water/open-spin
cases plus four tetramer HF/PBE cases), and all 14 raw auxiliary-g derivative
cases passed. The qualified core SHA-256 was
`bc046d6c64b870bbe504e5043295a7f1d7679985813db4d84961dd4ff75531b5`.
Host admission/ABI/dependency checks also passed. Subsequent warm-density
optimizations have separate qualification and binary identities.

## References

- [Original g value boundary](../numerics/2026-10-03-df-auxiliary-g-values.md)
- [Six-term nuclear sink](../numerics/2026-10-04-native-auxiliary-g-derivatives.md)
- [Full-DF HF derivative adapter](../numerics/2026-10-13-full-df-cbs-reference.md)
