# Experiment: combine bounded incremental J/K with frozen VV10 initialization

Status: default-off prototype; qualification pending
Date: 2026-10-14

The 96-atom frozen-VV10 trials still time out at the original 1800-second
whole-point limit. Their inner solves remove VV10 pair sweeps but repeat costly
range exchange. Freezing range exchange itself increases small-system target
anchors and is not promoted. A coarse-grid native WB seed also gives little
12-atom cold benefit and regresses moved work; those diagnostics remain retained.

With `GENERATIVEQC_VV10_INCREMENTAL_SEED=1`, the existing incremental Direct-J/K
policy may compose with the default-off frozen-VV10 initializer. This requires
both ordinary incremental J/K and frozen-VV10 opt-ins. J/K remains linear in the
current density, independent of the frozen nonlocal potential. The existing
screened policy still permits at most one delta build between full-density
refreshes; no tolerance or cadence is relaxed. A provisional stationary inner
operator cannot trigger the target's final energy-refinement epoch.

Range freezing remains excluded from the combination because it would reuse a
range matrix where delta reconstruction requires a fresh delta application.
The final target retains the original physical stationarity, energy, density,
consecutive full-density audit and total iteration gates. No provisional state
can become a successful warm seed or force result. This experiment needs
independent molecular E/F, changed geometry, finite-budget rejection, sanitizer
and complete original-protocol endpoint qualification before promotion.

The integration source also receives the same maximum point-count and absolute
coefficient cache guards already present in the SCF-reuse PR. Measured source
identities remain separate; earlier measurements are not relabeled.
