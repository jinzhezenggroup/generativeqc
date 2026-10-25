# Decision: qualify the fitted force bitmask at its own work crossover

Status: implemented
Date: 2026-10-25

## Problem

The fitted force policy inherited the sampled producer's 173,946,175,488
dense point-times-AO-squared crossover. Default-grid 48-atom r2SCAN-3c performs
159,559,221,248 units of that work, so it remained dense while the 96-atom
endpoint used native AO-jet bitmasks. An intrusive complete-force ledger
attributes about 3.77 seconds to 48-atom stationary geometry kernels and
0.98 seconds to density projection; these overlapping device/host diagnostics
are not additive endpoint timings.

## Decision

Use an independent `2**35` dense-work crossover for the existing fitted
`exact-jets-native-bitmask` producer, with versioned profile identity v2.
The native discovery cost differs from the older sampled producer. Preserve
the latter's policy and all other fitted selection constraints. No atom count,
method, basis or GPU-product special case enters dispatch.

Keep the existing `1e-16` AO-jet cutoff, all requested derivatives, every grid
point, grid response, final-state gates and DF integral response. Bitmask
"exactness" refers to its sampled-jet predicate, not unscreened mathematics.
Retain the 0.8 average active-fraction guard, optional 64 MiB map allowance,
complete host/device budget admission, dense fallback and geometry-bound
inventory invalidation. This changes scheduling admission, not the producer's
scientific contract.

## Evidence and alternatives

Same-binary diagnostic ablations on predecessor production source
`1a971cfc508c666f09ee8490039e74b6843c54f0`, RTX 5090 under finite Slurm,
compare default dense admission with an experimental lowered bitmask gate.
Complete cold/warm E+F observations for 48 atoms are 13.226/6.739 to
10.227/3.517 seconds for r2SCAN-3c, 7.849/3.160 to 7.420/2.674 for B3LYP,
and 6.378/2.346 to 5.968/1.860 for PBE. Maximum paired force changes are
below 2e-13 Hartree/Bohr. These one-repeat, ledger-instrumented observations
establish cause; publication uses separate clean three-repeat populations.

The experimental gate also enabled 12/24-atom endpoints. Their gains were
negligible, and 24-atom r2SCAN-3c slightly regressed, so enabling every workload
was rejected. The conservative new crossover retains these dense defaults.
Intermediate 36/42-atom experiments test the workload interval independently
of README sizes; full qualification and raw observations are retained in the
associated benchmark publication.

This supersedes only the admission threshold in
[the original fitted-force decision](2026-10-10-df-force-active-ao.md).
Revisit the crossover with complete cold/warm/moved measurements on additional
hardware/workloads; reduced projection work alone does not prove endpoint gains.

## Final qualification

[The retained publication](../../../../benchmarks/results/df-dft48-work-20261025/README.md)
contains 192 fresh README populations / 768 complete E+F calls on source
`2ed9b394a2889b3a373dc91ebbaaed03931056eb`. Six clean same-binary controls
retain three frozen-density warm calls: r2SCAN-3c Standard changes from
13.733/7.320 s cold/warm to 11.001/4.200 s, and Fast from 11.521/5.317 s
to 10.120/3.628 s. These are observations on separately allocated cards;
the earlier single-repeat ablation is not substituted for the formal cohort.
All 24 control sample pairs have equal energies/Fock counts and maximum force
change below 1.8e-13 Eh/Bohr. Selected point-times-AO-squared work is reduced
to about one quarter while grid/Becke work remains present.

All 263 related tests pass after supplying the five initially absent packaged
LDA/UKS test artifacts; the original setup failures remain in the evidence.
Independent matched-grid LDA/PBE/PBE0/r2SCAN RKS/UKS oracles, geometry rebinding
and both resource/occupancy dense fallbacks retain their gates. Eighteen
Standard/Fast moved-geometry and property-transition checks also pass. The
whole README retains its original 58 qualified / 6 failed cases per profile.

The 36/42-atom r2SCAN-3c diagnostics still spend about 4.51/8.13 seconds in
prepared DF integral derivatives, versus 1.12 seconds at 48 atoms. That
separate response-planning discontinuity remains a follow-up; its costs
are not attributed to AO masks or hidden from the ablation records.
