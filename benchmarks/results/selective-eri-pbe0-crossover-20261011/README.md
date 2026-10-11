# PBE0 partial/full cache negative crossover

This is two exploratory ABBA pairs, not formal/default qualification.
`summary.json.gz` pins master, DSOs, exact samples, work, gates and raw hashes.
Slurm n1 jobs 7538 (endpoints/six independent GPU probes) and 7545 (completed
zero-error memcheck) preserve assigned visibility. No disabled-cache ablation:
stock master uses unset full residency; candidate uses only mask 32752.

96-atom water, PBE0/def2-SVP/RKS, 768 spherical/800 Cartesian AOs, strict FP64,
moving grid 48/16/32, screening 1e-12, energy/density gates 1e-12/1e-10.
Median complete Cold/Warm/Moved/Moved-Warm seconds:

| Arm | Cold | Warm | Moved | Moved-warm |
| --- | ---: | ---: | ---: | ---: |
| Master full | 75.560 | 13.998 | 61.543 | 14.050 |
| Partial | 112.395 | 14.022 | 61.684 | 14.027 |

Cold regresses 48.75%; sequence regresses 22.39%. ERI value storage decreases
24.80% (14,985,773,728 to 11,268,906,016 bytes), not a measured total VRAM peak.
Both arms perform 18/1/12/1 Focks; cold includes 17 resident actions and one
incumbent handoff. Partial evaluates 464,608,464 uncached ERIs per action.
All 16 E+F gates pass; maximum energy/force errors are 9.55e-12/3.79e-11.

Keep full residency default. No PBE0 speedup or partial-default claim.
Reproduce with `benchmarks.selective_eri_pbe0` and the existing independent
reference in `hybrid-phase-source-default-20261011/reference.json.gz`, using
fresh processes/finite Slurm allocations and unset versus mask 32752.
Full force vectors, traces and logs remain under ignored
`.artifacts/selective-eri-cache/pbe0/results/`; this compact negative summary
does not substitute for a future complete promotion bundle.
