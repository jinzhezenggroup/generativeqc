# Investigation: stationary grid batches need independent workload planning

Status: proposed; experimental overrides are not production defaults
Date: 2026-10-31

## Measurements

A100 complete README energy-plus-force endpoints exposed substantial force-grid
batching cost. Force-only experiments left SCF tile settings and all quadratures
unchanged. Each population retained one cold and three warm calls. The combined
J/K and portable Rys candidates are separate changes; timings below include
both and are not attributable to batching alone.

| 96-atom DF method | 256 points, original cap | 1024 points, 64 MiB cap | 2048 points, 64 MiB cap | 4096 points, 64 MiB cap |
| --- | ---: | ---: | ---: | ---: |
| PBE0 | 4.250577 | 3.249567 | 3.174769 | 3.216306 |
| PBE | 3.672774 | 2.670052 | 2.580178 | 2.621811 |

The original 16-MiB geometry scratch cap cannot supply one lane per grid point
at 96 atoms and 2048 points (`144 * atoms * lanes` bytes). Merely increasing the
tile then disables the phased geometry path and regresses: the earlier separate
J/K PBE0 experiment took 6.152853 seconds at 1024 points and 8.178570 at 2048.
The expanded-cap experiment used 4096 maximum lanes and 64 MiB of geometry
scratch; public memory budgets still applied.

The larger setting is not universally beneficial. WB97M-V DF96 with the expanded
cap and 2048-point force tiles took 81.674956 seconds warm. With the clean
production tile/cap and the same Rys candidate it took 31.959681 seconds. The
earlier production control took 34.379809 seconds. Energy, force qualification,
one warm SCF iteration and one Fock build were preserved. Cold times did not
show the same large regression. A same-build clean legacy control took 34.386219
seconds, while the clean Rys candidate took 31.959681 seconds. This isolates the
large regression to the grid experiment rather than portable Rys alone. More
detailed work/cache diagnostics are required to identify its exact cause.

All 16 expanded-cap experimental populations passed the unchanged independent
README oracle gates. This establishes numerical parity for these static samples,
not a complete default-configuration or cross-device performance promotion.

## Design constraints

`KsOptions.tile_points` is an explicit public bound used by SCF and force
planning. Do not silently replace an explicitly supplied value, or ship the
diagnostic `GENERATIVEQC_EXPERIMENT_STATIONARY_TILE` environment variable.
An automatic force planner needs an explicit policy/contract and must preserve
the public memory budget, actual arena sizing and bounded allocation fallback.

The native controller currently shares `tile_points_` between semilocal geometry,
nonlocal geometry and VV10 pair workspace. Their preferred schedules and reuse
constraints need not coincide. Independent batching is a candidate design, not
yet a diagnosed fix. Keep exact quadrature identity and AO-map span semantics
when changing those loops; do not reuse a map for a different tile partition.

## Retained evidence

Raw receipts and experimental source snapshots are retained under
`.artifacts/a100-catchup-20261031/grid64/`, with independent analysis in
`grid64-all-independent-analysis.json`. The exact transfer archive SHA256 is
`1ce3bba48e2430972a1ca32fc6b8a3a5fde1f6bc7240c63345da9e914b43a192`.
Clean portable-candidate results retain separate source/binary identities.
The [compact evidence bundle](../../../benchmarks/results/ecnu-portable-df-20261031/README.md)
retains the experimental populations and independent gate outcomes separately
from clean/default integration.

Do not promote the 2048-point/64-MiB combination globally, relabel the pilot as a
clean default build, or infer a GPU4PySCF win from the narrowly faster PBE0/B3LYP
samples without repeated same-device controls.
