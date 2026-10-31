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

## Follow-up diagnosis: response headroom, not VV10 pair tiling

A subsequent same-binary A100 diagnostic applied the separately qualified
molecular VV10 schedule and compared 2048 against 256 force points under the
same 64-MiB lane cap. One cold and one warm endpoint per arm were traced; these
are component diagnostics, not endpoint-ranking populations.

The 2048-point warm call took 75.572 seconds versus 25.964 at 256 points. SCF
remained one iteration and one Fock, taking about 2.45 seconds in both. The
larger tile actually improved semilocal geometry (2.914 to 1.474 seconds) and
nonlocal geometry (2.308 to 1.407 seconds). VV10 force pairs remained about
8.69 seconds in both arms. No optional SCF storage was retired by
`prepare_force_memory`.

The retained controller grew from 388,545,756 to 889,391,164 bytes. Warm free
headroom fell from 1,457,848,320 to 954,531,840 bytes. At 256 points the occupied
DF exchange response retained 740,556,800 bytes and recycled 2,415,919,104 bytes
of the authenticated final projection; its trace took 0.924 seconds. At 2048
points the retained factors no longer fitted alongside mandatory response
storage. The first occupied attempt was refused, and its bounded fallback
performed **4,489 fitted-panel GEMMs** over 67 auxiliary panels, taking
**52.846 seconds**. This explains the warm-only cliff despite faster grids and
unchanged SCF work. The old raw-row VV10 experiment had an additional pair cost;
the new diagnostic does not relabel those historical bytes or timings.

Automatic force batching must therefore account for retained storage across
complete replays and the minimum space needed to avoid repeated fitting. Merely
checking that the grid arena fits, or freeing SCF caches opportunistically, is
insufficient. Consider sharing dead grid scratch with the subsequent response,
or preserving response headroom in joint admission. The existing authenticated
projection recycling already works; it cannot eliminate the separate live
occupied-factor and metric storage requirement. Do not globally raise the grid
cap or add a functional/GPU-name exception to hide this resource interaction.

Raw diagnostic receipts, work counters, source snapshots and the rejected
candidate build are under `.artifacts/vv10-native-mask-20261031/`. Production
keeps its original tile and geometry-memory limits.
The [compact follow-up evidence](../../../benchmarks/results/native-vv10-molecular-20261031/README.md)
retains the original diagnostic receipt/source/trace hashes and aggregated warm
phase/work observations separately from the qualified three-repeat populations.

## Follow-up: remove dead metric scratch before joint grid planning

The [occupied-response workspace decision](../implemented/performance/2026-10-31-occupied-response-metric-workspace.md)
eliminates three unused metric matrices in the admitted explicit-root consumer.
At the same 954,531,840-byte warm headroom, the 2048-point/64-MiB diagnostic now
keeps its occupied factors: 0.921 seconds in that response and 23.626 seconds
for the complete endpoint, with zero repeated fitted-panel GEMMs. The 256-point
arm takes 25.948 seconds. Both diagnostic populations pass the unchanged
independent numerical gates; ordinary cross-device ABBA controls remain within
0.2%. The old failure and its measurements above remain valid historical evidence.

This removes the demonstrated memory cliff without increasing a production cap.
The larger diagnostic tile remains unpromoted: an automatic force policy still
needs to preserve explicit tile requests and bound replay storage jointly with
response. The earlier method-wide reserve still covers raw/truncated consumers.
