# Fast anchor continuous default domain

For #2602, unset admits continuous **192–768 AO**, not a measured-size whitelist.
Explicit `1` retains smaller scientifically eligible opt-in; `0` disables.
Gates are unchanged; this is no optimal crossover fit or universal speed claim.

## Complete endpoints

WB97M-V/def2-SVP water-24, RTX 5090, 48×16×32 moving grid, fresh processes/owners,
2 GiB optional source allowance. Cold construction/preparation/SCF/analytic
forces/export/teardown are included, without first-use subtraction.

| Cohort, chronological order | Seconds | Focks | VV10 sweeps |
| --- | ---: | ---: | ---: |
| Screen off | 67.000341 | 18 | 3 |
| Screen on (`1`) | 37.312540 | 16 | 2 |
| Screen on (`1`) | 37.392634 | 16 | 2 |
| Screen off | 44.766241 | 18 | 3 |
| Final off | 67.663079 | 18 | 3 |
| Final auto | 37.334334 | 16 | 2 |
| Final auto | 37.357717 | 16 | 2 |
| Final off | 44.735551 | 18 | 3 |

Reverse-order pairs improve **16.47% / 16.49%**; forward pairs 44.31% / 44.82%.
These are ordered observations, not statistical bounds. Strict J/K applications
stay at 4/8; mixed J and each K source fall 14→12. Full work is in the packet.

All eight pass independent GPU4PySCF full-density/matched-grid gates. Energy
errors ≤7e-10 Eh; max/RMS force errors ≤5.25e-6/1.52e-6 Eh/Bohr. Limits remain
1e-5 Eh and 1e-4/3e-5 Eh/Bohr. Paired controls, geometry, basis and final states
are checked; oracle settings/hashes are in the packet.

## Provenance and limits

`qualification.json.xz` retains reductions, raw hashes, gates, work, state
comparisons and build bindings; its storage companion binds exact bytes.
Screening uses explicit on/off on the earlier 768-only-default build; only the
final cohort qualifies unset at 192 AO. Builds use verified sccache 0.16.0.

Exclusive node4 jobs 1143/1147 preserve Slurm visibility. The 384-AO oracle
times out at 900 s; independent coarse-seed retry job 1146 at 1000 s. Neither
publishes a reference or runs production endpoints: no accepted interior timing.
Historical 96-AO losses and the 768-AO 20.4504% win remain in
[the original packet](../fast-anchor-default-20261018/README.md).

Raw receipts/arrays/logs/reducer remain in
`.artifacts/fast-anchor-size-domain-20261018/`; remote roots are in the packet.

## Reproduction

Use `benchmarks.fast_anchor_default.reference` in finite Slurm, then authenticated
`run-endpoint.sh fast 24` in fresh `0/auto/auto/0` order. Reduce with
`analyze.reduce_sample`, `bind_samples_to_build` and `compare_pair`.
The historical six-arm CLI contract is unchanged.
