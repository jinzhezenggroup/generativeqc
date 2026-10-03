# Source-scoped AO-force-screen qualification follow-up

These are two distinct completed qualification campaigns for draft #1798,
not replacements for the original prototype publication and not incremental-KS
results from #1803. Each retains 168 native plus 84 fresh-reference complete
energy/analytic-force calls: 3/6/12/24/48/96 atoms and H2CO, with cold, five
warm, moved and five moved-warm calls. Every native call is checked against all
six matching-geometry reference calls, without a timing/iteration filter.

| Campaign | Published matching source | Native source identity |
| --- | --- | --- |
| pr-head | 81392aaca326f4d2acf4bc8b418a6a0c7fe18827 | b319ecfb7c5437621d1b9fc3bb13f51dc99c5411ccc38dc1a6803b3dac7aea45 |
| integrated | c830a0369837ccb90290de1370d2483bb65399b4 | 1db3843899d84a5c6c0529d74dd132d981a656242d1ec213a56273bd5138a621 |

Receipts retain the checkout commit actually reported by the measurement,
including dirty-base records, rather than being rewritten to these later
matching commits. Library identities and actual reference XCfun routing remain
per campaign. The integrated source includes master 701e00db7 / #1773, not later
837c2a51c or 38fc52352. Source-generation and executable hashes are distinct.

## Complete endpoint observations

| Integrated campaign | OFF | ON |
| --- | ---: | ---: |
| 48 warm median | 17.891551 s | 17.078026 s |
| 96 warm median | 76.512178 s | 73.203162 s |
| 96 moved | 255.914455 s / 12 iterations | 267.583077 s / 13 iterations |

All repeats, cold/moved costs, different SCF trajectories and regressions are
retained. These ordered runs share caches; they do not establish interleaved
endpoint causality. A force-only predicate cannot earn credit for different
SCF trajectories. The observed warm reduction is about 4.5%/4.3%, not 35%.
Actual KS Fock-count observations are recorded with their source; historical
null counters in the older publication are not backfilled.

## Work/source controls and limitations

Both source-specific producer censuses retain independent source-channel arrays
and all classes. Generic admitted AO quartets decrease from 595,220,532 to
385,596,186 on this one fixed density. These are not primitive/root/FLOP counts;
low-order weighted roots remain unobserved. Instrumented durations are never
substituted for clean timing. The separately interleaved integral-source runs
also retain every replay and independent comparison; they are not full E/F.

The exact 13,759,241-byte fixed-density text input remains ignored/local. Its
digest and independent snapshot are retained, but this is not an exact offline
census replay package. Gzip/xz of that input still exceed the current review
budget; no cap change, release upload or external backup is used. Re-solving
the molecule creates a new input identity and must not inherit these counts.

The controls record separate water/water-cation finite-difference results and
the failed pre-endpoint launcher / pending-only relocation. The original OH
SCF failures remain in `../pbe0-ao-density-force-20261004/controls.json.gz`.
Passing these completed endpoints does not resolve that convergence problem,
qualify broad UKS, promote an optional selector or constitute review approval.
Peak memory, transfers and other unobserved work are not inferred to be zero.

## Reproduction

Check out the matching published source, verify its source identity, build the
sm_120 AOT profile with CUDA 12.9.1 and explicit CXX/CUDA ccache launchers, and
set `GENERATIVEQC_LIBRARY` to the resulting library. Exact loaded-library hashes
are retained in each measurement, not predicted for a new toolchain.

Use `python -m benchmarks.readme_pbe0 reference --atoms 96 --basis-file
benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json --repeats 5 --output
.artifacts/reference.json`, then `native` with `--reference` pointing to it.
Set `GENERATIVEQC_BOUNDED_FORCE_AO_DENSITY=0` or `1` explicitly and preserve
`GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=0`. On the local Slurm machines, every
GPU invocation requires a finite `srun --partition=main --gres=gpu:5090:1`.
The retained `.py.txt` wrappers observe actual backend/binary identity and
adapt the H2CO fixture; copy to `.py` only in a reproduction checkout.

No release, default promotion, convergence closure or merge approval is claimed.
