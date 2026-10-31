# Decision: retain superseded H100 payloads in existing Git history

Status: implemented
Date: 2026-10-31

At base `5bf1d00a9f4de2f47e381d2c9fba66fd43bf51ba`, the 80 MiB aggregate
evidence budget left 258,147 bytes for new H100 qualification.
Move only seven historical raw/storage files to an exact, already merged Git
snapshot. Keep all original negative decisions, numerical summaries, thresholds,
source identities and test fixtures in the normal checkout. Verify stock offline
restoration before retiring any representation; add explicit historical-reader
payload arguments rather than automatic fetching or hidden fallbacks.

The old baseline's 19/30 exploratory ranking, eight slower native PBE0 groups and
incomplete small CBS matrix remain unchanged. The old CBS publication manifest is
also in the snapshot, because leaving a self-contained publication descriptor
without its sample capsule would misstate the checkout contract. Current DF
provenance and WB97MV's real offline CI input remain tracked. No budget increases,
Release assets, tags, external archive, history rewrite or numerical gate waiver.

Recovery commands and all exact byte identities are under
`benchmarks/results/h100-history-retention-20261031/`. Revisit only when one of
these raw packets becomes a required ordinary-CI or current production input.
