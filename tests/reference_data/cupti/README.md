# Scheduled scientific residency replay inputs

These are permanent, CPU-replayed regression fixtures, not routine run archives.
Each small archive contains only one test's exact raw source/API/activity inputs
and scientific numerical controls. Readers never extract paths or trust saved
diagnostic verdicts. No CUDA binary is retained or executed by these tests.

- `n4-1506`: H2/LiH/water/F-HeH generated-DF compatibility with independent
  energy/amplitude fixtures, consumed by `test_cupti_df_compatibility_replay.py`.
- `n4-1511`: H2/LiH/water actual metric host-transform links and RHF/UHF prepared
  resident-boundary contracts, consumed by `test_cupti_qualified_production_replay.py`.

Every member is byte-preserved from the recorded source capture. Per-member and
container hashes, original archive identities, and historical/local recovery
locations are recorded in
`manifests/evidence_migrations/mp2-residency-20261106.json`. This explicit storage
migration replaces oversized mixed benchmark bundles without dropping numerical,
failure, callback or independent-reference coverage. Non-test full run/build
artifacts remain in ignored local storage and, for job 1506, existing Git history;
they are not moved here to evade benchmark-retention guards.
