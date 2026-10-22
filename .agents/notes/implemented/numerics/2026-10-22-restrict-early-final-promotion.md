# Decision: restrict automatic final-state probing to qualified models

Status: implemented
Date: 2026-10-22

## Problem

The broader admission in the [original stopping decision](../performance/2026-10-22-final-state-aware-ks-stopping.md)
included M06-2X despite a retained default-grid RMS force error above the
unchanged README gate. Default and rollback agreed, so this was inherited
grid error, but agreement alone did not qualify a new automatic numerical path.

## Decision and invariants

Automatic probing requires the complete canonical PBE, PBE0, B3LYP or r2SCAN
electronic composition on the default grid. Generated split hybrids, custom
mixing and other grids keep the incumbent. Existing provider, precision,
spin, resource and final canonicality gates remain mandatory. Explicit
`GENERATIVEQC_KS_EARLY_FINAL_PROBE=1` is experimental opt-in; zero restores
the incumbent. No accuracy threshold is relaxed and failed samples remain
unranked. The ordinary stopping predicate is shared by production and host
behavioral tests rather than extracted from a local assignment block.

## Evidence and revisit conditions

The original M06-2X/3 errors and rollback comparison remain in the earlier
note and published evidence. Host contracts cover complete model identity and
strict stopping. The automatic domain reuses the existing independently qualified
geometry/force evidence; the guard does not change its arithmetic. Timings retain
their original source identities and are not relabeled as measurements of this
restricted policy. A future timing refresh must measure its own source revision.

Extend automatic model admission only with independent acceptance on its
full intended numerical domain. Do not use a size whitelist to conceal the
known small-system failure, or loosen the common energy/force gates.
