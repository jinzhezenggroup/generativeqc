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

HF work publication also needs an executed census: CUDA DF RHF increments
one stream-owned counter in each physical Fock assembly. Capture does not
increment it, but graph replays and inactive-lane work do. Reset occurs at
each solve; the existing allocation owner and budget include its eight bytes.
Direct HF owns a separate per-item device counter at its actual Fock builder
boundary, charged to the existing arena. Historical precision fields that
reconstruct counts from iterations are not used to certify this new export. Incomplete or discarded
attempts remain unavailable, never reconstructed from SCF iterations.

## Evidence and revisit conditions

The original M06-2X/3 errors and rollback comparison remain in the earlier
note and published evidence. Host contracts cover complete model identity,
strict stopping, missing/discarded work and failed attempts. Device tests
exercise counter capture, replay, reset and unchanged assembly arithmetic;
independent geometry/force qualification and a new complete README campaign
must accompany the changed native timing revision.

Extend automatic model admission only with independent acceptance on its
full intended numerical domain. Do not use a size whitelist to conceal the
known small-system failure, or loosen the common energy/force gates.
