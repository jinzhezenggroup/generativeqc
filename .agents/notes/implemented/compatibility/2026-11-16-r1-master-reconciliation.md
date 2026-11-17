# R1 master reconciliation without fresh campaigns

Status: implemented
Date: 2026-11-16

## Decision

Merge master `dab777e383c31c13cbd145b2ddefb27c93b9f685` into the R1
branch at `f68543214e80ef051efe25f26234d23a83b35be4`. Keep master’s
ordinary-CI policy from #3103: no standalone fresh Libxc acceptance gate.
Retire the [whole-file test-registration exemption](../architecture/2026-11-16-libxc-test-registration-scope.md).
This is not an assertion that changed production inputs match old measurements.

## Retained provenance

Both original native qualification manifests and complete compressed snapshots
are retained byte-exactly (manifests are losslessly compressed), and both
historical chains remain indexed. Existing campaign, resource, failed/rejected
endpoint, timing, and numerical receipts are untouched. The immediate predecessor
summary is extracted from the actual R1 head, not from an older accepted state.

The existing offline checker ingests all original campaigns against the merged
source inventory. Its exact generated transition tokens explicitly acknowledge
20 stage demotions across GGA_X_APBE, LDA_C_VWN_4, MGGA_X_GVT4 and MGGA_X_LTA.
The generated merged catalog has zero dependency-current products, 55 blockers,
and public DFT 0/222. This classifies old qualification as historical; it neither
invalidates the original measurements nor manufactures replacement results.

## Boundaries

No new GPU run, numerical threshold, force scope, source-hash exemption, or
capability promotion is introduced. The offline strict checker still exists.
Immutable receipt replay and independent numerical checks remain required.

## Final integration after #3095

The final integration uses actual merged master
`04db8c126ce7084d86866df979a885f4e4919bdc`. The approved aggregate budget
is exactly 95,845,458 bytes, with no extra headroom or other relaxed limits.
Regenerate the Libxc historical view from the same original f6854321 accepted
summary and the unmeasured CUDA AOT audit from actual merged source inputs.
Neither operation adds execution evidence or promotes current qualification.
