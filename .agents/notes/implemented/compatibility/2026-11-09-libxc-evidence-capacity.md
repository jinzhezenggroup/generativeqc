# Decision: retain current-source Libxc evidence with a 256 KiB cap adjustment

Status: implemented in this change
Date: 2026-11-09

## Reason and decision

The PR requested to complete #1118 must retain fresh source/binary-bound
independent numerical receipts without rewriting the preceding measurements.
The compact campaign, resource acceptance and reproduction README require
247,680 bytes (about 242 KiB).
Integrating master `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95` leaves
78,392 bytes under the existing 85 MiB aggregate cap (89,050,568 retained bytes).
The initial 192 KiB proposal, from 89,128,960 to 89,325,568 bytes,
accommodated the campaign on that base without changing historical bytes. An intermediate master revision
would have required only 32 KiB; that proposal no longer fits after integrating
the independently qualified DFT streaming evidence.

Keep this explicit policy change visible in the PR rather than weakening the
checker, silently dropping historical receipts, or migrating unrelated evidence
to make room. The earlier
`2026-11-07-evidence-capacity-85-mib.md` decision remains the rationale for the
original 85 MiB allowance, not authority for future increases.

## Invariants and verification

The hard 1 MiB per-file cap, 2 MiB incoming-change review budget, hash-pinned
large-evidence reviews and all scientific gates remain unchanged. The allowance
does not qualify a functional, relax a numerical tolerance, or authorize a
Release, tag, external backup or distribution workflow.

Run the actual staged-index retention and change checks against the integration
base. Existing aggregate overflow, invalid-policy and workflow tests retain
their behavioral checks; no size guard or scientific coverage is removed.
At the original source head, the indexed benchmark total was 89,298,248 bytes,
leaving 27,320 bytes under the initial proposal. Later master evidence raised the
integrated total to 89,358,810 bytes, including explicit stale-source reporting
in the campaign README. A bounded search of 24 lossless compression/layout
combinations could save at most 12,636 bytes, insufficient for that overflow.

The repository owner explicitly approved an additional 64 KiB, making this
change's total increase 256 KiB and the final cap 89,391,104 bytes. All historical
measurements and the new campaign retain their original bytes; unrelated
campaigns are not rewritten to fit. The final staged-index checks include any
subsequently merged evidence and remain the authority for the exact total.
With the matrix-profile evidence in integration base `c6967d7a8`, the staged
total is 89,368,739 bytes, leaving 22,365 bytes under the approved cap.

Integrating later native/compiler changes also makes the retained Libxc
campaign stale for the new tree. The prior qualified summary is preserved as
the transition baseline and its 66 exact stage demotions are acknowledged. The
ordinary catalog check verifies archival consistency; `--require-current`
continues to reject stale evidence. This storage allowance does not resolve
#1118 or qualify the later source.
