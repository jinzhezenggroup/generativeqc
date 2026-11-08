# Decision: retain current-source Libxc evidence with a 192 KiB cap adjustment

Status: implemented in this change
Date: 2026-11-09

## Reason and decision

The PR requested to complete #1118 must retain fresh source/binary-bound
independent numerical receipts without rewriting the preceding measurements.
The compact campaign, resource acceptance and reproduction README require
247,680 bytes (about 242 KiB).
Integrating master `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95` leaves
78,392 bytes under the existing 85 MiB aggregate cap (89,050,568 retained bytes).
A 192 KiB adjustment, from 89,128,960 to 89,325,568 bytes, accommodates the new
campaign without changing historical bytes. An intermediate master revision
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
The resulting indexed benchmark total is 89,298,248 bytes, leaving 27,320 bytes
under the proposed cap. The earlier published campaign remains byte-identical.
