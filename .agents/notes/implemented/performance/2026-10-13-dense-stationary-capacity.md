# Decision: reserve indexed force storage only for indexed consumers

Status: implemented; large endpoint qualification pending
Date: 2026-10-13

## Problem

After the SCF map upload fix, issue #2416's 96-atom r2SCAN-3c
calculation reaches a converged SCF state, exposing a second failure:
`stationary additional-host byte budget exceeded`. Its 1088 AOs exceed the
ordinary force owner's 256-MiB host allowance even at a one-point tile.
The grid planner always reserves full-capacity indexed-density packing and
validation, although a dense fallback could serve the same force when indexed capacity is unavailable.

## Decision and invariants

Pass the actual AO-map demand into dry grid admission. Without a cutoff,
plan and construct the existing dense CudaGrid contract, with no indexed AO
capacity. Both prepared and one-shot construction consume the selected plan's
capacity. Requested maps retain the complete indexed allowance. The density,
point/AO domains, precision, derivatives and 256-MiB host/512-MiB device limits
are unchanged. The native integral provider keeps its concurrent reserve.

This removes actual optional packing buffers as well as their planning charge;
it does not discount storage that will still be allocated. Prepared identities
already bind the cutoff, tile and resource inventory. Explicit insufficient
host/device budgets continue to reject before native allocation.

## Evidence and limits

The production dry admission accepts the dense 1088-AO case under the existing
caps, rejects its indexed counterpart and rejects a host budget one byte below
the dense bound. Existing ECP, publication-failure cleanup, geometry-resource
and tile-search tests retain their gates. Source-matched complete GPU forces
and the 12-call endpoint remain required before claiming issue resolution.

## Dense read-only feature leases

Planning alone was insufficient: the old native/Python feature lease required
an indexed grid even when no AO map was selected. Source `877775209` makes the
dense identity map explicit and lends its features and contracted density jets
without allocating local potential/scatter storage. Writing consumers still
require an indexed owner; explicit AO maps on dense plans fail admission.

The real-device test compares dense, identity, subset and empty selections with
independent stored CPU AO/density arrays at 7/3/1 points, including lease expiry
and dense scatter rejection. All four cases pass after correcting the fixture's
expected dense map-admission error (`233349989`, test-only). The initial fixture
failure is retained. Complete 3/12-atom M06-2X and r2SCAN-3c populations also pass.
This is a read-only capability extension, not a change to the force mathematics.

## Public auto-selection correction

The first 96-atom complete rerun at `877775209` still failed: public policy
requested a qualified indexed producer before capacity admission. The small
read-only lease test and dry dense layout test alone could not expose this.
The 48-atom public population passed all 12 calls (89.351-s cold, 27.071-s warm
median); the 96-atom direct call reached converged SCF in a 485.120-s failed
complete attempt and retained the host-capacity rejection.

The runtime now tries the complete indexed tile search first, then the complete
dense search if optional maps cannot be admitted. It validates cutoff/cache
arguments before either search, preserves explicit tile requests and every
resource/work cap, and records the original indexed rejection and requested
cutoff in telemetry. This retry is dry; it never repeats native execution or
hides a failed GPU operation. Map owners and prepared identities use the chosen
actual dense/indexed capacity. Full large-endpoint qualification remains required.

## Keep indexed features without unused host potential staging

A second capacity reduction preserves the sparse consumer: `features_only=True`
adds an explicit grid capability that excludes host local-potential validation
and scatter/publication buffers from the host bound. Indexed density gathering
is performed on the device. The force owner only reads features and contracted
AO jets, so it selects this capability in both prepared and one-shot construction.
`scatter` and `xc` reject before inspecting or converting a host array. All native
device allocation charges and full-density host validation allowances remain.
The capability is bound into resource/prepared identities; ordinary grids keep
their old writable contract. At 1088 AOs the complete indexed force layout now
fits the original caps without changing the AO cutoff or omitting any device
allocation. Dense fallback remains available if even this indexed layout cannot
fit. Host gates cover both actual complete layouts and rejection before packing;
real-device and complete endpoint qualification are still required.
