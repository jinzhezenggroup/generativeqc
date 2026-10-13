# Decision: reserve indexed force storage only for indexed consumers

Status: implemented; large endpoint qualification pending
Date: 2026-10-13

## Problem

After the SCF map upload fix, issue #2416's 96-atom r2SCAN-3c
calculation reaches a converged SCF state, exposing a second failure:
`stationary additional-host byte budget exceeded`. Its 1088 AOs exceed the
ordinary force owner's 256-MiB host allowance even at a one-point tile.
The grid planner always reserves full-capacity indexed-density packing and
validation, although this workload has no qualified force AO-map request.

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
