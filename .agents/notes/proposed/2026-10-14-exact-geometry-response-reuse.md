# Proposal: exact density-keyed composite geometry response reuse

Status: experimental, default-off; endpoint qualification pending
Date: 2026-10-14

The retained expanded-cache WB97M-V/96 diagnostic spends about 22.8 seconds per
warm call in the semilocal/nonlocal geometry pass. Its density is repeated while
its weighted density changes. Whole-force caching would conflate these sources
and incorrectly reuse Pulay response.

The optional `GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE=1` route retains only
successfully completed nuclear and semilocal/nonlocal geometry components in
the existing immutable prepared geometry/operator/grid owner. Every call still
authenticates the native snapshot and evaluates current integral/Pulay sources.
All spin-density bits form the key; signed zeros and one-ULP differences miss.
Geometry, grid, operator, resource or owner changes rebuild and clear retention.
Failure and explicit disable also clear it. Active-AO experiments keep executing
their own routes, as do explicit stationary execution controls. No new host density export or reference calculation is added.

One host slab owns the density and component vectors. Its admission includes
NumPy's largest boolean comparison/finite-check temporary and is bounded by
64 MiB, one quarter of the host allowance and remaining admitted host capacity.
Optional allocation/lookup failure returns to ordinary computation. Hits emit
zero executed geometry/traffic/pair-sweep counts, not the producer's old work or
timing receipts. Native one-electron and integral work retain separate owners.

Host tests cover exact/changed keys, shape, finite output, immutable retention,
capacity rejection and allocation recovery. Independent E/F/displaced-energy,
actual device hits, moved geometry, failure isolation and original-protocol
complete populations remain required before any default promotion.
