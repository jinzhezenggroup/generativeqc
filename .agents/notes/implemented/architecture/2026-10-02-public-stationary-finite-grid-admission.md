# Decision: finite public stationary work with bounded submission windows

Status: implemented; real-device numerical/performance qualification pending
Date: 2026-10-02

## Problem

At master `2aae79a21a58fe63540721928c31beffca07cb2f`, the generic public CUDA
stationary wrapper retained 32-atom/128-AO/4096-primitive diagnostic limits, while
its native source owner and compiler allocation planner supported
128 atoms/1024 AOs/16384 primitives. Public calls also inherited 1M-point and
100M-total-pair-work diagnostic guards. The frozen 24-atom PBE0 grid already
requires 325,583,124 pair visits. Merely raising those constants would hide total
work and could select an impractical bounded-storage AO^4 fallback.

## Decision

Keep scientific semantics and all native allocation limits unchanged. A shared,
method-neutral compiler plan records finite complete point/tile/chunk/pair work.
Public complete-force execution explicitly declines the diagnostic whole-grid
work guards, but retains mandatory per-window work and tile limits and the
existing additional host/device numeric budgets. The native kernel equations,
SCF provider, derivative source, point order and resident-grid identities are
unchanged. Each submission window drains the same native geometry owner; no
per-window graph/source specialization or coordinate-list materialization exists.

Enlarged domains require the existing prepared native integral derivative
provider. Capability is checked before compilation/allocation; actual provider
availability is checked before geometry/task submission. Failures cannot silently
select the old AO-descriptor fallback. Small-domain diagnostic fallback remains.
Both native point counts and pair counts are checked before result publication.

The benchmark retains normalized per-endpoint force work rather than only the
last moved-warm sample. Planned capacity/work remains separate from executed
counters; missing measurements remain absent/null.

## Rejected alternatives

- Raising every cap: conflates total work with bounded capacity and hides cliffs.
- Removing guards while accepting any bounded-memory AO fallback: an AO^4 task
  loop may be technically finite but unusable at hundreds of AOs.
- PBE0-specific planner, water special cases, new derivative algebra or copied
  native integral executor: duplicates shared compiler/HF/DFT ownership.
- Extending/promoting cooperative Becke beyond 32 atoms: no independent evidence.
- Coarse grids, DF, lower precision, relaxed tolerances or unconverged references:
  change the requested problem or invalidate the acceptance evidence.

## Evidence and limitations

Host CPU NativeAO packing of the frozen spherical def2-SVP H/O basis gives
24/48/96 atoms, 192/384/768 AOs and 176/352/704 packed primitives. The actual
production resource-admission expressions yield additional-host numeric bounds
24,714,848 / 65,054,240 / 197,047,712 bytes, all inside the unchanged 256 MiB cap, including explicit concurrent Direct
paired-host staging reserves of 328,064 / 1,245,888 / 4,851,008 bytes.
The geometry/source owner reserves 2,229,248 / 4,938,496 / 13,978,880 bytes at the
ordinary 256-point tile. Its allocations and shared one-electron reserve are
host-compiled directly from the native sources, including exact/cap-minus-one
admission and large-shape arena offsets. This is allocation proof, not a GPU run.

Tests cover complete census, tail/order/offset preservation for host and resident
grids, profile and ordinary routes, deterministic stops on deferred failures,
provider failure/no-budget fallback rejection, explicit diagnostic caps, native
128/1024 boundaries, and the unchanged cooperative fallback. Existing stationary
source/compiler/host suites remain applicable.

No NVIDIA device/NVCC was available for this change. The historical independent
24/48/96 PBE0 references did not converge in 100 iterations; this change neither
relaxes nor solves that separate validation problem. The exact current-head GPU
ladder is in `docs/maintainer/stationary_large_domain_qualification.md`.

## Consequences / revisit

Extra chunk error fences can add overhead. No speedup or large-case numerical
pass is claimed; retain complete endpoint comparisons before tuning these bounds.
21.5 billion grid pair visits at 96 atoms remain real work. Further optimization
must use measured profiles while preserving complete scientific coverage.

Refs #1481, #1186, #1187, #1659, #1672, #1684, #1685.

Agent: dot
