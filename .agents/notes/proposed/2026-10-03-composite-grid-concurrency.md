# Proposal: admit composite grid concurrency under the whole force budget

Status: proposed; independent device tests pass, endpoint comparison in progress
Date: 2026-10-03

## Problem

The complete WB97M-V force driver fixed AO tiles at 256 points and did not request
its existing cooperative Becke source. On the 24-atom full grid, an intrusive
profile attributed about 9.8 seconds to geometry kernels out of a 50.5-second
force. Increasing per-tile work must account for both coexisting stationary
accumulators and the complete-grid nonlocal arena; individual-owner admission
alone is insufficient.

## Decision

Move the existing composite live-capacity inventory into the method compiler's
dry planner. Prefer 1024 points, then try 256, 128, 64, 32, 16, 8, 4, 2 and 1.
Keep explicit tile requests exact or reject them. Request the existing cooperative
Becke lowering; its established resource planner retains scalar execution when
shared-memory or target constraints prevent the cooperative block.

The runtime supplies the AO owner's metadata-only capacity callback. The method
compiler must not import the DFT owner directly, load the runtime, or probe a GPU.
This preserves compiler dependency direction without duplicating AO formulas.
The native nonlocal capacity query remains authoritative, and its full-grid arena
is invariant under AO tiling. Device/host totals remain 1 GiB / 2 GiB.
The force report records selected points, tile count and planned concurrency.

## Evidence

Node1 Slurm job 5373 reused one converged full24 state (192 spherical def2-SVP
AOs, 589824 points) and measured warm force scheduling only:

| Becke | Tile | Warm force seconds |
| --- | ---: | ---: |
| scalar | 256 | 49.403 |
| cooperative | 256 | 42.855 |
| scalar | 1024 | 41.667 |
| cooperative | 1024 | 40.892 |

All forces differ by less than 5.5e-13 Eh/Bohr. This is a fixed-state scheduling
diagnostic, not a complete SCF/force endpoint or an independent oracle.
Artifacts are retained in the composed checkout's ignored
`.artifacts/wb97m-grid-geometry/` directory.

The candidate passes 33 host tests and the compiler structure checker (410
modules, zero dependency errors). Node1 job 5391 passes all six independent
complete RKS/UKS force and reconverged displaced-energy tests in 195.09 seconds.
Both candidate and baseline use the same native library SHA256
`f1253b0aa722ff119f6dc99ff61731f0da1ac473e38f53a8ef2233beb24a9c47`.
Metadata-only full-grid admission selects 1024 points for 24/48 atoms and 128
points for 96 atoms under the unchanged totals. This is capacity evidence only.
The paired full24 cold/warm comparator is node1 job 5392; its result is pending.
Every independent sample must pass 1e-8 Eh and 1e-7 Eh/Bohr gates.
Candidate source, patch, logs and results are retained under ignored
`.artifacts/wb97m-geometry/` and the authorized remote task directory.

## Rejected alternatives and limits

A fixed 1024-point tile can exceed whole-force totals. Falling back only to the
old 256-point tile can still reject the 96-atom grid, so smaller bounded tiles
remain necessary. Raising the force totals is not part of this change.
No equation, precision, grid, density mask or source sum changes. This scheduling
change does not remove quadratic nonlocal pairs or integral derivative work and
does not yet establish a large-system speed advantage over GPU4PySCF.
