# Candidate: align exact spatial grid cuts with XC execution tiles

Status: candidate; opt-in, complete endpoint qualification pending
Date: 2026-10-23

## Diagnosis

The current default visits nearby centers and groups points by nearest center,
but keeps their original order inside each group. A tile can still span distant
parts of a center's domain and carry the union of unrelated AO supports. Merely
changing the point-tile size did not produce a broad complete-endpoint gain.

A separate fixed-density experiment on the unchanged production level-3 grid
measures actual complete XC E/V evaluation, including input transfer and output
publication. It compares current grouping, within-group spatial bisection and
global spatial bisection, with identical FP64 math and AO cutoff. All three
retain every point/weight pair. On water96, def2-SVP, the global screen reduces
point-weighted AO-square work from 35,027,345,728 to 22,795,954,496. On the larger
def2-mTZVPP AO basis it falls from 74,247,183,680 to 48,553,277,184. All 72
evaluations pass E/V/population parity; maximum potential error is below
9e-15. Synthetic fixed density and component timings do not establish an SCF or
force endpoint gain, and the mTZVPP screen uses the same level-3 diagnostic grid.

## Candidate and invariants

`GENERATIVEQC_CUDA_XC_SPATIAL_ORDER=2` explicitly selects longest-axis balanced
bisection with cuts aligned to the owner's tile size. Unset/1 keeps the current
center-group schedule; 0 keeps canonical execution order. Each leaf orders its
points by original index for deterministic reduction within the tile. Coordinate
ties use original index. No point, weight, basis, grid prescription, cutoff or
precision changes. The native screen used an unsorted leaf; its original
receipts remain separate from subsequent candidate qualification.

The implementation needs one point-index array and logarithmic recursion, within
the existing conservative two-array spatial allowance. It preserves resource
refusal and allocation fallback. Only the semilocal SCF execution copy changes;
the canonical grid, point owners, moving-grid force views, nonlocal consumers
and public fleet admission remain unchanged. The original native overload keeps
its symbol and behavior. No molecule or functional name selects this schedule.

## Required evidence

Extend the executable spatial owner check to both schedules, with independent
CPU E/V, changed-density graph replay, partial tiles, exact resource bounds and
canonical force view ownership. Qualify current-source Standard/Fast complete
cold/warm/moved endpoints independently before any promotion. Retain initial
compile failure (a diagnostic pointer cast), all screen observations, provenance
and semantic work separately from clean repeated endpoints.

## Completed standalone qualification

The source-frozen five-pair water96 experiment and 272 independently gated
native calls are retained in
`benchmarks/results/xc-spatial-bisection-20261023/`. Cold gains pass the unchanged
median/MAD floor: PBE 3.904% DF / 4.050% Direct and r2SCAN-3c 7.561% DF /
6.637% Direct. PBE warm changes fail the floor; retain them without a warm claim.
The selector remains explicit. Independent CPU E/V, moved endpoints, permutation
and ownership/resource tests plus all three CUDA sanitizers pass. Combined
occupied-XC/WB/schedule qualification is a separate source population.
