# Decision: resident VV10 pair grids, Direct nonlocal tiles and cold VV10 anchors

Status: implemented
Date: 2026-11-17

## Problem

On H100 water-96 Direct WB97M-V the README cohort (`a7241b4fd`) measured a 48.3 s
cold endpoint against GPU4PySCF's 40.7 s, the only remaining WB97M-V loss (#3004).
Three independent causes were found and measured in one H100 Job; evidence is in
`benchmarks/results/h100-wb97mv-cold-20261117/`.

## Decisions

1. **Resident VV10 pair grids.** The bilateral pair kernels statically interleave
   a fixed tile-pair inventory over a hard-coded 680-block grid, exactly one wave
   of four 128-thread blocks on a 170-SM RTX 5090. On a 132-SM H100 the surplus
   blocks run as a partial second wave. The grid is now the executing device's
   multiprocessor count times the compiled kernel's occupancy
   (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`); no device names.
   `GENERATIVEQC_VV10_PAIR_GRID=fixed` restores 680. Same-image ABBA: -1.86 s cold.
2. **Direct nonlocal SCF tiles.** The automatic nonlocal SCF tile admitted only
   fitted owners on full-rate-FP64 devices (512); strict fused Direct owners fell
   back to 256. They now use 1024 under the same AO/grid/free-memory/device gates.
   Fitted owners keep 512, low-rate FP64 devices and resource promises 256.
   Measured: -2.6 s cold, -0.2 s warm with unchanged SCF work.
3. **Cold VV10 anchor.** Qualified strict Standard owners (Direct and fitted) no
   longer solve the semilocal/RSH-only provisional problem for a native cold
   guess; the first target evaluation anchors VV10 on the guess density, as
   transported densities already did. The seed history was discarded at the VV10
   handoff anyway. Water-96 Direct: 16 → 13 Focks, same four pair sweeps, -2.0 s.
   Mixed, budget, strict and other owners keep the seed stage;
   `GENERATIVEQC_VV10_COLD_SEED_STAGE=1` restores it.

Physical convergence, canonical final-state proofs, force gates, grids, screening
and precision are unchanged. All processes pass the original independent gates.

## Result and remaining gap

Final source: water-96 Direct cold 39.36–39.51 s against GPU4PySCF 38.07–40.03 s
(median paired ratio 1.010, parity), warm 1.6× faster; water-48 Direct 14.75 vs
16.49 s; DF water-96 23.90 vs 35.46 s. Water-12 Direct is within noise of the old
seed stage (one fewer Fock, one more pair sweep).

## Rejected alternatives

- **No resident range source** (`GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=0`):
  156.9 s cold. Direct paired range exchange is far slower than replaying the
  composite source.
- **A GPU-model or SM-count whitelist for the pair grid or tiles.** Occupancy and
  the existing FP64-throughput capability gate express the same facts portably.

## Revisit when

- A clear cold win on water-96 Direct is needed: GPU4PySCF screens VV10 points with
  |w| <= 1e-14 (about 9% of points, 17% of pairs). Adopting it changes the
  molecular density-domain policy and must cover CPU, CUDA, oracle and Hessian
  consumers together.
- A device with a different occupancy profile regresses: the explicit `fixed`
  grid selector provides the comparison.
