# Decision: Keep CUDA Direct Hessian work batched under bounded storage

Status: implemented
Date: 2026-11-22

## Problem

The #3133 RTX 5090 binary was already built with Release `-O3 -DNDEBUG`
for SM120. With a matched explicit moving quadrature it beat GPU4PySCF for
water, but lost in the warm Hessian phase for a water dimer. The profile put
Becke partition geometry and factorized second integrals at the top. A water
trimer also exposed a different, much larger launch cliff: the Python
endpoint's 64 MiB integral cap fit resident Direct jets, but not the full
resident batching plan. Retaining the unbatched jets was not useful in a
private frozen-column sweep, which never replays its last direction.

## Decisions

- Lower regular inverse norm/denominator powers after building the canonical
  CUDA geometry AD DAG. Keep the original expressions outside a range in which
  all new intermediates remain normal. Product jets still use one shared
  binary exponent; only exponent extraction and the common scale operation
  have a normal-binary64 fast path. Do not change contraction flags or switch
  to fast math.
- Dispatch the point-serial pair loops in single-warp blocks. Small default
  grids can keep per-lane scratch in at most 24 KiB of shared storage per CTA
  and dispatch the whole grid at once. Larger grids and explicit smaller
  tiles retain the previously admitted global scratch. Keeping the same arena
  plan makes this a launch/storage-location change, not a larger allocation.
- Assign each derivative output its own thread in the factorized integral
  fold. Each thread adds radial contributions in exactly the original order;
  no parallel tree reduction or floating atomic is used for the ordered sum.
- In a dense mode with an eligible frozen derivative cache, prefer canonical
  batched windows to unbatched resident jets when both plans do the same
  primitive work. Keep the resident plan when fitting it with batching, for
  paired HVP reuse, or if the optional windows cannot be admitted. The windows
  are bounded by the original byte, source, and cumulative work caps.
- Allow up to 4,096 one-electron graph tasks in the default dense mode. The
  trimer needs 2,475 tasks, above the previous 2,048-task limit. This keeps
  its existing parallel graph under the unchanged integral/workspace byte
  and work limits. HVP-only and scalar fallback schedules are unchanged.
- Compute the Treutler adjustment per atom pair, not per grid point. The
  bounded full-grid shared dispatch also admits the water trimer; explicit
  smaller tiles retain the tiled global-scratch route.
- Optionally retain public-AO `H_R` and `S_R` and the frozen weighted
  one-electron HVP for each Cartesian column. The same first derivatives
  supply the CPKS RHS and the response-weight contraction, so the primitive
  source is evaluated once per column. The cache has a separate 8 MiB cap
  inside the aggregate budget; single HVPs still use directional replay.
- Dense frozen Cartesian columns skip Direct sources whose centers all
  move together or do not move. The tagged frozen path validates the unit
  displacement and the zero response density. Later density-response terms
  use the retained directional Fock contraction rather than replaying the
  quartets. The parallel frozen-only FP64 lane reduction has a fixed order;
  arbitrary-direction actions keep the original sparse-list order.
- The AOT compiler instantiates a bounded s/p full-range ERI schedule from
  the same pair and Coulomb jets. Its axis/pair extents and Coulomb selectors
  are smaller, without new recurrences, fast math or reduced precision.
- Transpose the fixed-right semilocal TensorIR before contraction and retain
  its AO coefficients within the grid budget. When the full spatial AO jets
  are resident, dense unit columns select the necessary spatial derivatives
  and contract them in one CUDA kernel instead of publishing intermediate
  left/right/mixed AO panels. Becke weight derivatives are still evaluated
  independently for each ordered pair; the final Hessian is not symmetrized.
  The tiled AO and original forward grid consumers remain the bounded routes
  when resident storage or the adjoint is not admitted.

## Evidence

RTX 5090, Direct FP64 Cartesian STO-3G, 864 explicit points per atom, same
convergence settings and moving grid. `build/1905-gpu4pyscf/` retains timing
logs, matrices, binary hashes, compiler/cache logs, and work receipts.

- Slurm 1825 (two processes per engine/case, one cold plus three fresh-SCF warm
  calls per process): the candidate reduced dimer warm Hessian medians from
  1.741/1.763 s to 1.284/1.304 s (LDA/PBE). GPU4PySCF 1.9.0 took
  1.020/1.070 s. The candidate's warm complete endpoints were 1.421/1.479 s.
  Both native builds executed 18 directions, 9,290,376 Direct components and
  324 mixed grid contractions, with identical response action counts.
- Slurm 1824/1826 (different allocations, same node/card): trimer PBE reduced
  the warm Hessian from 128.414 s (one baseline repeat) to 19.207 s (median of
  four repeats) after windowing alone. The 64 MiB host admission plan
  went from 22,924,800 bytes without batching to 32,590,336 bytes in eight
  canonical windows. Native primitive visits stayed at 63,956,628. The
  independent CLI counted 3,024 factorized launches on the windowed route.
- Slurm 1829 alternated the final graph-cap candidate, GPU4PySCF and the
  window-only build on the same card. Trimer warm Hessian and end-to-end
  medians were 7.800/7.895 s for the final candidate and 1.561/3.397 s for
  GPU4PySCF. The previous single-warm baseline is not a paired measurement.
  The candidate still performs 27 directions, 63,956,628 primitive Direct
  components, 2,357,586 one-electron components and 729 grid contractions.
  The device receipt grew from 117,633,692 to 157,041,308 bytes to keep
  the graph, without raising the public budget. The dimer's route was
  unchanged by the graph cap.
- The raw native matrix maximum difference was below `2e-14 Ha/Bohr²` for
  the dimer and `1.1e-13 Ha/Bohr²` for the trimer window change. There was
  no postprocessing symmetrization. The node-5 dimer CPU analytic/displaced
  gradient oracle and live CUDA allocation gates passed; independent device
  geometry and Direct integral tests covered legacy/adjusted, resident,
  streamed, signed/zero/extended-domain and failure paths.

- Slurm 1832: Compact reduction entries admitted a single resident batched
  trimer schedule at 64 MiB. Its warm Hessian took 7.277 s versus 7.782 s
  without compaction. The schedule retained no windows. It made fewer
  integral launches (675 vs 3,024) but more projection/reduction launches
  (6,804 each vs 243 each); this does not measure window-source retention.
- Slurm 1834/1835: The one-electron cache and Treutler dispatch reduced
  warm Hessian medians to 1.228/1.240 s (LDA/PBE dimer) and 5.758 s (PBE
  trimer), versus 1.018/1.075/1.552 s for GPU4PySCF in the same allocations.
  One-electron primitive component counts fell from 508,032/2,357,586 to
  254,016/1,178,793 for dimer/trimer respectively; other semantic counts
  stayed unchanged. The raw matrices changed by at most `3.2e-14 Ha/Bohr²`
  relative to the compact build. Each median contains two warm samples.
- Slurm 1843/1845: Tiled then fused TensorIR transpose reduced water trimer
  PBE warm Hessian time to 2.069 s and 1.794 s. The final scope of Direct
  primitive work was 20,693,394 visits versus 63,956,628 before the frozen
  column specialization. The tiled trace showed 27 AO adjoint builds, 729
  mixed contractions, and no forward mixed-graph replays.
- Slurm 1847/1849/1850: A sparse ordered one-electron collector, specialized
  s/p compilation, and fixed frozen lane reductions lowered the trimer warm
  Hessian median to 1.752 s, 1.676 s, and 1.644 s, respectively.
- Slurm 1851: Fusing the dense Cartesian AO geometry into the transpose gave
  a 1.533 s warm Hessian pilot versus 1.545 s for GPU4PySCF and 1.643 s for
  the nonfused candidate on the same GPU. The largest raw native-to-native
  difference was `1.4e-14 Ha/Bohr²`; against GPU4PySCF it was `3.1e-8`.
- Slurm 1853: Four fresh processes per arm and case, alternating order;
  each process ran one cold call and two warm calls with fresh SCFs. Median
  warm complete Hessian stage seconds (excluding SCF):

  | Case | Tiled transpose | Final | GPU4PySCF 1.9.0 |
  | --- | ---: | ---: | ---: |
  | 6 atoms, LDA | 0.618 | 0.460 | 1.016 |
  | 6 atoms, PBE | 0.635 | 0.475 | 1.067 |
  | 9 atoms, PBE | 2.062 | 1.525 | 1.555 |

  The final raw Hessians differ from tiled transpose by at most `9.6e-14`
  and from GPU4PySCF by at most `3.2e-8 Ha/Bohr²`. The trimer executed
  27 directions, 144 physical induced Fock actions, and 729 mixed grid
  contractions. Slurm 1854 confirmed 27 AO adjoint builds, no forward mixed
  graph replays, and 52,907,904 materialized AO jet values (only the 27 right
  geometry calls). The allocated Hessian workspace fell from 238,970,772 to
  195,296,916 bytes versus the tiled transpose trimer. The tested Release
  SM120 shared library SHA-256 was
  `be9dca5389066c49fb084e68f0a17e5cda6a3e86b169a1fd7d1ef2ba45742c3c`.

## Revisit When

Do not extrapolate these workloads to ordinary force or 96-atom r²SCAN
timings. Pair geometry still repeats across left/right directions. The
trimer post-window trace (Slurm 1827) found about 109,000 nuclear-attraction
launches before the one-electron graph admission fix. This is a separate
scheduling layer from the Direct quartet source. A replacement that
batches those primitive visits or shares invariant derivatives must preserve
the live owner, ordered reductions, explicit workspace bounds, and raw
Hessian/HVP scientific gates.
