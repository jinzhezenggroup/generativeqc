# Decision: forward-mode analytic Becke partition derivative by default

Status: implemented
Date: 2026-11-19

## Problem

The phased stationary first derivative differentiates the normalized Becke
partition in reverse mode:

1. A pair phase evaluates each unordered pair's switch and logarithms and
   writes a four-word pair panel.
2. Atom phases form log products and normalize.
3. A reverse pair phase pulls back through the panel.
4. A gather sums incident pairs.

On H100, README Direct r2SCAN-3c96 spent about 440 ms of each warm call in
these seven launches per tile, over 1056 tiles. GPU4PySCF's forward-mode weight
derivative takes about 80 ms for the same work. r2SCAN-3c96 warm was the last
README warm loss (4.02 vs 3.68 s in the three-process cohort, 3.76 s here
before this change).

Earlier attempts did not pay or did not apply:

- Fused pair phases were equal or slower; pair evaluation is compute-bound.
- Primitive normalized adjoints and coefficients gave no gain.
- Binary-scaled products are ineligible for Treutler partitions.
- Radius-adjusted scaled products were rejected
  (`../../rejected/2026-11-09-radius-adjusted-scaled-becke.md`).

## Decision

Phased-storage owners now take a forward-mode route by default. It keeps the
phased distance phase, the generated `point_pair` switch and coordinate partials,
and the output layout. It replaces the six pair, atom and normalization phases
with four lanes:

1. **Products.** A (points, atom X) lane forms P_X = prod_{Y!=X} s_XY in the
   phased neighbor order. The product is a frexp-normalized mantissa and
   exponent of the nonzero factors, plus a zero count.
2. **Normalization.** One warp per point scales the products by the largest
   nonzero exponent and stores 1/sum.
3. **Motion.** A (points, atom G) lane, for G not the owner, sums over B the
   pair terms
   `(P_B/s_BG - P_G/s_GB) s' (c1 u_G - c2 e)`.
   - The expression has the same form in both pair orientations.
   - The u_G coefficient and the e vector are accumulated separately.
   - The owner pair is evaluated once more after the loop.
   - The lane writes `seed * dw/dG`.
   - One reciprocal of `s_BG s_GB` serves both scales while it is normal;
     otherwise exact division and the single-zero excluded product apply.
4. **Owner.** One warp per point writes `-sum_G` to the moving owner, by
   translation invariance.

Each (points, atom) block tables that atom's elemental Treutler adjustments
against every neighbor in shared memory. Owners above
`stationary_becke_normalize_max_atoms` atoms with radii keep the per-pair
adjustment through a separate fallback instantiation.

The motion kernel carries `__launch_bounds__(128, 6)`. At 76 registers, a
96-atom, 1024-point tile (768 blocks) fits in one wave on 132 SMs. At 86
registers it took 1.16 waves.

`GENERATIVEQC_STATIONARY_BECKE_ROUTE=analytic|phased` is captured at owner
creation; unset means `analytic`, `phased` is the rollback, and other values
reject. Several things still run the pair phases:

- primitive-adjoint selections;
- FP32 geometry batches, which use the bounded kernels as before;
- owners without phase storage, which keep the bounded kernels.

Binary-scaled products are a pair-route lowering. Their admission reports
inactive under the analytic route.

Work accounting follows the executed schedule:

- `becke_pair_state_evaluations` counts `2 * points * na * (na - 1)` switch
  states for analytic tiles, since the product and motion lanes each evaluate
  every ordered pair.
- Phase metrics count analytic tiles without pair domains or panels, at five
  launches per tile.
- `stationary_becke_analytic_metrics_v1` reports the selected route, the
  analytic tiles and points, and the switch states.

## Rejected alternatives

The route reached its final form in three H100 iterations:

- **v1** used per-call adjustments and serial normalization and owner lanes.
  It gained only 40 ms on r2SCAN-3c96.
- **v2** added the per-block adjustment table and warp folds, saving about
  110 ms. The motion kernel still ran 1.16 waves at 86 registers.
- **v3** split accumulation, kept the owner term out of the loop, hoisted the
  lane distance and product, and used the launch bound. It saved about 260 ms.

Other designs were considered and not taken:

- **Storing switch states for the motion lanes.** That is the pair panel again;
  pair evaluation is cheap compared with its traffic.
- **GPU4PySCF's two-reciprocal form without exact-zero semantics.** It would
  change single-zero and multiple-zero behavior that the existing derivative
  gates pin.
- **Larger or different KS force tiles.** They were measured slower on
  2026-11-18 with the pair route.

## Invariants

The following must be preserved:

- Exact-zero semantics. A single zero factor uses the excluded product;
  multiple zeros, and saturated switches (`s' = 0`), contribute exact zeros.
- Product order. The products multiply in neighbor order. Normalization and
  owner folds are fixed lane-strided warp reductions, so results are
  deterministic but not ordered left to right.
- Storage. The route uses only the already admitted phase reservation, and it
  never publishes after an invalid primal.
- Route capture at owner creation, and the rollback.
- Truthful work metrics for both routes.

## Evidence

`benchmarks/results/h100-stationary-becke-analytic-20261119/` (see its README).

- **Same-binary H100 A/B.** README Direct r2SCAN-3c96 warm improves by 0.25-0.28 s,
  and PBE0 Direct96 and B3LYP DF96 warm by about 0.30 s each. All 14 README DFT
  method/mode combinations at 96 and 48 atoms qualify on both routes, with
  identical iteration counts.
- **Host tests.** The analytic harness matches the phased reverse route to rtol
  1e-11, with and without adjustment or cached centers, up to 128 atoms.
  - It passes the shared edge cases: rounded zeros, signed zero seeds,
    coincident centers and point collisions.
  - It matches an independent Decimal finite-difference oracle and the
    translation and permutation invariances.
  - Dispatch, resource, accounting and stub contracts cover selection,
    fallback and metrics.
- **Device tests.** The shared-owner tests run under both routes, including
  the Treutler and binary-scaled wrappers, on the measured source.
  compute-sanitizer memcheck, racecheck, initcheck and synccheck run on
  analytic 96-atom owners.

## Consequences

The analytic route evaluates four times as many switch states as the phased
panel, but moves no pair data. It is compute-bound, about 185 µs per README
r2SCAN-3c96 tile, down from about 410 µs for the phased route. The planner still reserves the pair panel; the analytic route uses only
the distance, product and output fields.

## Revisit when

- The phase planner can admit analytic owners without the pair-panel
  reservation. That saves memory, not time.
- A non-H100 target needs a different launch bound, or its tiles no longer
  fit one wave.
- A Becke Hessian or response route needs these lanes. This is a
  first-derivative route only.

## References

- `docs/developer/dft_grid.md` (generated Becke atom adjoints).
- `../performance/2026-10-12-becke-binary-scaled-products.md`.
- `../../rejected/2026-11-09-radius-adjusted-scaled-becke.md`.
- GPU4PySCF `gpu4pyscf/lib/gdft/gen_grids.cu`, `GDFTgrid_weight_derivative_kernel`
  (forward formulation, for comparison only).
