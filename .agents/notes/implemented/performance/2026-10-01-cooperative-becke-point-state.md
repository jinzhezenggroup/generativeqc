# Decision: bounded cooperative Becke point state

Status: implemented, opt-in pending NVIDIA qualification
Date: 2026-10-01

## Problem

After #1672 distributes independent point workers and #1684 reuses center-pair
geometry, each worker still evaluates point-dependent Becke pair factors twice.
For 12 atoms that is 66 pairs in each of the forward and reverse passes. A
cooperative schedule must not reinterpret point-worker counts as CUDA threads,
change the discrete model, or hide its shared-memory requirement.

## Decision

The internal stationary CUDA owner accepts `cooperative_becke=True`. The compiler
admits one 32-thread block per independent point worker only for 2–32 atoms and a
sufficient target shared-memory limit. This remains opt-in: host tests cannot
establish NVIDIA performance or justify a default-promotion decision.

Atom distances and both pair passes are distributed over the block. Each forward
pair retains the generated ratio partials, clipped factor and slope, and the two
sides' generated log values/partials in an eight-double `PointPair`. Reverse work
consumes the retained state exactly once, then reuses four fields for edge
pullbacks. Per-atom gathers follow each atom's triangular traversal subsequence;
an ordered point-motion pass avoids racing updates to the selected owner center.
No floating-point atomic reductions, pair screening, or approximate formulas are
introduced. Summation/rounding equivalence is tolerance-gated, not asserted to be
bitwise invariant across CUDA schedules.

The generic and cooperative schedules share pair construction/adjoint,
normalized-product adjoint, and final point-motion helpers. Local norm, ratio,
log, and Becke polynomial algebra still comes from the existing scalar graphs.
The scalar fallback retains lazy reverse log evaluation, including the saturated
slope early exit. The AO/XC contraction is one shared generated device function;
only one thread runs it in the cooperative route. Parallelizing AO work is not
part of this slice.

## Resource and lifetime contracts

- `geometry_lanes` remains the number of independent point workers and retained
  partial/scratch panels. It is never a total thread count
- Shared admission is `16 + 64 * natom * (natom - 1) / 2` bytes per block, including
  conservative control-state alignment: 4,240 bytes at 12 atoms, 31,760 at 32
- Shared storage is on-chip launch storage, not an additional persistent device
  arena. Point-lane selection and the arena's bounded 8 MiB scratch policy remain
  unchanged; the retained center cache remains independently optional
- Native configuration rechecks actual device thread, grid and shared-memory caps.
  A cap mismatch retains the admitted generic route. Invalid resource contracts
  fail closed. Configuration is legal only before topology setup
- All eight geometry consumers enter one native launch dispatcher. Tail launches
  and reductions use live point workers; empty tiles never launch or read stale
  panels. Borrowed streams, sticky producer errors, reset, and transactional
  publication retain the existing owner rules
- Barriers delimit point-distance, pair-forward, atom-product, pair-reverse,
  atom-gather, and point-motion lifetimes. Every error return is collective.
  Point state is never retained across a point, tile, replay, or geometry reset
- Native metrics report actual threads per point/shared bytes and cumulative
  pair-state evaluations. Per-execution metric deltas include the new counter.
  Existing logical pair visits and center-distance work counters remain separate
- Source identities hash the changed compiler/native sources; resource provenance
  records the requested cooperative plan; native metrics report actual selection

## Evidence and acceptance gates

Host-thread tests execute the actual cooperative helper with 1, 7 and 32 lanes,
iterations 1/3/5, atom counts through 32, direct/prepared center geometry,
positive/negative zero seeds, saturation, rounded single-zero factors, nonfinite
and coincident inputs, collisions, tiny/huge coordinates, canaries, empty/ragged
point sets, and geometry replacement. Independent 60-digit Decimal directional
finite differences, translation, permutation, and tile composition check the
scientific result. Instrumented 12-atom work is 66 pair-state evaluations per point
rather than 132; this is a work count, not a device timing claim.

A separate threaded harness executes the actual emitted cooperative geometry
kernel with generated Becke code and explicit AO/XC stubs, comparing direct and
prepared geometry, explicit/implicit owners, external seeds, tails, stale capacity,
and sticky errors against the generic emitted kernel. Native configuration tests
execute the actual admission function with explicit device-property stubs.

The opt-in NVIDIA lane/replay matrix covers 2/12/33 atoms (including cap fallback),
LDA/PBE/r2SCAN/PBE0, both spins, cached/direct centers, generic/cooperative schedules,
32/256/2048 point capacities, irregular/empty tails, both reset entry points, and
changed-geometry replay. Run through `tools/run_stationary_cuda_validation.py`,
including Compute Sanitizer memcheck/initcheck/synccheck. Its
`--cooperative-becke` option selects the opt-in owner inside existing complete
force tests through a test-only fixture, preserves explicit matrix choices, and
fails qualification if an eligible owner silently takes a generic device fallback.
Before promotion also run full independent-force oracles and complete cold/warm/changed-geometry endpoint
measurements on the exact integrated head. Host execution or CUDA compilation is
not a substitute for those device gates. No GPU speedup or 12-atom target wall time
is established here.

The final broad host/source suite passed 790 tests with 14 device/optional skips.
Twenty-two complete CPU analytic/reconverged-force and late-state checks also
passed, using B's existing CPU native library with C's freshly generated helpers
(the changed native owner header is CUDA-only).
The full changed-file pre-commit suite passed, including compiler/SCF/electronic
structure boundaries, CUDA ownership, complexity, formatting and evidence gates.

Independent review compiled the prior B and current C scalar native ABI and
compared 6,300 cases across iterations 1–5, 1/2/3/7/16/32 atoms, empty and ragged
point sets, direct/cached budgets, coordinate scales from 1e-300 to 1e300,
collisions, signed-zero and huge seeds. All status codes and output bytes matched
(4,100 successful cases; 2,200 rejected/nonfinite cases). This scalar compatibility
witness does not assert bitwise CUDA equivalence. The review also caught and
fixed eager reverse-log work, cumulative metric deltas, repeated configuration
fallback, and the single-atom helper's final barrier before publication.

## Rejected alternatives and revisit conditions

A fixed maximum-sized shared array would penalize small systems; dynamic pair
storage charges only the actual shape. Floating-point atomics would obscure
reproducibility and error diagnosis. Unbounded atom counts or shared allocation
would remove the explicit generic fallback. Default selection is deferred until
real-device correctness, sanitizer, resource occupancy and endpoint performance
measurements support it. A future shared compiler schedule IR can subsume the
bounded team abstraction without replacing the scientific scalar graphs.

References: #1479, #1672, #1684; `test_becke_cooperative.py`,
`test_stationary_cooperative_kernel_host.py`, `test_dft_complete_cuda.py`
